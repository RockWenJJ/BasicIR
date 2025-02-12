import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicir.models.archs import define_network
from basicir.models.base_model import BaseModel
from basicir.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicir.models.losses')
metric_module = importlib.import_module('basicir.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
from basicir.models.losses.losses import ColorCastLoss, SaturatedLoss
from torch import nn
from basicir.models.losses.losses import SSIMLoss

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class UnsImageRestorationModel(BaseModel):
    """Base model for single image restoration."""

    def __init__(self, opt):
        super(UnsImageRestorationModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        # construct losses
        self.cri_recon = nn.L1Loss()
        self.cri_color_cast = ColorCastLoss()
        self.cri_saturated = SaturatedLoss()
        self.cri_ssim = SSIMLoss()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('loss_opt'):
            loss_types = train_opt['loss_opt'].pop('types')
            loss_weights = train_opt['loss_opt'].pop('loss_weights')
            reductions = train_opt['loss_opt'].pop('reductions')
            self.cri_losses = []
            for loss_type, loss_weight, reduction in zip(loss_types, loss_weights, reductions):
                cri_loss = getattr(loss_module, loss_type)(
                    loss_weight=loss_weight, reduction=reduction).to(self.device)
                
                self.cri_losses.append({'type':loss_type, 'loss':cri_loss})
        else:
            raise ValueError('loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq_path = data['lq_path']
        self.in_air_mask = torch.zeros((self.lq.size(0), 1, 1, 1)).to(self.device)
        self.in_air_count = 0
        for i in range(self.lq.size(0)):
            path = self.lq_path[i]
            if 'in-air' in path:
                self.in_air_mask[i, 0] = 1
                self.in_air_count += 1
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        clear, back, trans = self.net_g(self.lq)

        # redgrade the image
        alpha = torch.rand(1).to(self.device)
        redeg_lq = clear * (trans * alpha + 1 - alpha) + back * alpha
        new_clear, new_back, new_trans = self.net_g(redeg_lq.detach())
        # new_back = back
        # new_trans = trans * alpha
        # new_clear = clear

        self.output = clear

        total_loss = 0.
        loss_dict = OrderedDict()
        # Reconstruction loss
        l_recon0 = torch.mean(torch.abs(clear - self.lq) * (1 - self.in_air_mask))
        if self.lq.size(0) > self.in_air_count:  # normalize by underwater image count
            l_recon0 = l_recon0 * self.lq.size(0) / (self.lq.size(0) - self.in_air_count)
        loss_dict['l_recon0'] = l_recon0
        total_loss += l_recon0

        l_recon1 = torch.mean(torch.abs(clear - back * trans - self.lq * (1 - trans)) * (1 - self.in_air_mask))
        if self.lq.size(0) > self.in_air_count:  # normalize by underwater image count
            l_recon1 = l_recon1 * self.lq.size(0) / (self.lq.size(0) - self.in_air_count)
        loss_dict['l_recon1'] = l_recon1
        total_loss += l_recon1

        # The clear reconstruction loss for in-air images
        l_recon_clear = F.l1_loss(clear * self.in_air_mask, self.lq * self.in_air_mask)
        if self.in_air_count > 0:
            l_recon_clear = l_recon_clear * self.lq.size(0) / self.in_air_count  # normalize by actual in-air image count
        loss_dict['l_recon_clear'] = l_recon_clear  # weight factor can be adjusted
        total_loss += l_recon_clear

        # Gray world assumption loss for clear image (only for underwater images)
        clear_means = torch.mean(clear, dim=[2, 3])  # [B, 3]
        l_gray_world = torch.mean(torch.sum(torch.abs(clear_means - 0.5), dim=1, keepdim=True) * (1 - self.in_air_mask.reshape(-1, 1)))
        if self.lq.size(0) > self.in_air_count:  # normalize by underwater image count
            l_gray_world = l_gray_world * self.lq.size(0) / (self.lq.size(0) - self.in_air_count)
        loss_dict['l_gray_world'] = l_gray_world   # weight factor can be adjusted
        total_loss += l_gray_world

        # alignment loss
        l_clear_align = F.l1_loss(clear, new_clear)
        loss_dict['l_clear_align'] = l_clear_align
        total_loss += l_clear_align
        l_back_align = F.l1_loss(back * alpha, new_back)
        loss_dict['l_back_align'] = l_back_align
        total_loss += l_back_align
        l_trans_align = F.l1_loss(trans * alpha + 1 - alpha, new_trans)
        loss_dict['l_trans_align'] = l_trans_align
        total_loss += l_trans_align

        # uw_var_loss
        l_uw_var = self.uw_var_loss(trans, back, self.in_air_mask, self.in_air_count)
        loss_dict['l_uw_var'] = l_uw_var
        total_loss += l_uw_var

        # uw_mean_loss
        l_uw_mean = self.uw_mean_loss(trans, back, self.in_air_mask, self.in_air_count)
        loss_dict['l_uw_mean'] = l_uw_mean
        total_loss += l_uw_mean
        

        # # penalize the transmission if its variance is too low (only for underwater images)
        # trans_mean = torch.mean(trans, dim=[2, 3], keepdim=True)  # [B,C,1,1]
        # trans_var = torch.mean((trans - trans_mean) ** 2, dim=[2, 3])  # [B,C]
        # trans_var = torch.mean(trans_var * (1 - self.in_air_mask.reshape(-1, 1))) # only consider underwater images
        # l_trans_var = torch.exp(-trans_var * 10.0)  # exponential penalty for low variance
        # if self.lq.size(0) > self.in_air_count:  # normalize by underwater image count
        #     l_trans_var = l_trans_var * self.lq.size(0) / (self.lq.size(0) - self.in_air_count)
        # loss_dict['l_trans_var'] = l_trans_var
        # total_loss += l_trans_var

        # # penalize the transmission if its mean is too high (only for underwater images)
        # trans_mean = torch.mean(trans, dim=[2, 3])  # [B, 1]
        # l_trans_mean = torch.mean(torch.sum(torch.relu(trans_mean - 0.7), dim=1, keepdim=True) * (1 - self.in_air_mask.reshape(-1, 1)))
        # if self.lq.size(0) > self.in_air_count:  # normalize by underwater image count
        #     l_trans_mean = l_trans_mean * self.lq.size(0) / (self.lq.size(0) - self.in_air_count)
        # loss_dict['l_trans_mean'] = l_trans_mean
        # total_loss += l_trans_mean

        # # penalize the backscatter if its variance is too low (only for underwater images)
        # back_mean = torch.mean(back, dim=[2, 3], keepdim=True)  # [B,C,1,1]
        # back_var = torch.mean((back - back_mean) ** 2, dim=[2, 3])  # [B,C]
        # back_var = torch.mean(back_var * (1 - self.in_air_mask.reshape(-1, 1)))  # only consider underwater images
        # l_back_var = torch.exp(-back_var * 10.0)  # exponential penalty for low variance
        # if self.lq.size(0) > self.in_air_count:  # normalize by underwater image count
        #     l_back_var = l_back_var * self.lq.size(0) / (self.lq.size(0) - self.in_air_count)
        # loss_dict['l_back_var'] = l_back_var
        # total_loss += l_back_var

        # # penalize the backscatter if its mean deviates from input image mean (only for underwater images)
        # back_mean = torch.mean(back, dim=[2, 3])  # [B, c]
        # input_mean = torch.mean(self.lq, dim=[2, 3])  # [B, c]
        # l_back_mean = torch.mean(torch.sum(torch.abs(back_mean - input_mean), dim=1, keepdim=True) * (1 - self.in_air_mask.reshape(-1, 1)))
        # if self.lq.size(0) > self.in_air_count:  # normalize by underwater image count
        #     l_back_mean = l_back_mean * self.lq.size(0) / (self.lq.size(0) - self.in_air_count)
        # loss_dict['l_back_mean'] = l_back_mean
        # total_loss += l_back_mean

        # # align backscatter with complementary transmission scaled by input mean (only for underwater images)
        # input_mean = torch.mean(self.lq, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        # target_back = (1 - trans) * input_mean  # [B, C, H, W]
        # l_trans_back_align = torch.mean(torch.abs(back - target_back) * (1 - self.in_air_mask))
        # if self.lq.size(0) > self.in_air_count:  # normalize by underwater image count
        #     l_trans_back_align = l_trans_back_align * self.lq.size(0) / (self.lq.size(0) - self.in_air_count)
        # loss_dict['l_trans_back_align'] = l_trans_back_align
        # total_loss += l_trans_back_align

        # 

        # # transmision loss
        # l_trans_loss_air = ((trans - 1.0)**2) * self.in_air_mask
        # l_trans_loss_air = torch.mean(l_trans_loss_air.reshape(l_trans_loss_air.size(0), -1), dim=1)
        # l_trans_loss_air = torch.sum(l_trans_loss_air) / self.in_air_count
        # loss_dict['l_trans_loss_air'] = l_trans_loss_air
        # total_loss += l_trans_loss_air
        # # # penalize the transmision when the underwater images' transmission is 1
        # # l_trans_loss_uw = ((trans)**2) * (1 - self.in_air_mask)
        # # l_trans_loss_uw = torch.mean(l_trans_loss_uw.reshape(l_trans_loss_uw.size(0), -1), dim=1)
        # # l_trans_loss_uw = torch.sum(l_trans_loss_uw) / (self.lq.size(0) - self.in_air_count) * 0.01
        # # loss_dict['l_trans_loss_uw'] = l_trans_loss_uw
        # # total_loss += l_trans_loss_uw

        # # backscattering loss
        # l_back_loss_air = (back**2) * self.in_air_mask
        # l_back_loss_air = torch.mean(l_back_loss_air.reshape(l_back_loss_air.size(0), -1), dim=1)
        # l_back_loss_air = torch.sum(l_back_loss_air) / self.in_air_count
        # loss_dict['l_back_loss_air'] = l_back_loss_air
        # total_loss += l_back_loss_air

        # # # penalize the backscattering when the underwater images' backscattering is 0
        # # l_back_loss_uw = ((back-1)**2) * (1 - self.in_air_mask)
        # # l_back_loss_uw = torch.mean(l_back_loss_uw.reshape(l_back_loss_uw.size(0), -1), dim=1)
        # # l_back_loss_uw = torch.sum(l_back_loss_uw) / (self.lq.size(0) - self.in_air_count) * 0.01
        # # loss_dict['l_back_loss_uw'] = l_back_loss_uw
        # # total_loss += l_back_loss_uw

        loss_dict['total_loss'] = total_loss

        total_loss.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def uw_var_loss(self, trans, back, in_air_mask, in_air_count, lambda_var=0.1, min_variance=0.01):
        """
        连续型的水下图像方差损失，鼓励水下图像的 T 和 B 的方差保持较大。
        
        Args:
            trans (torch.Tensor): 形状为 (B, C, H, W) 的预测传输图。
            back (torch.Tensor): 形状为 (B, C, H, W) 的预测散射图。
            in_air_mask (torch.Tensor): 形状为 (B, 1, 1, 1) 的二值 mask，1 表示 in-air 图像。
            in_air_count (int): batch 中 in-air 图像的数量。
            lambda_var (float): 方差损失权重。
            min_variance (float): 期望的最小方差（目标值）。
            
        Returns:
            torch.Tensor: 标量损失。
        """
        # 计算 underwater mask 与数量
        uw_mask = 1 - in_air_mask
        uw_count = trans.shape[0] - in_air_count

        # 计算 underwater 图像的像素级均值和方差
        def compute_variance(x, mask, count):
            # 按 mask 计算均值（保留通道维度）
            mean_x = (x * mask).sum(dim=(0, 2, 3), keepdim=True) / (count + 1e-6)
            # 计算方差
            var_x = ((x - mean_x) ** 2 * mask).sum(dim=(0, 2, 3)) / (count + 1e-6)
            return var_x.mean()  # 返回标量

        var_trans_uw = compute_variance(trans, uw_mask, uw_count)
        var_back_uw = compute_variance(back, uw_mask, uw_count)

        # 定义连续型损失：1/(variance) 的形式
        l_var_trans = min_variance / (var_trans_uw + 1e-6)
        l_var_back  = min_variance / (var_back_uw + 1e-6)

        # 总的水下方差损失
        uw_loss = lambda_var * (l_var_trans + l_var_back)
        return uw_loss
    


    def uw_mean_loss(self, trans, back, in_air_mask, in_air_count, lambda_mean=0.1, target_margin=0.3):
        """
        连续型的水下图像均值损失，鼓励 underwater 图像的 T 和 B 均值与 in-air 图像有较大差距。
        """
        # 计算 underwater mask 与数量
        uw_mask = 1 - in_air_mask
        uw_count = trans.shape[0] - in_air_count

        def compute_mean(x, mask, count):
            # 确保返回的是每个通道的均值
            return (x * mask).sum(dim=(0, 2, 3)) / (count + 1e-6)  # [C]

        # 对于传输图，我们期望 in-air 的均值大于 underwater 的均值
        trans_air_mean = compute_mean(trans, in_air_mask, in_air_count)  # [1] or [C]
        trans_uw_mean  = compute_mean(trans, uw_mask, uw_count)          # [1] or [C]
        
        # 对于散射图，我们期望 underwater 的均值大于 in-air 的均值
        back_air_mean = compute_mean(back, in_air_mask, in_air_count)    # [C]
        back_uw_mean  = compute_mean(back, uw_mask, uw_count)           # [C]

        # 计算均值差异（注意差值可能为负，因此取 ReLU 保证非负）
        diff_trans = F.relu(trans_air_mean - trans_uw_mean)  # 希望尽可能大
        diff_back  = F.relu(back_uw_mean - back_air_mean)   # 希望尽可能大

        # 定义连续型均值损失，差值越小，损失越大；差值越大，损失越小（但不完全为0）
        l_mean_trans = target_margin / (diff_trans.mean() + target_margin + 1e-6)
        l_mean_back  = target_margin / (diff_back.mean() + target_margin + 1e-6)

        # 返回标量损失
        return lambda_mean * (l_mean_trans + l_mean_back)


    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()
        return self.output

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            try:
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

                self.feed_data(val_data)
                test()

                visuals = self.get_current_visuals()
                sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                    del self.gt

                # tentative for out of GPU memory
                del self.lq
                del self.output
                torch.cuda.empty_cache()

                if save_img:
                    
                    if self.opt['is_train']:
                        
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}.png')
                        
                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                    else:
                        
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')
                        
                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

                if with_metrics:
                    # calculate metrics
                    opt_metric = deepcopy(self.opt['val']['metrics'])
                    if use_image:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop('type')
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(sr_img, gt_img, **opt_)
                    else:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop('type')
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

                cnt += 1
            except Exception as e:
                print(e)
                continue

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

