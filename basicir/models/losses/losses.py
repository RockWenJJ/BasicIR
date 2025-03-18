import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicir.models.losses.loss_util import weighted_loss
from pytorch_msssim import ssim

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

@weighted_loss
def ssim_loss(pred, target):
    return 1 - ssim(pred, target, data_range=1.0, size_average=False)

class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) loss.

    Args:
        loss_weight (float): Loss weight for SSIM loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SSIMLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * ssim_loss(
            pred, target, weight, reduction=self.reduction)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class SaturatedLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SaturatedLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, x):
        if self.reduction == 'mean':
            positive_saturation = torch.mean(torch.max(x, torch.tensor(1.0).to(x.device)))
            negative_saturation = torch.mean(torch.min(x, torch.tensor(0.0).to(x.device)))
            loss = positive_saturation - negative_saturation
        elif self.reduction == 'sum':
            positive_saturation = torch.sum(torch.max(x, torch.tensor(1.0).to(x.device)))
            negative_saturation = torch.sum(torch.min(x, torch.tensor(0.0).to(x.device)))
            loss = positive_saturation - negative_saturation
        return loss

class ColorCastLoss(nn.Module):
    def __init__(self):
        super(ColorCastLoss, self).__init__()
        # Define channel pairs (R, G), (G, B), (B, R)
        self.channel_pairs = [(0, 1), (1, 2), (2, 0)]

    def forward(self, enhanced_image):
        """
        Args:
            enhanced_image (torch.Tensor): A tensor of shape (N, C, H, W), where
                                           N = batch size,
                                           C = number of channels (3 for RGB),
                                           H, W = height and width of the image.

        Returns:
            torch.Tensor: The computed color cast loss (scalar).
        """
        # Ensure the input is a 4D tensor
        assert enhanced_image.ndim == 4, "Input must be a 4D tensor (N, C, H, W)"
        assert enhanced_image.size(1) == 3, "Input must have 3 color channels (RGB)"
        
        loss = 0.0
        # Iterate over channel pairs (R, G), (G, B), (B, R)
        for c1, c2 in self.channel_pairs:
            # Compute the mean of each channel
            mean_c1 = enhanced_image[:, c1, :, :].mean(dim=(1, 2))  # Mean of channel c1
            mean_c2 = enhanced_image[:, c2, :, :].mean(dim=(1, 2))  # Mean of channel c2
            
            # Compute the squared difference between the means
            loss += torch.mean((mean_c1 - mean_c2) ** 2)
        
        loss = loss / 3.0
        return loss

class ReconstructionLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ReconstructionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, target, J, T, B):
        I_hat = J*T + B
        if self.reduction == 'mean':
            loss = F.l1_loss(I_hat, target, reduction='mean')
        elif self.reduction == 'sum':
            loss = F.l1_loss(I_hat, target, reduction='none')
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features.
    
    Args:
        layer_weights (dict): Weights for different VGG feature layers.
        vgg_type (str): Type of VGG network to use, only 'vgg16' is supported for now.
        use_input_norm (bool): Whether to normalize input using ImageNet mean and std.
        range_norm (bool): Whether to normalize input to range [0, 1].
        loss_weight (float): Loss weight for perceptual loss.
        criterion (str): Criterion for computing perceptual loss. Default: 'l1'.
    """
    
    def __init__(self,
                 layer_weights={'conv4_3': 1.0},
                 vgg_type='vgg16',
                 use_input_norm=True,
                 range_norm=False,
                 loss_weight=1.0,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.layer_weights = layer_weights
        self.vgg_type = vgg_type
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        self.loss_weight = loss_weight
        self.criterion = criterion
        
        # Initialize VGG network
        from torchvision import models
        
        # VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Get the VGG features
        self.vgg_features = vgg.features
        self.vgg_features.eval()
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        
        # Layer name -> layer index mapping (for VGG16)
        self.layer_mapping = {
            'conv1_1': 0, 'relu1_1': 1, 'conv1_2': 2, 'relu1_2': 3, 'pool1': 4,
            'conv2_1': 5, 'relu2_1': 6, 'conv2_2': 7, 'relu2_2': 8, 'pool2': 9,
            'conv3_1': 10, 'relu3_1': 11, 'conv3_2': 12, 'relu3_2': 13, 'conv3_3': 14, 'relu3_3': 15, 'pool3': 16,
            'conv4_1': 17, 'relu4_1': 18, 'conv4_2': 19, 'relu4_2': 20, 'conv4_3': 21, 'relu4_3': 22, 'pool4': 23,
            'conv5_1': 24, 'relu5_1': 25, 'conv5_2': 26, 'relu5_2': 27, 'conv5_3': 28, 'relu5_3': 29, 'pool5': 30
        }
        
        # Define normalization using ImageNet mean and std
        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x, gt):
        """
        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).
            gt (Tensor): Ground truth tensor with shape (N, C, H, W).
            
        Returns:
            Tensor: Perceptual loss.
        """
        # Normalize input to [0, 1] if needed
        if self.range_norm:
            x = (x + 1) / 2
            gt = (gt + 1) / 2
        
        # Apply ImageNet normalization if needed
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            gt = (gt - self.mean) / self.std
        
        # Get VGG features
        x_features = self.extract_features(x)
        gt_features = self.extract_features(gt)
        
        # Compute perceptual loss
        loss = 0
        for layer_name, weight in self.layer_weights.items():
            layer_idx = self.layer_mapping[layer_name]
            x_feat = x_features[layer_idx]
            gt_feat = gt_features[layer_idx]
            
            if self.criterion == 'l1':
                layer_loss = F.l1_loss(x_feat, gt_feat)
            elif self.criterion == 'l2' or self.criterion == 'mse':
                layer_loss = F.mse_loss(x_feat, gt_feat)
            else:
                raise ValueError(f'Unsupported criterion: {self.criterion}')
            
            loss += weight * layer_loss
        
        return self.loss_weight * loss
    
    def extract_features(self, x):
        """Extract VGG features.
        
        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).
            
        Returns:
            dict: Extracted VGG features.
        """
        features = []
        for i, layer in enumerate(self.vgg_features):
            x = layer(x)
            features.append(x)
        
        return features
