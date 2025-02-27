import torch
import cv2
import torch.nn as nn
import numpy as np
from torchvision import transforms
from copy import deepcopy
from torch import Tensor
from typing import Dict, List, Tuple, Union, Optional

import torch.nn.functional as F
import numbers

from einops import rearrange

class IFEBranch(nn.Module):
    def __init__(self, inp_channels=3, out_channels=512):
        super(IFEBranch, self).__init__()
        
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
        )
        self.LL, self.LH, self.HL, self.HH = self.get_wave(3)
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        
    def rgb_uv_hist(self, inp_img, n_bins=32):
        """ Computes an RGB-uv histogram tensor for a batch of images.
        """
        batch_size, channels, height, width = inp_img.shape
        hist = torch.zeros(batch_size, 3, n_bins, n_bins, device=inp_img.device)
        
        for b in range(batch_size):
            img = inp_img[b]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = cv2.resize(img_np, (n_bins, n_bins), interpolation=cv2.INTER_NEAREST)
            
            img_reshaped = img_np[(img_np > 0).all(axis=2)]
            eps = 6.4 / 256
            Iy = np.linalg.norm(img_reshaped, axis=1)
            
            for i in range(3):
                r = [j for j in range(3) if j != i]
                Iu = np.log(img_reshaped[:, i] / img_reshaped[:, r[1]])
                Iv = np.log(img_reshaped[:, i] / img_reshaped[:, r[0]])
                
                hist_2d, _, _ = np.histogram2d(Iu, Iv, bins=n_bins, 
                                           range=((-3.2 - eps / 2, 3.2 - eps / 2),) * 2, 
                                           weights=Iy)
                norm_ = hist_2d.sum()
                hist[b, i, :, :] = torch.clip(torch.sqrt(torch.tensor(hist_2d / (norm_ + 1e-6))), 0, 1)
                
        hist = hist.view(hist.size(0), -1).detach()
        hist_feature = self.mlp(hist)
        return hist_feature
    
    def get_wave(self, in_channels, pool=True):
        """wavelet decomposition using conv2d. component should be 'high/low/all' """
        harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

        harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
        harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
        harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
        harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

        filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
        filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
        filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
        filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

        if pool:
            net = nn.Conv2d
        else:
            net = nn.ConvTranspose2d

        LL = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)
        LH = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)
        HL = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)
        HH = net(in_channels, in_channels,
                kernel_size=2, stride=2, padding=0, bias=False,
                groups=in_channels)

        LL.weight.requires_grad = False
        LH.weight.requires_grad = False
        HL.weight.requires_grad = False
        HH.weight.requires_grad = False

        LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
        LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
        HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
        HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
        
        return LL, LH, HL, HH
    
    def get_wavelet_feature(self, inp_img):
        ll = self.LL(inp_img)
        lh = self.LH(inp_img)
        hl = self.HL(inp_img)
        hh = self.HH(inp_img)
        x = torch.cat([ll, lh, hl, hh], 1)
        x = self.cnn(x)
        return x
    
    def forward(self, inp_img):
        hist_feature = self.rgb_uv_hist(inp_img)
        wave_feature = self.pool(self.get_wavelet_feature(inp_img))
        return wave_feature * hist_feature[..., None, None]
            
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class SSDUIE(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 32,
        num_blocks = [2,2,2,2], 
        num_refinement_blocks = 2,
        heads = [1,2,2,2],
        ffn_expansion_factor = 1.33,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(SSDUIE, self).__init__()
        
        self.ife_branch = IFEBranch(inp_channels=3)
        # self.ife_branch = SimplifiedIFEBranch()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, **kwargs):
        
        ife_feature = self.ife_branch(inp_img)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
        
        latent = latent + ife_feature
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

class SimplifiedIFEBranch(nn.Module):
    def __init__(self, inp_channels=3, out_channels=512):
        super(SimplifiedIFEBranch, self).__init__()
        
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
        )
        # self.LL, self.LH, self.HL, self.HH = self.get_wave(3)
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        
    def rgb_uv_hist(self, inp_img, n_bins=32):
        """ Computes an RGB-uv histogram tensor for a batch of images.
        """
        batch_size, channels, height, width = inp_img.shape
        hist = torch.zeros(batch_size, 3, n_bins, n_bins, device=inp_img.device)
        
        for b in range(batch_size):
            img = inp_img[b]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = cv2.resize(img_np, (n_bins, n_bins), interpolation=cv2.INTER_NEAREST)
            
            img_reshaped = img_np[(img_np > 0).all(axis=2)]
            eps = 6.4 / 256
            Iy = np.linalg.norm(img_reshaped, axis=1)
            
            for i in range(3):
                r = [j for j in range(3) if j != i]
                Iu = np.log(img_reshaped[:, i] / img_reshaped[:, r[1]])
                Iv = np.log(img_reshaped[:, i] / img_reshaped[:, r[0]])
                
                hist_2d, _, _ = np.histogram2d(Iu, Iv, bins=n_bins, 
                                           range=((-3.2 - eps / 2, 3.2 - eps / 2),) * 2, 
                                           weights=Iy)
                norm_ = hist_2d.sum()
                hist[b, i, :, :] = torch.sqrt(torch.tensor(hist_2d / norm_))
                
        hist = hist.view(hist.size(0), -1).detach()
        hist_feature = self.mlp(hist)
        return hist_feature
    
    def forward(self, inp_img):
        hist_feature = self.rgb_uv_hist(inp_img)
        # wave_feature = self.pool(self.get_wavelet_feature(inp_img))
        return hist_feature[..., None, None]

class IFEBranchNoHist(IFEBranch):
    """IFE Branch without histogram features"""
    def __init__(self, inp_channels=3, out_channels=512):
        super(IFEBranch, self).__init__()
        
        # Remove MLP and histogram-related components
        self.LL, self.LH, self.HL, self.HH = self.get_wave(3)
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inp_img):
        # Only use wavelet features
        wave_feature = self.pool(self.get_wavelet_feature(inp_img))
        return wave_feature

class IFEBranchNoWave(IFEBranch):
    """IFE Branch without wavelet features"""
    def __init__(self, inp_channels=3, out_channels=512):
        super(IFEBranch, self).__init__()
        
        # Remove wavelet components and keep histogram MLP
        self.mlp = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, inp_img):
        # Only use histogram features
        hist_feature = self.rgb_uv_hist(inp_img)
        return hist_feature[..., None, None]  # Add spatial dimensions

class SSDNoHist(SSDUIE):
    """SSDUIE variant without histogram features in IFE branch"""
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=32,
        num_blocks=[2,2,2,2], 
        num_refinement_blocks=2,
        heads=[1,2,2,2],
        ffn_expansion_factor=1.33,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    ):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task
        )
        # Replace IFE branch with NoHist version
        self.ife_branch = IFEBranchNoHist(inp_channels=inp_channels)

class SSDNoWave(SSDUIE):
    """SSDUIE variant without wavelet features in IFE branch"""
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=32,
        num_blocks=[2,2,2,2], 
        num_refinement_blocks=2,
        heads=[1,2,2,2],
        ffn_expansion_factor=1.33,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    ):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task
        )
        # Replace IFE branch with NoWave version
        self.ife_branch = IFEBranchNoWave(inp_channels=inp_channels)