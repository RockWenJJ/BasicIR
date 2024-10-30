import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import numpy as np

from einops import rearrange
from basicir.models.archs.components import CompressedHE, EdgeEnhance

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

##########################################################################
def window_partition(x, window_size: int, h, w):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - w % window_size) % window_size
    pad_b = (window_size - h % window_size) % window_size
    x = F.pad(x, [pad_l, pad_r, pad_t, pad_b])
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    H = H + pad_b
    W = W + pad_r
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    windows = F.pad(x, [pad_l, -pad_r, pad_t, -pad_b])
    return windows

class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()
        
        hidden_features = int(dim * 3)
        
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x

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
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
class GLTransBlock(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GLTransBlock, self).__init__()
        
        self.norm1 = LayerNorm(dim)
        # global transformer
        self.glob_attn = GlobalAttention(dim, num_heads, bias)
        
        # cnn layers
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.local_attn = conv_block(dim, dim)
        
        # feed forward
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias)
    
    def forward(self, x):
        
        B, C, H, W = x.shape
        shortcut = x
        
        x_norm = self.norm1(x)

        y1 = self.glob_attn(x_norm)
        y1 = shortcut + y1
        
        y2 = self.local_attn(y1)
        
        y = y1 + self.ffn(self.norm2(y2))
        
        return y


class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GlobalAttention, self).__init__()
        self.num_heads = num_heads
        
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(int(c / self.num_heads))
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = self.project_out(out)
        return out
    
class CrossSpaceAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossSpaceAttention, self).__init__()
        self.num_heads = num_heads

        self.q_conv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)

        self.q_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.k_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.v_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)

    def forward(self, x0, x1, x2):
        b, c, h, w = x0.shape
        
        q = self.q_dwconv(self.q_conv(x0))
        k = self.k_dwconv(self.k_conv(x1))
        v = self.v_dwconv(self.v_conv(x2))
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(int(c / self.num_heads))
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = self.project_out(out)
        return out



class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    
    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat, n_out):
        super(Upsample, self).__init__()
        
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feat, n_out * 4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2))
    
    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)

def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    
    return x


##########################################################################
##---------- CrossSpaceFormer -----------------------
class CrossSpaceFormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 36,
        num_blocks = [2,4,4,2], 
        heads = [2,2,2,2],
        bias = False
    ):

        super(CrossSpaceFormer, self).__init__()

        self.white_balance = CompressedHE()
        self.edge_enhance = EdgeEnhance()

        self.patch_embed_wb = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed_ee = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed_ori = OverlapPatchEmbed(inp_channels, dim)

        self.cross_space_attn = CrossSpaceAttention(dim, 6, bias)

        self.encoder_level1 = nn.Sequential(*[
            GLTransBlock(dim=dim, num_heads=heads[0], bias=bias) for
            i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            GLTransBlock(dim=dim * 2 ** 1, num_heads=heads[1], bias=bias) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            GLTransBlock(dim=dim * 2 ** 2, num_heads=heads[2], bias=bias) for i in range(num_blocks[2])])
        
        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            GLTransBlock(dim=dim * 2 ** 3, num_heads=heads[3], bias=bias) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim * 2 ** 3), int(dim * 2**2))  ## From Level 4 to Level 3
        self.skip_connect3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            GLTransBlock(dim=int(dim * 2 ** 3), num_heads=heads[2], bias=bias) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 3), int(dim * 2))  ## From Level 3 to Level 2
        self.skip_connect2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            GLTransBlock(dim=int(dim * 2 ** 2), num_heads=heads[1], bias=bias) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim * 2 **2), int(dim))  ## From Level 2 to Level 1
        self.skip_connect1 = nn.Conv2d(int(dim), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            GLTransBlock(dim=int(dim * 2), num_heads=heads[0], bias=bias) for i in range(num_blocks[0])])
        
        self.output = nn.Conv2d(int(dim*2), out_channels, kernel_size=1, bias=bias)

    def forward(self, inp_img):

        # get white balance and edge enhance
        inp_img_wb = self.white_balance(inp_img)
        inp_img_ee = self.edge_enhance(inp_img)
        # inp_img_ee = inp_img

        # patch embed
        inp_enc_lvl1_wb = self.patch_embed_wb(inp_img_wb)
        inp_enc_lvl1_ee = self.patch_embed_ee(inp_img_ee)
        inp_enc_lvl1_ori = self.patch_embed_ori(inp_img)

        inp_enc_levl1 = self.cross_space_attn(inp_enc_lvl1_wb, inp_enc_lvl1_ee, inp_enc_lvl1_ori)
        
        out_enc_level1 = self.encoder_level1(inp_enc_levl1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        
        latent = self.latent(inp_enc_level4)
        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = cat(inp_dec_level3, self.skip_connect3(out_enc_level3))
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = cat(inp_dec_level2, self.skip_connect2(out_enc_level2))
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = cat(inp_dec_level1, self.skip_connect1(out_enc_level1))
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        ref_out = out_dec_level1
        
        out = self.output(ref_out) + inp_img

        return out