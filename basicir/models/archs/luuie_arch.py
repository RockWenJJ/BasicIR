import torch.nn as nn
import torch
import math
from basicir.models.archs.restormer_arch import OverlapPatchEmbed, TransformerBlock, Upsample, Downsample

import numpy as np

def get_low_wav_conv(in_channels):
    """wavelet decomposition using conv2d with same padding"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    # harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    # filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    class WaveletConv(nn.Module):
        def __init__(self, filter_type):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, in_channels,
                                kernel_size=2, stride=1, padding=1,  # Changed padding to 1
                                bias=False, groups=in_channels)
            self.conv.weight.requires_grad = False
            self.conv.weight.data = filter_type.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
            
        def forward(self, x):
            # Handle the padding: remove extra padded values
            out = self.conv(x)
            return out[:, :, :-1, :-1]  # Remove extra padded row and column

    LL = WaveletConv(filter_LL)
    LH = WaveletConv(filter_LH)
    HL = WaveletConv(filter_HL)
    # HH = net(in_channels, in_channels,
    #          kernel_size=2, stride=2, padding=0, bias=False,
    #          groups=in_channels)

    return LL, LH, HL

def get_high_wav_conv(in_channels):
    """wavelet decomposition using conv2d with same padding"""
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    filter_H = torch.from_numpy(harr_wav_H).unsqueeze(0)

    class WaveletConv(nn.Module):
        def __init__(self, filter_type):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, in_channels,
                                kernel_size=2, stride=1, padding=1,  # Changed padding to 1
                                bias=False, groups=in_channels)
            self.conv.weight.requires_grad = False
            self.conv.weight.data = filter_type.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
            
        def forward(self, x):
            # Handle the padding: remove extra padded values
            out = self.conv(x)
            return out[:, :, :-1, :-1]  # Remove extra padded row and column

    HH = WaveletConv(filter_H)

    return HH

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3):
        super(ConvBlock, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False, groups=in_channels)
        self.hori_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, ks), stride=1, padding=(0, 1), dilation=1, bias=False, groups=in_channels)
        self.vert_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(ks, 1), stride=1, padding=(1, 0), dilation=1, bias=False, groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.dw_conv(x)
        x1 = self.hori_conv(x)
        x2 = self.vert_conv(x)
        x = x1 + x2 + x0
        x = self.relu(self.bn(x))
        out = self.relu(self.pw_conv(x))
        return out

# class ConvBlock(nn.Module):
#     """YOLOv8 Conv Block"""
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None):
#         super().__init__()
#         if padding is None:
#             padding = kernel_size // 2
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = nn.ReLU()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    """YOLOv8 C2f module"""
    def __init__(self, in_channels, out_channels, n_bottlenecks=1, shortcut=True):
        super().__init__()
        # self.conv_down = ConvBlock(in_channels, in_channels)
        self.conv_up = ConvBlock(in_channels, out_channels)
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_channels, in_channels//4),
                ConvBlock(in_channels//4, in_channels)
            ) for _ in range(n_bottlenecks)
        ])
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        # y = self.conv_down(x)
        # y1, y2 = torch.chunk(y, 2, dim=1)
        for m in self.bottlenecks:
            x = m(x) + x if self.shortcut else m(x)
        return self.conv_up(x)

class EncoderBlockv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlockv1, self).__init__()
        self.c2f = C2f(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.c2f(x)
        x0 = x
        x = self.downsample(x)
        return x0, x

class DecoderBlockv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlockv1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.c2f = C2f(in_channels*2, out_channels)

    def forward(self, x, x0):
        x = self.upsample(x)
        x = torch.cat((x, x0), dim=1)
        x = self.c2f(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        
        # Global average pooling for spatial information
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Channel attention with dimensionality reduction
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.query_enc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.key_enc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.value_enc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling parameter
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Global average pooling
        y = self.global_avg_pool(x).view(batch_size, channels)
        
        # Generate attention weights through MLP
        mlp_attention = self.attention(y).view(batch_size, channels, 1, 1)
        
        # Channel-wise self-attention
        # Reshape for self-attention computation
        proj_query = self.query_enc(y).view(batch_size, channels, 1)  # B x C x 1
        proj_key = self.key_enc(y).view(batch_size, channels, 1).permute(0, 2, 1)  # B x 1 x C
        energy = torch.bmm(proj_query, proj_key)  # B x C x C
        attention = self.softmax(energy)  # B x C x C
        
        proj_value = self.value_enc(y).view(batch_size, channels, 1)  # B x C x 1
        sa_attention = torch.bmm(attention, proj_value)  # B x C x 1
        

        # Combine MLP attention and self-attention
        sa_attention = sa_attention.view(batch_size, channels, 1, 1)
        attention_weights = self.gamma * sa_attention + mlp_attention
        
        # Apply attention weights
        out = x * attention_weights.expand_as(x) + x
        return out

class CALayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CALayer, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.global_avg_pool(x).view(b, c)
        w = self.mlp(w).view(b, c, 1, 1)
        return x * w
    
class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))

class EncoderBlockv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlockv2, self).__init__()
        self.axial_depthwise_conv = ConvBlock(in_channels, out_channels)
        self.calayer = CALayer(out_channels)
        # self.pwconv = PointwiseConv(out_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.axial_depthwise_conv(x)
        x = self.calayer(x)
        out0 = x
        # x = self.pwconv(x)
        x = self.maxpool(x)
        return out0, x

class DecoderBlockv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlockv2, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pwconv = PointwiseConv(in_channels*2, in_channels*2)
        self.axial_depthwise_conv = ConvBlock(in_channels*2, out_channels)
        self.calayer = CALayer(out_channels)
        # self.pwconv2 = PointwiseConv(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, x0):
        x = self.upsample(x)
        x = torch.cat([x0, x], dim=1)
        x = self.pwconv(x)
        x = self.axial_depthwise_conv(x)
        x = self.calayer(x)
        # x = self.pwconv2(self.relu(x))
        return self.relu(x)
    
class EncoderBlockv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlockv3, self).__init__()
        self.axial_depthwise_conv = ConvBlock(in_channels, out_channels, ks=3)
        self.calayer = CALayer(out_channels)
        self.pwconv = PointwiseConv(out_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.axial_depthwise_conv(x)
        x = self.calayer(x)
        out0 = x
        x = self.pwconv(x)
        x = self.maxpool(x)
        return out0, x

class DecoderBlockv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlockv3, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pwconv = PointwiseConv(in_channels*2, in_channels*2)
        self.axial_depthwise_conv = ConvBlock(in_channels*2, out_channels)
        self.calayer = CALayer(out_channels)
        self.pwconv2 = PointwiseConv(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, x0):
        x = self.upsample(x)
        x = torch.cat([x0, x], dim=1)
        x = self.pwconv(x)
        x = self.axial_depthwise_conv(x)
        x = self.calayer(x)
        x = self.pwconv2(self.relu(x))
        return self.relu(x)

class LUUIEv1(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[16, 64, 96]):
        super(LUUIEv1, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, features[0], kernel_size=1, stride=1, padding=0)

        # encoder
        self.encoder1 = EncoderBlockv1(features[0], features[1])
        self.encoder2 = EncoderBlockv1(features[1], features[2])
        # bottleneck
        self.bottleneck = Bottleneck(features[2], reduction=32)
        # decoder
        self.decoder1 = DecoderBlockv1(features[2], features[1])
        self.decoder2 = DecoderBlockv1(features[1], features[0])
        # out_conv
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.in_conv(x)
        x0, x = self.encoder1(x)
        x1, x = self.encoder2(x)
        x = self.bottleneck(x)
        x = self.decoder1(x, x1)
        x = self.decoder2(x, x0)
        x = self.out_conv(x)
        return x

class LUUIEv2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[16, 64, 96]):
        super(LUUIEv2, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, features[0], kernel_size=1, stride=1, padding=0)

        # encoder
        self.encoder1 = EncoderBlockv2(features[0], features[1])
        self.encoder2 = EncoderBlockv2(features[1], features[2])
        # bottleneck
        self.bottleneck = Bottleneck(features[2], reduction=32)
        # decoder
        self.decoder1 = DecoderBlockv2(features[2], features[1])
        self.decoder2 = DecoderBlockv2(features[1], features[0])
        # out_conv
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.in_conv(x)
        x0, x = self.encoder1(x)
        x1, x = self.encoder2(x)
        x = self.bottleneck(x)
        x = self.decoder1(x, x1)
        x = self.decoder2(x, x0)
        x = self.out_conv(x)
        return x
    
class LUUIEv3(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 48, 96]):
        super(LUUIEv3, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1)

        # encoder
        self.encoder1 = EncoderBlockv3(features[0], features[1])
        self.encoder2 = EncoderBlockv3(features[1], features[2])
        # bottleneck
        self.bottleneck = Bottleneck(features[2], reduction=32)
        # decoder
        self.decoder1 = DecoderBlockv3(features[2], features[1])
        self.decoder2 = DecoderBlockv3(features[1], features[0])
        # out_conv
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.in_conv(x)
        x0, x = self.encoder1(x)
        x1, x = self.encoder2(x)
        x = self.bottleneck(x)
        x = self.decoder1(x, x1)
        x = self.decoder2(x, x0)
        x = self.out_conv(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * attention

class AdaptiveAxialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length=5):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels)
        self.hori_conv = nn.Conv2d(
            in_channels, in_channels, (1, kernel_length), 
            padding=(0, kernel_length//2), groups=in_channels
        )
        self.vert_conv = nn.Conv2d(
            in_channels, in_channels, (kernel_length, 1), 
            padding=(kernel_length//2, 0), groups=in_channels
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.ca = CALayer(in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        x1 = self.dw_conv(x)
        x2 = self.hori_conv(x)
        x3 = self.vert_conv(x)
        out = x1 + x2 + x3 + identity
        out = self.relu(self.bn(out))
        out = self.ca(out)
        out = self.pw_conv(out)
        return self.relu(out)

class EnhancedBottleneck(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.branch1 = AdaptiveAxialBlock(channels, channels, 3)
        self.branch2 = AdaptiveAxialBlock(channels, channels, 5)
        self.branch3 = AdaptiveAxialBlock(channels, channels, 7)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False)
        )
        self.ca = CALayer(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = self.fusion(torch.cat([b1, b2, b3], dim=1))
        out = self.ca(out)
        out = self.sa(out)
        return out + x

class EncoderBlockv4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.axial_block = AdaptiveAxialBlock(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.axial_block(x)
        out0 = x
        x = self.maxpool(x)
        return out0, x

class DecoderBlockv4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pwconv = PointwiseConv(in_channels*2, in_channels*2)
        self.axial_block = AdaptiveAxialBlock(in_channels*2, out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, x0):
        x = self.upsample(x)
        x = torch.cat([x0, x], dim=1)
        x = self.pwconv(x)
        x = self.axial_block(x)
        return self.relu(x)

class LUUIEv4(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 48, 96]):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=False)
        )
        
        # Encoder
        self.encoder1 = EncoderBlockv4(features[0], features[1])
        self.encoder2 = EncoderBlockv4(features[1], features[2])
        
        # Enhanced Bottleneck
        self.bottleneck = EnhancedBottleneck(features[2])
        
        # Decoder
        self.decoder1 = DecoderBlockv4(features[2], features[1])
        self.decoder2 = DecoderBlockv4(features[1], features[0])
        
        # Output conv
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Initial conv
        x = self.in_conv(x)
        
        # Encoder
        x0, x = self.encoder1(x)
        x1, x = self.encoder2(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.decoder1(x, x1)
        x = self.decoder2(x, x0)
        
        # Output
        x = self.out_conv(x)
        return x

class LightAdaptiveConvBlock(nn.Module):
    """Lightweight adaptive convolution block with efficient feature extraction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = max(in_channels // 2, 16)  # Reduce intermediate channels
        
        # Efficient depthwise separable convolutions
        self.dw_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pw_conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        
        # Lightweight axial attention
        self.hori_conv = nn.Conv2d(mid_channels, mid_channels, (1, 3), 
                                padding=(0, 1), groups=mid_channels)
        self.vert_conv = nn.Conv2d(mid_channels, mid_channels, (3, 1), 
                                padding=(1, 0), groups=mid_channels)
        
        self.bn = nn.BatchNorm2d(mid_channels)
        self.pw_conv2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        x = self.relu(self.pw_conv1(self.dw_conv(x)))
        x_h = self.hori_conv(x)
        x_v = self.vert_conv(x)
        out = x_h + x_v + x
        out = self.relu(self.bn(out))
        out = self.pw_conv2(out)
        return out + (identity if x.size(1) == out.size(1) else 0)

class LightBottleneck(nn.Module):
    """Lightweight bottleneck with efficient attention mechanism"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid_channels = channels // reduction
        
        # Simplified channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels),
            nn.Sigmoid()
        )
        
        # Lightweight feature processing
        self.conv_block = LightAdaptiveConvBlock(channels, channels)
        
    def forward(self, x):
        identity = x
        
        # Efficient channel attention
        b, c = x.size()[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y
        
        # Feature processing
        x = self.conv_block(x)
        
        return x + identity

class LightEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = LightAdaptiveConvBlock(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv_block(x)
        skip = x
        x = self.maxpool(x)
        return skip, x

class LightDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_block = LightAdaptiveConvBlock(in_channels*2, out_channels)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x

class LUUIEv5(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 48, 96]):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder with reduced channels
        self.encoder1 = LightEncoderBlock(features[0], features[1])
        self.encoder2 = LightEncoderBlock(features[1], features[2])
        
        # Lightweight Bottleneck
        self.bottleneck = LightBottleneck(features[2])
        
        # Decoder with reduced channels
        self.decoder1 = LightDecoderBlock(features[2], features[1])
        self.decoder2 = LightDecoderBlock(features[1], features[0])
        
        # Efficient output conv
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        identity = x
        
        # Initial features
        x = self.in_conv(x)
        
        # Encoder path
        s1, x = self.encoder1(x)
        s2, x = self.encoder2(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        x = self.decoder1(x, s2)
        x = self.decoder2(x, s1)
        
        # Output with residual connection
        x = self.out_conv(x)
        return x + identity

class LowFreqEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Get low frequency wavelet filters
        self.LL, self.LH, self.HL = get_low_wav_conv(in_channels)
        
        # Combine low frequency components
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ca = CALayer(out_channels)
        
    def forward(self, x):
        # Apply wavelet decomposition
        ll = self.LL(x)
        lh = self.LH(x)
        hl = self.HL(x)
        
        # Combine components
        x = torch.cat([ll, lh, hl], dim=1)
        x = self.fusion(x)
        x = self.ca(x)
        
        skip = x
        x = self.maxpool(x)
        return skip, x

class LowFreqDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Get low frequency wavelet filters for refinement
        self.LL, self.LH, self.HL = get_low_wav_conv(in_channels * 2)
        
        # Process concatenated features
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2 * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        
        self.ca = CALayer(out_channels)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        
        # Apply wavelet decomposition on concatenated features
        ll = self.LL(x)
        lh = self.LH(x)
        hl = self.HL(x)
        
        # Combine components
        x = torch.cat([ll, lh, hl], dim=1)
        x = self.fusion(x)
        x = self.ca(x)
        return x

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*math.pi*variance)) * \
                        torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / \
                        (2*variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, padding=kernel_size//2, groups=3, bias=False)
        
        kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(3, 1, 1, 1)
        self.gaussian_filter.weight.data = kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)

class LUUIEv6(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 48, 96]):
        super(LUUIEv6, self).__init__()
        
        # Gaussian blur for backscatter and transmission branches
        self.gaussian_blur = GaussianBlur(kernel_size=7, sigma=1.0)
        
        # Initial convolutions for each branch
        self.clear_in_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1)
        self.back_in_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1)
        self.trans_in_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1)

        # Clear image branch
        self.clear_encoder1 = EncoderBlockv3(features[0], features[1])
        self.clear_encoder2 = EncoderBlockv3(features[1], features[2])
        self.clear_bottleneck = Bottleneck(features[2], reduction=32)
        self.clear_decoder1 = DecoderBlockv3(features[2], features[1])
        self.clear_decoder2 = DecoderBlockv3(features[1], features[0])
        self.clear_out = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Backscatter branch with low frequency focus
        self.back_encoder1 = EncoderBlockv3(features[0], features[1])
        self.back_encoder2 = EncoderBlockv3(features[1], features[2])
        self.back_bottleneck = Bottleneck(features[2], reduction=32)
        self.back_decoder1 = DecoderBlockv3(features[2], features[1])
        self.back_decoder2 = DecoderBlockv3(features[1], features[0])
        self.back_out = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Transmission branch with low frequency focus
        self.trans_encoder1 = EncoderBlockv3(features[0], features[1])
        self.trans_encoder2 = EncoderBlockv3(features[1], features[2])
        self.trans_bottleneck = Bottleneck(features[2], reduction=32)
        self.trans_decoder1 = DecoderBlockv3(features[2], features[1])
        self.trans_decoder2 = DecoderBlockv3(features[1], features[0])
        self.trans_out = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # set the flag of whether to output the backscatter and transmission during testing
        self.output_all_components = False
    
    def _get_clear_image(self, x):
        # Clear image path
        x = self.clear_in_conv(x)
        x0, x = self.clear_encoder1(x)
        x1, x = self.clear_encoder2(x)
        x = self.clear_bottleneck(x)
        x = self.clear_decoder1(x, x1)
        x = self.clear_decoder2(x, x0)
        x = self.clear_out(x)
        return x
    
    def _get_backscatter(self, x):
        # Apply Gaussian blur before backscatter path
        x = self.gaussian_blur(x)
        x = self.back_in_conv(x)
        x0, x = self.back_encoder1(x)
        x1, x = self.back_encoder2(x)
        x = self.back_bottleneck(x)
        x = self.back_decoder1(x, x1)
        x = self.back_decoder2(x, x0)
        x = self.back_out(x)
        return x
    
    def _get_transmission(self, x):
        # Apply Gaussian blur before transmission path
        # x = self.gaussian_blur(x)
        x = self.trans_in_conv(x)
        x0, x = self.trans_encoder1(x)
        x1, x = self.trans_encoder2(x)
        x = self.trans_bottleneck(x)
        x = self.trans_decoder1(x, x1)
        x = self.trans_decoder2(x, x0)
        x = self.trans_out(x)
        return x
    
    def forward(self, x):
        # Get clear image
        clear = self._get_clear_image(x)
        
        # During training, also compute backscatter and transmission
        if self.training:
            back = self._get_backscatter(x)
            trans = self._get_transmission(x)
            return clear, back, trans
        
        # During testing, only return clear image unless output_all_components is True
        if self.output_all_components:
            back = self._get_backscatter(x)
            trans = self._get_transmission(x)
            return clear, back, trans
        else:
            return clear

class LUUIEv7(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128]):
        super(LUUIEv7, self).__init__()
        
        # Gaussian blur for backscatter and transmission branches
        self.gaussian_blur = GaussianBlur(kernel_size=5, sigma=1.0)
        
        # Clear Image Branch (Restormer-style)
        self.clear_patch_embed = OverlapPatchEmbed(in_channels, features[0])
        
        # Clear Image Encoder (2 stages)
        self.clear_encoder1 = nn.Sequential(*[TransformerBlock(dim=features[0], num_heads=1, 
            ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
            for i in range(2)])
        
        self.clear_down1_2 = Downsample(features[0])
        self.clear_encoder2 = nn.Sequential(*[TransformerBlock(dim=features[1], num_heads=2, 
            ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
            for i in range(2)])
        
        # Clear Image Bottleneck
        self.clear_bottleneck = nn.Sequential(*[TransformerBlock(dim=features[1], num_heads=2, 
            ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
            for i in range(2)])

        # Clear Image Decoder (2 stages)
        self.clear_up2_1 = Upsample(features[1])
        self.clear_reduce_chan_level1 = nn.Conv2d(features[1], features[0], kernel_size=1, bias=False)
        self.clear_decoder1 = nn.Sequential(*[TransformerBlock(dim=features[0], num_heads=1, 
            ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
            for i in range(2)])
        
        self.clear_refinement = nn.Sequential(*[TransformerBlock(dim=features[0], num_heads=1, 
            ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
            for i in range(2)])
        
        self.clear_output = nn.Conv2d(features[0], out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Backscatter Branch (Original LUUIEv6 style)
        self.back_in_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1)
        self.back_encoder1 = EncoderBlockv3(features[0], features[1])
        self.back_encoder2 = EncoderBlockv3(features[1], features[2])
        self.back_bottleneck = Bottleneck(features[2], reduction=32)
        self.back_decoder1 = DecoderBlockv3(features[2], features[1])
        self.back_decoder2 = DecoderBlockv3(features[1], features[0])
        self.back_out = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Transmission Branch (Original LUUIEv6 style)
        self.trans_in_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1)
        self.trans_encoder1 = EncoderBlockv3(features[0], features[1])
        self.trans_encoder2 = EncoderBlockv3(features[1], features[2])
        self.trans_bottleneck = Bottleneck(features[2], reduction=32)
        self.trans_decoder1 = DecoderBlockv3(features[2], features[1])
        self.trans_decoder2 = DecoderBlockv3(features[1], features[0])
        self.trans_out = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.output_all_components = False

    def _get_clear_image(self, x):
        # Encoder
        inp_enc_level1 = self.clear_patch_embed(x)
        out_enc_level1 = self.clear_encoder1(inp_enc_level1)
        
        inp_enc_level2 = self.clear_down1_2(out_enc_level1)
        out_enc_level2 = self.clear_encoder2(inp_enc_level2)

        # Bottleneck
        latent = self.clear_bottleneck(out_enc_level2)

        # Decoder
        inp_dec_level1 = self.clear_up2_1(latent)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.clear_reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.clear_decoder1(inp_dec_level1)
        
        out_dec_level1 = self.clear_refinement(out_dec_level1)
        out = self.clear_output(out_dec_level1)
        
        return out + x

    def _get_backscatter(self, x):
        x = self.back_in_conv(x)
        x0, x = self.back_encoder1(x)
        x1, x = self.back_encoder2(x)
        x = self.back_bottleneck(x)
        x = self.back_decoder1(x, x1)
        x = self.back_decoder2(x, x0)
        x = self.back_out(x)
        return x
    
    def _get_transmission(self, x):
        x = self.trans_in_conv(x)
        x0, x = self.trans_encoder1(x)
        x1, x = self.trans_encoder2(x)
        x = self.trans_bottleneck(x)
        x = self.trans_decoder1(x, x1)
        x = self.trans_decoder2(x, x0)
        x = self.trans_out(x)
        return x

    def forward(self, x):
        # Get clear image using Restormer structure
        clear = self._get_clear_image(x)
        
        if self.training or self.output_all_components:
            # Get backscatter and transmission using original LUUIEv6 structure
            back = self._get_backscatter(x)
            trans = self._get_transmission(x)
            return clear, back, trans
        else:
            return clear

class LUUIEv8(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[24, 48, 96]):
        super(LUUIEv8, self).__init__()
        
        # Create three identical Restormer-style branches
        self.clear_branch = self._create_branch(in_channels, out_channels, features)
        self.back_branch = self._create_branch(in_channels, out_channels, features)
        self.trans_branch = self._create_branch(in_channels, out_channels, features)  # transmission has 1 output channel

        self.output_all_components = False

    def _create_branch(self, in_channels, out_channels, features):
        return nn.ModuleDict({
            # Patch Embedding
            'patch_embed': OverlapPatchEmbed(in_channels, features[0]),
            
            # Encoder (2 stages)
            'encoder1': nn.Sequential(*[TransformerBlock(dim=features[0], num_heads=1, 
                ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
                for i in range(2)]),
            
            'down1_2': Downsample(features[0]),
            'encoder2': nn.Sequential(*[TransformerBlock(dim=features[1], num_heads=2, 
                ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
                for i in range(2)]),
            
            # Bottleneck
            'bottleneck': nn.Sequential(*[TransformerBlock(dim=features[1], num_heads=2, 
                ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
                for i in range(2)]),

            # Decoder (2 stages)
            'up2_1': Upsample(features[1]),
            'reduce_chan_level1': nn.Conv2d(features[1], features[0], kernel_size=1, bias=False),
            'decoder1': nn.Sequential(*[TransformerBlock(dim=features[0], num_heads=1, 
                ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
                for i in range(2)]),
            
            'refinement': nn.Sequential(*[TransformerBlock(dim=features[0], num_heads=1, 
                ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
                for i in range(2)]),
            
            'output': nn.Conv2d(features[0], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        })

    def _process_branch(self, x, branch, add_identity=False):
        # Encoder
        inp_enc_level1 = branch['patch_embed'](x)
        out_enc_level1 = branch['encoder1'](inp_enc_level1)
        
        inp_enc_level2 = branch['down1_2'](out_enc_level1)
        out_enc_level2 = branch['encoder2'](inp_enc_level2)

        # Bottleneck
        latent = branch['bottleneck'](out_enc_level2)

        # Decoder
        inp_dec_level1 = branch['up2_1'](latent)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = branch['reduce_chan_level1'](inp_dec_level1)
        out_dec_level1 = branch['decoder1'](inp_dec_level1)
        
        out_dec_level1 = branch['refinement'](out_dec_level1)
        out = branch['output'](out_dec_level1)
        
        return out + x if add_identity else out

    def forward(self, x):
        # Process clear image branch (with identity connection)
        clear = self._process_branch(x, self.clear_branch, add_identity=True)
        
        if self.training or self.output_all_components:
            # Process backscatter and transmission branches (without identity connection)
            back = self._process_branch(x, self.back_branch, add_identity=False)
            trans = self._process_branch(x, self.trans_branch, add_identity=False)
            return clear, back, trans
        else:
            return clear

if __name__ == '__main__':
    # Test LUUIEv6
    model = LUUIEv6()
    x = torch.randn(1, 3, 256, 256)
    
    # Test training mode
    model.train()
    train_outputs = model(x)
    print("\nTraining mode:")
    if isinstance(train_outputs, tuple):
        clear, back, trans = train_outputs
        print(f"Clear image shape: {clear.shape}")
        print(f"Backscatter shape: {back.shape}")
        print(f"Transmission shape: {trans.shape}")
    
    # Test eval mode
    model.eval()
    test_output = model(x)
    print("\nTesting mode:")
    print(f"Clear image shape: {test_output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")