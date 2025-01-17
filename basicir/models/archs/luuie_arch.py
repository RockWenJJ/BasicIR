import torch
import torch.nn as nn

import numpy as np

def get_low_wav_conv(in_channels):
    """wavelet decomposition using conv2d"""
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

    net = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=1, padding=1, bias=False, groups=in_channels)   

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=1, padding=1, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=1, padding=1, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=1, padding=1, bias=False,
             groups=in_channels)
    # HH = net(in_channels, in_channels,
    #          kernel_size=2, stride=2, padding=0, bias=False,
    #          groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    # HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    # HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL

def get_high_wav_conv(in_channels):
    """wavelet decomposition using conv2d"""
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    filter_H = torch.from_numpy(harr_wav_H).unsqueeze(0)

    HH = nn.Conv2d(in_channels, in_channels,
             kernel_size=2, stride=1, padding=1, bias=False,
             groups=in_channels)

    HH.weight.requires_grad = False
    HH.weight.data = filter_H.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

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

class LUUIEv6(nn.Module):
    def __init__(self, in_channels=3, features=[32, 48, 96]):
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
        
        # Three decoder heads
        # 1. Clear Image Decoder
        self.clear_decoder1 = LightDecoderBlock(features[2], features[1])
        self.clear_decoder2 = LightDecoderBlock(features[1], features[0])
        self.clear_out = nn.Conv2d(features[0], 3, kernel_size=1)
        
        # 2. Backscatter Decoder
        self.back_decoder1 = LightDecoderBlock(features[2], features[1])
        self.back_decoder2 = LightDecoderBlock(features[1], features[0])
        self.back_out = nn.Conv2d(features[0], 3, kernel_size=1)
        
        # 3. Transmission Decoder
        self.trans_decoder1 = LightDecoderBlock(features[2], features[1])
        self.trans_decoder2 = LightDecoderBlock(features[1], features[0])
        self.trans_out = nn.Conv2d(features[0], 1, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _get_clear_image(self, x, s1, s2):
        # Clear Image Branch
        c = self.clear_decoder1(x, s2)
        c = self.clear_decoder2(c, s1)
        clear = self.clear_out(c)
        clear = torch.sigmoid(clear)  # Ensure output is in [0,1]
        return clear
        
    def forward(self, x):
        # Initial features
        x = self.in_conv(x)
        
        # Encoder path
        s1, x = self.encoder1(x)
        s2, x = self.encoder2(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Get clear image
        clear = self._get_clear_image(x, s1, s2)
        
        # During training, also compute backscatter and transmission
        if self.training:
            # Backscatter Branch
            b = self.back_decoder1(x, s2)
            b = self.back_decoder2(b, s1)
            back = self.back_out(b)
            back = torch.sigmoid(back)  # Ensure output is in [0,1]
            
            # Transmission Branch
            t = self.trans_decoder1(x, s2)
            t = self.trans_decoder2(t, s1)
            trans = self.trans_out(t)
            trans = torch.sigmoid(trans)  # Ensure output is in [0,1]
            
            return clear, back, trans
        
        # During testing, only return clear image
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