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

    net = nn.Conv2d

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

if __name__ == '__main__':
    model = LUUIEv3()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)