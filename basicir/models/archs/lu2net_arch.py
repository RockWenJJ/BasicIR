import torch
import torch.nn as nn

class AxialDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length=5):
        super(AxialDepthwiseConv, self).__init__()
        self.horizontal_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), groups=in_channels)
        self.vertical_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_length, 1), padding=(kernel_length // 2, 0), groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h_out = self.horizontal_conv(x)
        v_out = self.vertical_conv(x)
        x = h_out + v_out + x  # Res connection
        out = self.conv(self.relu(x))
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
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.pointwise1 = PointwiseConv(in_channels, in_channels)
        self.axial_depthwise_conv1 = AxialDepthwiseConv(in_channels, out_channels)
        self.axial_depthwise_conv2 = AxialDepthwiseConv(in_channels, out_channels)
        self.axial_depthwise_conv3 = AxialDepthwiseConv(in_channels, out_channels)
        self.calayer1 = CALayer(out_channels)
        self.calayer2 = CALayer(out_channels)
        self.calayer3 = CALayer(out_channels)
        self.pointwise2 = PointwiseConv(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.pointwise1(x)
        x1 = self.axial_depthwise_conv1(x)
        x1 = self.calayer1(x1)
        x2 = self.axial_depthwise_conv2(x)
        x2 = self.calayer2(x2)
        x3 = self.axial_depthwise_conv3(x)
        x3 = self.calayer3(self.relu(x3))

        x = x + x1 + x2 + x3
        x = self.relu(self.pointwise2(self.relu(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.axial_depthwise_conv = AxialDepthwiseConv(in_channels, out_channels)
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

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pwconv = PointwiseConv(in_channels*2, in_channels*2)
        self.axial_depthwise_conv = AxialDepthwiseConv(in_channels*2, out_channels)
        self.calayer = CALayer(out_channels)
        self.pwconv2 = PointwiseConv(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x0, x):
        x = self.upsample(x)
        x = torch.cat([x0, x], dim=1)
        x = self.pwconv(x)
        x = self.axial_depthwise_conv(x)
        x = self.calayer(x)
        x = self.pwconv2(self.relu(x))
        return self.relu(x)

class LU2Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[16, 64, 96]):
        super(LU2Net, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1)
        self.encoder_block1 = EncoderBlock(features[0], features[1])
        self.encoder_block2 = EncoderBlock(features[1], features[2])
        self.bottleneck = Bottleneck(features[2], features[2])
        self.decoder_block1 = DecoderBlock(features[2], features[1])
        self.decoder_block2 = DecoderBlock(features[1], features[0])
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.in_conv(x)
        x0, x = self.encoder_block1(x)
        x1, x = self.encoder_block2(x)
        x = self.bottleneck(x)
        x = self.decoder_block1(x1, x)
        x = self.decoder_block2(x0, x)
        x = self.out_conv(x)
        return x

# ===================== LU2Net_Bottleneck =====================

# 1. 定义 MultiLatentAttention 模块
class MultiLatentAttention(nn.Module):
    """
    通过全局平均池化后映射到多个 latent 表示，聚合后再映射回通道注意力权重。
    参数量较小，可作为 CALayer 的补充。
    """
    def __init__(self, channels, num_latents=4, reduction=16):
        super(MultiLatentAttention, self).__init__()
        self.num_latents = num_latents
        self.latent_dim = channels // reduction  # latent 空间维度
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_latents = nn.Linear(channels, num_latents * self.latent_dim, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.fc_combine = nn.Linear(self.latent_dim, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        pooled = self.global_avg_pool(x).view(b, c)  # (b, c)
        # 映射到多个 latent 表示
        latents = self.fc_latents(pooled)  # (b, num_latents * latent_dim)
        latents = self.relu(latents)
        latents = latents.view(b, self.num_latents, self.latent_dim)  # (b, num_latents, latent_dim)
        # 聚合（此处简单取平均，可根据需要设计其他聚合方式）
        aggregated = latents.mean(dim=1)  # (b, latent_dim)
        # 映射回通道空间得到注意力权重
        channel_att = self.fc_combine(aggregated)  # (b, c)
        channel_att = self.sigmoid(channel_att).view(b, c, 1, 1)
        return channel_att  # 返回权重

# 2. 定义融合 CALayer 与 MLA 的模块
class MultiLatentCALayer(nn.Module):
    """
    将原有的 CALayer 与上面定义的 MultiLatentAttention 进行融合，
    二者计算得到的通道注意力取平均后作用于输入。
    """
    def __init__(self, channels, reduction=16, num_latents=4):
        super(MultiLatentCALayer, self).__init__()
        # 原 CALayer 部分
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cal_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        # 多 latent 注意力模块
        self.mla = MultiLatentAttention(channels, num_latents, reduction)

    def forward(self, x):
        b, c, _, _ = x.size()
        # 计算 CALayer 权重
        w_cal = self.global_avg_pool(x).view(b, c)
        w_cal = self.cal_mlp(w_cal).view(b, c, 1, 1)
        # 计算 MLA 权重
        w_mla = self.mla(x)  # (b, c, 1, 1)
        # 取平均组合（当然也可以设计其他融合方式）
        w = (w_cal + w_mla) / 2
        return x * w

# 3. 修改 Bottleneck 模块，命名为 Bottleneck_MLA
class Bottleneck_MLA(nn.Module):
    """
    在原有 Bottleneck 的基础上，用 MultiLatentCALayer 替换 CALayer，
    同时保持整体结构不变，从而引入多 latent 注意力机制，提高图像修复性能，
    并且参数量不会显著增加。
    """
    def __init__(self, in_channels, out_channels, num_latents=4, reduction=16):
        super(Bottleneck_MLA, self).__init__()
        self.pointwise1 = PointwiseConv(in_channels, in_channels)
        self.axial_depthwise_conv1 = AxialDepthwiseConv(in_channels, out_channels)
        self.axial_depthwise_conv2 = AxialDepthwiseConv(in_channels, out_channels)
        self.axial_depthwise_conv3 = AxialDepthwiseConv(in_channels, out_channels)
        self.attention1 = MultiLatentCALayer(out_channels, reduction=reduction, num_latents=num_latents)
        self.attention2 = MultiLatentCALayer(out_channels, reduction=reduction, num_latents=num_latents)
        self.attention3 = MultiLatentCALayer(out_channels, reduction=reduction, num_latents=num_latents)
        self.pointwise2 = PointwiseConv(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.pointwise1(x)
        x1 = self.axial_depthwise_conv1(x)
        x1 = self.attention1(x1)
        x2 = self.axial_depthwise_conv2(x)
        x2 = self.attention2(x2)
        x3 = self.axial_depthwise_conv3(x)
        x3 = self.attention3(self.relu(x3))
        x = x + x1 + x2 + x3
        x = self.relu(self.pointwise2(self.relu(x)))
        return x

# 4. 定义新的 LU2Net_Bottleneck 网络，其余模块保持不变，仅替换 Bottleneck 部分
class LU2Net_Bottleneck(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[16, 64, 96]):
        super(LU2Net_Bottleneck, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1)
        self.encoder_block1 = EncoderBlock(features[0], features[1])
        self.encoder_block2 = EncoderBlock(features[1], features[2])
        # 使用改进后的 Bottleneck_MLA
        self.bottleneck = Bottleneck_MLA(features[2], features[2])
        self.decoder_block1 = DecoderBlock(features[2], features[1])
        self.decoder_block2 = DecoderBlock(features[1], features[0])
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.in_conv(x)
        x0, x = self.encoder_block1(x)
        x1, x = self.encoder_block2(x)
        x = self.bottleneck(x)
        x = self.decoder_block1(x1, x)
        x = self.decoder_block2(x0, x)
        x = self.out_conv(x)
        return x


if __name__ == '__main__':
    model = LU2Net()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)

