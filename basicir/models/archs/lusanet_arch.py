import torch
import torch.nn as nn
import torch.nn.functional as F

class CrissCrossAttention(nn.Module):
    """轻量级交叉注意力机制"""
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 生成查询、键和值
        proj_query = self.query_conv(x)  # B x (C//8) x H x W
        proj_key = self.key_conv(x)      # B x (C//8) x H x W
        proj_value = self.value_conv(x)  # B x C x H x W
        
        # 水平方向注意力
        # 查询和键调整为合适的形状: B*W x (C//8) x H
        h_query = proj_query.permute(0, 3, 1, 2).contiguous().view(batch_size*width, -1, height)
        h_key = proj_key.permute(0, 3, 1, 2).contiguous().view(batch_size*width, -1, height)
        # 计算注意力分数: B*W x H x H
        h_attn = torch.bmm(h_query.transpose(1, 2), h_key)
        h_attn = F.softmax(h_attn, dim=2)
        
        # 值调整为合适的形状: B*W x C x H
        h_value = proj_value.permute(0, 3, 1, 2).contiguous().view(batch_size*width, channels, height)
        # 应用注意力权重: B*W x C x H
        h_out = torch.bmm(h_value, h_attn)
        # 重塑回原始维度: B x C x H x W
        h_out = h_out.view(batch_size, width, channels, height).permute(0, 2, 3, 1)
        
        # 垂直方向注意力
        # 查询和键调整为合适的形状: B*H x (C//8) x W
        v_query = proj_query.permute(0, 2, 1, 3).contiguous().view(batch_size*height, -1, width)
        v_key = proj_key.permute(0, 2, 1, 3).contiguous().view(batch_size*height, -1, width)
        # 计算注意力分数: B*H x W x W
        v_attn = torch.bmm(v_query.transpose(1, 2), v_key)
        v_attn = F.softmax(v_attn, dim=2)
        
        # 值调整为合适的形状: B*H x C x W
        v_value = proj_value.permute(0, 2, 1, 3).contiguous().view(batch_size*height, channels, width)
        # 应用注意力权重: B*H x C x W
        v_out = torch.bmm(v_value, v_attn)
        # 重塑回原始维度: B x C x H x W
        v_out = v_out.view(batch_size, height, channels, width).permute(0, 2, 1, 3)
        
        # 合并结果
        out = h_out + v_out
        out = self.gamma * out + x
        
        return out

class LUSANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[24, 48, 96]):
        super().__init__()
        # 初始卷积层
        self.in_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)
        
        # 编码器（浅层与深层）
        self.encoder1 = DeepEncoderBlock(features[0], features[1])  # 浅层
        self.encoder2 = DeepEncoderBlock(features[1], features[2])     # 深层
        
        # 瓶颈层（基于LU2Net改进，不使用自注意力）
        self.bottleneck = ImprovedBottleneck(features[2])
        
        # 解码器（与编码器对称）
        self.decoder1 = DeepDecoderBlock(features[2], features[1])     # 深层
        self.decoder2 = DeepDecoderBlock(features[1], features[0])  # 浅层
        
        # 输出层
        self.out_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        x = self.in_conv(x)
        
        # 编码路径
        x0, x = self.encoder1(x)
        x1, x = self.encoder2(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码路径
        x = self.decoder1(x1, x)
        x = self.decoder2(x0, x)
        
        return self.out_conv(x)

# 浅层编码器块 (axial_conv + norm + ca)
class ShallowEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # 轴向卷积
        self.axial_conv = AxialDepthwiseConv(in_c, out_c)
        
        # 混合归一化
        self.norm = HybridNormalization(out_c)
        
        # 通道注意力
        self.ca = CALayer(out_c)
        
        # 下采样
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.axial_conv(x)
        x = self.norm(x)
        x = self.ca(x)
        return x, self.down(x)

# 深层编码器块 (axial_conv + CrissCrossAttention)
class DeepEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # 轴向卷积
        self.axial_conv = AxialDepthwiseConv(in_c, out_c)
        
        # 混合归一化
        self.norm = HybridNormalization(out_c)
        
        # 交叉注意力
        self.cc_attn = CrissCrossAttention(out_c)
        
        # 下采样
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.axial_conv(x)
        x = self.norm(x)
        x = self.cc_attn(x)
        return x, self.down(x)

# 深层解码器块 (upsample + axial_conv + CrissCrossAttention) - 与深层编码器对称
class DeepDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 特征融合
        self.conv1 = nn.Conv2d(in_c * 2, in_c, 1)
        
        # 轴向卷积
        self.axial = AxialDepthwiseConv(in_c, out_c)
        
        # 交叉注意力
        self.cc_attn = CrissCrossAttention(out_c)
        
        # 最终卷积
        self.conv2 = nn.Conv2d(out_c, out_c, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, skip, x):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.axial(x)
        x = self.cc_attn(x)
        x = self.conv2(self.relu(x))
        return self.relu(x)

# 浅层解码器块 (upsample + axial_conv + CA) - 与浅层编码器对称
class ShallowDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 特征融合
        self.conv1 = nn.Conv2d(in_c * 2, in_c, 1)
        
        # 轴向卷积
        self.axial = AxialDepthwiseConv(in_c, out_c)
        
        # 混合归一化
        self.norm = HybridNormalization(out_c)
        
        # 通道注意力
        self.ca = CALayer(out_c)
        
        # 最终卷积
        self.conv2 = nn.Conv2d(out_c, out_c, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, skip, x):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.axial(x)
        x = self.norm(x)
        x = self.ca(x)
        x = self.conv2(self.relu(x))
        return self.relu(x)

# 改进的瓶颈层 - 基于LU2Net而不使用自注意力
class ImprovedBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 多分支结构
        self.branch1 = AxialDepthwiseConv(channels, channels, kernel_length=3)
        self.branch2 = AxialDepthwiseConv(channels, channels, kernel_length=5)
        self.branch3 = AxialDepthwiseConv(channels, channels, kernel_length=7)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False)
        )
        
        # 通道注意力
        self.ca = CALayer(channels)
        
        # 最终卷积
        self.conv = nn.Conv2d(channels, channels, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        x = self.fusion(torch.cat([b1, b2, b3], dim=1))
        x = self.ca(x)
        x = self.conv(self.relu(x))
        return x + identity

# 保留原有的辅助模块
class AxialDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length=5):
        super().__init__()
        self.horizontal_conv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=(1, kernel_length), 
            padding=(0, kernel_length//2),
            groups=in_channels
        )
        self.vertical_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(kernel_length, 1),
            padding=(kernel_length//2, 0),
            groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        h = self.horizontal_conv(x)
        v = self.vertical_conv(x)
        return self.pointwise(self.relu(h + v + x))

class HybridNormalization(nn.Module):
    def __init__(self, channels, ratio=0.7):
        super().__init__()
        self.ratio = ratio
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.batch_norm = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        return self.ratio * self.instance_norm(x) + (1-self.ratio) * self.batch_norm(x)

class CALayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
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

if __name__ == '__main__':
    model = LUSANet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
