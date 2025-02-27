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

class AreaSelfAttention(nn.Module):
    """区域自注意力机制"""
    def __init__(self, in_channels, reduction=8, area_size=8):
        super().__init__()
        self.area_size = area_size
        self.query = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 划分区域
        padded_h = ((h - 1) // self.area_size + 1) * self.area_size
        padded_w = ((w - 1) // self.area_size + 1) * self.area_size
        padded_x = F.pad(x, (0, padded_w - w, 0, padded_h - h))
        
        # 生成查询、键和值
        q = self.query(padded_x)
        k = self.key(padded_x)
        v = self.value(padded_x)
        
        # 重塑为区域
        q = q.view(b, -1, padded_h//self.area_size, self.area_size, padded_w//self.area_size, self.area_size)
        k = k.view(b, -1, padded_h//self.area_size, self.area_size, padded_w//self.area_size, self.area_size)
        v = v.view(b, -1, padded_h//self.area_size, self.area_size, padded_w//self.area_size, self.area_size)
        
        # 区域内自注意力
        q = q.permute(0, 2, 4, 1, 3, 5).contiguous().view(b*(padded_h//self.area_size)*(padded_w//self.area_size), -1, self.area_size*self.area_size)
        k = k.permute(0, 2, 4, 1, 3, 5).contiguous().view(b*(padded_h//self.area_size)*(padded_w//self.area_size), -1, self.area_size*self.area_size)
        v = v.permute(0, 2, 4, 1, 3, 5).contiguous().view(b*(padded_h//self.area_size)*(padded_w//self.area_size), -1, self.area_size*self.area_size)
        
        # 计算注意力
        attn = torch.bmm(q.transpose(1, 2), k)
        attn = F.softmax(attn, dim=2)
        
        # 应用注意力
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(b, padded_h//self.area_size, padded_w//self.area_size, -1, self.area_size, self.area_size)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, -1, padded_h, padded_w)
        
        # 裁剪回原始大小
        out = out[:, :, :h, :w]
        
        return self.gamma * out + x

class LUSANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[16, 64, 96]):
        super().__init__()
        # 初始卷积层（与LU2Net保持一致）
        self.in_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)
        
        # 编码器（结合LU2Net和LU2NetHybrid的优点）
        self.encoder1 = EnhancedEncoderBlock(features[0], features[1])
        self.encoder2 = EnhancedEncoderBlock(features[1], features[2])
        
        # 瓶颈层（添加区域自注意力）
        self.bottleneck = EnhancedBottleneck(features[2])
        
        # 解码器（增强版）
        self.decoder1 = EnhancedDecoderBlock(features[2], features[1])
        self.decoder2 = EnhancedDecoderBlock(features[1], features[0])
        
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

class EnhancedEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # 轴向卷积（从LU2Net）
        self.axial_conv = AxialDepthwiseConv(in_c, out_c)
        
        # 混合归一化（从LU2NetHybrid）
        self.norm = HybridNormalization(out_c)
        
        # 通道注意力（从LU2Net）
        self.ca = CALayer(out_c)
        
        # 下采样
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.axial_conv(x)
        x = self.norm(x)
        x = self.ca(x)
        return x, self.down(x)

class EnhancedBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 结合LU2Net的Bottleneck和LU2NetHybrid的LightDenseBottleneck
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.axial1 = AxialDepthwiseConv(channels, channels)
        self.axial2 = AxialDepthwiseConv(channels, channels)
        
        # 添加区域自注意力
        self.attention = AreaSelfAttention(channels)
        
        # 通道注意力
        self.ca = CALayer(channels)
        
        # 最终卷积
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x1 = self.axial1(x)
        x2 = self.axial2(x)
        x = x + x1 + x2
        x = self.attention(x)  # 区域自注意力
        x = self.ca(x)
        x = self.conv2(self.relu(x))
        return x + identity  # 残差连接

class EnhancedDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 特征融合
        self.conv1 = nn.Conv2d(in_c * 2, in_c, 1)
        
        # 轴向卷积
        self.axial = AxialDepthwiseConv(in_c, out_c)
        
        # 交叉注意力（轻量级）
        self.cc_attn = CrissCrossAttention(out_c)
        
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
        x = self.cc_attn(x)  # 交叉注意力
        x = self.ca(x)
        x = self.conv2(self.relu(x))
        return self.relu(x)

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
