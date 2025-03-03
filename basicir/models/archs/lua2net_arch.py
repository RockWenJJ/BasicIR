import torch
import torch.nn as nn
import torch.nn.functional as F

class AreaAttention(nn.Module):
    """区域注意力机制
    与Criss-cross类似，但扩展到多行多列的区域
    """
    def __init__(self, in_channels, area_width=3):
        super().__init__()
        self.area_width = area_width  # 区域宽度（行/列数）
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
        
        # 计算区域数量和填充
        num_h_areas = (height + self.area_width - 1) // self.area_width
        num_w_areas = (width + self.area_width - 1) // self.area_width
        
        # 水平方向区域注意力（使用unfold进行高效区域切分）
        # unfold将张量分为重叠区域 - 避免循环
        h_queries = []
        h_keys = []
        h_values = []
        h_sizes = []
        
        for i in range(num_h_areas):
            h_start = i * self.area_width
            h_end = min(h_start + self.area_width, height)
            area_h = h_end - h_start
            h_sizes.append(area_h)
            
            # 一次性提取所有区域特征
            h_queries.append(proj_query[:, :, h_start:h_end, :])
            h_keys.append(proj_key[:, :, h_start:h_end, :])
            h_values.append(proj_value[:, :, h_start:h_end, :])
        
        # 使用torch.stack一次性处理所有区域
        h_outs = []
        for i, (h_q, h_k, h_v, area_h) in enumerate(zip(h_queries, h_keys, h_values, h_sizes)):
            # 调整形状以计算区域内注意力
            h_q = h_q.permute(0, 3, 1, 2).contiguous().view(batch_size*width, -1, area_h)
            h_k = h_k.permute(0, 3, 1, 2).contiguous().view(batch_size*width, -1, area_h)
            h_v = h_v.permute(0, 3, 1, 2).contiguous().view(batch_size*width, channels, area_h)
            
            # 使用矩阵乘法计算注意力
            h_attn = torch.bmm(h_q.transpose(1, 2), h_k)
            h_attn = F.softmax(h_attn, dim=2)
            
            # 应用注意力
            h_out = torch.bmm(h_v, h_attn)
            h_out = h_out.view(batch_size, width, channels, area_h).permute(0, 2, 3, 1)
            h_outs.append(h_out)
        
        # 合并所有区域输出
        h_out = torch.cat(h_outs, dim=2)
        
        # 垂直方向区域注意力
        v_queries = []
        v_keys = []
        v_values = []
        v_sizes = []
        
        for i in range(num_w_areas):
            v_start = i * self.area_width
            v_end = min(v_start + self.area_width, width)
            area_w = v_end - v_start
            v_sizes.append(area_w)
            
            # 一次性提取所有区域特征
            v_queries.append(proj_query[:, :, :, v_start:v_end])
            v_keys.append(proj_key[:, :, :, v_start:v_end])
            v_values.append(proj_value[:, :, :, v_start:v_end])
            
        # 使用torch.stack一次性处理所有区域
        v_outs = []
        for i, (v_q, v_k, v_v, area_w) in enumerate(zip(v_queries, v_keys, v_values, v_sizes)):
            # 调整形状以计算区域内注意力
            v_q = v_q.permute(0, 2, 1, 3).contiguous().view(batch_size*height, -1, area_w)
            v_k = v_k.permute(0, 2, 1, 3).contiguous().view(batch_size*height, -1, area_w)
            v_v = v_v.permute(0, 2, 1, 3).contiguous().view(batch_size*height, channels, area_w)
            
            # 使用矩阵乘法计算注意力
            v_attn = torch.bmm(v_q.transpose(1, 2), v_k)
            v_attn = F.softmax(v_attn, dim=2)
            
            # 应用注意力
            v_out = torch.bmm(v_v, v_attn)
            v_out = v_out.view(batch_size, height, channels, area_w).permute(0, 2, 1, 3)
            v_outs.append(v_out)
            
        # 合并所有区域输出
        v_out = torch.cat(v_outs, dim=3)
        
        # 合并结果
        out = h_out + v_out
        out = self.gamma * out + x
        
        return out


class LUA2Net(nn.Module):
    """Light-weight Unsupervised Area Attention Net"""
    def __init__(self, in_channels=3, out_channels=3, features=[24, 48, 96], area_width=3):
        super().__init__()
        # 初始卷积层
        self.in_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)
        
        # 编码器（浅层与深层）
        self.encoder1 = ShallowEncoderBlock(features[0], features[1])  # 浅层
        self.encoder2 = DeepEncoderBlock(features[1], features[2], area_width)  # 深层
        
        # 瓶颈层（基于LU2Net改进，不使用自注意力）
        self.bottleneck = ImprovedBottleneck(features[2])
        
        # 解码器（与编码器对称）
        self.decoder1 = DeepDecoderBlock(features[2], features[1], area_width)  # 深层
        self.decoder2 = ShallowDecoderBlock(features[1], features[0])  # 浅层
        
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

# 深层编码器块 (axial_conv + AreaAttention)
class DeepEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, area_width=3):
        super().__init__()
        # 轴向卷积
        self.axial_conv = AxialDepthwiseConv(in_c, out_c)
        
        # 混合归一化
        self.norm = HybridNormalization(out_c)
        
        # 区域注意力替代交叉注意力
        self.area_attn = AreaAttention(out_c, area_width)
        
        # 下采样
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.axial_conv(x)
        x = self.norm(x)
        x = self.area_attn(x)
        return x, self.down(x)

# 深层解码器块 (upsample + axial_conv + AreaAttention)
class DeepDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, area_width=3):
        super().__init__()
        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 特征融合
        self.conv1 = nn.Conv2d(in_c * 2, in_c, 1)
        
        # 轴向卷积
        self.axial = AxialDepthwiseConv(in_c, out_c)
        
        # 区域注意力替代交叉注意力
        self.area_attn = AreaAttention(out_c, area_width)
        
        # 最终卷积
        self.conv2 = nn.Conv2d(out_c, out_c, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, skip, x):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.axial(x)
        x = self.area_attn(x)
        x = self.conv2(self.relu(x))
        return self.relu(x)

# 浅层解码器块 (upsample + axial_conv + norm + CA)
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

# 改进的瓶颈层
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

# 辅助模块
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
    model = LUA2Net()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape) 