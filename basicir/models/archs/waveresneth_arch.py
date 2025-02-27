import torch
import torch.nn as nn

class WaveResNetH(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 48, 96]):
        super().__init__()
        self.norm = HybridNormalization(features[0])
        
        # Encoder with wavelet decomposition
        self.encoder1 = WaveletEncoderBlock(in_channels, features[0])
        self.encoder2 = WaveletEncoderBlock(features[0], features[1])
        self.encoder3 = WaveletEncoderBlock(features[1], features[2])
        
        # Bottleneck with residual dense blocks
        self.bottleneck = nn.Sequential(
            ResidualDenseBlock(features[2], 32),
            EnhancedChannelAttention(features[2]),
            ResidualDenseBlock(features[2], 32)
        )
        
        # Fix decoder input channels
        self.decoder3 = WaveletDecoderBlock(features[2], features[1])
        self.decoder2 = WaveletDecoderBlock(features[1] * 2, features[0])
        self.decoder1 = WaveletDecoderBlock(features[0] * 2, out_channels)
        
        # Final convolution
        self.final_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # Encoder path
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Decoder path with corrected skip connections
        x = self.decoder3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)
        
        return self.final_conv(x) + x

class WaveletEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            HybridNormalization(out_c),
            nn.GELU(),
            DepthWiseConv(out_c),
            ChannelGate(out_c)
        )
        self.down = nn.Conv2d(out_c, out_c, 3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv(x)
        return self.down(x)

class WaveletDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_c, out_c, 3, padding=1)
        )
        self.conv = ResidualDenseBlock(out_c, 16)
        
    def forward(self, x):
        return self.conv(self.up(x))

class HybridNormalization(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_norm = nn.InstanceNorm2d(channels)
        self.bn_norm = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        return 0.5*self.in_norm(x) + 0.5*self.bn_norm(x)

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2*growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3*growth_rate, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.relu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], 1))
        return x4 * 0.2 + x

class EnhancedChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1).squeeze(-1))
        scale = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale

class DepthWiseConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x):
        return self.dw_conv(x)

class ChannelGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x) 
    
if __name__ == "__main__":
    model = WaveResNetH()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)