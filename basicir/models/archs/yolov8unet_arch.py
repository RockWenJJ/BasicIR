import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """YOLOv8 Conv Block"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    """YOLOv8 C2f module"""
    def __init__(self, in_channels, out_channels, n_bottlenecks=1, shortcut=True):
        super().__init__()
        self.conv_down = ConvBlock(in_channels, 2 * out_channels, 1)
        self.conv_up = ConvBlock(2 * out_channels, out_channels, 1)
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(out_channels, out_channels, 3),
                ConvBlock(out_channels, out_channels, 3)
            ) for _ in range(n_bottlenecks)
        ])
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv_down(x)
        y1, y2 = torch.chunk(y, 2, dim=1)
        for m in self.bottlenecks:
            y1 = m(y1) + y1 if self.shortcut else m(y1)
        return self.conv_up(torch.cat((y1, y2), dim=1))

class YOLOv8Backbone(nn.Module):
    """Official YOLOv8 Backbone implementation"""
    def __init__(self, in_channels=3, width_mult=0.25):
        super().__init__()
        base_channels = int(64 * width_mult)  # base_channels = 16 for width_mult=0.25
        
        # Stem
        self.stem = ConvBlock(in_channels, base_channels, 3, 2)
        
        # Stages
        self.stage1 = nn.Sequential(
            ConvBlock(base_channels, 2 * base_channels, 3, 2),
            C2f(2 * base_channels, 2 * base_channels, n_bottlenecks=1)
        )  # P2/4
        
        self.stage2 = nn.Sequential(
            ConvBlock(2 * base_channels, 4 * base_channels, 3, 2),
            C2f(4 * base_channels, 4 * base_channels, n_bottlenecks=2)
        )  # P3/8
        
        self.stage3 = nn.Sequential(
            ConvBlock(4 * base_channels, 8 * base_channels, 3, 2),
            C2f(8 * base_channels, 8 * base_channels, n_bottlenecks=2)
        )  # P4/16
        
        self.stage4 = nn.Sequential(
            ConvBlock(8 * base_channels, 16 * base_channels, 3, 2),
            C2f(16 * base_channels, 16 * base_channels, n_bottlenecks=1)
        )  # P5/32

    def forward(self, x):
        features = []
        x = self.stem(x)
        features.append(x)  # P1/2
        
        x = self.stage1(x)
        features.append(x)  # P2/4
        
        x = self.stage2(x)
        features.append(x)  # P3/8
        
        x = self.stage3(x)
        features.append(x)  # P4/16
        
        x = self.stage4(x)
        features.append(x)  # P5/32
        
        return features

class DecoderBlock(nn.Module):
    """UNet Decoder Block"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv1 = ConvBlock(in_channels // 2 + skip_channels, out_channels, 3)
        self.conv2 = ConvBlock(out_channels, out_channels, 3)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class YOLOv8UNet(nn.Module):
    """UNet with YOLOv8 Backbone for Image Restoration"""
    def __init__(self, inp_channels=3, out_channels=3, width_mult=0.25):
        super().__init__()
        
        # Encoder (YOLOv8-n Backbone)
        self.backbone = YOLOv8Backbone(inp_channels, width_mult=width_mult)
        
        # Calculate base channels
        base_channels = int(64 * width_mult)  # 16 for width_mult=0.25
        
        # Decoder (channels are now half of previous version)
        self.decoder4 = DecoderBlock(16 * base_channels, 8 * base_channels, 8 * base_channels)  # 256 -> 128
        self.decoder3 = DecoderBlock(8 * base_channels, 4 * base_channels, 4 * base_channels)   # 128 -> 64
        self.decoder2 = DecoderBlock(4 * base_channels, 2 * base_channels, 2 * base_channels)   # 64 -> 32
        self.decoder1 = DecoderBlock(2 * base_channels, base_channels, base_channels)           # 32 -> 16
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 2, stride=2),
            ConvBlock(base_channels // 2, base_channels // 2, 3),
            nn.Conv2d(base_channels // 2, out_channels, 1),
            # nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        features = self.backbone(x)
        
        # Decoder
        x = self.decoder4(features[4], features[3])  # 256 + 128 -> 128
        x = self.decoder3(x, features[2])            # 128 + 64 -> 64
        x = self.decoder2(x, features[1])            # 64 + 32 -> 32
        x = self.decoder1(x, features[0])            # 32 + 16 -> 16
        
        # Final output
        x = self.final_conv(x)                       # 16 -> 8 -> 3
        
        return x

def test():
    # Test the model
    x = torch.randn(1, 3, 256, 256)
    model = YOLOv8UNet()
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
if __name__ == "__main__":
    test() 