import torch
import torch.nn as nn

class AxialDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length=5):
        super(AxialDepthwiseConv, self).__init__()
        self.horizontal_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), groups=in_channels)
        self.vertical_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_length, 1), padding=(kernel_length // 2, 0), groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
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
        self.relu = nn.ReLU(inplace=True)

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
        self.relu = nn.ReLU(inplace=True)

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
        self.relu = nn.ReLU(inplace=True)

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


if __name__ == '__main__':
    model = LU2Net()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)

