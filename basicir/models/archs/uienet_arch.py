import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

class CrissCrossAttention(nn.Module):
    """Lightweight Criss-Cross Attention Module"""
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, and value
        proj_query = self.query_conv(x)  # B x (C//8) x H x W
        proj_key = self.key_conv(x)      # B x (C//8) x H x W
        proj_value = self.value_conv(x)  # B x C x H x W
        
        # Horizontal attention
        # Reshape query and key to proper shape: B*W x (C//8) x H
        h_query = proj_query.permute(0, 3, 1, 2).contiguous().view(batch_size*width, -1, height)
        h_key = proj_key.permute(0, 3, 1, 2).contiguous().view(batch_size*width, -1, height)
        # Calculate attention scores: B*W x H x H
        h_attn = torch.bmm(h_query.transpose(1, 2), h_key)
        h_attn = F.softmax(h_attn, dim=2)
        
        # Reshape value to proper shape: B*W x C x H
        h_value = proj_value.permute(0, 3, 1, 2).contiguous().view(batch_size*width, channels, height)
        # Apply attention weights: B*W x C x H
        h_out = torch.bmm(h_value, h_attn)
        # Reshape back to original dimensions: B x C x H x W
        h_out = h_out.view(batch_size, width, channels, height).permute(0, 2, 3, 1)
        
        # Vertical attention
        # Reshape query and key to proper shape: B*H x (C//8) x W
        v_query = proj_query.permute(0, 2, 1, 3).contiguous().view(batch_size*height, -1, width)
        v_key = proj_key.permute(0, 2, 1, 3).contiguous().view(batch_size*height, -1, width)
        # Calculate attention scores: B*H x W x W
        v_attn = torch.bmm(v_query.transpose(1, 2), v_key)
        v_attn = F.softmax(v_attn, dim=2)
        
        # Reshape value to proper shape: B*H x C x W
        v_value = proj_value.permute(0, 2, 1, 3).contiguous().view(batch_size*height, channels, width)
        # Apply attention weights: B*H x C x W
        v_out = torch.bmm(v_value, v_attn)
        # Reshape back to original dimensions: B x C x H x W
        v_out = v_out.view(batch_size, height, channels, width).permute(0, 2, 1, 3)
        
        # Combine results
        out = h_out + v_out
        out = self.gamma * out + x
        
        return out

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
            nn.ReLU()  # Changed from sigmoid to ReLU as per requirements
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.global_avg_pool(x).view(b, c)
        w = self.mlp(w).view(b, c, 1, 1)
        return x * w

class DeepEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # Axial convolution
        self.axial_conv = AxialDepthwiseConv(in_c, out_c)
        
        # Hybrid normalization
        self.norm = HybridNormalization(out_c)
        
        # Cross-attention
        self.cc_attn = CrissCrossAttention(out_c)
        
        # Downsampling
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.axial_conv(x)
        x = self.norm(x)
        x = self.cc_attn(x)
        return x, self.down(x)

class DeepDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Feature fusion
        self.conv1 = nn.Conv2d(in_c * 2, in_c, 1)
        
        # Axial convolution
        self.axial = AxialDepthwiseConv(in_c, out_c)
        
        # Cross-attention
        self.cc_attn = CrissCrossAttention(out_c)
        
        # Final convolution
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

class ImprovedBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Multi-branch structure
        self.branch1 = AxialDepthwiseConv(channels, channels, kernel_length=3)
        self.branch2 = AxialDepthwiseConv(channels, channels, kernel_length=5)
        self.branch3 = AxialDepthwiseConv(channels, channels, kernel_length=7)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False)
        )
        
        # Channel attention
        self.ca = CALayer(channels)
        
        # Final convolution
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

class BranchNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[24, 48, 96]):
        super().__init__()
        # Input convolution
        self.in_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)
        
        # Encoder
        self.encoder1 = DeepEncoderBlock(features[0], features[1])
        self.encoder2 = DeepEncoderBlock(features[1], features[2])
        
        # Bottleneck
        self.bottleneck = ImprovedBottleneck(features[2])
        
        # Decoder
        self.decoder1 = DeepDecoderBlock(features[2], features[1])
        self.decoder2 = DeepDecoderBlock(features[1], features[0])
        
        # Output layer
        self.out_conv = nn.Conv2d(features[0], out_channels, 1)
        
    def forward(self, x):
        x = self.in_conv(x)
        
        # Encoder path
        x0, x = self.encoder1(x)
        x1, x = self.encoder2(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        x = self.decoder1(x1, x)
        x = self.decoder2(x0, x)
        
        return self.out_conv(x)

class WhitePointBranch(nn.Module):
    def __init__(self, in_channels=3, features=[24, 48, 96]):
        super().__init__()
        # Input convolution
        self.in_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)
        
        # Encoder
        self.encoder1 = DeepEncoderBlock(features[0], features[1])
        self.encoder2 = DeepEncoderBlock(features[1], features[2])
        
        # Bottleneck
        self.bottleneck = ImprovedBottleneck(features[2])
        
        # Global pooling and output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.out_conv = nn.Conv2d(features[2], 3, 1)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        x = self.in_conv(x)
        
        # Encoder path
        _, x = self.encoder1(x)
        _, x = self.encoder2(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Global pooling and output 
        x = self.global_pool(x)
        x = self.out_conv(x)
        x = self.relu(x)
        
        return x  # Shape: B x 3 x 1 x 1 -> Will be reshaped in main model

class UIENet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[24, 48, 96]):
        super(UIENet, self).__init__()

        self.gaussian_blur = GaussianBlur(kernel_size=25, sigma=13)
        
        # Create four branches based on LUSANet
        # 1. Clear image branch (J0)
        self.clear_branch = BranchNet(in_channels, out_channels, features)
        
        # 2. Backscatter branch (B)
        self.back_branch = BranchNet(in_channels, out_channels, features)
        
        # 3. Transmission branch (T)
        self.trans_branch = BranchNet(in_channels, out_channels, features)
        
        # 4. White point branch (W)
        self.white_branch = WhitePointBranch(in_channels, features)
        
        # Flag for testing vs training mode
        self.output_all_components = False
    
    def normalize_output(self, x):
        # Normalize output to be in reasonable range without using sigmoid
        x_min = x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        
        # Add small epsilon to avoid division by zero
        x = (x - x_min) / (x_max - x_min + 1e-8)
        return x
    
    def forward(self, x):
        # Get clear image (J0)
        clear = self.clear_branch(x)
        clear = self.normalize_output(clear)
        
        # Only compute other branches during training or if output_all_components is True
        if self.training or self.output_all_components:
            # Get backscatter (B)
            # back = self.gaussian_blur(x)
            back = self.back_branch(x)
            back = self.normalize_output(back)
            
            # Get transmission (T)
            trans = self.trans_branch(x)
            trans = self.normalize_output(trans)
            
            # Get white point (W)
            white = self.white_branch(x)  # Shape: B x 3 x 1 x 1
            white = white.squeeze(-1).squeeze(-1)  # Shape: B x 3
            white = F.relu(white)  # Ensure positive values
            
            return clear, back, trans, white
        else:
            # During inference, only return the clear image unless output_all_components is True
            return clear

if __name__ == '__main__':
    # Test the model
    model = UIENet()
    x = torch.randn(2, 3, 256, 256)
    
    # Test training mode
    model.train()
    outputs = model(x)
    if isinstance(outputs, tuple):
        clear, back, trans, white = outputs
        print(f"Training mode outputs:")
        print(f"- Clear image shape: {clear.shape}")
        print(f"- Backscatter shape: {back.shape}") 
        print(f"- Transmission shape: {trans.shape}")
        print(f"- White point shape: {white.shape}")
    
    # Test evaluation mode
    model.eval()
    output = model(x)
    if isinstance(output, tuple):
        print(f"\nTesting mode (output_all_components=True):")
        clear, back, trans, white = output
        print(f"- Clear image shape: {clear.shape}")
        print(f"- Backscatter shape: {back.shape}")
        print(f"- Transmission shape: {trans.shape}")
        print(f"- White point shape: {white.shape}")
    else:
        print(f"\nTesting mode (output_all_components=False):")
        print(f"- Clear image shape: {output.shape}")
    
    # Test with output_all_components flag
    model.eval()
    model.output_all_components = True
    output = model(x)
    if isinstance(output, tuple):
        print(f"\nTesting mode (output_all_components=True):")
        clear, back, trans, white = output
        print(f"- Clear image shape: {clear.shape}")
        print(f"- Backscatter shape: {back.shape}")
        print(f"- Transmission shape: {trans.shape}")
        print(f"- White point shape: {white.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}") 