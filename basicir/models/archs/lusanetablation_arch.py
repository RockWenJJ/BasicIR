import torch
import torch.nn as nn
import torch.nn.functional as F
from .lusanet_arch import LUSANet, DeepEncoderBlock, DeepDecoderBlock, ImprovedBottleneck
from .lusanet_arch import AxialDepthwiseConv, HybridNormalization, CrissCrossAttention, CALayer

def create_lusanet_reference(in_channels=3, out_channels=3, features=[24, 48, 96]):
    """
    Creates the original LUSANet model to use as reference
    """
    return LUSANet(in_channels, out_channels, features)

# Standard implementations to replace the innovative components

# Standard Conv2d to replace AxialDepthwiseConv
class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length=5):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))

# BatchNorm only to replace HybridNormalization
class StandardNorm(nn.Module):
    def __init__(self, channels, ratio=0.7):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        return self.norm(x)

# Self-attention module with standard implementation to replace CrissCrossAttention
class StandardAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Reshape for attention computation
        proj_query = self.query_conv(x).view(batch_size, -1, height*width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height*width)
        
        # Standard attention
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        
        proj_value = self.value_conv(x).view(batch_size, -1, height*width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        out = self.gamma * out + x
        return out

# Standard bottleneck without multi-branch structure
class StandardBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Simple residual block
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

# Simple channel scaling to replace CALayer
class StandardChannelScaling(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.ones(in_channels))

    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.global_avg_pool(x).view(b, c)
        return x * scale.view(b, c, 1, 1)

# Dilated Conv2d to replace AxialDepthwiseConv
class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length=5):
        super().__init__()
        # Use dilated convolution with dilation=2 to increase receptive field
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=2, dilation=2
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))

class LUSANetAblation(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[24, 48, 96], 
                 use_axial=True, use_hybrid_norm=True, use_cc_attn=True, 
                 use_improved_bottleneck=True, use_ca=True):
        """
        Configurable LUSANet for ablation studies
        
        Args:
            in_channels: Input image channels
            out_channels: Output image channels
            features: Channel dimensions at each level
            use_axial: Whether to use AxialDepthwiseConv or standard conv
            use_hybrid_norm: Whether to use HybridNormalization or standard BatchNorm
            use_cc_attn: Whether to use CrissCrossAttention or standard attention
            use_improved_bottleneck: Whether to use ImprovedBottleneck or standard bottleneck
            use_ca: Whether to use CALayer or standard channel scaling
        """
        super().__init__()
        
        # Configure convolution type
        self.conv_layer = AxialDepthwiseConv if use_axial else StandardConv
        
        # Configure normalization type
        self.norm_layer = HybridNormalization if use_hybrid_norm else StandardNorm
        
        # Configure attention type
        self.attn_layer = CrissCrossAttention if use_cc_attn else StandardAttention
        
        # Configure channel attention
        self.ca_layer = CALayer if use_ca else StandardChannelScaling
        
        # Configure bottleneck
        self.bottleneck_layer = ImprovedBottleneck if use_improved_bottleneck else StandardBottleneck
        
        # Initial convolution
        self.in_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)
        
        # Create encoder blocks with configured layers
        self.encoder1 = self._create_encoder_block(features[0], features[1])
        self.encoder2 = self._create_encoder_block(features[1], features[2])
        
        # Bottleneck
        self.bottleneck = self.bottleneck_layer(features[2])
        
        # Create decoder blocks with configured layers
        self.decoder1 = self._create_decoder_block(features[2], features[1])
        self.decoder2 = self._create_decoder_block(features[1], features[0])
        
        # Output layer
        self.out_conv = nn.Conv2d(features[0], out_channels, 1)
    
    def _create_encoder_block(self, in_c, out_c):
        """Helper to create encoder block with configured layers"""
        return nn.ModuleDict({
            'conv': self.conv_layer(in_c, out_c),
            'norm': self.norm_layer(out_c),
            'attn': self.attn_layer(out_c),
            'down': nn.MaxPool2d(kernel_size=2, stride=2)
        })
    
    def _create_decoder_block(self, in_c, out_c):
        """Helper to create decoder block with configured layers"""
        return nn.ModuleDict({
            'up': nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            'conv1': nn.Conv2d(in_c * 2, in_c, 1),
            'conv_main': self.conv_layer(in_c, out_c),
            'attn': self.attn_layer(out_c),
            'conv2': nn.Conv2d(out_c, out_c, 1),
            'relu': nn.ReLU(inplace=False)
        })
        
    def _run_encoder_block(self, x, block):
        # Apply convolution, normalization and attention
        x = block['conv'](x)
        x = block['norm'](x)
        x = block['attn'](x)
        # Return feature map and downsampled feature map
        return x, block['down'](x)
    
    def _run_decoder_block(self, skip, x, block):
        # Upsample and concatenate with skip connection
        x = block['up'](x)
        x = torch.cat([skip, x], dim=1)
        x = block['conv1'](x)
        
        # Apply main convolution and attention
        x = block['conv_main'](x)
        x = block['attn'](x)
        
        # Final processing
        x = block['conv2'](block['relu'](x))
        return block['relu'](x)
    
    def forward(self, x):
        x = self.in_conv(x)
        
        # Encoder path
        x0, x = self._run_encoder_block(x, self.encoder1)
        x1, x = self._run_encoder_block(x, self.encoder2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        x = self._run_decoder_block(x1, x, self.decoder1)
        x = self._run_decoder_block(x0, x, self.decoder2)
        
        return self.out_conv(x)

def create_lusanet_no_axial(in_channels=3, out_channels=3, features=[24, 48, 96]):
    """LUSANet without AxialDepthwiseConv (using standard convolution instead)"""
    return LUSANetAblation(in_channels, out_channels, features, use_axial=False)

def create_lusanet_no_hybrid_norm(in_channels=3, out_channels=3, features=[24, 48, 96]):
    """LUSANet without HybridNormalization (using standard BatchNorm instead)"""
    return LUSANetAblation(in_channels, out_channels, features, use_hybrid_norm=False)

def create_lusanet_no_cc_attn(in_channels=3, out_channels=3, features=[24, 48, 96]):
    """LUSANet without CrissCrossAttention (using standard attention instead)"""
    return LUSANetAblation(in_channels, out_channels, features, use_cc_attn=False)

def create_lusanet_no_improved_bottleneck(in_channels=3, out_channels=3, features=[24, 48, 96]):
    """LUSANet without ImprovedBottleneck (using standard bottleneck instead)"""
    return LUSANetAblation(in_channels, out_channels, features, use_improved_bottleneck=False)

def create_lusanet_no_ca(in_channels=3, out_channels=3, features=[24, 48, 96]):
    """LUSANet without CALayer (using standard channel scaling instead)"""
    return LUSANetAblation(in_channels, out_channels, features, use_ca=False)

def create_lusanet_vanilla(in_channels=3, out_channels=3, features=[24, 48, 96]):
    """LUSANet with all innovative components removed (baseline)"""
    return LUSANetAblation(
        in_channels, out_channels, features,
        use_axial=False,
        use_hybrid_norm=False,
        use_cc_attn=False,
        use_improved_bottleneck=False,
        use_ca=False
    )

# Define LUSANetDilatedConv as a global class
class LUSANetDilatedConv(LUSANetAblation):
    """
    LUSANet variant that uses dilated convolutions instead of AxialDepthwiseConv
    
    This variant tests whether the benefits of AxialDepthwiseConv come from
    the increased receptive field (which dilated convolutions also provide)
    or from the axial decomposition itself.
    """
    def __init__(self, in_channels=3, out_channels=3, features=[24, 48, 96]):
        super().__init__(
            in_channels, out_channels, features,
            use_axial=False,  # Don't use axial convolution
            use_hybrid_norm=True,
            use_cc_attn=True,
            use_improved_bottleneck=True,
            use_ca=True
        )
        # Override the conv_layer with DilatedConv
        self.conv_layer = DilatedConv
        
        # Reinitialize the encoder and decoder blocks with the new conv_layer
        self.encoder1 = self._create_encoder_block(features[0], features[1])
        self.encoder2 = self._create_encoder_block(features[1], features[2])
        self.decoder1 = self._create_decoder_block(features[2], features[1])
        self.decoder2 = self._create_decoder_block(features[1], features[0])

# Factory function now simply instantiates the global class
def create_lusanet_dilated_conv(in_channels=3, out_channels=3, features=[24, 48, 96]):
    """
    Creates LUSANet with dilated convolutions instead of AxialDepthwiseConv
    """
    return LUSANetDilatedConv(in_channels, out_channels, features)

# Updated example to include the dilated convolution ablation
def run_ablation_test():
    # Create sample input
    x = torch.randn(1, 3, 256, 256)
    
    # Dictionary to store models
    models = {
        'LUSANet (Reference)': create_lusanet_reference(),
        'LUSANet w/o AxialDepthwiseConv': create_lusanet_no_axial(),
        'LUSANet w/ DilatedConv': create_lusanet_dilated_conv(),
        'LUSANet w/o HybridNormalization': create_lusanet_no_hybrid_norm(),
        'LUSANet w/o CrissCrossAttention': create_lusanet_no_cc_attn(),
        'LUSANet w/o ImprovedBottleneck': create_lusanet_no_improved_bottleneck(),
        'LUSANet w/o CALayer': create_lusanet_no_ca(),
        'LUSANet Vanilla (Baseline)': create_lusanet_vanilla()
    }
    
    # Print model parameters and output shape
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        output = model(x)
        print(f"{name}: Parameters={params:,}, Output Shape={output.shape}")

if __name__ == '__main__':
    run_ablation_test() 