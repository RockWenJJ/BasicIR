import torch
from basicir.models.archs.lusanetablation_arch import (
    create_lusanet_reference,
    create_lusanet_no_axial,
    create_lusanet_no_hybrid_norm,
    create_lusanet_no_cc_attn,
    create_lusanet_no_improved_bottleneck,
    create_lusanet_no_ca,
    create_lusanet_vanilla
)

def run_ablation_test():
    # Create sample input
    x = torch.randn(1, 3, 256, 256)
    
    # Dictionary to store models
    models = {
        'LUSANet (Reference)': create_lusanet_reference(),
        'LUSANet w/o AxialDepthwiseConv': create_lusanet_no_axial(),
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