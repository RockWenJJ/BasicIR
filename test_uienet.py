import torch
from basicir.models.archs.uienet_arch import UIENet

def test_uienet():
    """Test the UIENet model"""
    # Create model
    model = UIENet(features=[32, 64, 128])
    
    # Create a test input tensor
    x = torch.randn(2, 3, 256, 256)
    
    # Test in training mode (should output all components)
    model.train()
    outputs = model(x)
    
    assert isinstance(outputs, tuple), "In training mode, model should return a tuple of outputs"
    assert len(outputs) == 4, "Model should return 4 outputs: clear, backscatter, transmission, white_point"
    
    clear, back, trans, white = outputs
    
    # Check shapes
    assert clear.shape == (2, 3, 256, 256), f"Clear image has incorrect shape: {clear.shape}"
    assert back.shape == (2, 3, 256, 256), f"Backscatter has incorrect shape: {back.shape}"
    assert trans.shape == (2, 3, 256, 256), f"Transmission has incorrect shape: {trans.shape}" 
    assert white.shape == (2, 3), f"White point has incorrect shape: {white.shape}"
    
    # Test in eval mode (should only output clear image)
    model.eval()
    output = model(x)
    
    assert not isinstance(output, tuple), "In evaluation mode, model should return only the clear image"
    assert output.shape == (2, 3, 256, 256), f"Clear image has incorrect shape: {output.shape}"
    
    # Test with output_all_components flag
    model.eval()
    model.output_all_components = True
    output = model(x)
    
    assert isinstance(output, tuple), "With output_all_components=True, model should return a tuple"
    assert len(output) == 4, "Model should return 4 outputs: clear, backscatter, transmission, white_point"
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    print("All tests passed!")

if __name__ == '__main__':
    test_uienet() 