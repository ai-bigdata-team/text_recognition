import torch
import sys
import os
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vgg_transformer import CNNNetwork

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            setattr(self, key, value)

def test_cnn_output_shape():
    """Test CNN output shape with different input sizes"""
    config = Config("vgg_transformer/config/cnn.yaml")
    model = CNNNetwork(config)
    
    # Test case 1: Standard input
    batch_size = 4
    height = 32
    width = 128
    x = torch.randn(batch_size, 3, height, width)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Calculate expected output dimensions
    # After 4 pooling layers with stride 2: H/16, W/16
    expected_h = height // (2 ** 4)
    expected_w = width // (2 ** 4)
    expected_shape = (batch_size, 512, expected_h, expected_w)
    
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"Test passed! Expected shape: {expected_shape}")
    
    # Test case 2: Different input size
    x2 = torch.randn(2, 3, 64, 256)
    output2 = model(x2)
    print(f"\nInput shape 2: {x2.shape}")
    print(f"Output shape 2: {output2.shape}")
    
    expected_shape2 = (2, 512, 64//16, 256//16)
    assert output2.shape == expected_shape2, f"Expected {expected_shape2}, got {output2.shape}"
    print(f"Test 2 passed! Expected shape: {expected_shape2}")

def test_cnn_layers():
    """Test number of layers in CNN"""
    config = Config("vgg_transformer/config/cnn.yaml")
    print(f"Configuration of model: {config}")
    model = CNNNetwork(config)
    
    # Count layers: 4 blocks * 3 layers (Conv + ReLU + MaxPool) = 12 layers
    expected_layers = 12
    actual_layers = len(model.cnn)
    
    print(f"\nNumber of layers: {actual_layers}")
    assert actual_layers == expected_layers, f"Expected {expected_layers} layers, got {actual_layers}"
    print(f"Layer count test passed!")

if __name__ == "__main__":
    print("Running CNN shape tests...\n")
    test_cnn_output_shape()
    test_cnn_layers()
    print("\nAll tests passed!")