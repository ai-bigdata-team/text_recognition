import torch
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from vgg_transformer import MultiheadAttention, scaled_dot_product, expand_mask

class Config:
    def __init__(self):
        self.d_model = 512
        self.d_kv = 512
        self.n_heads = 8

def test_scaled_dot_product():
    """Test scaled dot product attention"""
    print("Testing scaled_dot_product...")
    
    batch_size = 2
    n_heads = 4
    seq_len = 10
    d_head = 64
    
    q = torch.randn(batch_size, n_heads, seq_len, d_head)
    k = torch.randn(batch_size, n_heads, seq_len, d_head)
    v = torch.randn(batch_size, n_heads, seq_len, d_head)
    
    values, attention = scaled_dot_product(q, k, v)
    
    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    print(f"V shape: {v.shape}")
    print(f"Output values shape: {values.shape}")
    print(f"Attention weights shape: {attention.shape}")
    
    assert values.shape == (batch_size, n_heads, seq_len, d_head)
    assert attention.shape == (batch_size, n_heads, seq_len, seq_len)
    
    # Check attention weights sum to 1
    attention_sum = attention.sum(dim=-1)
    assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-6)
    
    print("✓ scaled_dot_product test passed!\n")

def test_expand_mask():
    """Test mask expansion"""
    print("Testing expand_mask...")
    
    # Test 2D mask -> 4D
    mask_2d = torch.ones(5, 10)
    expanded = expand_mask(mask_2d)
    print(f"2D mask shape: {mask_2d.shape} -> {expanded.shape}")
    assert expanded.ndim == 4
    
    # Test 3D mask -> 4D
    mask_3d = torch.ones(2, 5, 10)
    expanded = expand_mask(mask_3d)
    print(f"3D mask shape: {mask_3d.shape} -> {expanded.shape}")
    assert expanded.ndim == 4
    assert expanded.shape[1] == 1  # unsqueeze at dim 1
    
    print("✓ expand_mask test passed!\n")

def test_multihead_attention_shape():
    """Test MultiheadAttention output shape"""
    print("Testing MultiheadAttention shape...")
    
    config = Config()
    model = MultiheadAttention(config)
    
    batch_size = 4
    seq_len = 20
    d_model = 512
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test without mask
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ Output shape test passed!")
    
    # Test with mask
    mask = torch.ones(batch_size, seq_len, seq_len)
    output_masked = model(x, mask=mask)
    assert output_masked.shape == (batch_size, seq_len, d_model)
    print("✓ Masked output shape test passed!")
    
    # Test return attention
    output, attention = model(x, return_attention=True)
    print(f"Attention shape: {attention.shape}")
    assert attention.shape == (batch_size, config.n_heads, seq_len, seq_len)
    print("✓ Return attention test passed!\n")

def test_multihead_attention_parameters():
    """Test MultiheadAttention parameters"""
    print("Testing MultiheadAttention parameters...")
    
    config = Config()
    model = MultiheadAttention(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Check qkv_proj dimensions
    assert model.qkv_proj.weight.shape == (3 * config.d_kv, config.d_model)
    assert model.qkv_proj.bias.shape == (3 * config.d_kv,)
    
    # Check o_proj dimensions
    assert model.o_proj.weight.shape == (config.d_model, config.d_kv)
    assert model.o_proj.bias.shape == (config.d_model,)
    
    print("✓ Parameters test passed!\n")

def test_multihead_attention_different_configs():
    """Test with different configurations"""
    print("Testing different configurations...")
    
    configs = [
        {"d_model": 256, "d_kv": 256, "n_heads": 4},
        {"d_model": 512, "d_kv": 512, "n_heads": 8},
        {"d_model": 768, "d_kv": 768, "n_heads": 12},
    ]
    
    for cfg in configs:
        class TestConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        config = TestConfig(**cfg)
        model = MultiheadAttention(config)
        
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, cfg["d_model"])
        
        output = model(x)
        assert output.shape == (batch_size, seq_len, cfg["d_model"])
        print(f"✓ Config {cfg} passed!")
    
    print("✓ All configurations test passed!\n")

if __name__ == "__main__":
    print("Running MultiheadAttention tests...\n")
    print("=" * 60)
    
    test_scaled_dot_product()
    test_expand_mask()
    test_multihead_attention_shape()
    test_multihead_attention_parameters()
    test_multihead_attention_different_configs()
    
    print("=" * 60)
    print("✓ All tests passed successfully!")