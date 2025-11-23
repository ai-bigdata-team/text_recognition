import torch 
import torch.nn as nn
from attention import MultiheadAttention
from mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.self_attn = MultiheadAttention(config)
        self.fc = MLP(config)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x, mask = None):
        attn_out = self.self_attn(x, mask = mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        linear_out = self.fc(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

