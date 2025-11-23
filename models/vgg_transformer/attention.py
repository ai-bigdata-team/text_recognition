import torch 
import torch.nn  as nn
import math
import torch.nn.functional as F

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim = -1)
    values = torch.matmul(attention, v)
    return values, attention

def expand_mask(mask):
    assert mask.ndim >=2
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.n_heads 
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        # stack Q, K, V for efficiency
        self.qkv_proj = nn.Linear(self.d_model, 3*self.d_model) 
        self.o_proj = nn.Linear(self.d_model, self.d_model)
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x) # (B, T, D) -> (B, T, 3*D)
        qkv = qkv.reshape(batch_size, seq_length, self.n_heads, 3*self.d_kv)
        qkv = qkv.permute(0,2,1,3) # (B, H, T, 3*D_KV)
        q, k, v = qkv.chunk(3, dim = -1) # (B, H, T, D_KV)

        values, attention = scaled_dot_product(q,k,v, mask=mask) # (B, H, T, D_KV)
        values = values.permute(0,2,1,3) # (B, T, H, D_KV)
        values = values.reshape(batch_size, seq_length, self.d_model)
        o = self.o_proj(values)
        if return_attention:
            return o, attention
        else:
            return o
        
