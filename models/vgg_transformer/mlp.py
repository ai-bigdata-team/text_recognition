import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config):
        self.d_model = config.d_model
        self.ffn_hidden_dim = config.ffn_hidden_dim
        self.act_fn = config.act_fn
        self.fc_in = nn.Linear(self.d_model, self.ffn_hidden_dim)
        self.fc_out = nn.Linear(self.ffn_hidden_dim, self.d_model)
    def forward(self,x):
        x = self.fc_in(x)
        if self.act_fn.lower == "relu":
            x = F.relu(x)
        else:
            print(f"actication fucnct: {self.act_fn} is not supported")
        x = self.fc_out(x)
        return x
    