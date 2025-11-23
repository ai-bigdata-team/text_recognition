import torch
import torch.nn as nn

class CNNNetwork(nn.Module):
    def __init__(self, config):
        super(CNNNetwork, self).__init__()
        self.channels_list = config.channels_list # (n+1)x1
        self.kernel_size = config.kernel_size
        self.padding = config.padding 
        self.stride = config.stride
        self.n_layers = len(config.channels_list) - 1
        self.pooling_kernel_size = config.pooling_kernel_size
        self.pooling_stride = config.pooling_stride
        self.pooling_padding = config.pooling_padding
        self.layers = []
        for i in range (self.n_layers):
            conv_layer = nn.Conv2d(
                in_channels = self.channels_list[i],
                out_channels = self.channels_list[i+1],
                kernel_size = self.kernel_size,
                padding  = self.padding,
                stride = self.stride,
                bias = True
            )
            pooling_layer = nn.MaxPool2d(
                kernel_size = self.pooling_kernel_size,
                stride = self.pooling_stride,
                padding = self.pooling_padding,

            )
            self.layers.extend([conv_layer, nn.ReLU(), pooling_layer])
        self.cnn = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.cnn(x)
            