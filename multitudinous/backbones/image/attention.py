import torch
from torch import nn, Tensor
from abc import ABC

# abstract attention module
class AttentionModule(ABC):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class SqueezeAndExcitation(AttentionModule):
    
    def __init__(self,
                 in_channels: int,
                 reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )


    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvolutionalBlockAttentionModule(AttentionModule):
    
    def __init__(self,
                 in_channels,
                 reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x: Tensor) -> Tensor:

        max = self.max_pool(x) # max pooling
        avg = self.avg_pool(x) # average pooling

        # reshape to (C)
        max = max.squeeze()
        avg = avg.squeeze()

        # channel attention
        max = self.fc(max) # apply fully connected layer (W0, W1)
        avg = self.fc(avg) # apply fully connected layer (W0, W1)
        mc = self.gate_layer(max + avg)

        # reshape to (BxCx1x1)
        mc = mc.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # apply channel attention to the input tensor
        # (BxCxHxW) * (BxCx1x1)
        x1 = x * mc

        # spatial attention
        # (Bx1xHxW) - reduce dimension 1
        max_s, _ = torch.max(x1, dim=1, keepdim=True)
        avg_s = torch.mean(x1, dim=1, keepdim=True)

        pool_s = cat([max_s, avg_s], dim=1) # concatenate the two tensors
        ms = self.gate_layer(self.conv(pool_s)) # apply convolutional layer

        # apply spatial attention
        return x1 * ms
