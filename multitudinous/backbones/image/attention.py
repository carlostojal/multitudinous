import torch
from torch import nn, Tensor
from torch.ao.nn.quantized import FloatFunctional
from abc import ABC

# abstract attention module
class AttentionModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
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
        self.excitation = FloatFunctional()


    def forward(self, x: Tensor) -> Tensor:
        
        # squeeze
        f_sq = self.avg_pool(x) # average pooling (BxCx1x1)
        f_sq = f_sq.squeeze() # reshape to (C)

        # excitation
        f_ex = self.fc(f_sq)
        # reshape to (BxCx1x1)
        # if it doesn't have a batch dimension, add one
        if len(f_ex.shape) == 1:
            f_ex = f_ex.unsqueeze(0)
        f_ex = f_ex.unsqueeze(2).unsqueeze(3)

        return self.excitation.mul(x, f_ex) # multiply the input tensor by the excitation

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
        self.channel_attention = FloatFunctional()
        self.spatial_attention = FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:

        max = self.max_pool(x) # max pooling
        avg = self.avg_pool(x) # average pooling

        # reshape to (C)
        max = max.squeeze()
        avg = avg.squeeze()

        # channel attention
        max = self.fc(max) # apply fully connected layer (W0, W1)
        avg = self.fc(avg) # apply fully connected layer (W0, W1)
        mc = torch.functional.F.relu(max + avg)

        # reshape to (BxCx1x1)
        mc = mc.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # apply channel attention to the input tensor
        # (BxCxHxW) * (BxCx1x1)
        x1 = self.channel_attention.mul(x, mc)

        # spatial attention
        # (Bx1xHxW) - reduce dimension 1
        max_s, _ = torch.max(x1, dim=1, keepdim=True)
        avg_s = torch.mean(x1, dim=1, keepdim=True)

        pool_s = torch.cat([max_s, avg_s], dim=1) # concatenate the two tensors
        ms = torch.functional.F.relu(self.conv(pool_s)) # apply convolutional layer

        # apply spatial attention
        x1 = self.spatial_attention.mul(x1, ms)

        return x1
