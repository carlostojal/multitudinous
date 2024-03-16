import torch
from torch import nn, Tensor
from torch.quantization import QuantStub, DeQuantStub
from abc import ABC
from typing import Type, Union, Optional
from multitudinous.backbones.image.attention import SqueezeAndExcitation, ConvolutionalBlockAttentionModule

class BottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 expansion: int,
                 stride: int = 1):
        super().__init__()

        self.expansion = expansion

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels * self.expansion)

        self.shortcut = None
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * self.expansion)
            )

        self.relu = nn.ReLU(inplace=True)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    # returns the output of the block and the identity
    def _forward_impl(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # keep the identity
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        return out, identity

    def forward(self, x: Tensor) -> Tensor:

        # quantize output
        x = self.quant(x)

        x, identity = self._forward_impl(x)

        # add the residual connection
        out += identity

        out = self.relu(out)

        # dequantize output
        out = self.dequant(out)

        return out

    
class SEBottleneckBlock(BottleneckBlock):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 expansion: int,
                 stride: int = 1):
        super().__init__(in_channels, channels, expansion, stride)
        self.se = SqueezeAndExcitation(channels * expansion)

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)

        x, identity = self._forward_impl(x)

        # apply the squeeze and excitation
        x = self.se(x)

        # add the residual connection
        x += identity

        x = self.relu(x)

        x = self.dequant(x)

        return x


class CBAMBottleneckBlock(BottleneckBlock):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 expansion: int,
                 stride: int = 1):
        super().__init__(in_channels, channels, expansion, stride)
        self.cbam = ConvolutionalBlockAttentionModule(channels * expansion)

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)

        x, identity = self._forward_impl(x)

        # apply the convolutional block attention module
        x = self.cbam(x)

        # add the residual connection
        x += identity

        x = self.relu(x)

        x = self.dequant(x)

        return x


class ResNet(ABC):
    def __init__(self, block: Type[Union[BottleneckBlock,SEBottleneckBlock,CBAMBottleneckBlock]], in_channels: int):
        self.block = block
        self.in_channels = in_channels
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass

class ResNet50(ResNet):
    def __init__(self, block: Optional[Type[Union[BottleneckBlock,SEBottleneckBlock,CBAMBottleneckBlock]]], in_channels: int):
        super().__init__(block, in_channels)

        # 3, 4, 6, 3
        self.layers = [3, 4, 6, 3]

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 8, stride=2)

    def _make_layer(self, in_channels: int, channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(self.block(in_channels, channels, stride=stride))
        for _ in range(1, n_blocks):
            layers.append(self.block(channels, channels, stride=1))
        return nn.Sequential(*layers)
        

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.maxpool(x) # 64 channels

        x2 = self.layer1(x1) # 256 channels
        x3 = self.layer2(x2) # 512 channels
        x4 = self.layer3(x3) # 1024 channels
        out = self.layer4(x4) # 2048 channels

        return x1, x2, x3, x4, out

class SEResNet50(ResNet50):
    def __init__(self, in_channels: int):
        super().__init__(SEBottleneckBlock, in_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return super().forward(x)

class CBAMResNet50(ResNet50):
    def __init__(self, in_channels: int):
        super().__init__(CBAMBottleneckBlock, in_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return super().forward(x)
