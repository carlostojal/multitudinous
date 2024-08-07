import torch
from torch import nn, Tensor
from torch.quantization import QuantStub, DeQuantStub
from torch.ao.nn.quantized import FloatFunctional
from abc import ABC
from typing import Type, Union
from multitudinous.backbones.image.attention import SqueezeAndExcitation, ConvolutionalBlockAttentionModule

class BottleneckBlock(nn.Module):
    expansion: int = 4
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 expansion: int = 4,
                 stride: int = 1,
                 with_dropout: bool = True):
        super().__init__()

        self.expansion = expansion
        self.with_dropout = with_dropout

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        if with_dropout:
            self.dropout1 = nn.Dropout2d(p=0.5)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if with_dropout:
            self.dropout2 = nn.Dropout2d(p=0.5)

        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        if with_dropout:
            self.dropout3 = nn.Dropout2d(p=0.5)

        self.shortcut = None
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * self.expansion)
            )

        self.relu = nn.ReLU(inplace=True)

        self.residual = FloatFunctional()

    # returns the output of the block and the identity
    def _forward_impl(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # keep the identity
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.with_dropout:
            out = self.dropout1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.with_dropout:
            out = self.dropout2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.with_dropout:
            out = self.dropout3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        return out, identity

    def forward(self, x: Tensor) -> Tensor:

        x, identity = self._forward_impl(x)

        # add the residual connection
        x = self.residual.add(x, identity)

        x = self.relu(x)

        return x

    
class SEBottleneckBlock(BottleneckBlock):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 expansion: int = 4,
                 stride: int = 1,
                 with_dropout: bool = True):
        super().__init__(in_channels, channels, expansion, stride, with_dropout)
        self.se = SqueezeAndExcitation(channels * expansion)

    def forward(self, x: Tensor) -> Tensor:

        x, identity = self._forward_impl(x)

        # apply the squeeze and excitation
        x = self.se(x)

        # add the residual connection
        x = self.residual.add(x, identity)

        x = self.relu(x)

        return x


class CBAMBottleneckBlock(BottleneckBlock):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 expansion: int = 4,
                 stride: int = 1,
                 with_dropout: bool = True):
        super().__init__(in_channels, channels, expansion, stride, with_dropout)
        self.cbam = ConvolutionalBlockAttentionModule(channels * expansion)

    def forward(self, x: Tensor) -> Tensor:

        x, identity = self._forward_impl(x)

        # apply the convolutional block attention module
        x = self.cbam(x)

        # add the residual connection
        x = self.residual.add(x, identity)

        x = self.relu(x)

        return x


class ResNet(ABC, nn.Module):
    def __init__(self, block: Type[Union[BottleneckBlock,SEBottleneckBlock,CBAMBottleneckBlock]], in_channels: int = 4, with_dropout: bool = True):
        super().__init__()
        self.block: BottleneckBlock = block
        self.in_channels = in_channels
        self.with_dropout = with_dropout
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass

class ResNet50(ResNet):
    def __init__(self, block: Type[Union[BottleneckBlock,SEBottleneckBlock,CBAMBottleneckBlock]] = BottleneckBlock, in_channels: int = 4, with_dropout: bool = True):
        super().__init__(block, in_channels, with_dropout)

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = 64

        self.layer1 = self._make_layer(64, 3, stride=1) # 64 -> 256
        self.layer2 = self._make_layer(128, 4, stride=2) # 128 -> 512
        self.layer3 = self._make_layer(256, 6, stride=2) # 256 -> 1024
        self.layer4 = self._make_layer(512, 3, stride=2) # 512 -> 2048

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _make_layer(self, channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(self.block(self.inplanes, channels, stride=stride, with_dropout=self.with_dropout))
        self.inplanes = channels * self.block.expansion
        for _ in range(1, n_blocks):
            layers.append(self.block(self.inplanes, channels, stride=1))
        return nn.Sequential(*layers)
        

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        x = self.quant(x)
        
        x = self.conv1(x)
        x1 = self.bn1(x)
        x = self.maxpool(x1) # 64 channels

        x2 = self.layer1(x) # 256 channels
        x3 = self.layer2(x2) # 512 channels
        x4 = self.layer3(x3) # 1024 channels
        out = self.layer4(x4) # 2048 channels

        x1 = self.dequant(x1)
        x2 = self.dequant(x2)
        x3 = self.dequant(x3)
        x4 = self.dequant(x4)
        out = self.dequant(out)

        return x1, x2, x3, x4, out

class SEResNet50(ResNet50):
    def __init__(self, in_channels: int):
        super().__init__(block=SEBottleneckBlock, in_channels=in_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return super().forward(x)

class CBAMResNet50(ResNet50):
    def __init__(self, in_channels: int):
        super().__init__(block=CBAMBottleneckBlock, in_channels=in_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return super().forward(x)
