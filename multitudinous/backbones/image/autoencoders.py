import torch
from torch import nn, Tensor
from typing import Type, Union
from multitudinous.backbones.image.resnet import ResNet50, SEResNet50, CBAMResNet50
    
class ResNetAutoEncoder(nn.Module):
    def __init__(self, encoder: Type[Union[ResNet50, SEResNet50, CBAMResNet50]], out_channels: int = 3, with_residuals: bool = False) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.with_residuals = with_residuals

        self.block1 = self._make_block(2048, 1024)
        self.block2 = self._make_block(1024, 512)
        self.block3 = self._make_block(512, 256)
        self.block4 = self._make_block(256, 64)

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False), # 2x2 upsample
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), # 3x3 conv
            nn.BatchNorm2d(out_channels), # batch norm
            nn.ReLU(inplace=True)
        )


    def forward(self, x: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:

        x1, x2, x3, x4, x5 = x

        if self.with_residuals:
            out = self.block1(x5) + x4
            out = self.block2(out) + x3
            out = self.block3(out) + x2
            out = self.block4(out) + x1
        else:
            out = self.block1(x5)
            out = self.block2(out)
            out = self.block3(out)
            out = self.block4(out)

        return out
