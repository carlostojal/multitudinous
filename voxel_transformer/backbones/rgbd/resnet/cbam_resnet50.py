import torch

from voxel_transformer.backbones.rgbd.resnet.abstract_resnet50 import AbstractResNet50
from voxel_transformer.backbones.rgbd.attention.convolutional_block_attention_module import CBAM


class CBAMResNet50(AbstractResNet50[CBAM]):

    def __init__(self, out_features: int):
        super().__init__(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
