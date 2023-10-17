import torch

from voxel_transformer.backbones.rgbd.resnet.abstract_resnet50 import AbstractResNet50
from voxel_transformer.backbones.rgbd.attention.squeeze_and_excitation import SE


class SEResNet50(AbstractResNet50[SE]):

    def __init__(self, out_features: int):
        super().__init__(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
