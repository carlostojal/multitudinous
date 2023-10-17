import torch
from abstract_resnet50 import AbstractResNet50

# the vanilla resnet (no attention) uses None as attention type
class ResNet50(AbstractResNet50[None]):

    def __init__(self, out_feature_len: int):
        super().__init__(out_feature_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
