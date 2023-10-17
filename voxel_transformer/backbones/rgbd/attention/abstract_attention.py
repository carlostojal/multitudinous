import torch

from abc import ABC


# abstract attention module
class AbstractAttention(ABC, torch.nn.Module):

    def __init__(self, in_size: int):

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass