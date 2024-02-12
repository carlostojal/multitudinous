import torch
from abc import ABC, abstractmethod

class Head(ABC, torch.nn.Module):

    @abstractmethod
    def forward(self, fused_features):
        pass
