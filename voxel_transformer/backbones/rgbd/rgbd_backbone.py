import torch
from abc import ABC, abstractmethod

class RGBDBackbone(ABC, torch.nn.Module):
    
    @abstractmethod
    def forward(self, rgbd: torch.Tensor):
        pass