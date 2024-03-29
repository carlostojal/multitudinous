import torch
from abc import ABC, abstractmethod

# Point cloud backbone abstract class. Used to extract features from point clouds.
class PointCloudBackbone(ABC, torch.nn.Module):
    
    @abstractmethod
    def forward(self, point_cloud):
        pass
