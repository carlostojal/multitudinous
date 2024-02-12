import torch

from abc import ABC, abstractmethod

class Neck(ABC, torch.nn.Module):

    @abstractmethod
    def forward(self, point_cloud_features, rgbd_features):
        pass
