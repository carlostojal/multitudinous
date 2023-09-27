import torch

from abc import abstractmethod

from voxel_transformer import AbstractVoxelTransformer

# import the point cloud backbone
from backbones.point_cloud import PointCloudBackbone, PointTransformer

# import the rgbd backbone
from backbones.rgbd import RGBDBackbone, ResNet50

# import the neck
from necks import Neck, ConcatNeck

# import the occupancy head
from heads import Head, DeconvolutionalHead

class VoxelTransformer_ResNet(AbstractVoxelTransformer):

    @property
    def point_cloud_backbone(self) -> PointCloudBackbone:
        return PointTransformer()

    @property
    @abstractmethod
    def rgbd_backbone(self) -> RGBDBackbone:
        return ResNet50()

    @property
    @abstractmethod
    def neck(self) -> Neck:
        return ConcatNeck()

    @property
    @abstractmethod
    def occupancy_head(self) -> Head:
        return DeconvolutionalHead()


    def __init__(self):
        super().__init__()
