import torch

from abc import abstractmethod

from voxel_transformer.model_zoo.abstract_voxel_transformer import AbstractVoxelTransformer

# import the point cloud backbone
from voxel_transformer.backbones.point_cloud.point_cloud_backbone import PointCloudBackbone
from voxel_transformer.backbones.point_cloud.point_transformer import PointTransformer

# import the rgbd backbone
from voxel_transformer.backbones.rgbd.rgbd_backbone import RGBDBackbone
from voxel_transformer.backbones.rgbd.resnet50 import ResNet50

# import the neck
from voxel_transformer.necks.neck import Neck
from voxel_transformer.necks.concat import ConcatNeck

# import the occupancy head
from voxel_transformer.heads.head import Head
from voxel_transformer.heads.deconvolutional import DeconvolutionalHead

class VoxelTransformer_ResNet(AbstractVoxelTransformer):

    @property
    def point_cloud_backbone(self) -> PointCloudBackbone:
        return PointTransformer()
 
    @property
    def rgbd_backbone(self) -> RGBDBackbone:
        return ResNet50()

    @property
    def neck(self) -> Neck:
        return ConcatNeck()

    @property
    def occupancy_head(self) -> Head:
        return DeconvolutionalHead()


    def __init__(self):
        super().__init__()
