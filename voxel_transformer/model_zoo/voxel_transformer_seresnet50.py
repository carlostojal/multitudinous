from voxel_transformer.model_zoo.abstract_voxel_transformer import AbstractVoxelTransformer

# import the point cloud backbone
from voxel_transformer.backbones.point_cloud.point_transformer import PointTransformer

# import the rgbd backbone
from voxel_transformer.backbones.rgbd.resnet.resnet50 import ResNet50

# import the neck
from voxel_transformer.necks.concat import ConcatNeck

# import the occupancy head
from voxel_transformer.heads.deconvolutional import DeconvolutionalHead

from voxel_transformer.backbones.rgbd.attention.squeeze_and_excitation import SE

from typing import Type


class VoxelTransformer_SEResNet50(AbstractVoxelTransformer[PointTransformer, ResNet50, ConcatNeck, DeconvolutionalHead]):

    def __init__(self):
        super().__init__()

        self.point_cloud_backbone = PointTransformer()
        self.rgbd_backbone = ResNet50(1000, attention_type=SE)
        self.neck = ConcatNeck()
        self.occupancy_head = DeconvolutionalHead()
