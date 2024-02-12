from typing import Type

import torch

from multitudinous.model_zoo.abstract_multitudinous import AbstractVoxelTransformer, PointCloudBackboneT

# import the point cloud backbone
from multitudinous.backbones.point_cloud.point_transformer import PointTransformer

# import the rgbd backbone
from multitudinous.backbones.rgbd.resnet50 import ResNet50

# import the neck
from multitudinous.necks.concat import ConcatNeck

# import the occupancy head
from multitudinous.heads.deconvolutional import DeconvolutionalHead


class VoxelTransformer_ResNet50(AbstractVoxelTransformer[PointTransformer, ResNet50, ConcatNeck, DeconvolutionalHead]):

    def __init__(self):
        super().__init__()

        self.point_cloud_backbone = PointTransformer()
        self.rgbd_backbone = ResNet50(out_feature_len=1000)
        self.neck = ConcatNeck()
        self.occupancy_head = DeconvolutionalHead()
