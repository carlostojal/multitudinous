import torch

from abc import abstractmethod, ABC
from typing import Type, TypeVar, Generic

# import the point cloud backbones
from voxel_transformer.backbones.point_cloud.point_cloud_backbone import PointCloudBackbone

# import the rgbd backbones
from voxel_transformer.backbones.rgbd.rgbd_backbone import RGBDBackbone

# import the necks
from voxel_transformer.necks.neck import Neck

# import the heads
from voxel_transformer.heads.head import Head

# define abstract types bound to the abstract networks
PointCloudBackboneT = TypeVar('PointCloudBackboneT', bound=PointCloudBackbone)
RGBDBackboneT = TypeVar('RGBDBackboneT', bound=RGBDBackbone)
NeckT = TypeVar('NeckT', bound=Neck)
HeadT = TypeVar('HeadT', bound=Head)


# Abstract voxel transformer implementation. Can have any combination of backbones, neck and heads.
class AbstractVoxelTransformer(ABC, torch.nn.Module, Generic[PointCloudBackboneT, RGBDBackboneT, NeckT, HeadT]):

    def forward(self, point_cloud, rgbd):

        # extract point cloud features using the point cloud backbone
        point_cloud_features = self.point_cloud_backbone(point_cloud)

        # extract rgbd features using the rgbd backbone
        rgbd_features = self.rgbd_backbone(rgbd)

        # fuse the features using the neck
        fused_features = self.neck(point_cloud_features, rgbd_features)

        # reconstruct the occupancy using the occupancy head
        occupancy = self.occupancy_head(fused_features)

        return occupancy
