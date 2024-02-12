import torch

from abc import abstractmethod, ABC
from typing import Type, TypeVar, Generic
from threading import Thread

# import the point cloud backbones
from multitudinous.backbones.point_cloud.point_cloud_backbone import PointCloudBackbone

# import the rgbd backbones
from multitudinous.backbones.rgbd.rgbd_backbone import RGBDBackbone

# import the necks
from multitudinous.necks.neck import Neck

# import the heads
from multitudinous.heads.head import Head

# define abstract types bound to the abstract networks
PointCloudBackboneT = TypeVar('PointCloudBackboneT', bound=PointCloudBackbone)
RGBDBackboneT = TypeVar('RGBDBackboneT', bound=RGBDBackbone)
NeckT = TypeVar('NeckT', bound=Neck)
HeadT = TypeVar('HeadT', bound=Head)


# Abstract voxel transformer implementation. Can have any combination of backbones, neck and heads.
class AbstractVoxelTransformer(ABC, torch.nn.Module, Generic[PointCloudBackboneT, RGBDBackboneT, NeckT, HeadT]):

    def forward(self, point_cloud: torch.Tensor, rgbd: torch.Tensor) -> torch.Tensor:

        point_cloud_features = None
        rgbd_features = None

        # a routine to extract point cloud features
        def run_point_cloud_backbone():
            nonlocal point_cloud_features
            point_cloud_features = self.point_cloud_backbone(point_cloud)

        # a routine to extract rgbd features
        def run_rgbd_backbone():
            nonlocal rgbd_features
            rgbd_features = self.rgbd_backbone(rgbd)

        # create a thread to extract point cloud features and another to extract rgbd features
        point_cloud_thread = Thread(target=run_point_cloud_backbone)
        rgbd_thread = Thread(target=run_rgbd_backbone)

        # start the threads concurrently
        point_cloud_thread.start()
        rgbd_thread.start()

        # wait for the threads
        point_cloud_thread.join()
        rgbd_thread.join()

        # fuse the features using the neck
        fused_features = self.neck(point_cloud_features, rgbd_features)

        # reconstruct the occupancy using the occupancy head
        occupancy = self.occupancy_head(fused_features)

        return occupancy
