import torch

from abc import abstractmethod, ABC

# import the point cloud backbones
from backbones.point_cloud import PointCloudBackbone

# import the rgbd backbones
from backbones.rgbd import RGBDBackbone

# import the necks
from necks import Neck

# import the heads
from heads import Head

# Abstract voxel transformer implementation. Can have any combination of backbones, neck and heads.
class AbstractVoxelTransformer(ABC, torch.nn.Module):

    @property
    @abstractmethod
    def point_cloud_backbone(self) -> PointCloudBackbone:
        pass

    @property
    @abstractmethod
    def rgbd_backbone(self) -> RGBDBackbone:
        pass

    @property
    @abstractmethod
    def neck(self) -> Neck:
        pass

    @property
    @abstractmethod
    def occupancy_head(self) -> Head:
        pass

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
