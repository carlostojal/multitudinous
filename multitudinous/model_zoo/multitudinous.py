import torch
from torch import nn
from multitudinous.necks.concat import ConcatNeck
from multitudinous.heads.deconv import DeconvHead
from threading import Thread

# Abstract voxel transformer implementation. Can have any combination of backbones, neck and heads.
class Multitudinous(torch.nn.Module):

    def __init__(self, img_backbone: nn.Module, point_cloud_backbone: nn.Module) -> None:
        super().__init__()

        self.img_backbone = img_backbone
        self.point_cloud_backbone = point_cloud_backbone
        self.neck = ConcatNeck()
        self.head = DeconvHead()


    def forward(self, point_cloud: torch.Tensor, rgbd: torch.Tensor) -> torch.Tensor:

        point_cloud_features = None
        rgbd_features = None

        # a routine to extract point cloud features
        def run_point_cloud_backbone():
            nonlocal point_cloud_features
            point_cloud_features = self.point_cloud_backbone(point_cloud)

        # a routine to extract rgbd features
        def run_img_backbone():
            nonlocal rgbd_features
            rgbd_features = self.img_backbone(rgbd)

        # create a thread to extract point cloud features and another to extract rgbd features
        point_cloud_thread = Thread(target=run_point_cloud_backbone)
        rgbd_thread = Thread(target=run_img_backbone)

        # start the threads concurrently
        point_cloud_thread.start()
        rgbd_thread.start()

        # wait for the threads
        point_cloud_thread.join()
        rgbd_thread.join()

        # fuse the features using the neck
        fused_features = self.neck(point_cloud_features, rgbd_features)

        # reconstruct the occupancy using the occupancy head
        occupancy = self.head(fused_features)

        return occupancy
