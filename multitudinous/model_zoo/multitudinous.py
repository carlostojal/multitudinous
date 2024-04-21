import torch
from torch import nn
from threading import Thread

# Multitudinous implementation
class Multitudinous(torch.nn.Module):

    def __init__(self, img_backbone: nn.Module, point_cloud_backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()

        self.img_backbone = img_backbone
        self.point_cloud_backbone = point_cloud_backbone
        self.neck = neck
        self.head = head


    def forward(self, point_cloud: torch.Tensor, rgbd: torch.Tensor) -> torch.Tensor:

        # initialize the features
        rgbd_features = None
        point_cloud_features = None

        # a routine to extract rgbd features
        def run_img_backbone():
            nonlocal rgbd_features
            rgbd_features = self.img_backbone(rgbd)

        # a routine to extract point cloud features
        def run_point_cloud_backbone():
            nonlocal point_cloud_features
            point_cloud_features = self.point_cloud_backbone(point_cloud)

        # create a thread to extract point cloud features and another to extract rgbd features
        rgbd_thread = Thread(target=run_img_backbone)
        point_cloud_thread = Thread(target=run_point_cloud_backbone)

        # start the threads concurrently
        rgbd_thread.start()
        point_cloud_thread.start()

        # wait for the threads
        rgbd_thread.join()
        point_cloud_thread.join()

        # fuse the features using the neck
        rgbd_embeddings, point_cloud_embeddings = self.neck(point_cloud_features, rgbd_features)

        # decode the end task using the head, be it grid or cones
        out = self.head(rgbd_embeddings, point_cloud_embeddings)

        return out
