import torch
from torch import nn
from threading import Thread
from multitudinous.backbones.point_cloud.NDT_Netpp.ndtnetpp.preprocessing.ndtnet_preprocessing import ndt_preprocessing

# Multitudinous implementation
class Multitudinous(torch.nn.Module):

    def __init__(self, img_backbone: nn.Module, point_cloud_backbone: nn.Module, neck: nn.Module, head: nn.Module, embedding_dim: int = 768,
                 num_point_features: int = 1000, num_img_features: int = 300) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_point_features = num_point_features
        self.num_img_features = num_img_features
        self.img_backbone = img_backbone
        self.point_cloud_backbone = point_cloud_backbone
        self.neck = neck
        self.head = head

        # check if the model is on the GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, point_cloud: torch.Tensor, rgbd: torch.Tensor) -> torch.Tensor:

        # initialize the features
        rgbd_features = None
        point_cloud_features = None

        # ---- RGBD FEATURE EXTRACTION ----
        _, _, _, _, rgbd_features = self.img_backbone(rgbd)

        # flatten the features to (batch_size, n, embedding_dim)
        n_rgbd_features = (rgbd_features.shape[0] * rgbd_features.shape[1] * rgbd_features.shape[2] * rgbd_features.shape[3] // self.embedding_dim) * self.embedding_dim
        rgbd_features = torch.flatten(rgbd_features)[:n_rgbd_features].reshape(rgbd_features.shape[0], -1, self.embedding_dim)

        # ---- POINT CLOUD FEATURE EXTRACTION ----
        # if NDT-Net, use NDT preprocessing
        if self.point_cloud_backbone.__class__.__name__ == 'NDTNet':
            # apply NDT preprocessing
            point_cloud, covariances, _ = ndt_preprocessing(self.num_point_features, point_cloud)
            point_cloud_features, _ = self.point_cloud_backbone(point_cloud, covariances)
        else:
            point_cloud_features = self.point_cloud_backbone(point_cloud)

        # flatten the features to (batch_size, n, embedding_dim)
        n_point_cloud_features = (point_cloud_features.shape[0] * point_cloud_features.shape[1] * point_cloud_features.shape[2] // self.embedding_dim) * self.embedding_dim
        point_cloud_features = torch.flatten(point_cloud_features)[:n_point_cloud_features].reshape(point_cloud_features.shape[0], -1, self.embedding_dim)

        """
        # TODO: remove this after testing
        point_cloud_features = torch.rand((1, 1000, 768)).to(self.device)
        rgbd_features = torch.rand((1, 300, 768)).to(self.device)
        """

        # fuse the features using the neck
        rgbd_embeddings, point_cloud_embeddings = self.neck(point_cloud_features, rgbd_features)

        # decode the end task using the head, be it grid or cones
        out = self.head(rgbd_embeddings, point_cloud_embeddings)

        return out
