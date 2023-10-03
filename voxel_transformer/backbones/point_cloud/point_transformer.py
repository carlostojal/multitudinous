import torch
from voxel_transformer.backbones.point_cloud.point_cloud_backbone import PointCloudBackbone

# Point cloud backbone using the "Point Transformer" architecture.
class PointTransformer(PointCloudBackbone):

    def forward(self, point_cloud):
        print("PointTransformer forward")

        pass
