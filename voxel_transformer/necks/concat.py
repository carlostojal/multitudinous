import torch

from voxel_transformer.necks.neck import Neck

# Simple concatenation feature fusion neck

class ConcatNeck(Neck):
    
    def forward(self, point_cloud_features, rgbd_features):
        print("Neck forward")

        pass
