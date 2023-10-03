import torch

from voxel_transformer.heads.head import Head

# Occupancy head based on a 3D deconvolutional network

class DeconvolutionalHead(Head):
    
    def forward(self, fused_features):
        print("Deconvolutional head forward")

        pass
