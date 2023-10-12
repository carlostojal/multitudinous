import torch
from voxel_transformer.model_zoo.voxel_transformer_resnet50 import VoxelTransformer_ResNet50

# Run the training

if __name__ == "__main__":

    model = VoxelTransformer_ResNet50()

    # TODO

    # generate random input tensors
    random_pcl = torch.rand((1, 10000))
    random_rgbd = torch.rand((3, 224, 224))

    # test forward pass with random tensors (with correct dimensions)
    out = model(random_pcl, random_rgbd)
