import torch

import sys
sys.path.append(".")

from voxel_transformer.model_zoo.voxel_transformer_resnet50 import VoxelTransformer_ResNet50
from voxel_transformer.model_zoo.voxel_transformer_seresnet50 import VoxelTransformer_SEResNet50

# Run the training

if __name__ == "__main__":

    # detect cuda availability
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # generate random input tensors
    random_pcl = torch.rand((1, 10000)).to(device)
    random_rgbd = torch.rand((1, 4, 224, 224)).to(device)

    print("------- ResNet-50 -------")

    # initialize the model
    model = VoxelTransformer_ResNet50()

    # transfer the model to the cpu
    model.to(device)

    print(model)

    # test forward pass with random tensors (with correct dimensions)
    out = model(random_pcl, random_rgbd)


    print("------- SE-ResNet-50 -------")

    model1 = VoxelTransformer_SEResNet50()
    model1.to(device)

    print(model1)

    out1 = model1(random_pcl, random_rgbd)

