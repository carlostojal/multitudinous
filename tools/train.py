import torch

import sys
sys.path.append(".")

from multitudinous.model_zoo.multitudinous_resnet50 import VoxelTransformer_ResNet50

# Run the training

if __name__ == "__main__":

    # detect cuda availability
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # initialize the model
    model = VoxelTransformer_ResNet50()

    # transfer the model to the cpu
    model.to(device)

    print(model)

    # TODO

    # generate random input tensors
    random_pcl = torch.rand((1, 10000)).to(device)
    random_rgbd = torch.rand((1, 4, 224, 224)).to(device)

    # test forward pass with random tensors (with correct dimensions)
    out = model(random_pcl, random_rgbd)
