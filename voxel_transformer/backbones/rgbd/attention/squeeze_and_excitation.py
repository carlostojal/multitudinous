import torch
from enum import Enum

# Squeeze-and-excitation attention module immplementation

class PoolingMethod(Enum):
    MAX_POOL = 1
    AVG_POOL = 2

class SE(torch.nn.Module):

    def __init__(self, in_size: int, reduction_ratio: float, pooling_method: PoolingMethod):
        super().__init__()

        # -- squeeze --
        # TODO: review kernel sizes and stride
        if pooling_method == PoolingMethod.MAX_POOL:
            self.pool = torch.nn.MaxPool2d(in_size)
        elif pooling_method == PoolingMethod.AVG_POOL:
            self.pool = torch.nn.AvgPool2d(in_size)
        else:
            raise Exception("Pooling method not supported!")

        # -- excitation --

        # first fully-connected layer
        self.fc1 = torch.nn.Linear(in_features=in_size, out_features=int(in_size/reduction_ratio))

        # relu activation
        self.relu = torch.nn.ReLU()

        # second fully-connected layer
        self.fc2 = torch.nn.Linear(in_features=int(in_size/reduction_ratio), out_features=in_size)

        # sigmoid activation
        self.sigmoid = torch.nn.Sigmoid()
