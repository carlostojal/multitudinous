import torch
from enum import Enum
from voxel_transformer.backbones.rgbd.attention.abstract_attention import AbstractAttention

# Squeeze-and-excitation attention module immplementation


class SEPoolingMethod(Enum):
    MAX_POOL = 1
    AVG_POOL = 2


class SE(AbstractAttention):

    def __init__(self, in_size: int):
        super().__init__(in_size)

        # -- hyperparameters --

        # set the pooling method
        pooling_method = SEPoolingMethod.AVG_POOL
        # set the reduction ratio
        reduction_ratio = 2.0

        # -- squeeze --
        if pooling_method == SEPoolingMethod.MAX_POOL:
            self.pool = torch.nn.MaxPool2d(in_size)
        elif pooling_method == SEPoolingMethod.AVG_POOL:
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

    def forward(self, x):

        # squeeze
        x = self.pool(x)

        # excitation
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
