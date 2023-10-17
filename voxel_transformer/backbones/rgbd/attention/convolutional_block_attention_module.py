import torch
from voxel_transformer.backbones.rgbd.attention.abstract_attention import AbstractAttention

# "Convolutional Block Attention Module" attention


class CBAM(AbstractAttention):
    def __init__(self, in_size: int):
        super().__init__(in_size)

        # -- hyperparameters --
        reduction_ratio = 2.0


        # -- channel attention module
        # max and avg pool
        self.maxpool = torch.nn.MaxPool2d(in_size)
        self.avgpool = torch.nn.AvgPool2d(in_size)
        # mlp
        self.fc1 = torch.nn.Linear(in_features=in_size, out_features=int(in_size/reduction_ratio))
        self.fc2 = torch.nn.Linear(in_features=int(in_size/reduction_ratio), out_features=in_size)
        # sigmoid
        self.sigmoid = torch.nn.Sigmoid()

        # -- spatial attention module --
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_avg = self.avgpool(x)
        f_max = self.maxpool(x)

        f_avg1 = self.sigmoid(self.fc2(self.fc1(f_avg)))
        f_max1 = self.fc2(self.fc1(f_max))

        # compute channel attention
        channel_attention = f_avg1 + f_max1

        # compute average-pooled and max-pooled feature maps
        f_avg2 = x * f_avg
        f_max2 = x * f_max

        # compute spatial attention
        # concatenating on channel 0 generates a tensor of 2xWxH
        spatial_attention = self.sigmoid(self.conv1(torch.cat((f_avg2, f_max2), dim=0)))

        # apply channel attention
        x *= channel_attention

        x *= spatial_attention

        return x
