import torch
from voxel_transformer.backbones.rgbd.rgbd_backbone import RGBDBackbone

# RGBD backbone using the "ResNet 50" architecture - convolutional.

class ResNet50(RGBDBackbone):
    # bottleneck block class (1x1 -> 3x3 -> 1x1 convolutions)
    class BottleneckBlock(torch.nn.Module):

        def __init__(self, input_size: int, first_layer_stride: int = 1):
            super().__init__()

            # outputs are 4x times larger than inputs
            self.expansion: int = 4

            self.relu = torch.nn.ReLU()

            # 1x1
            self.bn1 = torch.nn.BatchNorm2d(input_size)
            self.conv1 = torch.nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=1,
                                         stride=first_layer_stride)

            # 3x3
            self.bn2 = torch.nn.BatchNorm2d(input_size)
            self.conv2 = torch.nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=3, stride=1)

            # 1x1
            self.bn3 = torch.nn.BatchNorm2d(input_size)
            self.conv3 = torch.nn.Conv2d(in_channels=input_size, out_channels=input_size * self.expansion,
                                         kernel_size=1,
                                         stride=1)

        def forward(self, x):
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv1(x)

            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv2(x)

            x = self.bn3(x)
            x = self.relu(x)
            x = self.conv3(x)

            return x

    # feature_len is the output feature vector length
    def __init__(self, feature_len: int):
        super().__init__()

        # the first layer has 4 input channels (RGBD)
        self.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # count of each convolutional block
        self.block_counts: list[int] = [3, 4, 6, 3]
        # convolutional block implementations (4 implementations)
        self.conv_blocks: list[torch.nn.Sequential] = []

        # TODO

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(1024, feature_len)

        self.softmax = torch.nn.Softmax2d()

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        pass