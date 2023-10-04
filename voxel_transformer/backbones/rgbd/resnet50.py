import torch
from voxel_transformer.backbones.rgbd.rgbd_backbone import RGBDBackbone

# RGBD backbone using the "ResNet 50" architecture - convolutional.
class ResNet50(RGBDBackbone):

    # bottleneck block class (1x1 -> 3x3 -> 1x1 convolutions)
    class BottleneckBlock(torch.nn.Module):
        def __init__(self, in_features: int, out_features: list[int], first_layer_stride=1):
            super().__init__()

            self.relu = torch.nn.ReLU()

            self.bn1 = torch.nn.BatchNorm2d(in_features)
            self.conv1 = torch.nn.Conv2d(in_features, out_features[0], kernel_size=1, stride=first_layer_stride)

            self.bn2 = torch.nn.BatchNorm2d(out_features[0])
            self.conv2 = torch.nn.Conv2d(out_features[0], out_features[1], kernel_size=3)

            self.bn3 = torch.nn.BatchNorm2d(out_features[1])
            self.conv3 = torch.nn.Conv2d(out_features[1], out_features[2], kernel_size=1)

        def forward(self, x):

            # 1x1 convolution
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv1(x)

            # 3x3 convolution
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv2(x)

            # 1x1 convolution
            x = self.bn3(x)
            x = self.relu(x)
            x = self.conv3(x)

            return x

    # feature_len is the output feature vector length
    def __init__(self, feature_len: int):
        super().__init__()

        # the first layer has 4 input channels (RGBD)
        # TODO: review padding and bias. the rest is fine
        self.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # count of each convolutional block
        self.block_counts: list[int] = [3, 4, 6, 3]
        # convolutional block implementations (4 implementations)
        self.conv_blocks: list[torch.nn.Sequential] = []

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(1024, feature_len)

        self.softmax = torch.nn.Softmax2d()

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        pass