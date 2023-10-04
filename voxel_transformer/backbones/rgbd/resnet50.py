import torch
from voxel_transformer.backbones.rgbd.rgbd_backbone import RGBDBackbone

# RGBD backbone using the "ResNet 50" architecture - convolutional.
class ResNet50(RGBDBackbone):

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

        # first convolutional block
        # x3
        self.conv_blocks[0] = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 256, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        # second convolutional block
        # x4
        self.conv_blocks[1] = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 512, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )

        # third convolutional block
        # x6
        self.conv_blocks[2] = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1024, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        )

        # forth convolutional layer
        # x3
        self.conv_blocks[3] = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 2048, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU()
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(1024, feature_len)

        self.softmax = torch.nn.Softmax2d()

    def forward(self, rgbd: torch.Tensor):
        pass