import torch
from voxel_transformer.backbones.rgbd.rgbd_backbone import RGBDBackbone

# RGBD backbone using the "ResNet 50" architecture - convolutional.


class Shortcut(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()

        # "when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2."
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn1(self.conv1(x))


# residual block class
class ResBlock(torch.nn.Module):

    def __init__(self, in_channels: int, true_in_channels: int, stride: int = 1, attention_module: torch.nn.Module = None):
        super().__init__()

        self.expansion: int = 4

        self.in_channels: int = in_channels
        self.out_channels: int = in_channels * self.expansion

        self.stride = stride

        self.block = BottleneckBlock(in_channels, true_in_channels, stride)

        self.shortcut = Shortcut(in_channels=true_in_channels, out_channels=self.out_channels, stride=stride)

        self.sigmoid = torch.nn.Sigmoid()
        
        self.attention = attention_module

    def forward(self, x: torch.Tensor):

        # save the residual value
        residual: torch.Tensor = x

        # do the forward pass
        x = self.block(x)

        # adapt dimensions with a 1x1 conv layer
        residual = self.shortcut(residual)

        # if an attention module was set
        if self.attention != None:
            # compute attention
            x_attn = self.attention(x)
            # apply attention
            x *= x_attn

        # concatenate the new feature map and the residual connection
        return self.sigmoid(x + residual)


# bottleneck block class (1x1 -> 3x3 -> 1x1 convolutions)
class BottleneckBlock(torch.nn.Module):

    def __init__(self, in_channels: int, true_in_channels: int, stride: int = 1):
        super().__init__()

        # outputs are 4x times larger than inputs
        self.expansion: int = 4

        self.relu = torch.nn.ReLU()

        # "Downsampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2."

        # 1x1
        self.conv1 = torch.nn.Conv2d(in_channels=true_in_channels, out_channels=in_channels, kernel_size=1,
                                     stride=stride, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(in_channels)

        # 3x3
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                     padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(in_channels)

        # 1x1
        self.conv3 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels * self.expansion,
                                     kernel_size=1, stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(in_channels * self.expansion)

    def forward(self, x):

        # "We adopt batch normalization (BN) [16] right after each convolution and before activation, following [16]."
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x


class ResNet50(RGBDBackbone):

    # feature_len is the output feature vector length
    def __init__(self, out_feature_len: int, attention_module: torch.nn.Module = None):
        super().__init__()

        # the first layer has 4 input channels (RGBD)
        self.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # count of each convolutional block
        self.block_counts: list[int] = [3, 4, 6, 3]

        # convolutional block implementations (4 blocks)
        self.res_blocks: list[ResBlock] = []

        # the first block has layers with 64 input size
        input_size: int = 64
        last_input_size: int = input_size
        last_output_size: int = 64

        block_idx: int = 0

        # create the convolutional blocks
        for block_count in self.block_counts:

            for _ in range(block_count):

                # "(...) when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2."
                if last_input_size != input_size:
                    stride = 2
                else:
                    stride = 1

                new_block = ResBlock(in_channels=input_size, true_in_channels=last_output_size,
                                     attention_module=attention_module, stride=stride)

                # save the previous input size
                last_input_size = input_size

                last_output_size = new_block.in_channels * new_block.expansion

                # create a bottleneck block
                self.res_blocks.append(new_block)

            # at each block, the input size doubles
            input_size *= 2

            block_idx += 1

        self.block = torch.nn.Sequential(*self.res_blocks)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(1, out_feature_len)

        self.softmax = torch.nn.Softmax2d()

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:

        # input shape must be 3x224x224
        if rgbd.size() != (1, 4, 224, 224):
            raise ValueError("Invalid RGB-D tensor shape: must be 4x224x224! ({} was provided)".format(rgbd.size()))

        x = self.conv1(rgbd)
        x = self.bn1(x)

        x = self.maxpool(x)

        x = self.block(x)

        x = self.avgpool(x)

        x = self.fc(x)

        return x

