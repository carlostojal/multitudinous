import torch

# "Convolutional Block Attention Module" attention

class CBAM(torch.nn.Module):
    def __init__(self, in_size: int, reduction_ratio: float):
        super().__init__()

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

