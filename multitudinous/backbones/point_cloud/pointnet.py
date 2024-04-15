import torch
from torch import nn
from torch.autograd import Variable

class TNet(nn.Module):
    """
    Transformation Network
    """

    def __init__(self, in_dim: int = 64) -> None:
        super().__init__()

        self.in_dim = in_dim

        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, in_dim**2)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformation Network

        Args:
        - x (torch.Tensor): the input point cloud

        Returns:
        - torch.Tensor: the output of the Transformation Network
        """

        # MLP
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # FC layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity matrix
        x += torch.eye(self.in_dim).view(1, self.in_dim**2).repeat(x.size(0), 1)
        x = x.view(-1, self.in_dim, self.in_dim)

        return x


class PointNet(nn.Module):
    """
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    """

    def __init__(self, point_dim: int = 3) -> None:
        super().__init__()

        self.point_dim = point_dim

        self.conv1 = nn.Conv1d(self.point_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.t1 = TNet(in_dim=3)
        self.t2 = TNet(in_dim=64)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PointNet

        Args:
        - x (torch.Tensor): the input point

        Returns:
        - torch.Tensor: the output of the PointNet
        """

        x = x.transpose(2, 1)

        # input transform
        t = self.t1(x)
        x *= t

        # MLP
        x = self.bn1(self.conv1(x))

        # feature transform
        t = self.t2(x)
        x *= t

        # MLP
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))

        # return a tensor with shape (batch_size, 1024, num_points)
        return x
