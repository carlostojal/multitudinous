import torch
from torch import nn

class GridDecoder(nn.Module):

    """
    Decodes a voxel grid from the embeddings. 
    For generalized 3-dimensional environment perception.
    """

    def __init__(self, embedding_dim: int = 762) -> None:
        super().__init__()

        # TODO: have considerations on the kernel size and stride considering the hidden dimension and desired voxel grid size (e.g. 32x32x32)

        # initialize the deconv3d layers
        self.deconv1 = nn.ConvTranspose3d(embedding_dim, embedding_dim // 2, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(embedding_dim // 2, embedding_dim // 4, kernel_size=3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(embedding_dim // 4, 1, kernel_size=3, stride=2, padding=1)


    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the voxel grid decoder.

        Args:
        - embeddings: the embeddings

        Returns:
        - torch.Tensor: the voxel grid
        """

        # apply the deconv3d layers
        grid = self.deconv1(embeddings)
        grid = self.deconv2(grid)
        grid = self.deconv3(grid)

        return grid
    
class ConesDecoder(nn.Module):

    """
    Decodes a set of 2-dimensional cone positions and classes from the embeddings.
    For Formula Student Driverless.
    """

    def __init__(self, embedding_dim: int = 762) -> None:
        super().__init__()

        # TODO

        raise NotImplementedError("Cone decoder is not implemented yet.")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the cone decoder.

        Args:
        - embeddings: the embeddings

        Returns:
        - torch.Tensor: the cone positions and classes
        """

        # TODO

        raise NotImplementedError("Cone decoder is not implemented yet.")
