import torch
from torch import nn

class ViT_Embedding(torch.nn.Module):
    """
    Vision Transformer Embedding Layer

    Args:
    - patch_size (int): the size of the patch
    - feature_dim (torch.Size): the dimension of the input feature map
    - embedding_dim (int): the dimension of the embedding
    """
    def __init__(self, patch_size: int = 16, embedding_dim: int = 768, feature_dim: torch.Size = torch.Size(1, 2048, 15, 20)) -> None:
        super().__init__()
        
        # initialize the patch size
        self.patch_size = patch_size
        if self.patch_size < 1:
            raise ValueError("Patch size should be greater or equal to 1")
        
        # initialize the embedding dimension
        self.embedding_dim = embedding_dim
        if self.embedding_dim < 1:
            raise ValueError("Embedding dimension should be greater or equal to 1")
        
        # initialize the feature dimension
        self.feature_dim = feature_dim
        
        # initialize the convolutional layer
        self.conv = nn.Conv2d(in_channels=self.feature_dim[1], out_channels=self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer Embedding Layer

        Args:
        - x (torch.Tensor): the input feature map

        Returns:
        - torch.Tensor: the embedded feature map
        """

        # batch size, channels, height, width
        n, c, h, w = x.shape

        # verify the input tensor shape
        if x.shape != self.feature_dim:
            raise ValueError(f"Input tensor shape should be {self.feature_dim}, but got {x.shape}")

        # calculate the number of patches in the height and width
        n_h = h // self.patch_size
        n_w = w // self.patch_size

        # apply the convolutional layer to the input tensor (embedding the feature map)
        # (n, c, h, w) to (n, embedding_dim, n_h, n_w)
        x = self.conv(x)

        # reshape to (n, embedding_dim, n_h * n_w). sequence length = n_h * n_w = number of patches
        x = x.reshape(n, self.embedding_dim, n_h * n_w)


        # transpose to (n, n_h * n_w, embedding_dim)
        # (batch size, sequence length, embedding dimension)
        x = x.permute(0, 2, 1)

        return x
        