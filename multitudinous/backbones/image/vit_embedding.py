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
    def __init__(self, patch_size: int = 16, feature_dim: torch.Size = torch.Size(2048, 15, 20), embedding_dim: int = 512) -> None:
        super().__init__()
        
        # initialize the patch size
        self.patch_size = patch_size
        if self.patch_size < 1:
            raise ValueError("Patch size should be greater or equal to 1")

        # initialize the feature dimension - dimension of the feature map
        self.feature_dim = feature_dim
        num_features = self.feature_dim[0] * self.feature_dim[1] * self.feature_dim[2] # total number of elements on the feature map
        
        # initialize the embedding dimension
        self.embedding_dim = embedding_dim
        if self.embedding_dim < 1:
            raise ValueError("Embedding dimension should be greater or equal to 1")
        # initialize the linear layer
        self.fc = nn.Linear(num_features, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer Embedding Layer

        Args:
        - x (torch.Tensor): the input feature map

        Returns:
        - torch.Tensor: the embedded feature map
        """

        # verify the input tensor shape
        if x.shape[1:] != self.feature_dim:
            raise ValueError(f"Input feature map shape {x.shape[1:]} does not match the expected feature dimension {self.feature_dim}")
        
        pass

