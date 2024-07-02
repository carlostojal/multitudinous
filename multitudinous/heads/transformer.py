import torch
from torch import nn
from multitudinous.tasks import Task
from typing import Tuple
from math import ceil
import numpy as np

class TransformerHead(nn.Module):

    """
    Transformer-based decoder with a task head for grid prediction or cone detection.
    """

    class GridDecoder(nn.Module):

        """
        Decodes a voxel grid from the embeddings. 
        For generalized 3-dimensional environment perception.
        """

        def __init__(self, embedding_dim: int = 768, seq_len: int = 876, output_shape: Tuple[int] = (200, 200, 16)) -> None:
            super().__init__()

            self.embedding_dim = embedding_dim
            self.seq_len = seq_len
            self.output_shape = output_shape

            self.conv3d = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=(1,4,2), stride=2, padding=0),
                nn.ReLU(),
                nn.Conv3d(16, 32, kernel_size=(1,4,2), stride=2, padding=0),
                nn.ReLU(),
                nn.Conv3d(32, 16, kernel_size=(1,3,2), stride=1, padding=0)
            )


        def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the voxel grid decoder.

            Args:
            - embeddings: the embeddings

            Returns:
            - torch.Tensor: the voxel grid
            """

            # add two dimensions to the embeddings
            x = embeddings.unsqueeze(1).unsqueeze(1)

            # TODO: apply the conv3d layers
            x = self.conv3d(x) # (B, 16, 1, 216, 190)

            # swap the channels to the end
            x = x.permute(0, 2, 3, 4, 1)

            # interpolate the output to the desired shape
            x = nn.functional.interpolate(x, size=self.output_shape, mode='trilinear')

            # remove the channel dimension
            x = x.squeeze(1)

            return x
        
    class ConesDecoder(nn.Module):

        """
        Decodes a set of 2-dimensional cone positions and classes from the embeddings.
        For Formula Student Driverless.
        """

        def __init__(self, embedding_dim: int = 768) -> None:
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

    def __init__(self, embedding_dim: int = 768, num_heads: int = 12, num_layers: int = 12, task: Task = Task.GRID) -> None:
        super().__init__()

        # initialize the decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embedding_dim, num_heads),
            num_layers
        )

        # initialize the task
        self.task = task

        # initialize the task head from the task
        self.task_head = None
        if task == Task.GRID:
            self.task_head = TransformerHead.GridDecoder(embedding_dim)
        elif task == Task.CONES:
            self.task_head = TransformerHead.ConesDecoder(embedding_dim)

    def forward(self, img_embeddings: torch.Tensor, pcl_embeddings: torch.Tensor) -> torch.Tensor:

        # verify the input shapes
        if img_embeddings.shape[2] != pcl_embeddings.shape[2]:
            raise ValueError("The image and point cloud embeddings must have the same embedding dimension!")
        
        # concatenate the image and point cloud embeddings, as in UniT (https://arxiv.org/abs/2102.10772)
        fused_embeddings = torch.cat((img_embeddings, pcl_embeddings), dim=1)

        # initialize the task embedding, as in UniT (https://arxiv.org/abs/2102.10772)
        task_embedding = torch.zeros_like(fused_embeddings)
        if self.task == Task.GRID:
            task_embedding[..., -1] = 1
        elif self.task == Task.CONES:
            task_embedding[..., -2] = 1

        # apply the transformer decoder
        fused_embeddings = self.transformer_decoder(task_embedding, fused_embeddings)

        # apply the task head
        return self.task_head(fused_embeddings)
