import torch
from torch import nn
from multitudinous.tasks import Task

class TransformerHead(nn.Module):

    """
    Transformer-based decoder with a task head for grid prediction or cone detection.
    """

    class GridDecoder(nn.Module):

        """
        Decodes a voxel grid from the embeddings. 
        For generalized 3-dimensional environment perception.
        """

        def __init__(self, embedding_dim: int = 1024) -> None:
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

        def __init__(self, embedding_dim: int = 1024) -> None:
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

    def __init__(self, embedding_dim: int = 1024, num_heads: int = 12, num_layers: int = 12, task: Task = Task.GRID) -> None:
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
        if img_embeddings.shape != pcl_embeddings.shape:
            raise ValueError("The image and point cloud embeddings must have the same shape!")
        
        # concatenate the image and point cloud embeddings, as in UniT (https://arxiv.org/abs/2102.10772)
        fused_embeddings = torch.cat((img_embeddings, pcl_embeddings), dim=-1)

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
