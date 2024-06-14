import torch
from torch import nn
from torch.nn.modules.normalization import LayerNorm

class ViLBERT_Encoder(nn.Module):
    """
    Vision-and-Language BERT Encoder with modality cross-attention

    Args:
    - embedding_dim (int): the dimension of the embedding
    - num_heads (int): the number of heads in the multi-head attention
    """

    def __init__(self, embedding_dim: int = 1024, num_heads: int = 12, masking_prob: float = 0.15, mask_prob: float = 0.8, random_prob: float = 0.1, unchanged_prob: float = 0.1) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # verify the masking probabilities
        if masking_prob < 0 or masking_prob > 1 or mask_prob < 0 or mask_prob > 1 or random_prob < 0 or random_prob > 1 or unchanged_prob < 0 or unchanged_prob > 1:
            raise ValueError("Probabilities must be between 0 and 1!")
        if mask_prob + random_prob + unchanged_prob != 1:
            raise ValueError("The sum of the probabilities must be equal to 1!")

        self.masking_prob = masking_prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.unchanged_prob = unchanged_prob

        # initialize the multi-head attention
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

        # initialize the layer normalization
        self.ln = LayerNorm(embedding_dim)

        # initialize the feed-forward layer
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim), # in BERT, the feed-forward layer has 4 times the embedding dimension
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, q: torch.Tensor, k_v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision-and-Language BERT Encoder with cross-attention

        Args:
        - q: the query tensor, shaped as (seq_len, batch_size, embedding_dim)
        - k_v: the key and value tensor

        Returns:
        - torch.Tensor: the output hidden states
        """
        
        # verify if model is in training mode to apply ViLBERT maskings
        if self.training:
            # apply ViLBERT maskings
            # 15% of the tokens are masked
            # 80% of the masked tokens are replaced with [MASK]
            # 10% of the masked tokens are replaced with random tokens
            # 10% of the tokens are left unchanged

            # generate the random mask, selecting 15% of the tokens
            masked_tokens = torch.rand(q.shape) < self.masking_prob

            # generate the mask for the tokens to be masked, selecting 80% of the 15% of the tokens
            mask_tokens = masked_tokens & (torch.rand(q.shape) < self.mask_prob)

            # generate the mask for the tokens to be replaced with random tokens, selecting 10% of the 15% of the tokens
            random_tokens = masked_tokens & (torch.rand(q.shape) < self.random_prob)

            # generate the mask for the tokens to be left unchanged, selecting 10% of the tokens
            unchanged_tokens = ~masked_tokens

            # apply the maskings
            q[mask_tokens] = 0
            q[random_tokens] = torch.randint(0, q.shape[-1], random_tokens.sum())
            q[unchanged_tokens] = q[unchanged_tokens]

        q_residual = q # query residual connection

        # apply the multi-head attention
        q, _ = self.mha(q, k_v, k_v)

        # apply the layer normalization (add & norm)
        q = self.ln(q + q_residual)

        # store the query residual connection
        q_residual = q

        # apply the feed-forward layer
        q = self.ff(q)

        # apply the layer normalization (add & norm)
        q = self.ln(q + q_residual)

        return q

class ViLBERT(nn.Module):
    """
    Vision-and-Language BERT

    Args:
    - embedding_dim (int): the dimension of the embedding
    - num_heads (int): the number of heads in the multi-head attention
    - num_layers (int): the number of layers in the encoder

    """

    def __init__(self, embedding_dim: int = 1024, num_heads: int = 12, num_layers: int = 12) -> None:
        super().__init__()

        # initialize the encoder layers
        img_encoders = []
        pcl_encoders = []
        for _ in range(num_layers):
            img_encoders.append(ViLBERT_Encoder(embedding_dim, num_heads))
            pcl_encoders.append(ViLBERT_Encoder(embedding_dim, num_heads))
        self.img_encoders = nn.Sequential(*img_encoders)
        self.pcl_encoders = nn.Sequential(*pcl_encoders)



    def forward(self, img_embeddings: torch.Tensor, pcl_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Vision-and-Language BERT

        Args:
        - img_embeddings: the image embeddings
        - pcl_embeddings: the point cloud embeddings

        Returns:
        - tuple[torch.Tensor, torch.Tensor]: the image and point cloud embeddings
        """

        # verify the input shapes
        if img_embeddings.shape != pcl_embeddings.shape:
            raise ValueError("The image and point cloud embeddings must have the same shape!")

        # apply the image encoder
        img_embeddings = self.img_encoders(img_embeddings, pcl_embeddings)

        # apply the point cloud encoder
        pcl_embeddings = self.pcl_encoders(pcl_embeddings, img_embeddings)

        return img_embeddings, pcl_embeddings
