import torch
from torch import nn
from torch.nn import Dropout

class ModalityWeighing(nn.Module):
    """
    Modality Weighing Layer. Gives a weight (0-100%) to the embeddings of each modality.

    Args:
    - dropout_prob (float): the dropout probability
    """
    def __init__(self, dropout_prob: float = 0.7) -> None:
        super().__init__()

        # initialize the dropout probability
        self.dropout_prob = dropout_prob
        if self.dropout_prob < 0 or self.dropout_prob > 1:
            raise ValueError("Dropout probability should be between 0 and 1")
        
        # initialize the dropout layer
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, img_embedding: torch.Tensor, pcl_embedding: torch.Tensor) -> torch.Tensor:
        # TODO
        pass