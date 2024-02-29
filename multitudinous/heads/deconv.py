from torch import nn, Tensor

class DeconvHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(x: Tensor) -> Tensor:

        raise NotImplementedError("Deconvolutional head not implemented")
