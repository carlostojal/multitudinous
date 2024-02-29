from torch import nn, Tensor, cat

class ConcatNeck(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return cat(x, y)
