from torch import nn, optim
from torch.utils.data import DataLoader

class PreTrainer:
    def __init__(self, model: nn.Module, optimizer: optim, loss, device: str) -> None:
        self.model: nn.Module = model
        self.optimizer: optim = optimizer
        self.loss = loss
        self.device: str = device

    def train(self, data: DataLoader, epochs: int) -> nn.Module:
        self.model.to(self.device)
        self.model.train()
        for _ in range(epochs):
            for batch in data:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.loss(output, batch)
                loss.backward()
                self.optimizer.step()
        self.model.eval()
        self.model.to('cpu')

        return self.model
