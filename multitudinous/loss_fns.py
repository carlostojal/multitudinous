import torch

# pixel loss function
def pixel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    # squeeze the tensors
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)

    # verify that the tensors are 2-d
    if len(pred.shape) != 2 or len(target.shape) != 2:
        raise ValueError('The input tensors must be 2-d.')
    
    # compare dimensions
    if pred.shape != target.shape:
        raise ValueError(f'The input tensors must have the same shape. Got {pred.shape} and {target.shape}.')
    
    return torch.nn.MSELoss()(pred, target)
