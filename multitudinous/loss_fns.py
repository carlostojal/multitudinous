import torch

# rooted mean squared error (RMSE)
def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    # squeeze the tensors
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)

    # verify that the tensors are 2-d
    if len(pred.shape) != 2 or len(target.shape) != 2:
        raise ValueError('The input tensors must be 2-d.')
    
    # compare dimensions
    if pred.shape != target.shape:
        raise ValueError(f'The input tensors must have the same shape. Got {pred.shape} and {target.shape}.')
    
    return torch.sqrt(torch.nn.MSELoss()(pred, target))

# mean absolute error (MAE)
def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    # squeeze the tensors
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)

    # verify that the tensors are 2-d
    if len(pred.shape) != 2 or len(target.shape) != 2:
        raise ValueError('The input tensors must be 2-d.')
    
    # compare dimensions
    if pred.shape != target.shape:
        raise ValueError(f'The input tensors must have the same shape. Got {pred.shape} and {target.shape}.')
    
    target = torch.where(target == 0, target, 1e-9) # avoid division by zero
    return torch.mean(torch.abs(pred - target) / target)

# log10 error
def log10_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    # squeeze the tensors
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)

    # verify that the tensors are 2-d
    if len(pred.shape) != 2 or len(target.shape) != 2:
        raise ValueError('The input tensors must be 2-d.')
    
    # compare dimensions
    if pred.shape != target.shape:
        raise ValueError(f'The input tensors must have the same shape. Got {pred.shape} and {target.shape}.')
    
    return torch.mean(torch.log10(torch.abs(pred - target) + 1e-9))

# huber loss
def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:

    # squeeze the tensors
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)

    # verify that the tensors are 2-d
    if len(pred.shape) != 2 or len(target.shape) != 2:
        raise ValueError('The input tensors must be 2-d.')
    
    # compare dimensions
    if pred.shape != target.shape:
        raise ValueError(f'The input tensors must have the same shape. Got {pred.shape} and {target.shape}.')
    
    # calculate the loss
    return torch.nn.SmoothL1Loss(reduction='none')(pred, target)
