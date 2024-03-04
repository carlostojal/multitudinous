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
    
    num_pixels = pred.shape[0] * pred.shape[1]

    # calculate the difference
    diff = pred - target

    # calculate the l2 norm
    l2 = torch.linalg.matrix_norm(diff, ord=2)

    # calculate the pixel loss
    return (l2**2) / num_pixels
