from ..loss_index import img_loss_fns

def build_loss_fn(loss: str) -> callable:

    # verify that the loss function is configured
    if loss not in img_loss_fns:
        raise ValueError(f"Loss function {loss} not configured")
    
    return img_loss_fns[loss]
