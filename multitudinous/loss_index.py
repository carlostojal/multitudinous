from .loss_fns import rmse, rel, delta
import torch.nn.functional as F

img_loss_fns = {
    'rmse': rmse,
    'rel': rel,
    'delta': delta,
    'cross_entropy': F.cross_entropy
}
