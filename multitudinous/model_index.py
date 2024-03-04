import sys
sys.path.append(".")
from torchvision.models import resnet50, se_resnet50, cbam_resnet50, resnet50_unet, se_resnet50_unet, cbam_resnet50_unet
from multitudinous.heads.deconv import DeconvHead
from multitudinous.necks.concat import ConcatNeck

# image backbone models
img_backbones = {
    'resnet50': resnet50(),
    'se_resnet50': se_resnet50(),
    'cbam_resnet50': cbam_resnet50()
}

# pre-training specific models
pretraining = {
    'resnet50_unet': resnet50_unet(),
    'se_resnet50_unet': se_resnet50_unet(),
    'cbam_resnet50_unet': cbam_resnet50_unet()
}

# point cloud backbone models
point_cloud_backbones = {
    'point_transformer': None,
    'pointnet': None
}

# neck models
necks = {
    'concat': ConcatNeck,
}

# head models
heads = {
    'deconvolutional': DeconvHead
}
