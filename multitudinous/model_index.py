import sys
sys.path.append(".")
from multitudinous.heads.deconv import DeconvHead
from multitudinous.necks.concat import ConcatNeck
from multitudinous.backbones.image.resnet import ResNet50, SEResNet50, CBAMResNet50
from multitudinous.backbones.image.autoencoders import ResNet50AE, SEResNet50AE, CBAMResNet50AE, ResNet50UNet, SEResNet50UNet, CBAMResNet50UNet

# image backbone models
img_backbones = {
    'resnet50': ResNet50,
    'se_resnet50': SEResNet50,
    'cbam_resnet50': CBAMResNet50
}

# pre-training specific models
pretraining = {
    'resnet50_unet': ResNet50UNet,
    'se_resnet50_unet': SEResNet50UNet,
    'cbam_resnet50_unet': CBAMResNet50UNet,
    "resnet50_ae": ResNet50AE,
    "se_resnet50_ae": SEResNet50AE,
    "cbam_resnet50_ae": CBAMResNet50AE
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
