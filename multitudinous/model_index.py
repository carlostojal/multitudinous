import sys
sys.path.append(".")
from multitudinous.backbones.image.resnet import ResNet50, SEResNet50, CBAMResNet50, ResNet50Embedding, SEResNet50Embedding, CBAMResNet50Embedding
from multitudinous.backbones.image.vit_embedding import ViT_Embedding
from multitudinous.backbones.image.autoencoders import ResNet50AE, SEResNet50AE, CBAMResNet50AE, ResNet50UNet, SEResNet50UNet, CBAMResNet50UNet
from multitudinous.backbones.point_cloud.pointnet import PointNet, PointNetEmbedding, PointNetSegmentation
from multitudinous.backbones.point_cloud.NDT_Netpp.ndtnetpp.models.ndtnet import NDTNet, NDTNetSegmentation, NDTNetEmbedding
from multitudinous.necks.vilbert import ViLBERT, ViLBERT_Encoder
from multitudinous.heads.transformer import TransformerHead, Task

# image backbone models
img_backbones = {
    'resnet50': ResNet50,
    'resnet50_embedding': ResNet50Embedding,
    'se_resnet50': SEResNet50,
    'se_resnet50_embedding': SEResNet50Embedding,
    'cbam_resnet50': CBAMResNet50,
    'cbam_resnet50_embedding': CBAMResNet50Embedding,
    'vit_embedding': ViT_Embedding
}

# image pre-training specific models
img_pretraining = {
    'resnet50_unet': ResNet50UNet,
    'se_resnet50_unet': SEResNet50UNet,
    'cbam_resnet50_unet': CBAMResNet50UNet,
    "resnet50_ae": ResNet50AE,
    "se_resnet50_ae": SEResNet50AE,
    "cbam_resnet50_ae": CBAMResNet50AE
}

# point cloud pre-training specific models
point_cloud_pretraining = {
    'ndtnet_seg': NDTNetSegmentation
}

# point cloud backbone models
point_cloud_backbones = {
    'ndtnet': NDTNet,
    'ndtnet_embedding': NDTNetEmbedding
}

# neck models
necks = {
    'vilbert': ViLBERT,
}

# head models
heads = {
    'grid': TransformerHead.GridDecoder
}
