from torchvision.models import resnet50, se_resnet50, cbam_resnet50

# image backbone models
img_backbones = {
    'resnet50': resnet50,
    'se_resnet50': se_resnet50,
    'cbam_resnet50': cbam_resnet50
}

# point cloud backbone models
point_cloud_backbones = {
    'point_transformer': None,
    'point_net': None
}

# neck models
necks = {
    'concat': None,
    'add': None
}

# head models
heads = {
    'deconvolutional': None
}
