import torch
from multitudinous.model_index import img_backbones, point_cloud_backbones, pretraining, necks, heads
from multitudinous.model_zoo.multitudinous import Multitudinous

# ensemble the multitudinous model
def build_multitudinous(img_backbone: str, point_cloud_backbone: str,
                        img_backbone_weights_path: str = None, point_cloud_backbone_weights_path: str = None) -> Multitudinous:
    
    # get the image backbone
    img_b = build_img_backbone(img_backbone, img_backbone_weights_path)

    # get the point cloud backbone
    point_cloud_b = build_point_cloud_backbone(point_cloud_backbone, point_cloud_backbone_weights_path)

    # create the model
    model = Multitudinous(img_b, point_cloud_b)

    return model

# build the image backbone
def build_img_backbone(img_backbone: str, in_channels: int, weights_path: str = None) -> torch.nn.Module:
    if img_backbone not in img_backbones:
        raise ValueError(f'Image backbone {img_backbone} not found. Available image backbones are {list(img_backbones.keys())}.')
    img_b = img_backbones[img_backbone](in_channels=in_channels)
    if weights_path is not None:
        img_b.load_state_dict(torch.load(weights_path))
    return img_b

# build the image pre-training model
def build_img_pretraining(img_pretraining: str, in_channels: int, weights_path: str = None) -> torch.nn.Module:
    if img_pretraining not in pretraining:
        raise ValueError(f'Image pre-training model {img_pretraining} not found. Available image pre-training models are {list(pretraining.keys())}.')
    img_p = pretraining[img_pretraining](in_channels=in_channels)
    if weights_path is not None:
        img_p.load_state_dict(torch.load(weights_path))
    return img_p

# build the point cloud backbone
def build_point_cloud_backbone(point_cloud_backbone: str, weights_path: str = None) -> torch.nn.Module:
    if point_cloud_backbone not in point_cloud_backbones:
        raise ValueError(f'Point cloud backbone {point_cloud_backbone} not found. Available point cloud backbones are {list(point_cloud_backbones.keys())}.')
    point_cloud_b = point_cloud_backbones[point_cloud_backbone]
    if weights_path is not None:
        point_cloud_b.load_state_dict(torch.load(weights_path))
    return point_cloud_b
