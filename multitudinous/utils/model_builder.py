import torch
from torch import nn
from multitudinous.model_index import img_backbones, point_cloud_backbones, img_pretraining, point_cloud_pretraining, necks, heads
from multitudinous.tasks import Task
from multitudinous.configs.model.ModelConfig import ImgBackboneConfig, PointCloudBackboneConfig, NeckConfig, HeadConfig
from multitudinous.model_zoo.multitudinous import Multitudinous

# ensemble the multitudinous model
def build_multitudinous(img_backbone: ImgBackboneConfig, point_cloud_backbone: PointCloudBackboneConfig,
                        neck: NeckConfig, head: HeadConfig,
                        img_backbone_weights_path: str = None, point_cloud_backbone_weights_path: str = None,
                        neck_weights_path: str = None, head_weights_path: str = None,
                        embedding_dim: int = 768,
                        sequence_len: int = 2048) -> Multitudinous:
    
    # TODO: add a task parameter to the model builder
    
    # create the image backbone
    img_b = build_img_backbone(img_backbone=img_backbone, embedding_dim=embedding_dim, sequence_len=sequence_len, weights_path=img_backbone_weights_path)

    # create the point cloud backbone
    point_cloud_b = build_point_cloud_backbone(point_cloud_backbone=point_cloud_backbone, weights_path=point_cloud_backbone_weights_path)

    # create the neck
    neck = build_neck(neck=neck, embedding_dim=embedding_dim, weights_path=neck_weights_path)

    # create the head
    head = build_head(head=head, embedding_dim=embedding_dim, weights_path=head_weights_path)

    # create the model
    model = Multitudinous(img_b, point_cloud_b, neck, head)

    return model

# build the image backbone
def build_img_backbone(img_backbone: ImgBackboneConfig, embedding_dim: int = 768, sequence_len: int = 2048, weights_path: str = None) -> torch.nn.Module:
    if img_backbone.name not in img_backbones:
        raise ValueError(f'Image backbone {img_backbone.name} not found. Available image backbones are {list(img_backbones.keys())}.')
    img_b = img_backbones[img_backbone.name](in_channels=int(img_backbone.in_channels))
    if weights_path is not None:
        img_b.load_state_dict(torch.load(weights_path))

    return img_b

# build the image pre-training model
def build_img_pretraining(img_pretraining_model: str, in_channels: int = 4, weights_path: str = None) -> torch.nn.Module:
    if img_pretraining_model not in img_pretraining_model:
        raise ValueError(f'Image pre-training model {img_pretraining_model} not found. Available image pre-training models are {list(img_pretraining.keys())}.')
    img_p = img_pretraining[img_pretraining_model](in_channels=in_channels)
    if weights_path is not None:
        img_p.load_state_dict(torch.load(weights_path))
    return img_p

# build the point cloud backbone
def build_point_cloud_backbone(point_cloud_backbone: PointCloudBackboneConfig, embedding_dim: int = 768, sequence_len: int = 1024, weights_path: str = None) -> torch.nn.Module:
    if point_cloud_backbone.name not in point_cloud_backbones:
        raise ValueError(f'Point cloud backbone {point_cloud_backbone.name} not found. Available point cloud backbones are {list(point_cloud_backbones.keys())}.')
    point_cloud_b = point_cloud_backbones[point_cloud_backbone.name](point_dim=point_cloud_backbone.point_dim, 
                                                                     feature_dim=point_cloud_backbone.feature_dim)
    if weights_path is not None:
        point_cloud_b.load_state_dict(torch.load(weights_path))
    return point_cloud_b

# build the point cloud pre-training model (segmentation)
def build_point_cloud_pretraining(point_cloud_pretraining_model: str, point_dim: int = 3, num_classes: int = 16, weights_path: str = None) -> torch.nn.Module:
    if point_cloud_pretraining_model not in point_cloud_pretraining:
        raise ValueError(f'Point cloud pre-training model {point_cloud_pretraining_model} not found. Available point cloud pre-training models are {list(point_cloud_pretraining.keys())}.')
    point_cloud_p = point_cloud_pretraining[point_cloud_pretraining_model](point_dim=point_dim, num_classes=num_classes)
    if weights_path is not None:
        point_cloud_p.load_state_dict(torch.load(weights_path))
    return point_cloud_p

# build the neck
def build_neck(neck: NeckConfig, embedding_dim: int = 768, weights_path: str = None) -> torch.nn.Module:
    if neck.name not in necks:
        raise ValueError(f'Neck {neck.name} not found. Available necks are {list(necks.keys())}.')
    neck = necks[neck.name](embedding_dim=embedding_dim)
    if weights_path is not None:
        neck.load_state_dict(torch.load(weights_path))
    return neck

# build the head
def build_head(head: HeadConfig, embedding_dim: int = 768, task: Task = Task.GRID, weights_path: str = None) -> torch.nn.Module:
    if head.name not in heads:
        raise ValueError(f'Head {head.name} not found. Available heads are {list(heads.keys())}.')
    head = heads[head.name](embedding_dim=embedding_dim, task=task)
    if weights_path is not None:
        head.load_state_dict(torch.load(weights_path))
    return head
