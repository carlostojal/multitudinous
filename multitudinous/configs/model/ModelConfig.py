import yaml
from ..Config import Config
from typing import Tuple

class ImgBackboneConfig:
    def __init__(self) -> None:
        self.name: str = None
        self.weights_path: str = None
        self.in_channels: int = None
        self.img_width: int = None
        self.img_height: int = None
    
    def __str__(self) -> str:
        return f"ImgBackboneConfig(name={self.name}, weights_path={self.weights_path}, in_channels={self.in_channels}, img_width={self.img_width}, img_height={self.img_height})"

class PointCloudBackboneConfig:
    def __init__(self) -> None:
        self.name: str = None
        self.weights_path: str = None
        self.point_dim: int = None
        self.num_points: int = None
        self.feature_dim: int = None

    def __str__(self) -> str:
        return f"PointCloudBackboneConfig(name={self.name}, weights_path={self.weights_path}, point_dim={self.point_dim}, num_points={self.num_points}, feature_dim={self.feature_dim})"

class NeckConfig:
    def __init__(self) -> None:
        self.name: str = None
        self.weights_path: str = None

    def __str__(self) -> str:
        return f"NeckConfig(name={self.name}, weights_path={self.weights_path})"

class HeadConfig:
    def __init__(self) -> None:
        self.name: str = None
        self.weights_path: str = None
        self.grid_x: int = None
        self.grid_y: int = None
        self.grid_z: int = None

    def __str__(self) -> str:
        return f"HeadConfig(name={self.name}, weights_path={self.weights_path})"

class ModelConfig(Config):

    def __init__(self) -> None:
        self.name: str = None
        self.batch_size: int = None
        self.embedding_dim: int = None
        self.sequence_len: int = None
        self.img_backbone: ImgBackboneConfig = None
        self.point_cloud_backbone: PointCloudBackboneConfig = None
        self.neck: NeckConfig = None
        self.head: HeadConfig = None


    def parse_from_file(self, filename: str):

        conf = None
        
        with open(filename, 'r') as f:
            conf = yaml.safe_load(f)

            self.name = conf['name']
            self.embedding_dim = conf['embedding_dim']
            self.sequence_len = conf['sequence_len']

            self.img_backbone: ImgBackboneConfig = ImgBackboneConfig()
            self.img_backbone.name = conf['img_backbone']['name']
            self.img_backbone.in_channels = conf['img_backbone']['in_channels']
            self.img_backbone.img_width = conf['img_backbone']['img_width']
            self.img_backbone.img_height = conf['img_backbone']['img_height']

            self.point_cloud_backbone: PointCloudBackboneConfig = PointCloudBackboneConfig()
            self.point_cloud_backbone.name = conf['point_cloud_backbone']['name']
            self.point_cloud_backbone.num_points = conf['point_cloud_backbone']['num_points']
            self.point_cloud_backbone.point_dim = conf['point_cloud_backbone']['point_dim']
            self.point_cloud_backbone.feature_dim = conf['point_cloud_backbone']['feature_dim']

            self.neck: NeckConfig = NeckConfig()
            self.neck.name = conf['neck']['name']

            self.head: HeadConfig = HeadConfig()
            self.head.name = conf['head']['name']
            self.head.grid_x = conf['head']['grid_x']
            self.head.grid_y = conf['head']['grid_y']
            self.head.grid_z = conf['head']['grid_z']
    
    def __str__(self) -> str:
        return f"ModelConfig(name={self.name}, embedding_dim={self.embedding_dim}, sequence_len={self.sequence_len}, img_backbone={self.img_backbone}, point_cloud_backbone={self.point_cloud_backbone}, neck={self.neck}, head={self.head})"
