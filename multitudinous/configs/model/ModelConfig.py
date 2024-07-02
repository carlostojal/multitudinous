import yaml
from ..Config import Config
from typing import Tuple

class ImgBackboneConfig:
    def __init__(self, name: str = None, weights_path: str = None, in_channels: int = None,
                 img_width: int = None, num_img_features: int = None, img_height: int = None) -> None:
        self.name = name
        self.weights_path = weights_path
        self.in_channels = in_channels
        self.img_width = img_width
        self.img_height = img_height
        self.num_img_features = num_img_features
    
    def __str__(self) -> str:
        return f"ImgBackboneConfig(name={self.name}, weights_path={self.weights_path}, in_channels={self.in_channels}, img_width={self.img_width}, img_height={self.img_height}, num_img_features={self.num_img_features})"

class PointCloudBackboneConfig:
    def __init__(self, name: str = None, weights_path: str = None, point_dim: int = None,
                 num_points: int = None, num_point_features: int = None, feature_dim: int = None) -> None:
        self.name = name
        self.weights_path = weights_path
        self.point_dim = point_dim
        self.num_points = num_points
        self.num_point_features = num_point_features
        self.feature_dim = feature_dim

    def __str__(self) -> str:
        return f"PointCloudBackboneConfig(name={self.name}, weights_path={self.weights_path}, point_dim={self.point_dim}, num_points={self.num_points}, num_point_features={self.num_point_features}, feature_dim={self.feature_dim})"

class NeckConfig:
    def __init__(self, name: str = None, weights_path: str = None) -> None:
        self.name = name
        self.weights_path = weights_path

    def __str__(self) -> str:
        return f"NeckConfig(name={self.name}, weights_path={self.weights_path})"

class HeadConfig:
    def __init__(self, name: str = None, weights_path: str = None,
                 grid_x: int = None, grid_y: int = None, grid_z: int = None) -> None:
        self.name = name
        self.weights_path = weights_path
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z

    def __str__(self) -> str:
        return f"HeadConfig(name={self.name}, weights_path={self.weights_path})"

class ModelConfig(Config):

    def __init__(self, name: str = None, batch_size: int = None,
                 embedding_dim: int = None, sequence_len: int = None,
                 img_backbone: ImgBackboneConfig = None,
                 point_cloud_backbone: PointCloudBackboneConfig = None,
                 neck: NeckConfig = None,
                 head: HeadConfig = None) -> None:
        self.name = name
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        self.img_backbone = img_backbone
        self.point_cloud_backbone = point_cloud_backbone
        self.neck = neck
        self.head = head

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
            self.img_backbone.num_img_features = conf['img_backbone']['num_img_features']

            self.point_cloud_backbone: PointCloudBackboneConfig = PointCloudBackboneConfig()
            self.point_cloud_backbone.name = conf['point_cloud_backbone']['name']
            self.point_cloud_backbone.num_points = conf['point_cloud_backbone']['num_points']
            self.point_cloud_backbone.point_dim = conf['point_cloud_backbone']['point_dim']
            self.point_cloud_backbone.feature_dim = conf['point_cloud_backbone']['feature_dim']
            self.point_cloud_backbone.num_point_features = conf['point_cloud_backbone']['num_point_features']

            self.neck: NeckConfig = NeckConfig()
            self.neck.name = conf['neck']['name']

            self.head: HeadConfig = HeadConfig()
            self.head.name = conf['head']['name']
            self.head.grid_x = conf['head']['grid_x']
            self.head.grid_y = conf['head']['grid_y']
            self.head.grid_z = conf['head']['grid_z']
    
    def __str__(self) -> str:
        return f"ModelConfig(name={self.name}, embedding_dim={self.embedding_dim}, sequence_len={self.sequence_len}, img_backbone={self.img_backbone}, point_cloud_backbone={self.point_cloud_backbone}, neck={self.neck}, head={self.head})"
