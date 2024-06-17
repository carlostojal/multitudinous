import yaml
from ..Config import Config

class ImgBackboneConfig:
    def __init__(self) -> None:
        self.name = None
        self.weights_path = None
        self.img_width = None
        self.img_height = None

class PointCloudBackboneConfig:
    def __init__(self) -> None:
        self.name = None
        self.weights_path = None
        self.num_points = None

class ModelConfig(Config):

    def __init__(self) -> None:
        self.name = None
        self.batch_size = None
        self.embedding_dim = None
        self.sequence_len = None
        self.img_backbone = None
        self.point_cloud_backbone = None

    def parse_from_file(self, filename: str):

        conf = None
        
        with open(filename, 'r') as f:
            conf = yaml.safe_load(f)

            self.name = conf['name']
            self.embedding_dim = conf['embedding_dim']
            self.sequence_len = conf['sequence_len']

            self.img_backbone = ImgBackboneConfig()
            self.img_backbone.name = conf['img_backbone']['name']
            self.img_backbone.img_width = conf['img_backbone']['img_width']
            self.img_backbone.img_height = conf['img_backbone']['img_height']

            self.point_cloud_backbone = conf['point_cloud_backbone']['name']
            self.point_cloud_backbone.num_points = conf['point_cloud_backbone']['num_points']
