import yaml

class ModelConfig:

    def __init__(self) -> None:
        self.name = None
        self.img_backbone = None
        self.point_cloud_backbone = None

    def parse_from_file(self, filename: str):

        conf = None
        
        with open(filename, 'r') as f:
            conf = yaml.safe_load(f)

            self.name = conf['name']
            self.img_backbone = conf['img_backbone']['name']
            self.point_cloud_backbone = conf['point_cloud_backbone']['name']
