import yaml
from ..Config import Config

class EncoderConfig(Config):
    def __init__(self) -> None:
        self.name = None
        self.img_width = None
        self.img_height = None

    def parse_from_file(self, filename: str):
        pass

class DecoderConfig:
    def __init__(self) -> None:
        self.name = None
        self.num_points = None

    def parse_from_file(self, filename: str):
        pass

class PreTrainingConfig:

    def __init__(self) -> None:
        self.name = None
        self.batch_size = None
        self.optimizer = None
        self.epochs = None
        self.train_percent = None
        self.learning_rate = None
        self.momentum = None
        self.loss_fn = None
        self.encoder = None
        self.img_backbone = None
        self.point_cloud_backbone = None

    def parse_from_file(self, filename: str):
            
        conf = None
        
        with open(filename, 'r') as f:
            conf = yaml.safe_load(f)

            self.name = conf['name']
            self.batch_size = conf['batch_size']
            self.optimizer = conf['optimizer']
            self.epochs = conf['epochs']
            self.train_percent = conf['train_percent']
            self.learning_rate = conf['learning_rate']
            self.momentum = conf['momentum']
            self.loss_fn = conf['loss_fn']

            self.encoder = EncoderConfig()
            self.encoder.name = conf['encoder']['name']
            self.encoder.img_width = conf['encoder']['img_width']
            self.encoder.img_height = conf['encoder']['img_height']
        
            self.decoder = DecoderConfig()
            self.decoder.attention = conf['decoder']['attention']
