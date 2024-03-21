import yaml
from ..Config import Config

class DatasetConfig(Config):
    def __init__(self) -> None:
        self.name = None
        self.base_path = None
        self.train_path = None
        self.val_path = None
        self.test_path = None

    def parse_from_file(self, filename: str) -> None:
        
        conf = None

        with open(filename, 'r') as f:
            conf = yaml.safe_load(f)

            self.name = conf['name']
            self.base_path = conf['base_path']
            self.train_path = conf['train_path']
            self.val_path = conf['val_path']
            self.test_path = conf['test_path']
