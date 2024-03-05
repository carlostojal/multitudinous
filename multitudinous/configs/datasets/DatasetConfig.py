import yaml
from ..Config import Config

class DatasetConfig(Config):
    def __init__(self) -> None:
        self.name = None
        self.path = None

    def parse_from_file(self, filename: str) -> None:
        
        conf = None

        with open(filename, 'r') as f:
            conf = yaml.safe_load(f)

            self.name = conf['name']
            self.path = conf['path']
