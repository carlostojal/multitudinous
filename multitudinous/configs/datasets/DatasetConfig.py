import yaml
from ..Config import Config
from enum import Enum

class SubSet(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3

class DatasetConfig(Config):
    def __init__(self) -> None:
        self.name = None
        self.base_path = None
        self.train_path = None
        self.val_path = None
        self.test_path = None

        # point cloud specific parameters
        self.min_points_threshold = None
        self.n_pcl_classes = None

    def parse_from_file(self, filename: str) -> None:
        
        conf = None

        with open(filename, 'r') as f:
            conf = yaml.safe_load(f)

            self.name = conf['name']
            self.base_path = conf['base_path']
            self.train_path = conf['train_path']
            self.val_path = conf['val_path']
            self.test_path = conf['test_path']

            if 'min_points_threshold' in conf:
                self.min_points_threshold = conf['min_points_threshold']
            if 'n_pcl_classes' in conf:
                self.n_pcl_classes = conf['n_pcl_classes']
