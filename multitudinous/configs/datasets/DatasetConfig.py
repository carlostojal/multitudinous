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
        self.num_points = None
        self.n_pcl_classes = None

        # camera specific parameters
        self.img_shape = None

    def parse_from_file(self, filename: str) -> None:
        
        conf = None

        with open(filename, 'r') as f:
            conf = yaml.safe_load(f)

            self.name = conf['name']
            self.base_path = conf['base_path']
            self.train_path = conf['train_path']
            self.val_path = conf['val_path']
            self.test_path = conf['test_path']

            # point cloud specific
            if 'num_points' in conf:
                self.num_points = conf['num_points']
            if 'n_pcl_classes' in conf:
                self.n_pcl_classes = conf['n_pcl_classes']

            # camera specific
            if 'img_width' in conf:
                self.img_width = conf['img_width']
            if 'img_height' in conf:
                self.img_height = conf['img_height']
