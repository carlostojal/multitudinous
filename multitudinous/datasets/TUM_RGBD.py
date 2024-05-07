import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from ..configs.datasets.DatasetConfig import DatasetConfig, SubSet
import numpy as np

BIT_PRECISION_16 = (2**16)-1

class TUM_RGBD(Dataset):
    def __init__(self, config: DatasetConfig, subset: SubSet):

        subdir: str = None
        if subset == SubSet.TRAIN:
            subdir = config.train_path
        elif subset == SubSet.VAL:
            subdir = config.val_path
        elif subset == SubSet.TEST:
            subdir = config.test_path
        else:
            raise ValueError(f"Invalid subset {subset}")
        
        self.root = os.path.join(config.base_path, subdir)
        self.shape = config.img_shape

        # load the rgb images from "root/rgb" to a list
        self.rgb = []
        fullpath = os.path.join(self.root, "rgb")
        rgb_imgs = os.listdir(fullpath)
        rgb_imgs.sort()
        for file in rgb_imgs:
            # append the filename to the list
            self.rgb.append(os.path.join(fullpath, file))

        # load the depth images from "root/depth" to a list
        self.depth = []
        fullpath = os.path.join(self.root, "depth")
        depth_imgs = os.listdir(fullpath)
        depth_imgs.sort()
        for file in depth_imgs:
            # append the filenamer to the list
            self.depth.append(os.path.join(fullpath, file))


    def __len__(self):
        return len(min(self.rgb, self.depth)) # the same length as the depth images

    def __getitem__(self, idx):

        # check bounds
        if idx >= len(self.rgb) or idx >= len(self.depth) or idx < 0:
            return

        rgb_filename = self.rgb[idx]

        # open the rgb image
        rgb_f = None
        try:
            # verify that the file exists
            rgb_f = open(rgb_filename, 'rb')
        except FileNotFoundError:
            return
        rgb_img = Image.open(rgb_f)
        # resize the image to shape
        rgb_img = rgb_img.resize(self.shape)
        rgb_img = np.array(rgb_img, dtype=np.float32)
        rgb_img = rgb_img / 255.0
        rgb_img = torch.from_numpy(rgb_img)
        rgb_img = rgb_img.float()   
        # squeeze the tensor
        rgb_img = torch.squeeze(rgb_img, dim=0)
        # permute the dimensions
        rgb_img = rgb_img.permute(2, 0, 1)
        rgb_f.close()

        depth_filename = self.depth[idx]

        # open the depth image
        depth_f = None
        try:
            # verify that the file exists
            depth_f = open(depth_filename, 'rb')
        except FileNotFoundError:
            return
        depth_img = Image.open(depth_f)
        # resize the image to shape
        depth_img = depth_img.resize(self.shape)
        depth_img = np.array(depth_img, dtype=np.float32)
        depth_img = depth_img / BIT_PRECISION_16
        depth_img = torch.from_numpy(depth_img)
        depth_img = depth_img.float()
        depth_f.close()

        return rgb_img, depth_img
