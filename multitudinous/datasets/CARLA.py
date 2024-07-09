import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from ..configs.datasets.DatasetConfig import DatasetConfig, SubSet
import numpy as np

BIT_PRECISION_16 = (2**16)-1

class CARLA(Dataset):
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
        
        self.root = os.path.join(config.base_path, subdir) # dataset root directory
        self.num_point_samples = config.num_point_samples # number of points to sample from the point cloud

        # load the lidar files from "root/pcl" to a list
        self.lidar = [] # point cloud file paths list
        fullpath = os.path.join(self.root, "lidar")
        lidar_files = os.listdir(fullpath)
        lidar_files.sort()
        for file in lidar_files:
            # append the filename to the list
            self.lidar.append(os.path.join(fullpath, file))

        # load the rgb files from "root/rgb" to a list
        self.rgb = [] # RGB file paths list
        fullpath = os.path.join(self.root, "rgb")
        rgb_files = os.listdir(fullpath)
        rgb_files.sort()
        for file in rgb_files:
            # append the filename to the list
            self.rgb.append(os.path.join(fullpath, file))

        # load the depth files from "root/depth" to a list
        self.depth = [] # depth file paths list
        fullpath = os.path.join(self.root, "depth")
        depth_files = os.listdir(fullpath)
        depth_files.sort()
        for file in depth_files:
            # append the filename to the list
            self.depth.append(os.path.join(fullpath, file))

        # load the ground truth files from "root/ground_truth" to a list
        self.gt = []
        fullpath = os.path.join(self.root, "ground_truth")
        gt_files = os.listdir(fullpath)
        gt_files.sort()
        for file in gt_files:
            # append the filename to the list
            self.gt.append(os.path.join(fullpath, file))

    def __len__(self):
        return len(min(self.pcl, self.rgb, self.depth, self.gt))

    def __getitem__(self, idx):

        # check bounds
        if id >= len (self.pcl) or idx >= len(self.rgb) or idx >= len(self.depth) or idx < 0:
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
        # remove the alpha channel
        rgb_img = rgb_img[:-1, :, :]
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

        # concatenate the rgb and depth images
        rgbd = torch.cat((rgb_img, depth_img), dim=0)

        # load the lidar file
        lidar_filename = self.lidar[idx]
        lidar = self.get_lidar_data(lidar_filename)

        # load the ground truth file
        gt_filename = self.gt[idx]
        gt = np.load(gt_filename)['arr_0']
        gt = torch.from_numpy(gt)
        gt = gt.float()

        return rgbd, lidar, gt


    def get_lidar_data(self, lidar_filename) -> torch.Tensor:

        points = [] # point coordinates list
        self.n_points = 0 # number of points in the point cloud
        
        with open(lidar_filename, 'r') as f:
            pcl = f.readlines()

        for point in pcl[10:]: # skip no data lines
            data = point.strip().split()
            x = float(data[0])
            y = float(data[1])
            z = float(data[2])
            points.append([x, y, z])
            self.n_points += 1

        if self.n_points < self.num_point_samples:
            raise RuntimeError(f"Expected at least {self.num_point_samples} points! Got {self.n_points}.")

        # randomly sample points from the point cloud
        if self.num_point_samples < self.n_points:
            points = np.random.choice(points, size=self.num_point_samples)

        return torch.tensor(points)
