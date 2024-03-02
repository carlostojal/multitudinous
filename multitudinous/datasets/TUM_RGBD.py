import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

class TUM_RGBD(Dataset):
    def __init__(self, root: str):
        self.root = root

        # load the rgb images from "root/rgb" to a list
        self.rgb = []
        fullpath = os.path.join(root, "rgb")
        rgb_imgs = os.listdir(fullpath)
        rgb_imgs.sort()
        for file in rgb_imgs:
            # load the image to a tensor
            f = None
            try:
                # verify that the file exists
                f = open(os.path.join(fullpath, file), 'rb')
            except FileNotFoundError:
                return
            rgb_img = Image.open(f)
            rgb_img = rgb_img.convert('RGB')
            rgb_img = np.array(rgb_img)
            rgb_img = rgb_img / 255.0 # normalize the image
            rgb_img = torch.from_numpy(rgb_img)
            rgb_img = rgb_img.float()
            f.close()

            self.rgb.append(rgb_img)

        # load the depth images from "root/depth" to a list
        self.depth = []
        fullpath = os.path.join(root, "depth")
        depth_imgs = os.listdir(fullpath)
        depth_imgs.sort()
        for file in depth_imgs:
            f = None
            try:
                # verify that the file exists
                f = open(os.path.join(fullpath, file), 'rb')
            except FileNotFoundError:
                return
            d_img = Image.open(f)
            d_img = np.array(d_img, dtype=np.float32)
            d_img = d_img / float((2**16)-1) # normalize the image (16-bit)
            d_img = torch.from_numpy(d_img)
            d_img = d_img.float()
            f.close()

            self.depth.append(d_img)


    def __len__(self):
        return len(self.rgb) # the same length as the depth images

    def __getitem__(self, idx):
        
        # get the rgb and depth images
        rgb = self.rgb[idx]
        depth = self.depth[idx]

        return rgb, depth
