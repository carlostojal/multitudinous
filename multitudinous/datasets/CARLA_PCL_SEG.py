from torch.utils.data import Dataset
import os
import torch
import numpy as np
from random import randint
from ..configs.datasets.DatasetConfig import DatasetConfig, SubSet
from ..utils.pointclouds import farthest_point_sampling

class CARLA_PCL_SEG(Dataset):
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
        self.pcl = [] # file paths list
        self.num_points = config.num_points # randomly remove extra points on each sample
        self.n_classes = config.n_pcl_classes # number of segmentation classes
        
        pcl_files = os.listdir(self.root)
        pcl_files.sort()
        
        for file in pcl_files:
            sample = None
            gt = None
            try:
                sample, gt = self.get_data_pcl(os.path.join(self.root, file))
                del sample
                del gt
            except RuntimeError as e:
                print(e)
                continue # skip invalid point clouds
            self.pcl.append(os.path.join(self.root, file))
            
            
    def __len__(self):
        return len(self.pcl)
    
    
    def __getitem__(self, idx):

        pcl_filename = self.pcl[idx]

        # Verify that the file exists
        try:
            f = open(pcl_filename, 'rb')
            f.close()
        except FileNotFoundError:
            return
        
        # Get the number of points, their coordinates, and the class tag of each
        pcl_tensor, ground_truth_tensor = self.get_data_pcl(pcl_filename)

        return pcl_tensor, ground_truth_tensor
    
    
    def get_data_pcl(self, pcl_file):
        points = []                 # Points coordinates
        matrix_classes = []         # Class tag of the points
        self.n_points = 0           # Number of points in the point cloud

        
        with open(pcl_file, 'r') as f:
            pcl = f.readlines()     
    
        for point in pcl[10:]:      # Skip no data lines
            
            # Matrix with the number of classes dim to store the class tag of the point
            matrix_class = np.zeros((self.n_classes))   
            
            data = point.strip().split()
            x = float(data[0])
            y = float(data[1])
            z = float(data[2])
            class_tag = int(data[-1]) # the class tag is the last element

            matrix_class[class_tag-1] = 1               # Put 1 in the class tag position in the matrix
            
            points.append([x, y, z])                    # Append the point to the list of points
            matrix_classes.append(matrix_class)         # Append the class tag of the point to the list of class tags

            self.n_points += 1                          # Count the number of points in the point cloud

        # verify if the point cloud dimension is smaller than the threshold
        if self.n_points < self.num_points:
            raise RuntimeError(f"Expected at least {self.num_points} points! Got {len(self.n_points)}.")
        
        # remove random points until the threshold is reached
        max_index = len(points) - 1
        n_points_to_remove = len(points) - self.num_points
        for _ in range(n_points_to_remove):
            index_to_remove = randint(0, max_index) # sample a random integer in the range of the point count
            del points[index_to_remove] # remove the point
            del matrix_classes[index_to_remove] # remove the ground truth classes
            max_index -= 1 # given a point has been removed, the maximum index has decreased

        # sample the points using the farthest point sampling algorithm
        """
        points = farthest_point_sampling(np.asarray(points), n_points=self.num_points)
        """
            
        # finally, convert the arrays to tensors
        return torch.Tensor(points), torch.Tensor(np.asarray(matrix_classes))
    
    
    def get_dataset(self):
        
        for i in range(self.batch_size):
            pcl, ground_truth = self.__getitem__(i)
            
            pcl_batch = torch.Tensor(self.batch_size, self.n_points, 3)
            ground_truth_batch = torch.Tensor(self.batch_size, self.n_points, self.n_classes)
            
            pcl_batch[i] = pcl
            ground_truth_batch[i] = ground_truth
            
        return pcl_batch, ground_truth_batch
    
# ------------------------------------------------------
# TODO: REMOVE AFTER TESTING
if __name__ == "__main__":    
    dataset = CARLA_PCL_SEG("../../../Carla/PythonAPI/projeto_informatico/_out", batch_size=2, n_classes=28)
    
    # ---> VAI BUSCAR SÓ A PRIMEIRA PCL E DEVOLVE OS TENSORES COM BATCH_SIZE = 1
    pcl, ground_truth = dataset[0]
    print(f"PCL: {pcl} | Shape: {pcl.shape} \nGround Truth: {ground_truth} | Shape: {ground_truth.shape}")
    
    # ---> VAI BUSCAR AS DUAS PCLS E DEVOLVE OS TENSORES COM BATCH_SIZE = 2 (Possivel problema no n_points que pode ser diff nas duas PCLs)
    #pcl_batch, ground_truth_batch = dataset.get_dataset()
    #print(f"PCL Batch: {pcl_batch} | Shape: {pcl_batch.shape} \nGround Truth Batch: {ground_truth_batch} | Shape: {ground_truth_batch.shape}")