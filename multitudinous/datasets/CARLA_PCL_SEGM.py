from torch.utils.data import Dataset
import os
import torch
import numpy as np

class CARLA_PCL_SEGM(Dataset):
    def __init__(self, root: str, batch_size:int = 1, n_classes:int = 28):
        self.root = root
        self.pcl = []
        self.batch_size = batch_size
        self.n_classes = n_classes
        
        fullpath = os.path.join(root, "lidarSegm")
        pcl_files = os.listdir(fullpath)
        pcl_files.sort()
        
        for file in pcl_files:
            # append the filename to the list
            self.pcl.append(os.path.join(fullpath, file))
            
            
    def __len__(self):
        return len(self.pcl) 
    
    
    def __getitem__(self, idx):
        
        # Check bounds
        if idx >= len(self.pcl) or idx < 0:
            return

        pcl_filename = self.pcl[idx]

        # Verify that the file exists
        try:
            open(pcl_filename, 'rb')
        except FileNotFoundError:
            return
        
        # Get the number of points, their coordinates, and the class tag of each
        pcl_tensor, ground_truth_tensor = self.get_data_pcl(pcl_filename)
        print(f"N_points: {self.n_points}")

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
            class_tag = int(data[-1])

            matrix_class[class_tag-1] = 1               # Put 1 in the class tag position in the matrix
            
            points.append([x, y, z])                    # Append the point to the list of points
            matrix_classes.append(matrix_class)         # Append the class tag of the point to the list of class tags

            self.n_points += 1                          # Count the number of points in the point cloud

        return torch.Tensor(np.asarray(points)).unsqueeze(0), torch.Tensor(np.asarray(matrix_classes)).unsqueeze(0)
    
    
    def get_dataset(self):
        
        for i in range(self.batch_size):
            pcl, ground_truth = self.__getitem__(i)
            
            pcl_batch = torch.Tensor(self.batch_size, self.n_points, 3)
            ground_truth_batch = torch.Tensor(self.batch_size, self.n_points, self.n_classes)
            
            pcl_batch[i] = pcl
            ground_truth_batch[i] = ground_truth
            
        return pcl_batch, ground_truth_batch
    

if __name__ == "__main__":    
    dataset = CARLA_PCL_SEGM("../../../Carla/PythonAPI/projeto_informatico/_out", batch_size=2, n_classes=28)
    
    # ---> VAI BUSCAR SÓ A PRIMEIRA PCL E DEVOLVE OS TENSORES COM BATCH_SIZE = 1
    pcl, ground_truth = dataset[0]
    print(f"PCL: {pcl} | Shape: {pcl.shape} \nGround Truth: {ground_truth} | Shape: {ground_truth.shape}")
    
    # ---> VAI BUSCAR AS DUAS PCLS E DEVOLVE OS TENSORES COM BATCH_SIZE = 2 (Possivel problema no n_points que pode ser diff nas duas PCLs)
    #pcl_batch, ground_truth_batch = dataset.get_dataset()
    #print(f"PCL Batch: {pcl_batch} | Shape: {pcl_batch.shape} \nGround Truth Batch: {ground_truth_batch} | Shape: {ground_truth_batch.shape}")