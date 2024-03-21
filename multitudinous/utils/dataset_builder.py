import torch
from multitudinous.dataset_index import datasets
import os

def build_img_dataset(dataset: str, base_path: str, train_path: str, val_path: str, test_path: str) -> torch.utils.data.Dataset:

    # verify that the dataset is configured
    if dataset not in datasets:
        raise ValueError(f"Dataset {dataset} not configured")
    
    return datasets[dataset](os.path.join(base_path, train_path)), datasets[dataset](os.path.join(base_path, val_path)), datasets[dataset](os.path.join(base_path, test_path))
    