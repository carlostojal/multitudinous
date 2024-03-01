import torch
from multitudinous.dataset_index import datasets

def build_img_dataset(dataset: str, path: str) -> torch.utils.data.Dataset:

    # verify that the dataset is configured
    if dataset not in datasets:
        raise ValueError(f"Dataset {dataset} not configured")
    
    return datasets[dataset](path)
    