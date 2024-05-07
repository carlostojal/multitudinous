import torch
from multitudinous.dataset_index import datasets
from ..configs.datasets.DatasetConfig import DatasetConfig, SubSet
import os

def build_dataset(config: DatasetConfig) -> torch.utils.data.Dataset:

    # verify that the dataset is configured
    if config.name not in datasets:
        raise ValueError(f"Dataset {config.name} not configured")
    
    return datasets[config.name](config, SubSet.TRAIN), datasets[config.name](config, SubSet.VAL), datasets[config.name](config, SubSet.TEST)
    