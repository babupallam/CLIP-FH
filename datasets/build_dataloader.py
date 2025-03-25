"""
📍 Location: CLIP-FH/datasets/build_dataloader.py

🔧 What to Implement:
    - get_train_loader(config)
    - get_test_loader(config)
    - Handle both 11k and HD datasets.
    - Use image folder paths from config (dataset_11k.yml, dataset_hd.yml).
    - Return standard PyTorch DataLoader.

"""
"""
build_dataloader.py

Creates PyTorch DataLoaders for training, validation, testing, query, and gallery.
"""

import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from .transforms import build_transforms

def get_dataloader(data_dir, batch_size=64, shuffle=False, num_workers=4, train=True):
    """
    Generic dataloader builder.

    Args:
        data_dir (str): Path to dataset (e.g. train/val/test/query/gallery).
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for loading.
        train (bool): Whether it's train (apply data augmentation).

    Returns:
        DataLoader
    """
    transform = build_transforms(train=train)
    dataset = ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
