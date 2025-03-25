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
from torchvision import datasets, transforms

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


def get_train_val_loaders(config):
    dataset = config["dataset"]
    aspect = config["aspect"]
    batch_size = config["batch_size"]

    if dataset == "11k":
        base_path = f"./datasets/11khands/train_val_test_split_{aspect}"
    elif dataset == "hd":
        base_path = f"./datasets/HD/Original Images/train_val_test_split"
    else:
        raise ValueError("Unsupported dataset in config.")

    train_dir = os.path.join(base_path, "train")
    val_dir = os.path.join(base_path, "val")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)
    return train_loader, val_loader, num_classes
