"""
dataloaders.py

Purpose:
- Prepares PyTorch DataLoaders for hand image datasets (11k or HD).
- Supports loading query/gallery/test sets using custom transforms.
- Also provides train-validation splits with standard transforms.

Functions:
1. get_dataloader() → for generic directory (query/gallery/test)
2. get_train_val_loaders() → for loading training and validation data
"""

# ====== Imports ======

import os                                            # For file and path management
from torch.utils.data import DataLoader              # PyTorch DataLoader for batching and parallel loading
from torchvision.datasets import ImageFolder         # Loads images from directory structure as labeled dataset
from .transforms import build_transforms             # Project-specific image augmentation pipeline
from torchvision import datasets, transforms         # Standard torchvision tools for loading and transforming images
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import Subset

def get_dataloader(data_dir, batch_size=64, shuffle=False, num_workers=4, train=True):
    """
    Loads images from a given directory using CLIP-compatible preprocessing.

    Args:
        data_dir (str)     : Directory path containing class-subfolders of images
        batch_size (int)   : Number of images per batch
        shuffle (bool)     : Whether to shuffle dataset (usually False for eval)
        num_workers (int)  : Number of subprocesses for data loading
        train (bool)       : Whether to apply training augmentations (flip, crop, etc.)

    Returns:
        loader (DataLoader): PyTorch DataLoader yielding (image_tensor, label)
    """

    transform = build_transforms(train=train)                 # Custom transform builder for training/testing
    dataset = ImageFolder(root=data_dir, transform=transform) # Wraps images in directory into labeled dataset
    loader = DataLoader(dataset,                              # Create PyTorch DataLoader
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers)
    return loader


def get_train_val_loaders(config):
    """
    Loads training and validation DataLoaders.
    Temporarily replaces 'val' with query0+gallery0 for ReID-based validation.

    Returns:
        train_loader (DataLoader)
        val_loader (DataLoader)  : Combined query0 + gallery0
        num_classes (int)
    """
    dataset = config["dataset"]
    aspect = config["aspect"]
    batch_size = config["batch_size"]

    # Define base path
    if dataset == "11k":
        base_path = f"./datasets/11khands/train_val_test_split_{aspect}"
    elif dataset == "hd":
        base_path = f"./datasets/HD/Original Images/train_val_test_split"
    else:
        raise ValueError("Unsupported dataset in config.")

    train_dir = os.path.join(base_path, "train")
    query_dir = os.path.join(base_path, "query0")
    gallery_dir = os.path.join(base_path, "gallery0")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    query_dataset = datasets.ImageFolder(query_dir, transform=transform)
    gallery_dataset = datasets.ImageFolder(gallery_dir, transform=transform)

    # ===== OPTIONAL: Restrict gallery to max 2 samples per class =====
    # This is useful for testing or faster validation. Comment to disable.
    restrict_gallery = True # -- make it false later
    if restrict_gallery:
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(gallery_dataset.samples):
            class_to_indices[label].append(idx)

        selected_indices = []
        for label, indices in class_to_indices.items():
            selected_indices.extend(indices[:2])  # take first 2 samples

        gallery_dataset = Subset(gallery_dataset, selected_indices)
        #print(f"[DEBUG] Reduced gallery to {len(gallery_dataset)} samples (max 2 per class)")


    #-------------

    # Combine query0 and gallery0 into one validation set
    val_dataset = ConcatDataset([query_dataset, gallery_dataset])

    # Create DataLoaders
    num_workers = config.get("num_workers", 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_dataset.classes)
    return train_loader, val_loader, num_classes

def get_test_loader(config):
    """
    Loads the test set DataLoader for inference or final evaluation.

    Args:
        config (dict): Should contain keys:
            - dataset (str): "11k" or "hd"
            - aspect  (str): e.g., "dorsal", "palmar"
            - batch_size (int): number of images per batch

    Returns:
        test_loader (DataLoader): DataLoader for the test split
    """
    dataset = config["dataset"]
    aspect = config["aspect"]
    batch_size = config["batch_size"]

    # Define dataset path based on config
    if dataset == "11k":
        base_path = f"./datasets/11khands/train_val_test_split_{aspect}"
    elif dataset == "hd":
        base_path = f"./datasets/HD/Original Images/train_val_test_split"
    else:
        raise ValueError("Unsupported dataset in config.")

    test_dir = os.path.join(base_path, "test")  # Path to test folder

    # Use standard resize and tensor conversion for inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return test_loader


def get_train_all_loader(config):
    """
    Loads all training data (train + val combined) for final training before deployment.

    Args:
        config (dict): Should contain keys:
            - dataset (str): "11k" or "hd"
            - aspect  (str): e.g., "dorsal", "palmar"
            - batch_size (int): number of images per batch

    Returns:
        train_all_loader (DataLoader): DataLoader with full training set
        num_classes (int)             : Number of classes in the combined dataset
    """
    dataset = config["dataset"]
    aspect = config["aspect"]
    batch_size = config["batch_size"]

    # Determine correct path for selected dataset
    if dataset == "11k":
        base_path = f"./datasets/11khands/train_val_test_split_{aspect}"
    elif dataset == "hd":
        base_path = f"./datasets/HD/Original Images/train_val_test_split"
    else:
        raise ValueError("Unsupported dataset in config.")

    trainall_dir = os.path.join(base_path, "train_all")  # Folder containing all training data

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_all_dataset = datasets.ImageFolder(trainall_dir, transform=transform)
    train_all_loader = DataLoader(train_all_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    num_classes = len(train_all_dataset.classes)

    return train_all_loader, num_classes



def get_train_loader_all(config):
    """
    Loads the full training set as a standard PyTorch DataLoader.
    This is used for CLIP-ReID style Stage 1, without P×K batching.
    """
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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    num_classes = len(train_dataset.classes)
    return train_loader, num_classes
