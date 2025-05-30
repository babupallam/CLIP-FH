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
    # Read dataset name, hand aspect (e.g., dorsal_r), and batch size from config
    dataset = config["dataset"]
    aspect = config["aspect"]
    batch_size = config["batch_size"]

    # Set the base folder path based on dataset type
    if dataset == "11k":
        # For the 11k hands dataset, include the aspect in the folder name
        base_path = f"./datasets/11khands/train_val_test_split_{aspect}"
    elif dataset == "hd":
        # For the HD dataset, the folder path is static
        base_path = f"./datasets/HD/Original Images/train_val_test_split"
    else:
        # Raise error if unknown dataset is specified
        raise ValueError("Unsupported dataset in config.")

    # Define full paths for training, query, and gallery subdirectories
    train_dir = os.path.join(base_path, "train")
    query_dir = os.path.join(base_path, "query0")
    gallery_dir = os.path.join(base_path, "gallery0")

    # Define image transformation:
    # 1. Resize all images to 224x224
    # 2. Convert images to PyTorch tensors (CHW format, scaled to [0.0, 1.0])
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.Resize((224, 128)), # for v11
        transforms.ToTensor(),
    ])

    # Create datasets using ImageFolder
    # Each subfolder inside these directories is treated as a separate class
    # Labels are automatically assigned based on folder names
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    query_dataset = datasets.ImageFolder(query_dir, transform=transform)
    gallery_dataset = datasets.ImageFolder(gallery_dir, transform=transform)

    # ===== OPTIONAL STEP: Limit gallery to 2 samples per class for faster validation =====
    restrict_gallery = True  # Set to False if you want to use the full gallery
    if restrict_gallery:
        # Create a dictionary to collect sample indices per class
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(gallery_dataset.samples):
            class_to_indices[label].append(idx)

        # Select only the first 2 samples per class
        selected_indices = []
        for label, indices in class_to_indices.items():
            selected_indices.extend(indices[:4])  # Keep max 2 images for each class

        # Use only the selected samples in the new gallery dataset
        gallery_dataset = Subset(gallery_dataset, selected_indices)

    # ===== VALIDATION SETUP =====
    # Combine query0 and (possibly reduced) gallery0 into a single validation set
    val_dataset = ConcatDataset([query_dataset, gallery_dataset])

    # Get number of workers for DataLoader (used for parallel data loading)
    num_workers = config.get("num_workers", 4)

    # Create DataLoader for training:
    # - Shuffling enabled for training
    # - Loads data in batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Create DataLoader for validation:
    # - No shuffling (order matters in evaluation)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Get the number of classes (inferred from subfolders in training directory)
    num_classes = len(train_dataset.classes)

    # Return DataLoaders and number of classes to the training pipeline
    return train_loader, val_loader, num_classes

def get_test_loader(config):

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
