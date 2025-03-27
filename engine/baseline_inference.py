"""
baseline_inference.py

Purpose:
- Extracts features using CLIP image encoder (ViT-B/16 or RN50).
- Computes cosine similarity between query and gallery features.
- Used for evaluating Re-Identification (ReID) performance via mAP/CMC metrics.

Dependencies:
- torch        : for tensor operations and GPU support
- clip         : OpenAI's CLIP model
- tqdm         : progress bar for data loading
- normalize    : normalizes feature vectors to unit length
"""

import torch  # PyTorch library for tensor computations
import clip  # OpenAI CLIP model (ViT-B/16 or RN50)
from tqdm import tqdm  # Progress bar for iterations
from torch.nn.functional import normalize  # For L2-normalization of feature vectors


def extract_features(model, dataloader, device):
    """
    Step 1: Extract visual features from a dataloader using CLIP's image encoder.

    Args:
        model (clip.model.CLIP): Pretrained CLIP model
        dataloader (DataLoader): DataLoader yielding (images, labels)
        device (torch.device)  : Device to run inference on (CPU or CUDA)

    Returns:
        features (Tensor): Shape [N, D] where N = number of samples, D = embedding dimension
        labels   (Tensor): Shape [N]   where each entry is the class/ID label
    """

    all_features = []  # Will collect feature vectors from each batch
    all_labels = []  # Will collect corresponding labels

    with torch.no_grad():  # No gradients needed during inference (saves memory and computation)
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            # images: Tensor of shape [B, C, H, W]
            # labels: Tensor of shape [B]

            images = images.to(device)  # Move batch to device (CPU/GPU)
            features = model.encode_image(images)  # Use CLIP's image encoder to extract features → [B, D]
            features = normalize(features, dim=1)  # Normalize each feature vector to unit length
            all_features.append(features)  # Store batch of features
            all_labels.append(labels)  # Store batch of labels

    # Concatenate all batches into one tensor
    return torch.cat(all_features), torch.cat(all_labels)
    # Output:
    #   features: [N, D]
    #   labels  : [N]


def compute_similarity_matrix(query_features, gallery_features):
    """
    Step 2: Compute cosine similarity matrix between query and gallery features.

    Args:
        query_features   (Tensor): Shape [Nq, D] → Nq = #query samples
        gallery_features (Tensor): Shape [Ng, D] → Ng = #gallery samples

    Returns:
        sim_matrix (Tensor): Shape [Nq, Ng] → cosine similarities between each query and gallery
    """
    # Cosine similarity is just the dot product if vectors are normalized (which they are)
    return torch.matmul(query_features, gallery_features.T)
