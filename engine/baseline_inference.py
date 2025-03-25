"""
baseline_inference.py

Extracts features using CLIP (ViT-B/16 or RN50) without fine-tuning.
Evaluates ReID performance using cosine similarity + mAP/CMC.
"""

import torch
import clip
from tqdm import tqdm
from torch.nn.functional import normalize

def extract_features(model, dataloader, device):
    """
    Extract features using CLIP image encoder.

    Returns:
        features (Tensor): [N, D]
        labels (Tensor): [N]
    """
    all_features, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model.encode_image(images)
            features = normalize(features, dim=1)
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)

def compute_similarity_matrix(query_features, gallery_features):
    """Computes cosine similarity matrix"""
    return torch.matmul(query_features, gallery_features.T)
