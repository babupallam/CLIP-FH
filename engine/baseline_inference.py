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



def extract_features(model, dataloader, device, use_flip=False):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, label in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            label = label.to(device)

            feats_orig = model.encode_image(images)

            if use_flip:
                images_flipped = torch.flip(images, dims=[3])
                feats_flip = model.encode_image(images_flipped)
                feats = (feats_orig + feats_flip) / 2.0
            else:
                feats = feats_orig

            feats = torch.nn.functional.normalize(feats, dim=1)
            features.append(feats)
            labels.append(label)

    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels


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



def test_clip_import():
    import sys
    print("📦 Inside baseline_inference:")
    print(" - sys.executable:", sys.executable)
    print(" - sys.path:")
    for p in sys.path:
        print("   >", p)

    try:
        import clip
        print("✅ Successfully imported clip in baseline_inference.")
    except ModuleNotFoundError as e:
        print("❌ Failed to import clip in baseline_inference:", e)
