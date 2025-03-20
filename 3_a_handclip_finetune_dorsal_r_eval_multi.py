"""
3_a_handclip_finetune_dorsal_r_eval_reid_multi.py

Evaluates Re-ID for all query/gallery splits: query0..query9 vs. gallery0..gallery9.
Then reports average Rank-1 accuracy and average mAP across these 10 runs.

Usage:
  python 3_a_handclip_finetune_dorsal_r_eval_reid_multi.py

Assumptions:
  - Model is saved in ./models/handclip_finetuned_model_dorsal_r.pth
  - The dataset folder structure is created by 1_prepare_train_val_test_11k_r_l.py.
  - We detect automatically if query/gallery have subfolders or are flat.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Datasets ---
class HandDataset(Dataset):
    """
    Subfolder-based dataset:
      folder/
        personA/*.jpg
        personB/*.jpg
      ...
    """
    def __init__(self, image_folder, preprocess, label_map):
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []

        if not os.path.isdir(image_folder):
            print(f"WARNING: {image_folder} not found or not a directory.")
            return

        for label_name, label_idx in label_map.items():
            folder_path = os.path.join(image_folder, label_name)
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.endswith('.jpg'):
                        full_path = os.path.join(folder_path, fname)
                        self.image_paths.append(full_path)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        img = self.preprocess(img)
        label = self.labels[idx]
        return img, label


class HandDatasetFlat(Dataset):
    """
    Flat dataset (no subfolders):
      folder/
        image1.jpg
        image2.jpg
      ...
    Person ID is parsed from filename if possible, else label=-1.
    """
    def __init__(self, image_folder, preprocess, label_map):
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []

        if not os.path.isdir(image_folder):
            print(f"WARNING: {image_folder} is not a directory.")
            return

        for fname in os.listdir(image_folder):
            if fname.endswith('.jpg'):
                full_path = os.path.join(image_folder, fname)
                self.image_paths.append(full_path)
                # Parse ID from "0000000_1.jpg" => "0000000"
                pid = fname.split('_')[0]
                label_idx = label_map.get(pid, -1)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        img = self.preprocess(img)
        label = self.labels[idx]
        return img, label


def has_subfolders(folder):
    """Check if 'folder' has subfolders or is flat."""
    if not os.path.isdir(folder):
        return False
    for item in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, item)):
            return True
    return False


def build_label_map_subfolders(folder):
    """
    Build a label map from subfolders:
      subfolder0 -> label 0
      subfolder1 -> label 1
    """
    label_map_local = {}
    idx = 0
    for item in sorted(os.listdir(folder)):
        path = os.path.join(folder, item)
        if os.path.isdir(path):
            label_map_local[item] = idx
            idx += 1
    return label_map_local


def extract_features(model, loader):
    """Extract embeddings from the model in 'return_features' mode."""
    feats_list, labs_list = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            feats = model(imgs, return_features=True)
            feats_list.append(feats.cpu())
            labs_list.append(labs)
    if not feats_list:
        return None, None
    all_feats = torch.cat(feats_list)
    all_labs = torch.cat(labs_list)
    return all_feats, all_labs


def main():
    # 1) Load CLIP & build model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.transformer.requires_grad_(False)
    image_encoder = clip_model.visual

    num_classes = 190
    classification_head = nn.Linear(image_encoder.output_dim, num_classes)

    class HandCLIPReID(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier
        def forward(self, images, return_features=False):
            feats = self.encoder(images)
            if return_features:
                return feats
            return self.classifier(feats)

    model = HandCLIPReID(image_encoder, classification_head).to(device)

    # 2) Load best model from training
    save_path = './models/handclip_finetuned_model_dorsal_r.pth'
    if not os.path.exists(save_path):
        print(f"No model found at {save_path}. Exiting.")
        return
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f"Loaded model from {save_path}")

    # 3) Evaluate on N=10 splits: query0..9 vs. gallery0..9
    #    We'll accumulate rank1, mAP across these splits.
    N = 10
    rank1_list = []
    map_list = []

    for i in range(N):
        query_dir = f'./11k/train_val_test_split_dorsal_r/query{i}'
        gallery_dir = f'./11k/train_val_test_split_dorsal_r/gallery{i}'

        if not os.path.isdir(query_dir) or not os.path.isdir(gallery_dir):
            print(f"Split {i}: {query_dir} or {gallery_dir} not found. Skipping.")
            continue

        # Check subfolders vs. flat
        if has_subfolders(query_dir):
            q_map = build_label_map_subfolders(query_dir)
            query_dataset = HandDataset(query_dir, preprocess, q_map)
        else:
            # If flat
            q_map = {}
            query_dataset = HandDatasetFlat(query_dir, preprocess, q_map)

        if has_subfolders(gallery_dir):
            g_map = build_label_map_subfolders(gallery_dir)
            gallery_dataset = HandDataset(gallery_dir, preprocess, g_map)
        else:
            g_map = {}
            gallery_dataset = HandDatasetFlat(gallery_dir, preprocess, g_map)

        query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
        gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)

        # Extract features
        query_feats, query_labels = extract_features(model, query_loader)
        gallery_feats, gallery_labels = extract_features(model, gallery_loader)

        if query_feats is None or gallery_feats is None:
            print(f"Split {i}: No images found in query or gallery. Skipping this split.")
            continue

        # Cosine similarity
        sim_matrix = cosine_similarity(query_feats.numpy(), gallery_feats.numpy())

        rank1_count = 0
        average_precisions = []

        for q_idx in range(len(query_labels)):
            q_label = query_labels[q_idx]
            sims = sim_matrix[q_idx]
            sorted_indices = np.argsort(-sims)  # descending

            # Rank-1
            if gallery_labels[sorted_indices[0]] == q_label:
                rank1_count += 1

            # AP
            correct_inds = np.where(gallery_labels.numpy() == q_label.item())[0]
            hits = 0
            sum_precisions = 0
            for rank, g_idx in enumerate(sorted_indices, start=1):
                if g_idx in correct_inds:
                    hits += 1
                    sum_precisions += hits / rank
            ap = sum_precisions / len(correct_inds) if len(correct_inds) > 0 else 0
            average_precisions.append(ap)

        rank1_acc = rank1_count / len(query_labels) if len(query_labels) > 0 else 0
        mAP = float(np.mean(average_precisions)) if average_precisions else 0

        rank1_list.append(rank1_acc)
        map_list.append(mAP)

        print(f"[Split {i}] Rank-1: {rank1_acc:.4f}, mAP: {mAP:.4f}")

    # 4) Final Averages
    if rank1_list and map_list:
        avg_rank1 = float(np.mean(rank1_list))
        avg_map = float(np.mean(map_list))
        print(f"\n=== Final Results over {len(rank1_list)} splits ===")
        print(f"Mean Rank-1: {avg_rank1:.4f}")
        print(f"Mean mAP:    {avg_map:.4f}")
    else:
        print("No valid splits processed. No final average computed.")

if __name__ == "__main__":
    main()
