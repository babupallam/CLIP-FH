"""
3_a_handclip_finetune_dorsal_r_eval_multi_gallery_all.py

Evaluate Re-ID over query0..query9 vs. gallery0_all..gallery9_all (cross-aspect).
Computes Rank-1 and mAP for each split and averages them.
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

# -----------------------------------
# Dataset Classes
# -----------------------------------
class IDFolderDataset(Dataset):
    def __init__(self, root_dir, preprocess, label_map):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.label_map = label_map
        self.samples = []

        if not os.path.isdir(root_dir):
            print(f"WARNING: {root_dir} not found.")
            return

        for folder_name, label_idx in label_map.items():
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.endswith(".jpg"):
                        full_path = os.path.join(folder_path, fname)
                        self.samples.append((full_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)
        return img, label

class FlatFolderDataset(Dataset):
    def __init__(self, folder, preprocess):
        self.folder = folder
        self.preprocess = preprocess
        self.samples = []

        if not os.path.isdir(folder):
            print(f"WARNING: {folder} not found.")
            return

        for fname in os.listdir(folder):
            if fname.endswith(".jpg"):
                full_path = os.path.join(folder, fname)
                pid = fname.split("_")[0]  # assuming filename starts with ID
                label = int(pid) if pid.isdigit() else -1
                self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)
        return img, label

def has_subfolders(folder_path):
    if not os.path.isdir(folder_path):
        return False
    for item in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, item)):
            return True
    return False

# -----------------------------------
# Model Class
# -----------------------------------
class HandCLIPReID(nn.Module):
    def __init__(self, clip_encoder, num_classes):
        super().__init__()
        self.encoder = clip_encoder
        self.classifier = nn.Linear(clip_encoder.output_dim, num_classes)

    def forward(self, x, return_features=False):
        feats = self.encoder(x)
        if return_features:
            return feats
        return self.classifier(feats)

# -----------------------------------
# Extract Features
# -----------------------------------
def extract_features(model, loader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            feats = model(imgs, return_features=True)
            features.append(feats.cpu())
            labels.append(labs)
    if not features:
        return None, None
    all_feats = torch.cat(features)
    all_labs = torch.cat(labels)
    return all_feats, all_labs

# -----------------------------------
# Main Evaluation Loop
# -----------------------------------
def main():
    save_path = "./models/handclip_finetuned_dorsal_r.pth"
    if not os.path.exists(save_path):
        print(f"Model not found at {save_path}. Exiting.")
        return

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.transformer.requires_grad_(False)
    image_encoder = clip_model.visual

    num_classes = 190
    model = HandCLIPReID(image_encoder, num_classes).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f"Loaded model from {save_path}")

    N = 10
    rank1_list = []
    map_list = []

    for i in range(N):
        query_dir = f"./11k/train_val_test_split_dorsal_r/query{i}"
        gallery_dir = f"./11k/train_val_test_split_dorsal_r/gallery{i}_all"

        if not os.path.isdir(query_dir) or not os.path.isdir(gallery_dir):
            print(f"Split {i}: query or gallery dir not found. Skipping.")
            continue

        # Query Dataset
        if has_subfolders(query_dir):
            q_map = {folder: idx for idx, folder in enumerate(sorted(os.listdir(query_dir))) if os.path.isdir(os.path.join(query_dir, folder))}
            query_dataset = IDFolderDataset(query_dir, preprocess, q_map)
        else:
            query_dataset = FlatFolderDataset(query_dir, preprocess)

        # Gallery Dataset (all aspects combined)
        if has_subfolders(gallery_dir):
            g_map = {folder: idx for idx, folder in enumerate(sorted(os.listdir(gallery_dir))) if os.path.isdir(os.path.join(gallery_dir, folder))}
            gallery_dataset = IDFolderDataset(gallery_dir, preprocess, g_map)
        else:
            gallery_dataset = FlatFolderDataset(gallery_dir, preprocess)

        query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
        gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)

        q_feats, q_labs = extract_features(model, query_loader)
        g_feats, g_labs = extract_features(model, gallery_loader)

        if q_feats is None or g_feats is None:
            print(f"Split {i}: No images found in query/gallery. Skipping.")
            continue

        sim_matrix = cosine_similarity(q_feats.numpy(), g_feats.numpy())

        rank1_count = 0
        average_precisions = []

        for q_idx in range(len(q_labs)):
            q_label = q_labs[q_idx]
            sims = sim_matrix[q_idx]
            sorted_indices = np.argsort(-sims)

            # Rank-1
            if g_labs[sorted_indices[0]] == q_label:
                rank1_count += 1

            # mAP calculation
            correct_indices = np.where(g_labs.numpy() == q_label.item())[0]
            hits = 0
            sum_precisions = 0
            for rank, g_idx in enumerate(sorted_indices, start=1):
                if g_idx in correct_indices:
                    hits += 1
                    sum_precisions += hits / rank
            ap = sum_precisions / len(correct_indices) if len(correct_indices) > 0 else 0
            average_precisions.append(ap)

        rank1 = rank1_count / len(q_labs) if len(q_labs) > 0 else 0
        mAP = float(np.mean(average_precisions)) if average_precisions else 0

        rank1_list.append(rank1)
        map_list.append(mAP)

        print(f"[Split {i}] Rank-1: {rank1:.4f}, mAP: {mAP:.4f}")

    if rank1_list and map_list:
        avg_rank1 = np.mean(rank1_list)
        avg_map = np.mean(map_list)
        print(f"\n=== Final Cross-Aspect Results over {len(rank1_list)} splits ===")
        print(f"Mean Rank-1: {avg_rank1:.4f}")
        print(f"Mean mAP:    {avg_map:.4f}")
    else:
        print("No valid splits processed. No final average computed.")

if __name__ == "__main__":
    main()
