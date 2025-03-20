"""
3_a_handclip_finetune_dorsal_r_eval.py

Evaluates Re-ID (query-gallery) using the best model from training.

If query0/ and gallery0/ have subfolders for IDs, we use HandDataset.
If they are flat, we use HandDatasetFlat.
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
    def __init__(self, image_folder, preprocess, label_map):
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
        self.label_map = label_map

        if not os.path.isdir(image_folder):
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
    def __init__(self, image_folder, preprocess, label_map):
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []

        if not os.path.isdir(image_folder):
            return

        for fname in os.listdir(image_folder):
            if fname.endswith('.jpg'):
                full_path = os.path.join(image_folder, fname)
                self.image_paths.append(full_path)

                # parse ID from filename if relevant
                # e.g. "0000000_1.jpg" => "0000000"
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
    if not os.path.isdir(folder):
        return False
    for item in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, item)):
            return True
    return False


def main():
    # Load CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.transformer.requires_grad_(False)
    image_encoder = clip_model.visual

    # Classification head
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

    # Load best model
    save_path = './models/handclip_finetuned_model_dorsal_r.pth'
    if not os.path.exists(save_path):
        print(f"No checkpoint found at {save_path}")
        return
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f"Loaded model from {save_path}")

    # Example: Evaluate on query0/ + gallery0/
    query_dir = './11k/train_val_test_split_dorsal_r/query0'
    gallery_dir = './11k/train_val_test_split_dorsal_r/gallery0'

    def build_label_map_subfolders(folder):
        """Create label_map from subfolders if they exist."""
        label_map_local = {}
        idx = 0
        for item in sorted(os.listdir(folder)):
            subf_path = os.path.join(folder, item)
            if os.path.isdir(subf_path):
                label_map_local[item] = idx
                idx += 1
        return label_map_local

    # Query
    if has_subfolders(query_dir):
        q_map = build_label_map_subfolders(query_dir)
        query_dataset = HandDataset(query_dir, preprocess, q_map)
    else:
        # If flat
        # Possibly we can reuse some global label_map or parse from filename
        # We'll define a small map e.g. label_map = {}
        q_map = {}
        query_dataset = HandDatasetFlat(query_dir, preprocess, q_map)

    # Gallery
    if has_subfolders(gallery_dir):
        g_map = build_label_map_subfolders(gallery_dir)
        gallery_dataset = HandDataset(gallery_dir, preprocess, g_map)
    else:
        g_map = {}
        gallery_dataset = HandDatasetFlat(gallery_dir, preprocess, g_map)

    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)

    # Extract Features
    def extract_features(loader):
        feats_list = []
        labs_list = []
        with torch.no_grad():
            for imgs, labs in loader:
                imgs = imgs.to(device)
                feats = model(imgs, return_features=True)
                feats_list.append(feats.cpu())
                labs_list.append(labs)
        if len(feats_list) == 0:
            return None, None
        all_feats = torch.cat(feats_list)
        all_labs = torch.cat(labs_list)
        return all_feats, all_labs

    query_feats, query_labels = extract_features(query_loader)
    gallery_feats, gallery_labels = extract_features(gallery_loader)

    if query_feats is None or gallery_feats is None:
        print("No images found in query or gallery. Aborting Re-ID evaluation.")
        return

    # Cosine Similarity
    query_feats_np = query_feats.numpy()
    gallery_feats_np = gallery_feats.numpy()
    sim_matrix = cosine_similarity(query_feats_np, gallery_feats_np)

    rank1_count = 0
    average_precisions = []

    for q_idx in range(len(query_labels)):
        q_label = query_labels[q_idx]
        sims = sim_matrix[q_idx]
        sorted_indices = np.argsort(-sims)  # Desc

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

    rank1 = rank1_count / len(query_labels) if len(query_labels) > 0 else 0
    mAP = float(np.mean(average_precisions)) if average_precisions else 0

    print(f"Rank-1 Accuracy: {rank1:.4f}")
    print(f"mAP: {mAP:.4f}")

if __name__ == "__main__":
    main()
