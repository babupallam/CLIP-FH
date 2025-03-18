"""
HandCLIP Fine-Tuned Model Evaluation Script for All Query-Gallery Splits
Evaluates query-gallery splits (gallery0/query0 to gallery9/query9).
Outputs: Rank-1 Accuracy and mAP for each split + Average metrics.
"""

# ----------------------------------------
# Step 1: Imports and Setup
# ----------------------------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import numpy as np
from tqdm import tqdm

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------------------
# Step 2: Dataset Class for Query and Gallery
# ----------------------------------------
class HandQueryGalleryDataset(Dataset):
    def __init__(self, image_folder, preprocess):
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []

        # Assumes folder names are labels (e.g., 1001)
        for label_name in sorted(os.listdir(image_folder)):
            label_dir = os.path.join(image_folder, label_name)
            if not os.path.isdir(label_dir):
                continue
            for img_name in os.listdir(label_dir):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(label_dir, img_name))
                    self.labels.append(int(label_name))  # folder name is identity label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.preprocess(img)
        label = self.labels[idx]
        return img, label

# ----------------------------------------
# Step 3: Load the Fine-Tuned HandCLIP Model
# ----------------------------------------
clip_model, preprocess = clip.load("ViT-B/32", device=device)

clip_model.transformer.requires_grad_(False)
image_encoder = clip_model.visual

num_classes = 190  # Your dataset class count
classification_head = nn.Linear(image_encoder.output_dim, num_classes)

class HandCLIPModel(nn.Module):
    def __init__(self, image_encoder, classifier):
        super(HandCLIPModel, self).__init__()
        self.encoder = image_encoder
        self.classifier = classifier

    def forward(self, images):
        features = self.encoder(images)
        logits = self.classifier(features)
        return logits

# Load model and weights
model = HandCLIPModel(image_encoder, classification_head).to(device)
model.load_state_dict(torch.load('handclip_finetuned_model.pth'))
model.eval()

# ----------------------------------------
# Step 4: Feature Extraction Function
# ----------------------------------------
def extract_embeddings(dataloader):
    embeddings = []
    labels = []
    for images, batch_labels in tqdm(dataloader):
        images = images.to(device)
        with torch.no_grad():
            features = model.encoder(images)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
        embeddings.append(features.cpu())
        labels.extend(batch_labels)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.tensor(labels)
    return embeddings, labels

# ----------------------------------------
# Step 5: Evaluation Functions (Rank-1 & mAP)
# ----------------------------------------
def compute_rank1_map(query_embeddings, query_labels, gallery_embeddings, gallery_labels):
    sim_matrix = F.cosine_similarity(
        query_embeddings.unsqueeze(1),
        gallery_embeddings.unsqueeze(0),
        dim=-1
    )

    # Rank-1 Accuracy
    top1_indices = sim_matrix.argmax(dim=1)
    top1_predictions = gallery_labels[top1_indices]
    rank1_correct = (top1_predictions == query_labels).sum().item()
    rank1_accuracy = rank1_correct / len(query_labels)

    # mAP
    mAP = compute_mean_average_precision(sim_matrix, query_labels, gallery_labels)

    return rank1_accuracy, mAP

def compute_mean_average_precision(sim_matrix, query_labels, gallery_labels):
    average_precisions = []
    query_labels = query_labels.numpy()
    gallery_labels = gallery_labels.numpy()

    for i in range(len(query_labels)):
        similarities = sim_matrix[i].cpu().numpy()
        target_label = query_labels[i]

        sorted_indices = np.argsort(-similarities)
        sorted_gallery_labels = gallery_labels[sorted_indices]

        correct_matches = (sorted_gallery_labels == target_label)
        ranks = np.where(correct_matches)[0]

        if len(ranks) == 0:
            average_precisions.append(0)
            continue

        precisions = []
        for rank_idx, rank in enumerate(ranks):
            precision_at_k = (rank_idx + 1) / (rank + 1)
            precisions.append(precision_at_k)

        average_precision = np.mean(precisions)
        average_precisions.append(average_precision)

    return np.mean(average_precisions)

# ----------------------------------------
# Step 6: Loop Over All Query-Gallery Splits
# ----------------------------------------
gallery_root = "./11k/train_val_test_split_dorsal_r"
query_root = "./11k/train_val_test_split_dorsal_r"

num_splits = 10
rank1_scores = []
map_scores = []

for i in range(num_splits):
    gallery_dir = os.path.join(gallery_root, f"gallery{i}")
    query_dir = os.path.join(query_root, f"query{i}")

    print(f"\n========== Evaluating Split {i} ==========\n")

    # Create datasets and loaders
    gallery_dataset = HandQueryGalleryDataset(gallery_dir, preprocess)
    query_dataset = HandQueryGalleryDataset(query_dir, preprocess)

    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)

    print(f"Extracting features from: {query_dir}")
    query_embeddings, query_labels = extract_embeddings(query_loader)

    print(f"Extracting features from: {gallery_dir}")
    gallery_embeddings, gallery_labels = extract_embeddings(gallery_loader)

    print("\nComputing similarity between query and gallery images...\n")
    rank1_acc, mAP_score = compute_rank1_map(query_embeddings, query_labels, gallery_embeddings, gallery_labels)

    rank1_scores.append(rank1_acc)
    map_scores.append(mAP_score)

    print(f"[Split {i}] Rank-1 Accuracy: {rank1_acc:.4f}")
    print(f"[Split {i}] Mean Average Precision (mAP): {mAP_score:.4f}")

# ----------------------------------------
# Step 7: Average Metrics Across Splits
# ----------------------------------------
avg_rank1 = np.mean(rank1_scores)
avg_map = np.mean(map_scores)

print("\n========== Final Averaged Results Across All Splits ==========\n")
for i in range(num_splits):
    print(f"Split {i}: Rank-1 Accuracy = {rank1_scores[i]:.4f}, mAP = {map_scores[i]:.4f}")

print("\nAverage Rank-1 Accuracy: {:.4f}".format(avg_rank1))
print("Average Mean Average Precision (mAP): {:.4f}".format(avg_map))

print("\nHandCLIP multi-split evaluation completed ✅")
