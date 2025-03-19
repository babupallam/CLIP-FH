import os
import torch
import clip
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score

# ----------------------------------------
# Step 1: Setup and Load CLIP Model
# ----------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# ----------------------------------------
# Step 2: Define Dataset Class
# ----------------------------------------
class HandImageDataset(Dataset):
    """ Custom dataset to load hand images for evaluation. """
    def __init__(self, image_folder, preprocess):
        self.image_paths = []
        self.image_ids = []
        self.preprocess = preprocess

        for person_id in os.listdir(image_folder):
            person_folder = os.path.join(image_folder, person_id)
            if os.path.isdir(person_folder):
                for img_name in os.listdir(person_folder):
                    if img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(person_folder, img_name))
                        self.image_ids.append(person_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.preprocess(img)
        return img, self.image_ids[idx]

# ----------------------------------------
# Step 3: Function to Extract Features
# ----------------------------------------
def extract_features(image_folder):
    """ Extracts features from a given folder using CLIP. """
    dataset = HandImageDataset(image_folder, preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    features = []
    labels = []

    with torch.no_grad():
        for images, ids in tqdm(dataloader, desc=f"Extracting features from: {image_folder}"):
            images = images.to(device)
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize features
            features.append(image_features.cpu().numpy())
            labels.extend(ids)

    features = np.vstack(features)  # Stack all features
    return features, labels

# ----------------------------------------
# Step 4: Compute Similarity and Evaluate
# ----------------------------------------
def compute_similarity(query_features, query_labels, gallery_features, gallery_labels):
    """ Computes cosine similarity and evaluates Rank-1 and mAP. """
    similarity_matrix = np.dot(query_features, gallery_features.T)

    rank_1_correct = 0
    average_precisions = []

    for i, query_label in enumerate(query_labels):
        scores = similarity_matrix[i]
        sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
        sorted_gallery_labels = np.array(gallery_labels)[sorted_indices]

        # Rank-1 Accuracy
        if sorted_gallery_labels[0] == query_label:
            rank_1_correct += 1

        # mAP Calculation
        relevance = (sorted_gallery_labels == query_label).astype(int)
        if np.sum(relevance) > 0:
            ap = average_precision_score(relevance, scores[sorted_indices])
            average_precisions.append(ap)

    rank1_acc = rank_1_correct / len(query_labels)
    mean_ap = np.mean(average_precisions) if average_precisions else 0

    return rank1_acc, mean_ap

# ----------------------------------------
# Step 5: Evaluate Across All Splits (0 to 9)
# ----------------------------------------
query_base = "./11k/train_val_test_split_dorsal_r/query"
gallery_base = "./11k/train_val_test_split_dorsal_r/gallery"

rank1_scores = []
map_scores = []

for i in range(10):
    query_folder = f"{query_base}{i}"
    gallery_folder = f"{gallery_base}{i}_all"  # Using galleryX_all

    print(f"\nEvaluating: Query={query_folder}, Gallery={gallery_folder}")

    query_features, query_labels = extract_features(query_folder)
    gallery_features, gallery_labels = extract_features(gallery_folder)

    rank1, mean_ap = compute_similarity(query_features, query_labels, gallery_features, gallery_labels)

    rank1_scores.append(rank1)
    map_scores.append(mean_ap)

    print(f"✅ [Query {i}] Rank-1 Accuracy: {rank1:.4f}, mAP: {mean_ap:.4f}")

# ----------------------------------------
# Step 6: Compute and Print Final Averages
# ----------------------------------------
avg_rank1 = np.mean(rank1_scores)
avg_map = np.mean(map_scores)

print("\n✅ Final Evaluation on `galleryX_all` (Dorsal Right)")
print(f"Average Rank-1 Accuracy: {avg_rank1:.4f}")
print(f"Average Mean Average Precision (mAP): {avg_map:.4f}")
