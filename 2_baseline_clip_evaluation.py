import os
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP Model (ViT-B/32)
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Define dataset paths (modify accordingly)
gallery_dir = "./11k/train_val_test_split_dorsal_r/gallery0"  # Example path
query_dir = "./11k/train_val_test_split_dorsal_r/query0"  # Example path


# Helper function to extract image embeddings
def extract_image_features(image_folder):
    features = []
    labels = []

    print(f"Extracting features from: {image_folder}")

    for person_id in tqdm(os.listdir(image_folder)):
        person_folder = os.path.join(image_folder, person_id)
        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            # Load and preprocess image
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

            # Get image feature from CLIP
            with torch.no_grad():
                img_feature = model.encode_image(image)
                img_feature /= img_feature.norm(dim=-1, keepdim=True)  # Normalize

            features.append(img_feature.cpu().numpy())
            labels.append(person_id)

    features = np.concatenate(features, axis=0)

    return features, labels


# Step 1: Extract gallery features
gallery_features, gallery_labels = extract_image_features(gallery_dir)

# Step 2: Extract query features
query_features, query_labels = extract_image_features(query_dir)

# Step 3: Compute cosine similarity
print("\nComputing similarity between query and gallery images...")


def compute_cosine_similarity(query_feats, gallery_feats):
    query_feats = torch.tensor(query_feats)
    gallery_feats = torch.tensor(gallery_feats)

    similarity_matrix = query_feats @ gallery_feats.T
    return similarity_matrix


similarity_matrix = compute_cosine_similarity(query_features, gallery_features)

# Step 4: Evaluate Rank-1 accuracy and mAP
print("\nEvaluating performance...")


def evaluate(sim_matrix, query_labels, gallery_labels):
    rank1_count = 0
    average_precisions = []

    for idx, query_label in enumerate(query_labels):
        sims = sim_matrix[idx].numpy()
        sorted_indices = np.argsort(-sims)  # Descending sort
        sorted_gallery_labels = np.array(gallery_labels)[sorted_indices]

        # Rank-1 accuracy
        if sorted_gallery_labels[0] == query_label:
            rank1_count += 1

        # mAP computation
        true_matches = (sorted_gallery_labels == query_label).astype(int)
        ap = average_precision_score(true_matches, sims[sorted_indices])
        average_precisions.append(ap)

    rank1_accuracy = rank1_count / len(query_labels)
    mean_ap = np.mean(average_precisions)

    print(f"Rank-1 Accuracy: {rank1_accuracy:.4f}")
    print(f"Mean Average Precision (mAP): {mean_ap:.4f}")


evaluate(similarity_matrix, query_labels, gallery_labels)

print("\nBaseline CLIP evaluation completed ✅")
