import os
import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)


# Function to extract image features from a folder
def extract_image_features(image_folder):
    image_features = []
    image_labels = []

    # Iterate through identity folders
    for identity in os.listdir(image_folder):
        identity_path = os.path.join(image_folder, identity)
        if not os.path.isdir(identity_path):
            continue

        # Iterate through images in the identity folder
        for img_file in os.listdir(identity_path):
            img_path = os.path.join(identity_path, img_file)
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model.encode_image(image)
                feature = F.normalize(feature, dim=-1)  # Normalize to unit length

            image_features.append(feature.cpu())
            image_labels.append(identity)

    # Stack into tensors
    image_features = torch.cat(image_features, dim=0)
    return image_features, image_labels


# Function to compute cosine similarity between query and gallery features
def compute_cosine_similarity(query_features, gallery_features):
    # Normalize features (already normalized, but double-check)
    query_features = F.normalize(query_features, dim=-1)
    gallery_features = F.normalize(gallery_features, dim=-1)

    similarity_matrix = query_features @ gallery_features.T  # Cosine similarity
    return similarity_matrix.cpu()


# Evaluation function that returns rank-1 and mAP
def evaluate_return(sim_matrix, query_labels, gallery_labels):
    rank1_count = 0
    average_precisions = []

    for idx, query_label in enumerate(query_labels):
        sims = sim_matrix[idx].numpy()
        sorted_indices = np.argsort(-sims)
        sorted_gallery_labels = np.array(gallery_labels)[sorted_indices]

        # Rank-1 check
        if sorted_gallery_labels[0] == query_label:
            rank1_count += 1

        # True/False matches for AP calculation
        true_matches = (sorted_gallery_labels == query_label).astype(int)
        ap = average_precision_score(true_matches, sims[sorted_indices])
        average_precisions.append(ap)

    rank1_accuracy = rank1_count / len(query_labels)
    mean_ap = np.mean(average_precisions)

    print(f"Rank-1 Accuracy: {rank1_accuracy:.4f}")
    print(f"Mean Average Precision (mAP): {mean_ap:.4f}")

    return rank1_accuracy, mean_ap


# Main function to loop over 10 splits and compute average performance
def main():
    rank1_scores = []
    map_scores = []

    # Loop over 10 splits (gallery0-query0 to gallery9-query9)
    for i in range(10):
        print(f"\nRunning split {i}...")

        # Modify path according to the dataset structure
        gallery_dir = f"./11k/train_val_test_split_dorsal_r/gallery{i}"
        query_dir = f"./11k/train_val_test_split_dorsal_r/query{i}"

        # Extract gallery features
        print(f"Extracting features from: {gallery_dir}")
        gallery_features, gallery_labels = extract_image_features(gallery_dir)

        # Extract query features
        print(f"Extracting features from: {query_dir}")
        query_features, query_labels = extract_image_features(query_dir)

        # Compute cosine similarity matrix
        print("Computing similarity between query and gallery images...")
        similarity_matrix = compute_cosine_similarity(query_features, gallery_features)

        # Evaluate performance (Rank-1 and mAP)
        print("Evaluating performance...")
        rank1, mean_ap = evaluate_return(similarity_matrix, query_labels, gallery_labels)

        # Save results
        rank1_scores.append(rank1)
        map_scores.append(mean_ap)

    # Compute final average results across all splits
    avg_rank1 = np.mean(rank1_scores)
    avg_map = np.mean(map_scores)

    print("\n✅ Baseline CLIP Evaluation Across All Splits Completed!")
    print(f"Average Rank-1 Accuracy: {avg_rank1:.4f}")
    print(f"Average Mean Average Precision (mAP): {avg_map:.4f}")


if __name__ == "__main__":
    main()
