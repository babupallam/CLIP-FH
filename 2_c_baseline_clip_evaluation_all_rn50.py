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

# Load CLIP RN50 model
print("🔹 Loading CLIP RN50 model...")
model, preprocess = clip.load("RN50", device=device)
model.eval()  # Put the model in evaluation mode


# Function to extract image features from a folder
def extract_image_features(image_folder):
    image_features = []
    image_labels = []

    print(f"Extracting features from: {image_folder}")

    # Loop through identities (classes)
    for identity in tqdm(os.listdir(image_folder)):
        identity_path = os.path.join(image_folder, identity)
        if not os.path.isdir(identity_path):
            continue

        # Loop through images of each identity
        for img_file in os.listdir(identity_path):
            img_path = os.path.join(identity_path, img_file)
            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature = model.encode_image(image)
                    feature = F.normalize(feature, dim=-1)  # Normalize embeddings
                image_features.append(feature.cpu())
                image_labels.append(identity)
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")

    # Combine features into one tensor
    image_features = torch.cat(image_features, dim=0)
    return image_features, image_labels


# Function to compute cosine similarity between query and gallery features
def compute_cosine_similarity(query_features, gallery_features):
    query_features = F.normalize(query_features, dim=-1)
    gallery_features = F.normalize(gallery_features, dim=-1)

    # Cosine similarity matrix (dot product of normalized vectors)
    similarity_matrix = query_features @ gallery_features.T
    return similarity_matrix.cpu()


# Evaluation function (returns rank-1 accuracy & mAP)
def evaluate_return(sim_matrix, query_labels, gallery_labels):
    rank1_count = 0
    average_precisions = []

    for idx, query_label in enumerate(query_labels):
        sims = sim_matrix[idx].numpy()
        sorted_indices = np.argsort(-sims)  # Sort in descending order of similarity
        sorted_gallery_labels = np.array(gallery_labels)[sorted_indices]

        # Rank-1 correct?
        if sorted_gallery_labels[0] == query_label:
            rank1_count += 1

        # Calculate AP for this query
        true_matches = (sorted_gallery_labels == query_label).astype(int)
        ap = average_precision_score(true_matches, sims[sorted_indices])
        average_precisions.append(ap)

    rank1_accuracy = rank1_count / len(query_labels)
    mean_ap = np.mean(average_precisions)

    print(f"Rank-1 Accuracy: {rank1_accuracy:.4f}")
    print(f"Mean Average Precision (mAP): {mean_ap:.4f}")

    return rank1_accuracy, mean_ap


# Evaluate over 10 query-gallery splits inside one folder
def evaluate_on_view(folder_path, view_name):
    print(f"\n🚀 Evaluating {view_name} view with RN50...")
    rank1_scores = []
    map_scores = []

    # Evaluate over 10 splits (query-gallery pairs)
    for i in range(10):
        print(f"\n🔹 Split {i} 🔹")
        gallery_dir = os.path.join(folder_path, f'gallery{i}')
        query_dir = os.path.join(folder_path, f'query{i}')

        # Extract features
        gallery_features, gallery_labels = extract_image_features(gallery_dir)
        query_features, query_labels = extract_image_features(query_dir)

        # Compute similarity
        print("Computing cosine similarity...")
        similarity_matrix = compute_cosine_similarity(query_features, gallery_features)

        # Evaluate retrieval performance
        print("Evaluating performance...")
        rank1, mean_ap = evaluate_return(similarity_matrix, query_labels, gallery_labels)

        rank1_scores.append(rank1)
        map_scores.append(mean_ap)

    # Average over 10 splits
    avg_rank1 = np.mean(rank1_scores)
    avg_map = np.mean(map_scores)

    print(f"\n✅ Finished evaluation on {view_name}")
    print(f"✅ Average Rank-1 Accuracy for {view_name}: {avg_rank1:.4f}")
    print(f"✅ Average mAP for {view_name}: {avg_map:.4f}")

    return avg_rank1, avg_map


def main():
    # Paths for each hand view (adjust if your folder names are different!)
    dataset_base = './11k'  # Change this path if dataset is elsewhere

    views = {
        "Dorsal Right": os.path.join(dataset_base, 'train_val_test_split_dorsal_r'),
        "Dorsal Left": os.path.join(dataset_base, 'train_val_test_split_dorsal_l'),
        "Palmar Right": os.path.join(dataset_base, 'train_val_test_split_palmar_r'),
        "Palmar Left": os.path.join(dataset_base, 'train_val_test_split_palmar_l')
    }

    final_rank1_scores = []
    final_map_scores = []

    # Evaluate each hand view directory
    for view_name, folder_path in views.items():
        rank1, map_score = evaluate_on_view(folder_path, view_name)
        final_rank1_scores.append(rank1)
        final_map_scores.append(map_score)

    # Final summary of all views
    print("\n🎉 All views evaluated!")
    for i, view_name in enumerate(views.keys()):
        print(f"{view_name}: Rank-1 = {final_rank1_scores[i]:.4f}, mAP = {final_map_scores[i]:.4f}")


if __name__ == "__main__":
    main()
