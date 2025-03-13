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

    for identity in os.listdir(image_folder):
        identity_path = os.path.join(image_folder, identity)
        if not os.path.isdir(identity_path):
            continue

        for img_file in os.listdir(identity_path):
            img_path = os.path.join(identity_path, img_file)
            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature = model.encode_image(image)
                    feature = F.normalize(feature, dim=-1)
                image_features.append(feature.cpu())
                image_labels.append(identity)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    image_features = torch.cat(image_features, dim=0)
    return image_features, image_labels


# Function to compute cosine similarity between query and gallery features
def compute_cosine_similarity(query_features, gallery_features):
    query_features = F.normalize(query_features, dim=-1)
    gallery_features = F.normalize(gallery_features, dim=-1)
    similarity_matrix = query_features @ gallery_features.T
    return similarity_matrix.cpu()


# Evaluation function (returns rank-1 accuracy & mAP)
def evaluate_return(sim_matrix, query_labels, gallery_labels):
    rank1_count = 0
    average_precisions = []

    for idx, query_label in enumerate(query_labels):
        sims = sim_matrix[idx].numpy()
        sorted_indices = np.argsort(-sims)
        sorted_gallery_labels = np.array(gallery_labels)[sorted_indices]

        if sorted_gallery_labels[0] == query_label:
            rank1_count += 1

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
    print(f"\n🚀 Evaluating {view_name} view...")
    rank1_scores = []
    map_scores = []

    for i in range(10):
        print(f"\n🔹 Split {i} 🔹")
        gallery_dir = os.path.join(folder_path, f'gallery{i}')
        query_dir = os.path.join(folder_path, f'query{i}')

        # Extract features
        print(f"Extracting gallery features from: {gallery_dir}")
        gallery_features, gallery_labels = extract_image_features(gallery_dir)

        print(f"Extracting query features from: {query_dir}")
        query_features, query_labels = extract_image_features(query_dir)

        # Compute similarity
        print("Computing cosine similarity...")
        similarity_matrix = compute_cosine_similarity(query_features, gallery_features)

        # Evaluate
        print("Evaluating...")
        rank1, mean_ap = evaluate_return(similarity_matrix, query_labels, gallery_labels)

        rank1_scores.append(rank1)
        map_scores.append(mean_ap)

    avg_rank1 = np.mean(rank1_scores)
    avg_map = np.mean(map_scores)

    print(f"\n✅ Finished evaluation on {view_name}")
    print(f"✅ Average Rank-1 Accuracy for {view_name}: {avg_rank1:.4f}")
    print(f"✅ Average mAP for {view_name}: {avg_map:.4f}")

    return avg_rank1, avg_map


def main():
    # Paths for each hand view (adjust if you renamed them!)
    dataset_base = './11k'  # Change if dataset root is elsewhere
    views = {
        "Dorsal Right": os.path.join(dataset_base, 'train_val_test_split_dorsal_r'),
        "Dorsal Left": os.path.join(dataset_base, 'train_val_test_split_dorsal_l'),
        "Palmar Right": os.path.join(dataset_base, 'train_val_test_split_palmar_r'),
        "Palmar Left": os.path.join(dataset_base, 'train_val_test_split_palmar_l')
    }

    final_rank1_scores = []
    final_map_scores = []

    # Evaluate each view
    for view_name, folder_path in views.items():
        rank1, map_score = evaluate_on_view(folder_path, view_name)
        final_rank1_scores.append(rank1)
        final_map_scores.append(map_score)

    # Summary of all views
    print("\n🎉 All views evaluated!")
    for i, view_name in enumerate(views.keys()):
        print(f"{view_name}: Rank-1 = {final_rank1_scores[i]:.4f}, mAP = {final_map_scores[i]:.4f}")


if __name__ == "__main__":
    main()
