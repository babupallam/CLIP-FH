# ----------------------------------------
# Step 1: Imports and Setup
# ----------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import clip

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------------------
# Step 2: Define Dataset Class (Same as Before)
# ----------------------------------------
class HandDataset(Dataset):
    def __init__(self, image_folder, preprocess, label_map):
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
        self.label_map = label_map  # Dict: folder names to label indices

        for label_name, label_idx in label_map.items():
            label_dir = os.path.join(image_folder, label_name)
            for img_name in os.listdir(label_dir):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(label_dir, img_name))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.preprocess(img)
        label = self.labels[idx]
        return img, label

# ----------------------------------------
# Step 3: Load CLIP Model and Modify
# ----------------------------------------
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Freeze text encoder (optional: not used here)
clip_model.transformer.requires_grad_(False)

# Extract image encoder only
image_encoder = clip_model.visual

# Add a classification head (linear layer)
num_classes = 190  # Change if dataset class count changes
classification_head = nn.Linear(image_encoder.output_dim, num_classes)

# Combine encoder and classifier into a single model
class HandCLIPReID(nn.Module):
    def __init__(self, image_encoder, classifier):
        super(HandCLIPReID, self).__init__()
        self.encoder = image_encoder
        self.classifier = classifier

    def forward(self, images, return_features=False):
        features = self.encoder(images)  # Extract features
        if return_features:
            return features  # Return embeddings for query/gallery matching
        logits = self.classifier(features)  # Return logits for classification
        return logits

model = HandCLIPReID(image_encoder, classification_head).to(device)

# ----------------------------------------
# Step 4: Define Datasets and DataLoaders
# ----------------------------------------
train_dir = './11k/train_val_test_split_dorsal_r/train'
val_dir = './11k/train_val_test_split_dorsal_r/val'

label_names = sorted(os.listdir(train_dir))
label_map = {label_name: idx for idx, label_name in enumerate(label_names)}

train_dataset = HandDataset(train_dir, preprocess, label_map)
val_dataset = HandDataset(val_dir, preprocess, label_map)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------------------
# Step 5: Define Label Smoothing Cross-Entropy Loss
# ----------------------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, preds, targets):
        log_probs = nn.functional.log_softmax(preds, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# ----------------------------------------
# Step 6: Optimizer and Scheduler
# ----------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
# Optional: CosineAnnealingLR or StepLR can be added here

# ----------------------------------------
# Step 7: Training Loop (CE + Label Smoothing)
# ----------------------------------------
num_epochs = 5
best_val_acc = 0.0
save_path = '/models/handclip_reid_label_smooth_dorsal_r.pth'

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # Returns logits
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

    # Validation Phase (Classification)
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"[Epoch {epoch + 1}/{num_epochs}] Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"✅ Best model saved with val_acc: {best_val_acc:.4f}")

print("Training completed. Best validation accuracy:", best_val_acc)

# ----------------------------------------
# Step 8: Query-Gallery Evaluation Using Embeddings (Re-ID)
# ----------------------------------------

def extract_features(data_loader):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = model(images, return_features=True)  # Get embeddings
            all_features.append(features.cpu())
            all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)

# Example query/gallery dataset directories
query_dir = './11k/train_val_test_split_dorsal_r/query0'
gallery_dir = './11k/train_val_test_split_dorsal_r/gallery0'

# Create query/gallery datasets
query_dataset = HandDataset(query_dir, preprocess, label_map)
gallery_dataset = HandDataset(gallery_dir, preprocess, label_map)

query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)

# Extract features
query_features, query_labels = extract_features(query_loader)
gallery_features, gallery_labels = extract_features(gallery_loader)

# Convert to numpy arrays for similarity search
query_features_np = query_features.numpy()
gallery_features_np = gallery_features.numpy()

# Compute cosine similarity between query and gallery
similarity_matrix = cosine_similarity(query_features_np, gallery_features_np)

# Rank gallery images for each query
rank1_count = 0
average_precisions = []

for q_idx in range(len(query_labels)):
    q_label = query_labels[q_idx]
    sims = similarity_matrix[q_idx]
    sorted_indices = np.argsort(-sims)

    # Rank-1 Accuracy
    if gallery_labels[sorted_indices[0]] == q_label:
        rank1_count += 1

    # Compute Average Precision (AP)
    correct_indices = np.where(gallery_labels.numpy() == q_label)[0]
    hits = 0
    sum_precisions = 0
    for rank, g_idx in enumerate(sorted_indices, 1):
        if g_idx in correct_indices:
            hits += 1
            sum_precisions += hits / rank
    average_precision = sum_precisions / len(correct_indices) if len(correct_indices) > 0 else 0
    average_precisions.append(average_precision)

rank1_acc = rank1_count / len(query_labels)
mean_ap = np.mean(average_precisions)

print(f"✅ Re-ID Evaluation Complete:\nRank-1 Accuracy: {rank1_acc:.4f}\nMean Average Precision (mAP): {mean_ap:.4f}")
