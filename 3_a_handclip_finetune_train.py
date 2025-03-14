"""

This script fine-tunes the CLIP image encoder for hand biometric recognition using Cross-Entropy Loss.
Dataset: Hand images from 11k dataset split into train, val, test.
No text prompts or contrastive learning included.
"""

# ----------------------------------------
# Step 1: Imports and Setup
# ----------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import clip

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ----------------------------------------
# Step 2: Define Dataset Class
# ----------------------------------------
class HandDataset(Dataset):
    """
    Custom PyTorch dataset for loading hand images and labels.
    """

    def __init__(self, image_folder, preprocess, label_map):
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
        self.label_map = label_map  # Dictionary mapping folder names to label indices

        # Load image paths and assign labels
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
# Load pre-trained CLIP model (ViT-B/32)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Freeze CLIP text encoder (optional: not used in this implementation)
clip_model.transformer.requires_grad_(False)

# Extract the image encoder
image_encoder = clip_model.visual

# Add a classification head (linear layer)
num_classes = 190  # Change this if your dataset has different class count
classification_head = nn.Linear(image_encoder.output_dim, num_classes)


# Combine encoder and classifier into a single model
class HandCLIPModel(nn.Module):
    def __init__(self, image_encoder, classifier):
        super(HandCLIPModel, self).__init__()
        self.encoder = image_encoder
        self.classifier = classifier

    def forward(self, images):
        features = self.encoder(images)
        logits = self.classifier(features)
        return logits


model = HandCLIPModel(image_encoder, classification_head).to(device)

# ----------------------------------------
# Step 4: Define Datasets and DataLoaders
# ----------------------------------------
# Paths to train/val datasets
train_dir = './11k/train_val_test_split_dorsal_r/train'
val_dir = './11k/train_val_test_split_dorsal_r/val'

# Label map based on directory names (e.g., '1001': 0, '1002': 1, etc.)
label_names = sorted(os.listdir(train_dir))
label_map = {label_name: idx for idx, label_name in enumerate(label_names)}

# Create datasets
train_dataset = HandDataset(train_dir, preprocess, label_map)
val_dataset = HandDataset(val_dir, preprocess, label_map)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------------------
# Step 5: Define Loss Function and Optimizer
# ----------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# ----------------------------------------
# Step 6: Training Loop
# ----------------------------------------
num_epochs = 50
best_val_acc = 0.0
save_path = 'handclip_finetuned_model.pth'

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

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

    # Validation phase
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

    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved with val_acc: {best_val_acc:.4f}")

print("Training completed. Best validation accuracy:", best_val_acc)
