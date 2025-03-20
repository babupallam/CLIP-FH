"""
3_a_handclip_finetune_dorsal_r_train.py

This script trains a CLIP-based model for dorsal_r
classification using label smoothing cross-entropy.
After training, it saves the best model checkpoint.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------
# Dataset Classes
# ---------------------------
class HandDataset(Dataset):
    """
    Subfolder-based dataset:
      train/
        0000000/*.jpg
        0000001/*.jpg
      ...
    """
    def __init__(self, image_folder, preprocess, label_map):
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
        self.label_map = label_map

        if not os.path.isdir(image_folder):
            print(f"WARNING: {image_folder} is not a directory. No images loaded.")
            return

        for label_name, label_idx in label_map.items():
            subfolder = os.path.join(image_folder, label_name)
            if os.path.isdir(subfolder):
                for img_name in os.listdir(subfolder):
                    if img_name.endswith('.jpg'):
                        path = os.path.join(subfolder, img_name)
                        self.image_paths.append(path)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)
        label = self.labels[idx]
        return img, label


# ---------------------------
# Label Smoothing CE
# ---------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, preds, targets):
        log_probs = nn.functional.log_softmax(preds, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# ---------------------------
# Main Training Code
# ---------------------------
def main():
    # Load CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Freeze text encoder
    clip_model.transformer.requires_grad_(False)

    # Extract image encoder
    image_encoder = clip_model.visual

    # Classification head
    num_classes = 190  # Adjust as needed
    classification_head = nn.Linear(image_encoder.output_dim, num_classes)

    # Combine
    class HandCLIPReID(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier

        def forward(self, images):
            features = self.encoder(images)
            logits = self.classifier(features)
            return logits

    model = HandCLIPReID(image_encoder, classification_head).to(device)

    # Prepare dataset (train/val)
    train_dir = './11k/train_val_test_split_dorsal_r/train'
    val_dir = './11k/train_val_test_split_dorsal_r/val'

    # Build label_map
    label_map = {}
    if os.path.isdir(train_dir):
        subfolders = sorted(os.listdir(train_dir))
        idx = 0
        for sf in subfolders:
            sf_path = os.path.join(train_dir, sf)
            if os.path.isdir(sf_path):
                label_map[sf] = idx
                idx += 1

    # Create datasets
    train_dataset = HandDataset(train_dir, preprocess, label_map)
    val_dataset = HandDataset(val_dir, preprocess, label_map)

    # DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss, Optim
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    # Training loop
    num_epochs = 5
    best_val_acc = 0.0
    save_path = './models/handclip_finetuned_model_dorsal_r.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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

        epoch_loss = running_loss / total if total else 0
        epoch_acc = correct / total if total else 0
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        # Validation
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

        val_acc = val_correct / val_total if val_total else 0
        print(f"[Epoch {epoch+1}/{num_epochs}] Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved with val_acc: {best_val_acc:.4f}")

    print("Training completed. Best val accuracy:", best_val_acc)

if __name__ == "__main__":
    main()
