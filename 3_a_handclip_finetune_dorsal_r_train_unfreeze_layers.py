"""
3_a_handclip_finetune_dorsal_r_train_unfreeze_layers.py

Trains a CLIP-based model on dorsal_r data using ID classification (CrossEntropy).
Unfreezes later layers of the image encoder for better fine-tuning.
After training, saves the best model checkpoint: ./models/handclip_finetuned_dorsal_r.pth
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
class IDFolderDataset(Dataset):
    def __init__(self, root_dir, preprocess, label_map):
        super().__init__()
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.label_map = label_map
        self.samples = []

        if not os.path.isdir(root_dir):
            print(f"WARNING: {root_dir} not found.")
            return

        for folder_name, label_idx in label_map.items():
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.endswith(".jpg"):
                        full_path = os.path.join(folder_path, fname)
                        self.samples.append((full_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)
        return img, label


# ---------------------------
# Model
# ---------------------------
class HandCLIPReID(nn.Module):
    def __init__(self, clip_encoder, num_classes):
        super().__init__()
        self.encoder = clip_encoder
        self.classifier = nn.Linear(clip_encoder.output_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


# ---------------------------
# Function to Freeze Layers
# ---------------------------
def freeze_clip_layers(clip_model, unfreeze_layers=[]):
    """
    Freezes all layers except the ones in unfreeze_layers list.
    Example: unfreeze_layers = ['transformer.resblocks.10', 'transformer.resblocks.11']
    """
    # Freeze everything first
    for name, param in clip_model.visual.named_parameters():
        param.requires_grad = False

    # Unfreeze selected layers
    for name, param in clip_model.visual.named_parameters():
        if any([name.startswith(layer) for layer in unfreeze_layers]):
            param.requires_grad = True
            print(f"Unfreezing layer: {name}")


def main():
    # ----------------------------------
    # 1) Load CLIP and create model
    # ----------------------------------
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Freeze text encoder completely
    clip_model.transformer.requires_grad_(False)

    # Strategy: Unfreeze the last 2 transformer blocks of the image encoder
    # These are often 'transformer.resblocks.10' and 'transformer.resblocks.11'
    unfreeze_layers = ['transformer.resblocks.10', 'transformer.resblocks.11']

    freeze_clip_layers(clip_model, unfreeze_layers=unfreeze_layers)

    image_encoder = clip_model.visual

    num_classes = 190  # adjust if needed
    model = HandCLIPReID(image_encoder, num_classes).to(device)

    # ----------------------------------
    # 2) Prepare train/val
    # ----------------------------------
    train_dir = './11k/train_val_test_split_dorsal_r/train'
    val_dir = './11k/train_val_test_split_dorsal_r/val'

    label_map = {}
    if os.path.isdir(train_dir):
        subfolders = sorted(os.listdir(train_dir))
        idx = 0
        for sf in subfolders:
            sf_path = os.path.join(train_dir, sf)
            if os.path.isdir(sf_path):
                label_map[sf] = idx
                idx += 1
    else:
        print(f"{train_dir} not found. Can't train.")
        return

    train_dataset = IDFolderDataset(train_dir, preprocess, label_map)
    val_dataset = IDFolderDataset(val_dir, preprocess, label_map)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ----------------------------------
    # 3) Loss & Optimizer
    # ----------------------------------
    criterion = nn.CrossEntropyLoss()

    # Collect all parameters that require gradients (unfrozen ones + classifier head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.01)

    # ----------------------------------
    # 4) Training Loop
    # ----------------------------------
    num_epochs = 5
    best_val_acc = 0.0
    save_path = './models/handclip_finetuned_dorsal_r.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total if total else 0
        epoch_acc = correct / total if total else 0
        print(f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                v_logits = model(images)
                _, v_preds = torch.max(v_logits, dim=1)
                val_correct += (v_preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total else 0
        print(f"[Epoch {epoch + 1}/{num_epochs}] Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved with val_acc: {best_val_acc:.4f}")

    print("\nTraining complete.")
    print("Best validation accuracy:", best_val_acc)


if __name__ == "__main__":
    main()
