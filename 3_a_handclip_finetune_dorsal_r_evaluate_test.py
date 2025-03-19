# ----------------------------------------
# Step 1: Imports and Setup
# ----------------------------------------
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------------------
# Step 2: Dataset Class for Test Set
# ----------------------------------------
class HandTestDataset(Dataset):
    """
    Custom Dataset for loading hand test images and their labels.
    """
    def __init__(self, test_folder, preprocess, label_map):
        self.image_paths = []
        self.labels = []
        self.preprocess = preprocess
        self.label_map = label_map

        for label_name, label_idx in label_map.items():
            label_dir = os.path.join(test_folder, label_name)
            for img_name in os.listdir(label_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(label_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.preprocess(img)
        label = self.labels[idx]
        return img, label

# ----------------------------------------
# Step 3: Load HandCLIP Model
# ----------------------------------------
import clip
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Freeze text encoder (not used)
clip_model.transformer.requires_grad_(False)

# Extract image encoder
image_encoder = clip_model.visual

# Classification head
num_classes = 190  # adjust to your dataset
classification_head = nn.Linear(image_encoder.output_dim, num_classes)

# Combine encoder and classifier
class HandCLIPModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(HandCLIPModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, images):
        features = self.encoder(images)
        logits = self.classifier(features)
        return logits

# Instantiate the model
model = HandCLIPModel(image_encoder, classification_head).to(device)

# Load best fine-tuned model weights
checkpoint_path = "handclip_finetuned_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ----------------------------------------
# Step 4: Create Test Dataset and Dataloader
# ----------------------------------------
# Define path to test folder
test_dir = './11k/train_val_test_split_dorsal_r/test'

# Create label map based on subfolder names
label_names = sorted(os.listdir(test_dir))
label_map = {label_name: idx for idx, label_name in enumerate(label_names)}

# Dataset and DataLoader
test_dataset = HandTestDataset(test_dir, preprocess, label_map)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------------------
# Step 5: Evaluation on Test Set
# ----------------------------------------
top1_correct = 0
top5_correct = 0
total_samples = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, top1_preds = outputs.topk(1, dim=1)
        _, top5_preds = outputs.topk(5, dim=1)

        # Top-1 accuracy
        top1_correct += (top1_preds.squeeze() == labels).sum().item()

        # Top-5 accuracy
        for i in range(labels.size(0)):
            if labels[i] in top5_preds[i]:
                top5_correct += 1

        total_samples += labels.size(0)

# ----------------------------------------
# Step 6: Report Results
# ----------------------------------------
top1_acc = top1_correct / total_samples
top5_acc = top5_correct / total_samples

print(f"✅ Classification Evaluation Completed on Test Set")
print(f"Top-1 Accuracy: {top1_acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")
