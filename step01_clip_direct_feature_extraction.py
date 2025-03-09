import torch
import clip
from PIL import Image
import os
import numpy as np

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Path to the dataset
image_folder = "./11k/Hands"

# Get list of images
image_filenames = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# Load and preprocess images
images = [preprocess(Image.open(os.path.join(image_folder, img))) for img in image_filenames]
image_tensors = torch.stack(images).to(device)

# Extract image features using CLIP
with torch.no_grad():
    image_features = model.encode_image(image_tensors)

# Normalize features
image_features /= image_features.norm(dim=-1, keepdim=True)

print("Extracted image features shape:", image_features.shape)

import pickle

# Convert to NumPy array and save
image_features_np = image_features.cpu().numpy()

# Save embeddings
np.save("image_embeddings.npy", image_features_np)

# Save image filenames for reference
with open("image_filenames.pkl", "wb") as f:
    pickle.dump(image_filenames, f)

print("Saved image embeddings and filenames.")
