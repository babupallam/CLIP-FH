"""
Decompose CLIP Model Architecture for Fine-Tuning and Understanding
-------------------------------------------------------------------

This script:
1. Loads CLIP (ViT-B/32).
2. Explores and prints each component in detail.
3. Demonstrates how to:
    - Freeze/unfreeze layers.
    - Replace layers (if needed).
    - Understand parameters (dimensions, layer types, etc.).
"""

import torch
import clip  # OpenAI CLIP
from collections import OrderedDict

# ---------------------------------------------
# Step 1: Setup Device and Load CLIP
# ---------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model (ViT-B/32) and preprocessing pipeline
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# ---------------------------------------------
# Step 2: Overview of CLIP Components
# ---------------------------------------------

"""
CLIP model has two main parts:
1. Visual (Image Encoder) -> clip_model.visual
2. Text (Text Encoder) -> clip_model.transformer (inside clip_model)

Other components:
- projection layers for text and image embeddings
- logit_scale (scales cosine similarity logits)
"""

# Display the CLIP model full architecture
print("\n================ FULL CLIP MODEL ================\n")
print(clip_model)

# ---------------------------------------------
# Step 3: Visual (Image Encoder) - Decomposition
# ---------------------------------------------
visual_encoder = clip_model.visual

print("\n================ IMAGE ENCODER =================\n")
print(visual_encoder)

"""
IMAGE ENCODER (ViT):
- conv1 (stem conv): Projects 3-channel RGB images to patch embeddings.
- class_embedding: Learnable [CLS] token.
- positional_embedding: Adds position information to patches.
- transformer: Standard Vision Transformer blocks.
- ln_post: Final LayerNorm.
- proj: Final linear projection to map features into CLIP space.

You can modify:
- Transformer blocks (unfreeze, replace)
- proj layer (change dimension or task-specific head)
"""

# Example: print transformer blocks
print("\n======= IMAGE TRANSFORMER BLOCKS =======\n")
for idx, block in enumerate(visual_encoder.transformer.resblocks):
    print(f"Block {idx}: {block}")

# Example: print the proj layer (projects CLS token to embedding)
print("\n======= IMAGE FINAL PROJECTION LAYER (proj) =======\n")
print(visual_encoder.proj)

# ---------------------------------------------
# Step 4: Text (Text Encoder) - Decomposition
# ---------------------------------------------
print("\n================ TEXT ENCODER ================\n")
text_encoder = clip_model.transformer

"""
TEXT ENCODER (Transformer):
- Uses a standard Transformer encoder (same as GPT)
- Builds representations for text inputs (tokenized)

Key components:
- token_embedding: Token embedding matrix (vocab_size x embed_dim)
- positional_embedding: Adds position info.
- resblocks: Transformer blocks (12 layers in ViT-B/32)
- ln_final: Final LayerNorm.
"""

# Example: print the resblocks of the text encoder
print("\n======= TEXT TRANSFORMER BLOCKS =======\n")
for idx, block in enumerate(text_encoder.resblocks):
    print(f"Block {idx}: {block}")


# ---------------------------------------------------------
# SECTION: Text Token Embedding & Positional Embedding
# ---------------------------------------------------------

print("\n================ TEXT TOKEN & POSITIONAL EMBEDDING ================\n")

# Text Token Embedding Layer
print("Text Token Embedding Layer (clip_model.token_embedding):")
print(f"Embedding Weight Shape (vocab_size x embed_dim): {clip_model.token_embedding.weight.shape}")
print(f"Example Tensor (first 5 tokens):\n{clip_model.token_embedding.weight[:5]}")

# Positional Embedding Layer (applied to text tokens)
print("\nText Positional Embedding (clip_model.positional_embedding):")
print(f"Shape (context_length x embed_dim): {clip_model.positional_embedding.shape}")
print(f"Example Tensor (first 5 positions):\n{clip_model.positional_embedding[:5]}")

# Summary
print("\n✅ Token Embedding maps each word/token into a 512-dim vector (ViT-B/32).")
print("✅ Positional Embedding adds sequential order information into the transformer input.\n")


# ---------------------------------------------
# Step 5: Learnable Projection Layers
# ---------------------------------------------
"""
proj (image): visual.proj
text_projection: clip_model.text_projection
logit_scale: clip_model.logit_scale
"""

print("\n================ PROJECTION & SCALE ================\n")
print(f"Image Projection (visual.proj): {visual_encoder.proj}")
print(f"Text Projection (clip_model.text_projection): {clip_model.text_projection}")
print(f"Logit Scale: {clip_model.logit_scale}")


# ---------------------------------------------
# Step 8: Summary of Dimensions and Parameters
# ---------------------------------------------

def count_parameters(model):
    """
    Counts trainable and total parameters of the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

# Count parameters of the entire CLIP model
print("\n================ PARAMETER COUNT =================\n")
count_parameters(clip_model)

# Count parameters for each encoder separately
print("\nImage Encoder Parameters:")
count_parameters(visual_encoder)

print("\nText Encoder Parameters:")
count_parameters(text_encoder)

# ---------------------------------------------------------
# SECTION: Forward Pass Walkthrough (End-to-End Data Flow)
# ---------------------------------------------------------

print("\n================ CLIP FORWARD PASS WALKTHROUGH ================\n")

# Example dummy input
dummy_text = ["a photo of a hand"]
dummy_image = torch.randn(1, 3, 224, 224).to(device)

# Preprocess and tokenize text
text_tokens = clip.tokenize(dummy_text).to(device)

# IMAGE PIPELINE
print("---- IMAGE PIPELINE ----")

# Step 1: Input image goes through conv1 to get patch embeddings
with torch.no_grad():
    img_patches = visual_encoder.conv1(dummy_image)
    print(f"Patch Embeddings (conv1) Shape: {img_patches.shape}")

# Step 2: Flatten patches + add class embedding + positional embedding + transformer
# (Handled internally in clip_model.encode_image)
with torch.no_grad():
    image_features = clip_model.encode_image(dummy_image)
    print(f"Encoded Image Features (before projection): {image_features.shape}")

# Step 3: Normalize image features (unit length)
normalized_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
print(f"Normalized Image Features: {normalized_image_features.shape}")

# TEXT PIPELINE
print("\n---- TEXT PIPELINE ----")

# Step 1: Input text tokens go through token embedding and positional embedding
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    print(f"Encoded Text Features (before projection): {text_features.shape}")

# Step 2: Normalize text features (unit length)
normalized_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
print(f"Normalized Text Features: {normalized_text_features.shape}")

print("\n✅ Both pipelines produce normalized embeddings in the same space.\n")




# ---------------------------------------------
# Step 6: FREEZE / UNFREEZE Example
# ---------------------------------------------
def freeze_or_unfreeze_layers(model, freeze=True):
    """
    Freezes or unfreezes all parameters in a given model component.
    """
    for name, param in model.named_parameters():
        param.requires_grad = not freeze
        print(f"{'Freezing' if freeze else 'Unfreezing'}: {name}")

# Example: Freeze entire image encoder
freeze_or_unfreeze_layers(visual_encoder, freeze=True)

# Example: Unfreeze last 2 transformer blocks in visual encoder
print("\n======= UNFREEZE LAST 2 IMAGE TRANSFORMER BLOCKS =======\n")
for name, param in visual_encoder.named_parameters():
    if any(name.startswith(f"transformer.resblocks.{i}") for i in [10, 11]):
        param.requires_grad = True
        print(f"Unfreezing: {name}")

# ---------------------------------------------
# Step 7: Replace Layers Example (Optional)
# ---------------------------------------------
"""
You may want to replace the classification/projection head for:
- Different output dimensions
- Task-specific adaptation
"""

import torch.nn as nn

# Get the current embedding size of CLIP image features
original_proj_dim = visual_encoder.proj.shape[0]  # This is typically 512 for ViT-B/32

# Create a new projection layer as a trainable parameter
new_proj = torch.nn.Parameter(torch.randn(256, original_proj_dim) * 0.02).to(device)

# Assign the new projection correctly
visual_encoder.proj = new_proj
print("\n======= REPLACED IMAGE PROJECTION LAYER =======\n")
print(visual_encoder.proj)


# ---------------------------------------------------------
# SECTION: Logit Scaling Operation (Similarity Computation)
# ---------------------------------------------------------

print("\n================ LOGIT SCALING OPERATION ================\n")

# Compute cosine similarity (dot product between normalized vectors)
with torch.no_grad():
    # Scale the similarity by logit_scale (exponential of logit_scale parameter)
    logit_scale = clip_model.logit_scale.exp()
    print(f"logit_scale (exp): {logit_scale.item():.4f}")

    # Similarity computation (batch_size_text x batch_size_image)
    logits_per_image = logit_scale * normalized_image_features @ normalized_text_features.t()
    logits_per_text = logits_per_image.t()

    print(f"\nLogits Per Image Shape: {logits_per_image.shape}")
    print(f"\nLogits Per Text Shape: {logits_per_text.shape}")

print("\n✅ Similarity logits computed using logit_scale and normalized embeddings.\n")


# ---------------------------------------------------------
# SECTION: Text and Image Output Embedding Dimension Alignment
# ---------------------------------------------------------

print("\n================ TEXT AND IMAGE OUTPUT EMBEDDING ALIGNMENT ================\n")

# ViT-B/32 image encoder outputs 512-dim embeddings after projection
image_output_dim = visual_encoder.proj.shape[0]
print(f"Image Output Embedding Dimension (proj): {image_output_dim}")

# Text encoder outputs 512-dim embeddings after text_projection
text_output_dim = clip_model.text_projection.shape[0]
print(f"Text Output Embedding Dimension (text_projection): {text_output_dim}")

# Confirm alignment (both should be 512 for ViT-B/32)
if image_output_dim == text_output_dim:
    print("\n✅ Image and Text embeddings are dimensionally aligned!")
else:
    print("\n⚠️ Embeddings have different dimensions. Alignment needed!")

# Confirm normalization ensures unit-length vectors for similarity computation
with torch.no_grad():
    norm_img = normalized_image_features.norm(dim=-1)
    norm_txt = normalized_text_features.norm(dim=-1)

print(f"\nImage Feature Norms (should be 1): {norm_img}")
print(f"Text Feature Norms (should be 1): {norm_txt}")

print("\n✅ Embeddings normalized to unit length before similarity comparison.\n")


# ---------------------------------------------------------
# SECTION: Differences Between Vision and Text Transformers
# ---------------------------------------------------------

print("\n================ DIFFERENCES BETWEEN VISION AND TEXT TRANSFORMERS ================\n")

# Vision Transformer Parameters
vision_embed_dim = visual_encoder.conv1.out_channels  # 768 for ViT-B/32
vision_num_layers = len(visual_encoder.transformer.resblocks)
vision_patch_size = visual_encoder.conv1.kernel_size  # (32, 32)

print(f"Vision Transformer (ViT-B/32):")
print(f"- Patch Embedding Dim: {vision_embed_dim}")
print(f"- Patch Size: {vision_patch_size}")
print(f"- Number of Transformer Blocks: {vision_num_layers}")
print(f"- Hidden Layer Width: {vision_embed_dim}")

# Text Transformer Parameters
text_embed_dim = clip_model.token_embedding.embedding_dim  # 512 for ViT-B/32
text_num_layers = len(text_encoder.resblocks)
context_length = clip_model.positional_embedding.shape[0]

print(f"\nText Transformer:")
print(f"- Token Embedding Dim: {text_embed_dim}")
print(f"- Context Length (max tokens): {context_length}")
print(f"- Number of Transformer Blocks: {text_num_layers}")
print(f"- Hidden Layer Width: {text_embed_dim}")

print("\n✅ Vision and Text Transformers are specialized:")
print("- Vision Transformer: Wider, operates on spatial patches (images).")
print("- Text Transformer: Narrower, processes sequences of tokens (text).\n")


# ---------------------------------------------------------
# SECTION: Visualization of Embeddings (Text and Image)
# ---------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

print("\n================ VISUALIZATION OF EMBEDDINGS ================\n")

# Use previously encoded embeddings
# Example dummy data already created in earlier sections
# normalized_image_features: (1, 512)
# normalized_text_features: (n_texts, 512)

# Reduce embeddings to 2D using PCA for visualization
from sklearn.decomposition import PCA

# Stack image and text features
combined_features = torch.cat([normalized_image_features.cpu(), normalized_text_features.cpu()], dim=0)

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(combined_features)

# Plot the points
plt.figure(figsize=(8, 6))
plt.scatter(reduced_features[0, 0], reduced_features[0, 1], marker='*', s=200, label='Image', color='red')

for idx in range(1, reduced_features.shape[0]):
    plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1], marker='o', label=f'Text {idx}')

plt.title('PCA Projection of Image and Text Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()

print("\n✅ Embeddings visualized in 2D PCA space.\n")


# ---------------------------------------------------------
# SECTION: Comparison with ResNet-based CLIP Encoder
# ---------------------------------------------------------

print("\n================ COMPARISON: ViT vs ResNet CLIP ENCODERS ================\n")

# Load ResNet-50-based CLIP model
clip_model_resnet, preprocess_resnet = clip.load("RN50", device=device)

# Visual Encoder Comparison
print("ViT-B/32 Vision Encoder:")
print(f"- Type: Vision Transformer")
print(f"- Patch size: 32x32")
print(f"- Embedding dimension: {visual_encoder.conv1.out_channels}")
print(f"- Number of transformer layers: {len(visual_encoder.transformer.resblocks)}\n")

print("ResNet-50 Vision Encoder:")
resnet_encoder = clip_model_resnet.visual
print(f"- Type: ResNet-50 backbone")
print(f"- Layers: conv1, layer1-4, attention_pool")
print(f"- Embedding dimension (final): {resnet_encoder.attnpool.c_proj.out_features}")

# ResNet block structure
print(f"- ResNet Blocks: {resnet_encoder.layer1}, {resnet_encoder.layer2}, {resnet_encoder.layer3}, {resnet_encoder.layer4}")

# Architecture difference summary
print("\n✅ ViT encodes the image by splitting it into fixed-size patches processed by transformers.")
print("✅ ResNet processes the entire image hierarchically using convolutional blocks and global attention pooling.\n")

# ---------------------------------------------------------
# SECTION: Explanation and Use of QuickGELU Activation
# ---------------------------------------------------------

print("\n================ EXPLANATION: QuickGELU vs GELU ================\n")

"""
QuickGELU is an optimized approximation of the GELU activation function.
- GELU (Gaussian Error Linear Unit) is used in transformers for smoother non-linearity.
- Standard GELU is computationally expensive.
- QuickGELU simplifies the computation:
    QuickGELU(x) = x * sigmoid(1.702 * x)
This approximation is faster and performs similarly in practice.

CLIP uses QuickGELU to speed up training and inference, especially in large-scale models.
"""

import torch.nn.functional as F

def standard_gelu(x):
    return F.gelu(x)

def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

# Example input tensor
x = torch.linspace(-3, 3, steps=100)

# Compute both activations
gelu_output = standard_gelu(x)
quick_gelu_output = quick_gelu(x)

# Plot comparison
plt.figure(figsize=(8, 6))
plt.plot(x.numpy(), gelu_output.numpy(), label='Standard GELU')
plt.plot(x.numpy(), quick_gelu_output.numpy(), label='QuickGELU', linestyle='--')
plt.title('Comparison: GELU vs QuickGELU')
plt.xlabel('Input')
plt.ylabel('Activation')
plt.legend()
plt.grid(True)
plt.show()

print("\n✅ QuickGELU approximates GELU for faster computation while maintaining performance.\n")

# ---------------------------------------------
# End
# ---------------------------------------------
print("\n✅ CLIP Decomposition Complete. You can now fine-tune specific parts!")
