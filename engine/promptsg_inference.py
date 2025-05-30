import torch
from torch.nn.functional import normalize, interpolate
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def maybe_patch_pos_embed(clip_model, input_hw):
    """
    Adapts CLIP positional embeddings if input size changes (e.g. 224x128).
    Works for both ViT and RN50 backbones.
    """
    h, w = input_hw  # Input height and width

    if hasattr(clip_model.visual, "positional_embedding"):
        # === For ViT-based CLIP models ===
        pos_embed = clip_model.visual.positional_embedding  # Shape: [N, D]
        patch_size = (16, 16)  # Fixed patch size used by ViT

        # Compute number of patches along height and width
        grid_h = h // patch_size[0]
        grid_w = w // patch_size[1]
        new_len = grid_h * grid_w + 1  # +1 for CLS token

        # Only update if the current embedding length doesn't match
        if pos_embed.shape[0] != new_len:
            print(f"[ViT] Interpolating pos_embed: {pos_embed.shape[0]}  {new_len}")
            cls_token = pos_embed[:1, :]         # First token is the CLS token
            spatial_tokens = pos_embed[1:, :]    # Remaining tokens are for image patches

            # Try to reshape spatial tokens into a square grid
            old_grid_len = spatial_tokens.shape[0]
            old_grid_h = old_grid_w = int(old_grid_len ** 0.5)
            assert old_grid_h * old_grid_w == old_grid_len, f"Cannot reshape {old_grid_len} tokens into square grid"

            # Reshape, interpolate, and flatten
            spatial = spatial_tokens.reshape(old_grid_h, old_grid_w, -1).permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
            resized = F.interpolate(spatial, size=(grid_h, grid_w), mode='bicubic', align_corners=False)
            new_spatial = resized.squeeze(0).permute(1, 2, 0).reshape(grid_h * grid_w, -1)  # [N, D]

            # Concatenate CLS token back
            new_pos_embed = torch.cat([cls_token, new_spatial], dim=0)  # [N+1, D]
            clip_model.visual.positional_embedding = nn.Parameter(new_pos_embed)

    elif hasattr(clip_model.visual, "attnpool") and hasattr(clip_model.visual.attnpool, "positional_embedding"):
        # === For RN50 and ResNet-based CLIP backbones ===
        pos_embed = clip_model.visual.attnpool.positional_embedding  # Shape: [N, D]

        # For CNNs, the feature map is smaller (assume 32-stride)
        feat_h, feat_w = h // 32, w // 32
        new_len = feat_h * feat_w + 1  # +1 for CLS token

        if pos_embed.shape[0] != new_len:
            print(f"[RN50] Interpolating attnpool pos_embed: {pos_embed.shape[0]}  {new_len}")
            cls_token = pos_embed[:1]  # [1, D]
            spatial = pos_embed[1:].T.unsqueeze(0)  # Convert to [1, D, N] for interpolation

            # Linearly interpolate to new number of spatial tokens
            resized = interpolate(spatial, size=new_len - 1, mode='linear', align_corners=False)
            new_spatial = resized.squeeze(0).T  # [N', D]

            # Combine CLS and resized spatial tokens
            new_pos_embed = torch.cat([cls_token, new_spatial], dim=0)
            clip_model.visual.attnpool.positional_embedding = nn.Parameter(new_pos_embed)




@torch.no_grad()
def extract_features_promptsg(
    clip_model, inversion_model, multimodal_module, classifier,
    loader, device, compose_prompt
):
    feats, labels = [], []  # Lists to collect features and labels from all batches

    # === Patch positional embeddings once based on first image ===
    sample_img = loader.dataset[0][0]  # (C, H, W) - first image from dataset
    img_hw = (sample_img.shape[1], sample_img.shape[2])  # Extract image height and width
    maybe_patch_pos_embed(clip_model, input_hw=img_hw)  # Adjust positional embeddings if needed

    # === Main feature extraction loop ===
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        # Step 1: Get image features from CLIP image encoder
        img_features = clip_model.encode_image(imgs).float()  # [B, D]

        # Step 2: Generate pseudo-text tokens using inversion model
        pseudo_tokens = inversion_model(img_features)         # [B, D'] or [B, context_len, D]

        # Step 3: Convert pseudo tokens into text embeddings using prompt composer
        text_emb = compose_prompt(clip_model.encode_text, pseudo_tokens, device=device)  # [B, L, D]

        # Step 4: Fuse text and image features using the multi-modal module
        fused = multimodal_module(text_emb, img_features.unsqueeze(1))  # [B, L, D]

        # Step 5: Average across sequence length (L) and normalize
        pooled = normalize(fused.mean(dim=1), dim=1)  # [B, D]

        # Collect features and labels
        feats.append(pooled.cpu())
        labels.append(lbls.cpu())

    # Combine all batches into a single tensor
    return torch.cat(feats), torch.cat(labels)
