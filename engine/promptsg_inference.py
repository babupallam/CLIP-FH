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
    h, w = input_hw
    if hasattr(clip_model.visual, "positional_embedding"):
        pos_embed = clip_model.visual.positional_embedding  # [N, D]
        patch_size = (16, 16)

        grid_h = h // patch_size[0]
        grid_w = w // patch_size[1]
        new_len = grid_h * grid_w + 1

        if pos_embed.shape[0] != new_len:
            print(f"[ViT] Interpolating pos_embed: {pos_embed.shape[0]} → {new_len}")
            cls_token = pos_embed[:1, :]  # [1, D]
            spatial_tokens = pos_embed[1:, :]  # [N-1, D]

            # Dynamically infer grid size
            old_grid_len = spatial_tokens.shape[0]
            old_grid_h = old_grid_w = int(old_grid_len ** 0.5)
            assert old_grid_h * old_grid_w == old_grid_len, f"Cannot reshape {old_grid_len} tokens into square grid"

            spatial = spatial_tokens.reshape(old_grid_h, old_grid_w, -1).permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
            resized = F.interpolate(spatial, size=(grid_h, grid_w), mode='bicubic', align_corners=False)
            new_spatial = resized.squeeze(0).permute(1, 2, 0).reshape(grid_h * grid_w, -1)  # [N, D]

            new_pos_embed = torch.cat([cls_token, new_spatial], dim=0)  # [N+1, D]
            clip_model.visual.positional_embedding = nn.Parameter(new_pos_embed)


    elif hasattr(clip_model.visual, "attnpool") and hasattr(clip_model.visual.attnpool, "positional_embedding"):
        # === For RN50 and ResNet-based CLIP backbones ===
        pos_embed = clip_model.visual.attnpool.positional_embedding  # [N, D]
        feat_h, feat_w = h // 32, w // 32  # Assume 32-stride CNN
        new_len = feat_h * feat_w + 1

        if pos_embed.shape[0] != new_len:
            print(f"[RN50] Interpolating attnpool pos_embed: {pos_embed.shape[0]} → {new_len}")
            cls_token = pos_embed[:1]        # [1, D]
            spatial = pos_embed[1:].T.unsqueeze(0)  # [1, D, N]
            resized = interpolate(spatial, size=new_len - 1, mode='linear', align_corners=False)
            new_spatial = resized.squeeze(0).T       # [N', D]
            new_pos_embed = torch.cat([cls_token, new_spatial], dim=0)
            clip_model.visual.attnpool.positional_embedding = nn.Parameter(new_pos_embed)

@torch.no_grad()
def extract_features_promptsg(
    clip_model, inversion_model, multimodal_module, classifier,
    loader, device, compose_prompt
):
    feats, labels = [], []

    # === Patch positional embeddings once based on first image ===
    sample_img = loader.dataset[0][0]  # (C, H, W)
    img_hw = (sample_img.shape[1], sample_img.shape[2])
    maybe_patch_pos_embed(clip_model, input_hw=img_hw)

    # === Main feature extraction loop ===
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        img_features = clip_model.encode_image(imgs).float()  # [B, D]
        pseudo_tokens = inversion_model(img_features)         # [B, D']
        text_emb = compose_prompt(clip_model.encode_text, pseudo_tokens, device=device)
        fused = multimodal_module(text_emb, img_features.unsqueeze(1))  # [B, L, D]
        pooled = normalize(fused.mean(dim=1), dim=1)           # [B, D]

        feats.append(pooled.cpu())
        labels.append(lbls.cpu())

    return torch.cat(feats), torch.cat(labels)
