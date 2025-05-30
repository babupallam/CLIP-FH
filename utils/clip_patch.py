# models/clip_patch.py

import clip

# Load a CLIP model by name, optionally freezing all its parameters
def load_clip_with_patch(model_type, device, freeze_all=True):
    # Map short names to CLIP model names
    model_map = {
        "vitb16": "ViT-B/16",
        "vitb32": "ViT-B/32",
        "rn50": "RN50",
        "rn101": "RN101",
        "rn50x4": "RN50x4",
        "rn50x16": "RN50x16",
        "rn50x64": "RN50x64"
    }

    # Get the full model name from the map
    model_name = model_map.get(model_type.lower())
    if model_name is None:
        raise ValueError(f" Unknown model type: {model_type}")

    # Load the model from CLIP
    model, _ = clip.load(model_name, device=device)

    # If freeze_all is True, stop gradient updates for all parameters
    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False

    return model, _


import torch.nn as nn

# This module defines a multi-modal interaction block using cross-attention
class MultiModalInteraction(nn.Module):
    def __init__(self, dim=512, depth=2, num_heads=8):
        super().__init__()

        # Cross-attention between text and visual patches
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=False)

        # Stack of transformer encoder blocks applied after attention
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
            for _ in range(depth)
        ])

    def forward(self, text_emb, visual_patches):
        # Expected input: [B, L, D]
        # Convert to: [L, B, D] for attention modules
        text_emb = text_emb.transpose(0, 1)              # [prompt_len, B, D]
        visual_patches = visual_patches.transpose(0, 1)  # [vis_len, B, D]

        # Apply cross-attention: text queries, visual keys/values
        attn_output, _ = self.cross_attention(text_emb, visual_patches, visual_patches)  # [prompt_len, B, D]

        # Pass through transformer encoder layers
        for block in self.transformer_blocks:
            attn_output = block(attn_output)  # still [prompt_len, B, D]

        # Transpose back to [B, prompt_len, D] format
        return attn_output.transpose(0, 1)
