# models/clip_patch.py

import clip


def load_clip_with_patch(model_type, device, freeze_all=True):
    model_map = {
        "vitb16": "ViT-B/16",
        "vitb32": "ViT-B/32",
        "rn50": "RN50",
        "rn101": "RN101",
        "rn50x4": "RN50x4",
        "rn50x16": "RN50x16",
        "rn50x64": "RN50x64"
    }

    model_name = model_map.get(model_type.lower())
    if model_name is None:
        raise ValueError(f"❌ Unknown model type: {model_type}")

    model, _ = clip.load(model_name, device=device)

    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False

    return model, _


import torch.nn as nn

class MultiModalInteraction(nn.Module):
    def __init__(self, dim=512, depth=2, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=False)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
            for _ in range(depth)
        ])

    def forward(self, text_emb, visual_patches):
        # Expected input: [B, L, D]
        # Convert to: [L, B, D]
        text_emb = text_emb.transpose(0, 1)          # [prompt_len, B, D]
        visual_patches = visual_patches.transpose(0, 1)  # [vis_len, B, D]

        # Cross-attention
        attn_output, _ = self.cross_attention(text_emb, visual_patches, visual_patches)  # [prompt_len, B, D]

        # Apply transformer layers (optional)
        for block in self.transformer_blocks:
            attn_output = block(attn_output)  # still [prompt_len, B, D]

        # Transpose back to [B, prompt_len, D]
        return attn_output.transpose(0, 1)
