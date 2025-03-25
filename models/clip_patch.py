import torch.nn.functional as F

def encode_text_from_embedding(self, prompt_embeddings):
    """
    Runs text transformer on manually provided prompt embeddings.
    Args:
        prompt_embeddings: (B, context_length, embed_dim)
    Returns:
        text_features: (B, embed_dim)
    """
    x = prompt_embeddings + self.positional_embedding  # Add positional embeddings
    x = x.permute(1, 0, 2)  # (B, L, D) → (L, B, D)
    x = self.transformer(x)  # Run through transformer
    x = x.permute(1, 0, 2)  # (L, B, D) → (B, L, D)

    x = self.ln_final(x).type(self.dtype)  # LayerNorm + dtype cast
    cls_token = x[:, 0, :]  # CLS token

    # Project to final feature space
    return F.normalize(cls_token @ self.text_projection, dim=-1)  # (B, embed_dim)


