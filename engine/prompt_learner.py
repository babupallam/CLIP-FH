# prompt_learner.py (Fixed Version)

import torch
import torch.nn as nn
import clip


import torch
import torch.nn as nn
import clip
import random


class PromptLearner(nn.Module):
    """
    CLIP-ReID style prompt learner with class-specific context vectors.
    Fully GPU-compatible.
    """

    def __init__(self, classnames, cfg, clip_model,
                 n_ctx: int = 4, ctx_init=None,
                 template="A photo of a X X X X hand.",
                 aspect=None, device="cuda"):
        super().__init__()

        self.device = device
        num_classes = len(classnames)
        dtype = clip_model.token_embedding.weight.dtype
        ctx_dim = clip_model.token_embedding.embedding_dim
        token_embedding = clip_model.token_embedding

        # for v2
        self.prompt_template_list = cfg.get("prompt_template_list", [cfg["prompt_template"]])
        self.current_template = self.prompt_template_list[0]  # default

        # --- 1. build frozen prefix and suffix embeddings ------------------
        tokenized = clip.tokenize(template).to(device)
        with torch.no_grad():
            embed = token_embedding(tokenized).type(dtype)

        # Save prefix: [SOS] + n_ctx slots
        self.register_buffer("token_prefix", embed[:, :n_ctx + 1, :].clone().to(device))

        # Save suffix: remaining tokens after context slots (CLS + "." + [EOS])
        self.register_buffer("token_suffix", embed[:, n_ctx + 1 + n_ctx:, :].clone().to(device))

        # --- 2. class-specific learnable context vectors -------------------
        # Create learnable context vectors for each class (C, n_ctx, dim)
        cls_ctx = torch.empty(num_classes, n_ctx, ctx_dim, dtype=dtype).uniform_(-0.02, 0.02)
        self.cls_ctx = nn.Parameter(cls_ctx.to(device))  # (C, n_ctx, dim)

        # Handle feature dimension differences between text and visual encoder
        text_out_dim = clip_model.visual.output_dim  # 1024 for RN50, 512 for ViT
        print(f"[PromptLearner] ctx_dim: {ctx_dim}, output_dim: {text_out_dim}")

        # Project context embeddings to match output dim if needed
        self.proj = nn.Linear(ctx_dim, text_out_dim) if ctx_dim != text_out_dim else nn.Identity()
        self.proj = self.proj.to(self.device)
        print(f"[PromptLearner] proj layer = {self.proj.__class__.__name__} | in: {ctx_dim}, out: {text_out_dim}")

    def set_template(self, template):
        # Dynamically change the prompt template
        tokenized = clip.tokenize(template).to(self.device)
        with torch.no_grad():
            embed = self.clip_model.token_embedding(tokenized).type(self.cls_ctx.dtype)
        self.token_prefix = embed[:, :self.n_ctx + 1, :].clone()
        self.token_suffix = embed[:, self.n_ctx + 1 + self.n_ctx:, :].clone()

    def forward_batch(self, labels):
        # Compose prompts for a batch of labels across all templates
        all_embeds = []
        for template in self.prompt_template_list:
            tokenized = clip.tokenize(template).to(self.device)
            with torch.no_grad():
                embed = self.clip_model.token_embedding(tokenized).type(self.cls_ctx.dtype)

            # Expand prefix/suffix to batch size
            prefix = embed[:, :self.n_ctx + 1, :].clone().expand(labels.size(0), -1, -1)
            suffix = embed[:, self.n_ctx + 1 + self.n_ctx:, :].clone().expand(labels.size(0), -1, -1)

            # Get class-specific context vectors
            ctx_part = self.cls_ctx[labels]  # (B, n_ctx, dim)

            # Concatenate to form full prompt: [prefix | ctx | suffix]
            prompt = torch.cat([prefix, ctx_part, suffix], dim=1)  # (B, L, D)
            all_embeds.append(prompt)

        # Average embeddings if multiple templates are used
        return torch.stack(all_embeds).mean(dim=0)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Returns a single prompt per class.
        Used during training (Stage 1, Stage 2).
        """
        labels = labels.to(self.cls_ctx.device)
        B = labels.size(0)

        # Expand prefix and suffix to batch size
        prefix = self.token_prefix.expand(B, -1, -1)
        suffix = self.token_suffix.expand(B, -1, -1)

        # Get the learnable context tokens for each label
        ctx_part = self.cls_ctx[labels]  # (B, n_ctx, dim)

        # Concatenate prefix, context, and suffix
        return torch.cat([prefix, ctx_part, suffix], dim=1)

    def forward_batch(self, labels):
        return self.forward(labels)


import torch.nn as nn


class TextualInversionMLP(nn.Module):
    """
    TextualInversionMLP:
    Implements the inversion network described in PromptSG.
     Purpose:
    - Converts CLIP image features into a pseudo-token.
    - This token is inserted between a fixed prompt prefix and suffix.
    - The result is a learnable text token that adapts to visual input.

    As per the paper (Section: Implementation Details):
    - The inversion network is a lightweight, 3-layer MLP.
    - Each hidden layer has 512 dimensions.
    - A Batch Normalization (BN) layer is added after the final output.
    - All layers use ReLU activation.
    - The network is randomly initialized and trained from scratch.
    """

    def __init__(self, in_dim, out_dim):
        """
        Args:
            in_dim (int): Input dimension (matches CLIP image feature dim, e.g. 512 or 1024)
            out_dim (int): Output dimension (pseudo-token dimension, typically 512 or 768)
        """
        super().__init__()

        # 3-layer MLP with ReLU activations and BatchNorm at the end
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),  # Layer 1: Project input to 512
            nn.ReLU(),  # Non-linearity
            nn.Linear(512, 512),  # Layer 2: Maintain 512 hidden state
            nn.ReLU(),  # Non-linearity
            nn.Linear(512, out_dim),  # Layer 3: Final projection to pseudo-token dimension
            nn.BatchNorm1d(out_dim)  #  As specified: BN after final layer
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Image features from CLIP image encoder (B, in_dim)
        Returns:
            Tensor: Pseudo-token for each image (B, out_dim)
        """
        return self.mlp(x)
