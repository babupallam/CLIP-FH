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
    It generates learnable prompts that replace real text for each identity class.
    """

    def __init__(self, classnames, cfg, clip_model,
                 n_ctx: int = 4, ctx_init=None,
                 template="A photo of a X X X X hand.",
                 aspect=None, device="cuda"):
        super().__init__()

        self.device = device
        num_classes = len(classnames)
        dtype = clip_model.token_embedding.weight.dtype  # e.g. float16
        ctx_dim = clip_model.token_embedding.embedding_dim  # Token embedding dimension
        token_embedding = clip_model.token_embedding

        # List of prompt templates (e.g., "A photo of a hand", "This is a {} hand", etc.)
        self.prompt_template_list = cfg.get("prompt_template_list", [cfg["prompt_template"]])
        self.current_template = self.prompt_template_list[0]  # Use the first one as default

        # === Step 1: Create frozen prompt prefix and suffix using the base template ===
        tokenized = clip.tokenize(template).to(device)
        with torch.no_grad():
            embed = token_embedding(tokenized).type(dtype)

        # Store prefix: [SOS] + n_ctx token slots (non-learnable)
        self.register_buffer("token_prefix", embed[:, :n_ctx + 1, :].clone().to(device))

        # Store suffix: everything after the context slots (e.g., ".", [EOS])
        self.register_buffer("token_suffix", embed[:, n_ctx + 1 + n_ctx:, :].clone().to(device))

        # === Step 2: Create learnable context vectors for each class ===
        # Shape: [num_classes, n_ctx, ctx_dim]
        cls_ctx = torch.empty(num_classes, n_ctx, ctx_dim, dtype=dtype).uniform_(-0.02, 0.02)
        self.cls_ctx = nn.Parameter(cls_ctx.to(device))

        # === Step 3: Handle dimensional mismatch between text and image encoders ===
        text_out_dim = clip_model.visual.output_dim  # 512 (ViT) or 1024 (RN50)
        print(f"[PromptLearner] ctx_dim: {ctx_dim}, output_dim: {text_out_dim}")

        # Add a projection layer if needed to match text output to image feature dim
        self.proj = nn.Linear(ctx_dim, text_out_dim) if ctx_dim != text_out_dim else nn.Identity()
        self.proj = self.proj.to(self.device)
        print(f"[PromptLearner] proj layer = {self.proj.__class__.__name__} | in: {ctx_dim}, out: {text_out_dim}")

    def set_template(self, template):
        """
        Dynamically change the prompt template and rebuild prefix/suffix embeddings.
        Useful during template ensemble or evaluation.
        """
        tokenized = clip.tokenize(template).to(self.device)
        with torch.no_grad():
            embed = self.clip_model.token_embedding(tokenized).type(self.cls_ctx.dtype)
        self.token_prefix = embed[:, :self.n_ctx + 1, :].clone()
        self.token_suffix = embed[:, self.n_ctx + 1 + self.n_ctx:, :].clone()

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of class labels, generate their corresponding prompt embeddings.
        This is the main method used during training and inference.
        """
        labels = labels.to(self.cls_ctx.device)
        B = labels.size(0)

        # Duplicate prefix and suffix for each item in the batch
        prefix = self.token_prefix.expand(B, -1, -1)
        suffix = self.token_suffix.expand(B, -1, -1)

        # Select the class-specific learnable context vectors
        ctx_part = self.cls_ctx[labels]  # Shape: [B, n_ctx, ctx_dim]

        # Combine prefix + context + suffix to form the final prompt
        return torch.cat([prefix, ctx_part, suffix], dim=1)  # Shape: [B, L, D]

    def forward_batch(self, labels):
        """
        If using multiple templates, generate and average prompt embeddings over them.
        Otherwise, just fall back to the single template.
        """
        all_embeds = []
        for template in self.prompt_template_list:
            tokenized = clip.tokenize(template).to(self.device)
            with torch.no_grad():
                embed = self.clip_model.token_embedding(tokenized).type(self.cls_ctx.dtype)

            # Prefix/suffix broadcasted to match batch size
            prefix = embed[:, :self.n_ctx + 1, :].clone().expand(labels.size(0), -1, -1)
            suffix = embed[:, self.n_ctx + 1 + self.n_ctx:, :].clone().expand(labels.size(0), -1, -1)

            # Class-specific context tokens
            ctx_part = self.cls_ctx[labels]

            # Full prompt = prefix + context + suffix
            prompt = torch.cat([prefix, ctx_part, suffix], dim=1)
            all_embeds.append(prompt)

        # Return the average across all templates
        return torch.stack(all_embeds).mean(dim=0)


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
