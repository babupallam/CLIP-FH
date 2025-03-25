import torch
import torch.nn as nn

class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, n_ctx=4, prefix="A photo of a", suffix="person."):
        super().__init__()
        self.class_names = class_names
        self.n_cls = len(class_names)
        self.n_ctx = n_ctx

        # Fixed prefix and suffix
        self.prefix = prefix
        self.suffix = suffix

        # Token embedding dimension from CLIP
        embed_dim = clip_model.token_embedding.weight.shape[1]

        # Context (learnable) prompt embeddings: [X1]...[Xn]
        self.ctx_vectors = nn.Parameter(torch.randn(n_ctx, embed_dim) * 0.02)

        # Store CLIP's token and positional embeddings
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding

        # The full prompt will be built dynamically, but we use the same structure for all classes
        # So we simulate a fixed context length
        self.context_length = 77  # CLIP default

        # Initialize full prompt embeddings buffer
        self.register_buffer("tokenized_prompt", torch.zeros(self.context_length, dtype=torch.long))

    def forward(self):
        """
        Returns:
            prompt_embeddings: (num_classes, context_length, embed_dim)
        """
        # Get token embeddings (fixed for prefix/suffix)
        prefix_tokens = self.token_embedding(self.tokenized_prompt).detach().clone()
        token_embed_dim = prefix_tokens.shape[-1]

        # Insert learnable context into prompt embedding
        prompt_embeddings = []

        for _ in range(self.n_cls):
            # Start with all zeros
            prompt = torch.zeros(self.context_length, token_embed_dim).to(self.ctx_vectors.device)

            # Insert prefix
            prefix_len = len(self.prefix.split())
            suffix_len = len(self.suffix.split())
            ctx_start = prefix_len
            ctx_end = prefix_len + self.n_ctx

            # Use token embedding from prefix/suffix
            prefix_embed = self.token_embedding(self.tokenized_prompt[:prefix_len])
            suffix_embed = self.token_embedding(self.tokenized_prompt[-suffix_len:])

            prompt[:prefix_len] = prefix_embed
            prompt[ctx_start:ctx_end] = self.ctx_vectors  # <-- Learnable tokens
            prompt[-suffix_len:] = suffix_embed

            prompt_embeddings.append(prompt)

        return torch.stack(prompt_embeddings)  # shape: (n_cls, context_len, embed_dim)
