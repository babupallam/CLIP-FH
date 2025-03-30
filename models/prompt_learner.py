import torch
import torch.nn as nn
import clip


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=8, ctx_init=None, prompt_template="A photo of a {}.", device="cuda"):
        super().__init__()

        self.classnames = classnames
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.device = device
        self.ctx_init = ctx_init
        self.prompt_template = prompt_template

        # Token embedding dimension from CLIP
        dtype = clip_model.token_embedding.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.context_length = clip_model.context_length  # typically 77
        self.tokenizer = clip.tokenize

        # Build class-specific prompts (e.g., "A photo of a {} hand.")
        self.prompts = [prompt_template.format(c.replace("_", " ")) for c in classnames]
        tokenized_prompts = self.tokenizer(self.prompts).to(device)  # shape: (num_classes, context_len)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        # ===== Learnable context embeddings =====
        if ctx_init:
            # Use initialized tokens
            init_token = clip.tokenize(ctx_init).to(device)  # shape (1, context_len)
            init_embedding = self.token_embedding(init_token).detach()[0, 1 : 1 + n_ctx]
            assert init_embedding.shape[0] == n_ctx, "Init context length doesn't match n_ctx"
            ctx_vectors = init_embedding
        else:
            # Random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).uniform_(-0.02, 0.02)

        self.ctx = nn.Parameter(ctx_vectors)  # shape: (n_ctx, dim)

        # Meta indices: [prefix_len, ctx_len, suffix_len] to reconstruct full prompt
        self.prefix_len = (self.tokenized_prompts == self.tokenizer("a")[0]).nonzero(as_tuple=True)[1][0].item()

    def forward(self):
        """
        Returns:
            prompts_embedded: (num_classes, context_length, embed_dim)
        """
        # Shape: (n_cls, ctx_len, dim)
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # Get full token embeddings (with padding)
        token_embeds = self.token_embedding(self.tokenized_prompts)  # (n_cls, context_len, dim)

        # Replace the context tokens [1 : 1+n_ctx] with learned ctx
        prefix = token_embeds[:, :1, :]                        # [SOS]
        suffix = token_embeds[:, 1 + self.n_ctx :, :]         # [tokens after prompt]

        # Rebuild full prompt embedding
        prompts_embedded = torch.cat([prefix, ctx, suffix], dim=1)  # (n_cls, context_len, dim)
        return prompts_embedded
