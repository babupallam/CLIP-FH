import torch
import torch.nn as nn
import clip


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=8, ctx_init=None,
                 prompt_template="A photo of a {}.", device="cuda"):
        super().__init__()

        # === Store metadata ===
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.n_ctx = n_ctx
        self.device = device
        self.ctx_init = ctx_init
        self.prompt_template = prompt_template

        # === Tokenizer and model components ===
        dtype = clip_model.token_embedding.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.context_length = clip_model.context_length  # usually 77
        self.tokenizer = clip.tokenize

        # === Prepare class-specific text prompts ===
        self.prompts = [prompt_template.format(c.replace("_", " ")) for c in classnames]
        tokenized_prompts = self.tokenizer(self.prompts).to(device)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        # === Initialize learnable context embeddings ===
        if ctx_init:
            init_token = clip.tokenize(ctx_init).to(device)
            init_embedding = self.token_embedding(init_token).detach()[0, 1:1 + n_ctx]
            assert init_embedding.shape[0] == n_ctx, "Init context length doesn't match n_ctx"
            ctx_vectors = init_embedding
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).uniform_(-0.02, 0.02)

        self.ctx = nn.Parameter(ctx_vectors)  # shape: (n_ctx, dim)

        # === Identify where the [CTX] will be inserted ===
        self.prefix_len = (self.tokenized_prompts == self.tokenizer("a")[0]).nonzero(as_tuple=True)[1][0].item()

    def forward(self):
        """
        Generate embedded prompts for all classes.

        Returns:
            Tensor of shape (num_classes, context_length, embed_dim)
        """
        # Expand learnable context for all classes
        ctx = self.ctx.unsqueeze(0).expand(self.num_classes, -1, -1)  # (num_classes, n_ctx, dim)

        # Get token embeddings for full prompts
        token_embeds = self.token_embedding(self.tokenized_prompts)  # (num_classes, context_len, dim)

        # Replace context tokens
        prefix = token_embeds[:, :1, :]                        # [SOS]
        suffix = token_embeds[:, 1 + self.n_ctx:, :]           # Suffix after context

        prompts_embedded = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts_embedded  # (num_classes, context_len, dim)

    def forward_batch(self, labels):
        """
        Fetch prompt embeddings for a specific batch of labels.

        Args:
            labels (Tensor): shape (B,) with integer class indices

        Returns:
            Tensor of shape (B, context_len, embed_dim)
        """
        all_prompts = self.forward()  # shape: (num_classes, context_len, dim)

        assert labels.max().item() < self.num_classes, \
            f"❌ Label {labels.max().item()} is out of bounds (prompt table size = {self.num_classes})"

        prompt_batch = all_prompts[labels, :, :]  # shape: (B, context_len, dim)
        return prompt_batch
