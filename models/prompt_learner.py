# prompt_learner.py (Fixed Version)

import torch
import torch.nn as nn
import clip


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=8, ctx_init=None,
                 prompt_template="A photo of {}'s {aspect} hand for identification.", aspect=None,device="cuda"):
        super().__init__()

        self.classnames = classnames
        self.num_classes = len(classnames)
        self.n_ctx = n_ctx
        self.device = device
        self.ctx_init = ctx_init
        self.prompt_template = prompt_template

        dtype = clip_model.token_embedding.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.context_length = clip_model.context_length
        self.tokenizer = clip.tokenize

        aspect_map = {
            "dorsal_r": "right dorsal",
            "dorsal_l": "left dorsal",
            "palmar_r": "right palmar",
            "palmar_l": "left palmar"
        }
        aspect_text = aspect_map[aspect]
        self.prompts = [prompt_template.format(cls_name, aspect=aspect_text) for cls_name in classnames]
        print("🔤 Generated prompts per class:")
        for i, p in enumerate(self.prompts[:5]):
            print(f"[{i}] {p}")
        if len(self.prompts) > 5:
            print(f"... (total {len(self.prompts)})")

        tokenized_prompts = self.tokenizer(self.prompts).to(device)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        # === Initialize learnable context embeddings ===
        if ctx_init:
            init_token = clip.tokenize(ctx_init).to(device)
            init_embedding = self.token_embedding(init_token)[0, 1:1 + n_ctx]
            assert init_embedding.shape[0] == n_ctx, "Init context length doesn't match n_ctx"
            ctx_vectors = init_embedding
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).uniform_(-0.02, 0.02)

        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx, dim)

    def forward_batch(self, labels):
        """
        Dynamically builds prompts for each label in the batch to maintain gradient flow.
        """
        B = labels.shape[0]
        ctx = self.ctx.unsqueeze(0).expand(B, -1, -1)  # (B, n_ctx, dim)

        # Get prompt tokens for this batch
        token_embeds = self.token_embedding(self.tokenized_prompts[labels])  # (B, context_len, dim)

        prefix = token_embeds[:, :1, :]                      # [SOS]
        suffix = token_embeds[:, 1 + self.n_ctx:, :]         # class name and period

        prompts_embedded = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts_embedded  # (B, context_len, dim)
