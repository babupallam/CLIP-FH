Below is a simple way to **partially unfreeze** the CLIP text encoder, focusing on the **last transformer block** and/or the **final layernorm**. This helps the text backbone adapt a bit more to your domain, while still keeping most of it frozen.

---

## 1) Identify Where You Freeze Everything

You likely have code in your trainer that does something like:

```python
# Currently in your code
for param in self.clip_model.transformer.parameters():
    param.requires_grad = False
for param in self.clip_model.token_embedding.parameters():
    param.requires_grad = False
for param in self.clip_model.visual.parameters():
    param.requires_grad = False
```

This “freezes” (turns off gradients) for the entire text transformer, token embeddings, and vision encoder.

---

## 2) Unfreeze the Last Transformer Block

In OpenAI’s CLIP model, the text Transformer is typically stored in something like `clip_model.transformer.resblocks`, which is a list of residual blocks. You can unfreeze (turn on gradients) for the last block like this:

```python
# UNFREEZE the last block in the text transformer
for param in self.clip_model.transformer.resblocks[-1].parameters():
    param.requires_grad = True
```

That tells PyTorch to **optimize** all parameters in that final block. You can do this for the last *few* blocks if you want deeper partial fine‐tuning:

```python
# Example: unfreeze the last TWO transformer blocks
for param in self.clip_model.transformer.resblocks[-2].parameters():
    param.requires_grad = True
for param in self.clip_model.transformer.resblocks[-1].parameters():
    param.requires_grad = True
```

---

## 3) Unfreeze the Final LayerNorm

CLIP also has a `ln_final` layernorm at the very end of the text Transformer. If you want to adapt it as well:

```python
# UNFREEZE ln_final
self.clip_model.ln_final.weight.requires_grad = True
self.clip_model.ln_final.bias.requires_grad = True
```

That allows the final normalization step to shift for your domain.

---

## 4) Verify You Didn’t Re‐Freeze Anything

Because of the order of operations, ensure that you **freeze everything first**, *then* selectively unfreeze. One clean approach is:

```python
# 1) Freeze everything
for param in self.clip_model.transformer.parameters():
    param.requires_grad = False

# 2) Freeze token embedding
for param in self.clip_model.token_embedding.parameters():
    param.requires_grad = False

# 3) Freeze vision encoder
for param in self.clip_model.visual.parameters():
    param.requires_grad = False

# 4) Unfreeze last transformer block
for param in self.clip_model.transformer.resblocks[-1].parameters():
    param.requires_grad = True

# 5) Unfreeze final layernorm
self.clip_model.ln_final.weight.requires_grad = True
self.clip_model.ln_final.bias.requires_grad = True
```

Now only the prompt tokens, the last block in the text transformer, and LN_final are trainable. Everything else remains frozen.

---

## 5) Done — Partial Unfreeze

After this, your training loop remains the same. The difference is your text encoder’s last block + LN can now learn along with your prompt. If you see more improvement or faster gradient flow, that’s exactly the intended effect of partial unfreezing.

---

### Potential Tweaks

- **Learning Rate**: Because you are now training more parameters than just the prompt, you might want to reduce the LR (e.g. 1e-5 or 5e-5 instead of 1e-4), or do some LR tuning.  
- **Number of Blocks**: You can choose to unfreeze the last 1, 2, or more blocks. More blocks = more capacity to learn, but also greater risk of overfitting on small data.  
- **Text Projection**: If you want, you could also unfreeze `self.clip_model.text_projection` so the final projection matrix can adapt. But usually partial blocks + LN_final is enough for a small domain shift.  
