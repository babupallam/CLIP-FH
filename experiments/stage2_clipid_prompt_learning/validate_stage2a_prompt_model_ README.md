### 📄 `validate_stage2a_prompt_model.md`

---

```markdown
# ✅ Stage 2a Prompt Model Validation (HandCLIP)

This document outlines the validation process for the **Stage 2a prompt learning model** in the HandCLIP project. The goal is to verify that the trained prompt model aligns correctly with image features before proceeding to Stage 2b (image encoder fine-tuning).

---

## 🔍 Purpose of Validation

- Confirm that the prompt model saved from Stage 2a training correctly generates meaningful text embeddings.
- Ensure these prompt-generated text features align well with frozen CLIP image features.
- Compute retrieval accuracy using cosine similarity: Image → Text and Text → Image.
- Sanity-check the prompt effectiveness on both **validation** and **training sets**.

---

## 🧠 What Is Validated?

- ✅ Correct loading of the CLIP backbone (`ViT-B/16` or other variants).
- ✅ PromptLearner weights loaded from Stage 2a checkpoint (`.pth`).
- ✅ Prompt tokens (`ctx`) are frozen during evaluation.
- ✅ Contrastive performance is measured via cosine similarity between:
  - **Image features**: Extracted from frozen `clip_model.encode_image()`
  - **Text features**: Generated from `PromptLearner.forward_batch()` and passed through the frozen text encoder.

---

## 🛠️ Validation Script

Script path:
```
experiments/stage2_clipid_prompt_learning/validate_stage2a_prompt_model.py
```

Usage:
```bash
python experiments/stage2_clipid_prompt_learning/validate_stage2a_prompt_model.py \
  --config configs/validate_stage2_clip_reid/validate_stage2a_validationset_vitb16_11k_dorsal_r.yml
```

Optional: Evaluate on training set:
```bash
python experiments/stage2_clipid_prompt_learning/validate_stage2a_prompt_model.py \
  --config configs/validate_stage2_clip_reid/validate_stage2a_trainset_vitb16_11k_dorsal_r.yml
```

---

## ⚙️ YAML Configs

### `validate_stage2a_vitb16_11k_dorsal_r.yml`

```yaml
experiment: validate_stage2a_prompt_vitb16_11k_dorsal_r
dataset: 11k
aspect: dorsal_r
model: vitb16
variant: clipid
n_ctx: 8
ctx_init: "a hand"
prompt_template: "A photo of {}'s {aspect} hand for identification."
batch_size: 32
val_split: ./datasets/11khands/train_val_test_split_dorsal_r/val
stage2a_ckpt: saved_models/stage2a_prompt_vitb16_11k_dorsal_r_vitb16_11k_dorsal_r_nctx8_e30_lr0001_bs32_ctxahand.pth
```

### `validate_stage2a_trainset_vitb16_11k_dorsal_r.yml`

```yaml
experiment: validate_stage2a_prompt_vitb16_11k_dorsal_r_trainset
dataset: 11k
aspect: dorsal_r
model: vitb16
variant: clipid
n_ctx: 8
ctx_init: "a hand"
prompt_template: "A photo of {}'s {aspect} hand for identification."
batch_size: 32
val_split: ./datasets/11khands/train_val_test_split_dorsal_r/train
stage2a_ckpt: saved_models/stage2a_prompt_vitb16_11k_dorsal_r_vitb16_11k_dorsal_r_nctx8_e30_lr0001_bs32_ctxahand.pth
```

---

## 📊 Output Metrics

The script reports:

- `Top-1 Accuracy`: Percentage of samples where the most similar text matches the correct label.
- `Top-5 Accuracy`: Percentage where the correct label is among top-5 similar texts.

Example output:
```
🎯 Validation Summary
Top-1 Accuracy : 0.0421
Top-5 Accuracy : 0.3149
```

---
