# CLIP-FH: Fine-Tuning CLIP for Hand-Based Identity Matching



```angular2html
CLIP-FH/
├── configs/
│   ├── dataset_11k.yml                 # 11k dataset configurations (paths, aspect selection, etc.)
│   ├── dataset_hd.yml                  # HD dataset configurations
│   ├── model_vitb16.yml                # Model config for CLIP-ViT-B/16
│   ├── model_res50.yml                 # Model config for CLIP-RN50
│   ├── train_baseline.yml              # For direct/zero-shot usage of CLIP
│   ├── train_finetune.yml              # For single-stage finetuning (frozen text encoder)
│   └── train_clipreid.yml              # For two-stage CLIP-ReID style fine-tuning

├── datasets/
│   ├── __init__.py
│   ├── data_preprocessing/
│   │   ├── prepare_11k.py             # <--- Replaces 1_a_prepare_train_val_test_11k_r_l.py
│   │   ├── prepare_hd.py              # <--- Replaces 1_b_prepare_train_val_test_hd.py
│   │   └── README.md                  # (Optional) documentation for data prep steps
│   ├── build_dataloader.py            # Creates PyTorch dataloaders for 11k & HD
│   └── transforms.py                  # Any custom transforms or augmentations

├── models/
│   ├── __init__.py
│   ├── clip_backbones.py              # Utilities to load pre-trained CLIP (ViT-B/16, RN50)
│   ├── handclip_fh_encoder.py         # Wraps the CLIP image encoder + classification heads
│   ├── prompt_learner.py              # (Future) Prompt learning module if adopting CLIP-ReID
│   ├── reid_heads.py                  # Additional classifier heads, e.g., ArcFace, BNNeck, etc.
│   └── ...

├── loss/
│   ├── __init__.py
│   ├── cross_entropy_loss.py
│   ├── triplet_loss.py
│   ├── center_loss.py
│   └── make_loss.py                   # Combines multiple losses for training (CE + triplet + center, etc.)

├── engine/
│   ├── __init__.py
│   ├── baseline_inference.py          # For direct/zero-shot usage of CLIP
│   ├── finetune_trainer.py            # For single-stage finetuning (frozen text encoder)
│   ├── clipreid_trainer.py            # For two-stage CLIP-ReID approach
│   ├── evaluator.py                   # mAP, CMC, re-ranking, etc.
│   ├── inference.py                   # Additional scripts for embedding extraction or deployment
│   └── ...

├── experiments/
│   ├── 1_baseline_vitb16.py           # Script for direct CLIP-ViT-B/16 usage
│   ├── 1_baseline_res50.py            # Script for direct CLIP-RN50 usage
│   ├── 2_finetune_vitb16.py           # Single-stage fine-tune for CLIP-ViT-B/16
│   ├── 2_finetune_res50.py            # Single-stage fine-tune for CLIP-RN50
│   ├── 3_clipreid_vitb16.py           # Two-stage CLIP-ReID approach for ViT-B/16
│   ├── 3_clipreid_res50.py            # Two-stage CLIP-ReID approach for RN50
│   └── ...                            # Additional scripts as needed

├── result_logs/
│   ├── baseline_vitb16_eval.log
│   ├── baseline_res50_eval.log
│   ├── finetune_vitb16.log
│   ├── finetune_res50.log
│   ├── clipreid_vitb16.log
│   ├── clipreid_res50.log
│   └── ...

├── saved_models/
│   ├── baseline_vitb16.pth
│   ├── baseline_res50.pth
│   ├── finetune_vitb16.pth
│   ├── finetune_res50.pth
│   ├── clipreid_vitb16_stage1.pth
│   ├── clipreid_vitb16_stage2.pth
│   └── ...
    
├── my_logs/                           # Additional debug outputs or custom logs
├── README.md
├── requirements.txt                   # Dependencies (PyTorch, clip, etc.)
└── main.py                            # (Optional) Master CLI script to run training/evaluation

```


***
***
***


# 🧠 **Overall Project Goal**
You’re building **CLIP-FH: Fine-Tuning CLIP for Hand-Based Identity Matching**, with:

1. **Baseline Evaluation** of CLIP on hand images.
2. **Single-Stage Fine-Tuning** (text encoder frozen).
3. **Two-Stage CLIP-ReID Style Fine-Tuning** with prompt learning.

This structure is modular, extensible, and logically grouped to support all three stages on multiple datasets (`11k`, `HD`) and backbones (`ViT-B/16`, `RN50`).

---

# 📂 **Folder-by-Folder and File-by-File Explanation**

---

## ✅ `configs/`  
**Purpose**: YAML-based configuration management for datasets, models, and training.

| File | What It Contains |
|------|------------------|
| `dataset_11k.yml` | Paths, image extensions, selected aspect(s) (e.g., dorsal_r) for the 11k dataset. |
| `dataset_hd.yml` | Paths for HD dataset (e.g., `1-501`, `502-712`, output structure). |
| `model_vitb16.yml` | Backbone-specific settings for CLIP ViT-B/16: dimensions, token configs. |
| `model_res50.yml` | Same as above for CLIP ResNet-50. |
| `train_baseline.yml` | Learning rate, optimizer, batch size, etc., for baseline testing. |
| `train_finetune.yml` | Used when fine-tuning the image encoder (text encoder frozen). |
| `train_clipreid.yml` | Two-stage training: Stage 1 (prompt learning), Stage 2 (encoder fine-tuning). |

🧩 Used by `experiments/*.py` scripts or via a `main.py` dispatcher.

---

## ✅ `datasets/`  
**Purpose**: Data preparation and loading.

### 📁 `data_preprocessing/`
| File | What It Does |
|------|--------------|
| `prepare_11k.py` | Prepares the 11k dataset with splits (train/val/test/query/gallery) by aspect (e.g., dorsal_r). |
| `prepare_hd.py` | Prepares the HD dataset with 213 extra subjects added to gallery, and 10 query/gallery splits. |
| `README.md` | (Optional) Guide for using data prep scripts. |

### 🧪 Other Scripts
| File | Purpose |
|------|---------|
| `build_dataloader.py` | Creates PyTorch `DataLoader` objects for train/test/val based on configs. |
| `transforms.py` | (Optional) Contains any custom image transforms (e.g., resizing, augmentations, normalization). |

---

## ✅ `models/`  
**Purpose**: Defines model architectures and encoders.

| File | What It Contains |
|------|------------------|
| `clip_backbones.py` | Loads CLIP with pre-trained weights (ViT-B/16 or RN50). |
| `handclip_fh_encoder.py` | Custom wrapper over CLIP's image encoder + additional layers (classification, projection). |
| `prompt_learner.py` | (Used in Stage 1 of CLIP-ReID): Learnable prompts `[X1][X2]...[XM]`. |
| `reid_heads.py` | Optional classification heads (e.g., ArcFace, BNNeck) for reID identity loss. |
| `__init__.py` | Package initializer for `models/`. |

---

## ✅ `loss/`  
**Purpose**: Custom loss functions and combiners.

| File | What It Does |
|------|--------------|
| `cross_entropy_loss.py` | CE loss for ID classification. |
| `triplet_loss.py` | Triplet loss with hard-negative mining. |
| `center_loss.py` | Optional feature-center regularization. |
| `make_loss.py` | Combines multiple losses (e.g., `total_loss = CE + Triplet + 0.0005 * Center`). |
| `__init__.py` | Package initializer. |

---

## ✅ `engine/`  
**Purpose**: Training and evaluation workflows (the heart of the pipeline).

| File | What It Does |
|------|--------------|
| `baseline_inference.py` | Runs inference using CLIP (zero-shot or feature extraction) for reID evaluation. |
| `finetune_trainer.py` | Handles single-stage fine-tuning where the text encoder is frozen. |
| `clipreid_trainer.py` | Two-stage CLIP-ReID training: Stage 1 (learn prompts), Stage 2 (fine-tune image encoder). |
| `evaluator.py` | Computes mAP, CMC rank-1, rank-5, etc. Uses distance metrics, optionally re-ranking. |
| `inference.py` | Embedding extraction or end-to-end test-time inference. |
| `__init__.py` | For module import structure. |

---

## ✅ `experiments/`  
**Purpose**: High-level experiment runners (organized by your 3-stage approach + backbones).

| File | What It Runs |
|------|---------------|
| `1_baseline_vitb16.py` | Stage 1: Evaluate pre-trained CLIP ViT-B/16 on hand images (zero-shot). |
| `1_baseline_res50.py` | Same as above for ResNet-50. |
| `2_finetune_vitb16.py` | Stage 2: Fine-tune image encoder (text encoder frozen) for ViT-B/16. |
| `2_finetune_res50.py` | Same as above for ResNet-50. |
| `3_clipreid_vitb16.py` | Stage 3: Two-stage CLIP-ReID training for ViT-B/16 (prompt learning + encoder tuning). |
| `3_clipreid_res50.py` | Same as above for ResNet-50. |

🔁 You can easily add:
- `1_baseline_vitb16_hd.py` if testing on HD.
- `2_finetune_vitb16_dorsal_l.py` if switching aspect.

---

## ✅ `result_logs/`  
**Purpose**: Store log files per experiment type and backbone.

| File | What It Logs |
|------|---------------|
| `baseline_vitb16_eval.log` | Output from `1_baseline_vitb16.py`. |
| `clipreid_res50.log` | From CLIP-ReID experiments with ResNet-50. |
| `finetune_res50.log` | Training progress for finetuning with frozen text encoder. |

🧪 Use this for training loss, val acc, and debugging.

---

## ✅ `saved_models/`  
**Purpose**: Store model checkpoints.

| File | What It Is |
|------|------------|
| `baseline_vitb16.pth` | Features extracted from CLIP (not fine-tuned). |
| `finetune_res50.pth` | Model after single-stage fine-tuning. |
| `clipreid_vitb16_stage1.pth` | Checkpoint after prompt learning (Stage 1 of CLIP-ReID). |
| `clipreid_vitb16_stage2.pth` | Final fine-tuned model after Stage 2. |

---

## ✅ `my_logs/`  
Optional scratchpad for:
- TensorBoard
- Console outputs
- Custom CSV logs
- Visual debug plots

---

## ✅ Root-Level Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, how to run experiments, dataset setup. |
| `requirements.txt` | Dependencies: `torch`, `clip-by-openai`, `scikit-learn`, `opencv`, etc. |
| `main.py` | Optional CLI dispatcher that uses config files to launch different stages. |
```python
# Sample main.py CLI routing
python main.py --mode finetune --config configs/train_finetune.yml
```

---

# ✅ Summary Table: Purpose by Layer

| Layer | Purpose |
|-------|---------|
| `configs/` | Modular experiment settings (model, data, training) |
| `datasets/` | Prepare, load, and transform hand datasets |
| `models/` | Define and extend CLIP models for reID |
| `loss/` | Modular losses: CE, Triplet, Center |
| `engine/` | Training & evaluation logic |
| `experiments/` | Scripts to run specific model-stage-dataset combinations |
| `result_logs/` | Output logs for training & testing |
| `saved_models/` | Store model weights after training |
| `my_logs/` | Optional debug/info logs |
| `README.md`, `main.py` | Entry points and documentation |

---