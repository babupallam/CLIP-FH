
# 🛠️ Utilities — CLIP-FH Project

The `utils/` directory provides reusable modules and helper functions essential to training, evaluation, logging, model handling, and data processing throughout the CLIP-FH pipeline.

---

## 📁 Directory Structure

```

utils/
├── loss/                         # All loss function definitions
│   ├── archives/                 # Deprecated or legacy loss function versions
│   ├── arcface.py               # ArcFace loss (angular margin softmax)
│   ├── center\_loss.py           # Center loss for feature compactness
│   ├── cross\_entropy\_loss.py    # Label-smoothing CrossEntropy loss
│   ├── make\_loss.py             # Factory method to construct combined losses
│   ├── supcon.py                # Supervised Contrastive loss (SupCon)
│   ├── triplet\_loss.py          # Triplet loss with optional margin
│   └── README.md                # Loss module documentation
│
├── clip\_patch.py                # Wrapper and patch for modifying CLIP modules
├── dataloaders.py               # Dataset loading and PK sampling logic
├── device\_selection.py          # Auto-detect CUDA or MPS device
├── eval\_utils.py                # Functions for ReID-style evaluation (Rank\@K, mAP)
├── feature\_cache.py             # Embedding cache utils for quick eval or zero-shot
├── logger.py                    # CSV and TXT logger utilities
├── naming.py                    # Filename and experiment naming helpers
├── save\_load\_models.py          # Model checkpoint save/load logic
├── train\_helpers.py             # Training logic wrappers and hooks
├── transforms.py                # Image transform pipelines
└── README.md                    # You are here 📘

````

---

## 🔍 Key Components

### 🔹 `loss/`
Contains all the loss functions used across the pipeline:
- `arcface.py`: Implements ArcFace margin-based classifier.
- `center_loss.py`: Minimizes intra-class variance in embedding space.
- `triplet_loss.py`: Distance-based learning with anchor, positive, and negative.
- `supcon.py`: Implements Supervised Contrastive Loss for P×K sampled batches.
- `make_loss.py`: Centralized function to combine multiple loss terms per config.
- `cross_entropy_loss.py`: Standard CE loss with optional smoothing.
- `archives/`: Backup versions of older loss implementations.

> Use `make_loss()` to dynamically assemble loss functions based on config.

---

### 🔹 Other Utility Scripts

- `clip_patch.py`: Custom patches and wrappers to modify OpenAI's CLIP modules (e.g., freezing layers, BNNeck).
- `dataloaders.py`: Includes dataset creation, PK sampling, class-balanced batch logic.
- `eval_utils.py`: mAP and Rank@K computation logic for ReID evaluation.
- `feature_cache.py`: Saves and loads image/text features for fast zero-shot or ablation testing.
- `logger.py`: Dual-mode logger writing to both CSV and console/txt.
- `train_helpers.py`: Hooks and helpers for loss scaling, gradient clipping, metric tracking.
- `save_load_models.py`: Load/save checkpoint handlers including best-model logic.
- `transforms.py`: Defines training/validation transforms (augmentations, normalization).
- `device_selection.py`: Automatically selects between CPU, CUDA, or MPS for PyTorch.

---

## ✅ Usage Example

You can import any utility in your training or evaluation script like so:

```python
from utils.loss.make_loss import make_loss
from utils.dataloaders import create_dataloader
from utils.eval_utils import compute_rank_k_accuracy
````

---

## 📬 Contact

For questions or suggestions, contact: [babupallam@gmail.com](mailto:babupallam@gmail.com)

```
