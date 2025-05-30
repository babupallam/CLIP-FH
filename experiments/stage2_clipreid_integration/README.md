
#  Stage 2  CLIP-ReID Joint Fine-Tuning

This folder implements **Stage 2** of the CLIP-FH pipeline: joint training of prompt learners and image encoders using CLIP-ReID principles. It contains training scripts, evaluation runners, and detailed version tracking reports.

---

##  Contents

| File                                 | Description |
|--------------------------------------|-------------|
| `train_stage2_joint.py`              | Main training script for joint prompt + image encoder fine-tuning using ArcFace, SupCon, Triplet, Center, and ID losses. |
| `eval_stage2_joint.py`               | Wrapper script to evaluate a trained Stage 2 model using a specified config. |
| `ANLAYSIS REPORT_UNFORMATTED.md`     | Raw experimental logs and summaries of each versions setup and result. |
| `ANLAYSIS REPORT_FORMATTED.md`       | Cleanly formatted markdown report comparing ViT-B/16 and RN50 runs across versions (v1v10). Includes key insights and configuration deltas. |

---

##  Key Components

###  `train_stage2_joint.py`

This is the main joint training script that performs:

- Prompt stage training (PromptSG-style with frozen encoder)
- Image encoder fine-tuning (ReID-style with ArcFace classifier)
- Supports staged unfreezing via `unfreeze_blocks`
- Supports multiple loss functions: SupCon, Triplet, Center, ArcFace, Cross-Entropy

 **Usage:**
```bash
python train_stage2_joint.py --config configs/train_stage2_clip_reid/train_joint_vitb16_11k_dorsal_r.yml
````

###  `eval_stage2_joint.py`

This is a wrapper to run evaluations using `run_eval_clip.py`.

 **Usage:**

```bash
python eval_stage2_joint.py configs/eval_stage2_clip_reid/eval_joint_vitb16_11k_dorsal_r.yml
```

---

##  Model Versions (`v1` to `v10`)

The experiments are organized as versioned runs (`v1`, `v2`, ..., `v10`) for both ViT-B/16 and RN50. Key differences include:

| Version | Change Highlights                                         |
| ------- | --------------------------------------------------------- |
| v1      | Baseline dual-stage, all losses, Cosine LR                |
| v2      | Added prompt diversity, AdamW, -normalization           |
| v3      | Switched to OneCycleLR                                    |
| v4      | Triplet and Center loss disabled                          |
| v5      | ArcFace softened (scale 20, margin 0.3); all losses ON    |
| v6      | Partial fine-tuning (`unfreeze_blocks=2`)                 |
| v7      | RN50-specific: reduced LR, ArcFace scale=25               |
| v8      | RN50-specific: deeper FT (`unfreeze_blocks=4`), Center ON |
| v9      | ArcFace extreme (scale=35), Triplet OFF                   |
| v10     | RN50 only: Tweaked v9 with Triplet ON, ArcFace scale=30   |

Detailed performance comparisons are available in:

*  `ANLAYSIS REPORT_FORMATTED.md`  tabular breakdown of results and settings
*  `ANLAYSIS REPORT_UNFORMATTED.md`  raw notes and YAML config summaries

---

##  Losses Supported

The training pipeline can combine the following loss functions based on YAML:

* `SupCon` (Supervised Contrastive Loss)
* `TripletLoss`
* `CenterLoss`
* `ArcFaceLoss`
* `CrossEntropyLoss`

Each can be toggled via:

```yaml
loss_use_supcon: true
loss_use_triplet: true
loss_use_center: true
loss_use_arcface: true
loss_use_id: true
```

---

##  Notes

* Each run saves logs to `train_logs/stage2-v*/` and evaluation to `eval_logs/stage2-v*/`
* Final results (Rank-1, mAP) for each split are summarized in the Stage 2 CSVs.
* Feature dimension alignment and gradient unfreezing are handled internally based on model type (ViT vs RN50).

---

##  Related Paths

*  Configs: `configs/train_stage2_clip_reid/` and `configs/eval_stage2_clip_reid/`
*  Result logs: `result_logs/stage2_vitb16_*.csv`, `stage2_rn50_*.csv`
*  Evaluation runner: `experiments/run_eval_clip.py`

---

##  Contact

For detailed questions or code walkthrough, email: [babupallam@gmail.com](mailto:babupallam@gmail.com)

```

---
