#  Stage 3  PromptSG Integration (CLIP-FH)

This folder implements **Stage 3** of the CLIP-FH pipeline, focused on joint training using **PromptSG**: a semantic-guided prompt tuning strategy that combines pseudo-token generation, multimodal fusion, and contrastive ReID losses.

---

##  Folder Contents

| File                            | Description |
|----------------------------------|-------------|
| `train_stage3_promptsg.py`       | Main training script using PromptSG methodology: joint pseudo-token learning, prompt composition, cross-attention, and multi-loss optimization. |
| `eval_stage3_promptsg.py`        | Evaluation runner for ReID-style Rank@K and mAP metrics using the best model across 10 query-gallery splits. |
| `ANLAYSIS REPORT_FORMATTED.md`   | Clean tabular summary of all PromptSG versions (`v1` to `v11`) for both ViT-B/16 and RN50, with configuration changes and results. |
| `ANLAYSIS REPORT_UNFORMATTED.md` | Full experimental notes, YAML config details, motivations, and architecture insights across all runs. |

---

##  Script Breakdown

###  `train_stage3_promptsg.py`

Implements PromptSG pipeline:
- Builds CLIP model with frozen text encoder
- Learns **TextualInversionMLP** for pseudo-token generation
- Dynamically composes prompts with pseudo tokens
- Applies **MultiModalInteraction** (cross-attention & transformers)
- Uses configurable classifier: `linear` or `BNNeck + ArcFace`
- Supports three loss types: CrossEntropy, Triplet, SupCon

 YAML fields include:
```yaml
classifier: arcface
bnneck_dim: 256
transformer_layers: 3
prompt_template: "A captured frame showing a person's {aspect} hand."
loss_id_weight: 1.0
loss_tri_weight: 1.0
supcon_loss_weight: 1.0
````

 Example usage:

```bash
python train_stage3_promptsg.py --config configs/train_stage3_promptsg/train_stage3_vitb16_11k_dorsal_r.yml
```

---

###  `eval_stage3_promptsg.py`

Launches ReID-style evaluation using the trained model:

* Supports all 10 query-gallery splits
* Logs Rank-1, Rank-5, Rank-10, and mAP
* Works with best saved `_BEST.pth` checkpoint

 Usage:

```bash
python eval_stage3_promptsg.py configs/eval_stage3_promptsg/eval_stage3_vitb16_11k_dorsal_r.yml
```

---

##  Model Versions (`v1``v11`)

All versions are detailed in `ANLAYSIS REPORT_FORMATTED.md`:

| Version | Highlights                            | Best Result (ViT-B/16)     |
| ------- | ------------------------------------- | -------------------------- |
| v1      | Baseline CE + Triplet + Linear Head   | R1 62.38 / mAP 71.57       |
| v4      | BNNeck + ArcFace Head                 | R1 42.25 / mAP 54.32       |
| v5      | Paper-faithful MLP + prompt fusion    | R1 41.45 / mAP 52.09       |
| v6      | Grad clip , LR  (1e-6)  huge boost | R1 80.83 / mAP 86.63       |
| v8      | BNNeck dim 1024 + no Cosine LR        |  **R1 85.86 / mAP 89.99** |
| v11     | Portrait resize (224128)             | R1 81.59 / mAP 86.93       |

> For RN50 track, version v8 also achieved the peak: R1 61.23 / mAP 69.75.

---

##  Technical Highlights

*  Fully frozen CLIP text encoder
*  Supports pseudo-token learning via `TextualInversionMLP`
*  Prompt composition: `"A captured frame showing a persons {aspect} hand"`
*  Cross-attention between image and prompt embeddings
*  ArcFace classifier with BNNeck + optional reduction
*  Rank\@K and mAP evaluation across all splits

---

##  Key Takeaways

| Insight                                                         | Explanation                                                       |
| --------------------------------------------------------------- | ----------------------------------------------------------------- |
|  **Contrastive-only** (Triplet + SupCon) works better for RN50 | ViT collapsed on it (v3), RN50 peaked (v3)                        |
|  **BNNeck + ArcFace** needs tuning                             | Improves results only when paired with lower LR and gradient norm |
|  **Portrait resize** hurts performance                        | v11 drop \~34 pp mAP                                             |
|  **Gradient clipping and tiny LR** stabilize training         | v6v8 gains                                                       |

---

##  Result Files

* CSV files: `result_logs/stage3_vitb16_remarks.csv`, `stage3_rn50_remarks.csv`
* Model checkpoints: Saved under `save_dir` as `_BEST.pth`
* Evaluation logs: Written to `eval_logs/`

---

##  Contact

For further clarifications, contact [babupallam@gmail.com](mailto:babupallam@gmail.com)

```
