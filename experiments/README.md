
# 🧪 CLIP-FH Experiments Directory

This folder contains all the experimental scripts used in the **CLIP-FH** project. The project is organized in modular stages, each representing a critical step in the fine-tuning and evaluation pipeline for hand-based identity matching using CLIP.

---

## 📁 Folder Structure Overview

```

experiments/
├── archived/                     # \[Optional] Deprecated or backup scripts from earlier versions
├── conclusion\_outputs/          # Post-training analysis: metric plots, log parsers, and CSV generators
├── stage1\_baseline\_inference/   # Stage 0/1: Baseline zero-shot evaluation and classifier fine-tuning
├── stage2\_clipreid\_integration/ # Stage 2: CLIP-ReID style training (joint prompt + image encoder tuning)
├── stage3\_promptsg\_integration/ # Stage 3: PromptSG (prompt tuning with semantic guidance)

```

---

## 📍 Subdirectory Descriptions

### 📂 `archived/`
- Contains experimental or deprecated scripts from earlier training runs.
- Not used in the current pipeline, but retained for reference.

### 📂 `conclusion_outputs/`
- Includes all scripts for:
  - Directory tree generation (`generate_tree.py`)
  - Log parsing for training/evaluation
  - Plotting training metrics
- Outputs CSV tables and `.png` plots to `result_logs/`.

### 📂 `stage1_baseline_inference/`
- Stage 0: Zero-shot baseline inference using pretrained CLIP.
- Stage 1: Fine-tuning classifier on top of a frozen image encoder.
- Example scripts:
  - `eval_baseline_clip_single.py`
  - `train_stage1_frozen_text.py`
  - `eval_stage1_frozen_text.py`

### 📂 `stage2_clipreid_integration/`
- Implements Stage 2 of the pipeline using CLIP-ReID methodology.
- Supports joint training of image encoder + prompt embeddings using ArcFace, Triplet, and Center loss.
- Example scripts:
  - `train_stage2_joint.py`
  - `eval_stage2_joint.py`

### 📂 `stage3_promptsg_integration/`
- Final stage integrating **PromptSG**: fine-tuning prompts with semantic guidance.
- Focuses on PromptEncoder and advanced supervision signals.
- Example scripts:
  - `train_stage3_promptsg.py`
  - `eval_stage3_promptsg.py`

---

## ✅ Usage Notes

- Each stage requires a YAML config file (under `configs/`) to run.
- Training outputs are logged to `train_logs/`, and evaluations to `eval_logs/`.
- Results and plots are saved to `result_logs/` by the scripts in `conclusion_outputs/`.

---

## 📬 Contact

For technical questions or collaboration, contact: [babupallam@gmail.com](mailto:babupallam@gmail.com)
