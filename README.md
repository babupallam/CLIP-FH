# CLIP-FH: Fine-Tuning CLIP for Hand-Based Identity Matching

This repository implements a multi-stage fine-tuning pipeline for using CLIP (Contrastive Language–Image Pretraining) on hand-based biometric identification. It supports baseline evaluation, classifier training, CLIP-ReID integration, prompt-based fine-tuning (PromptSG), and detailed performance analysis.

---

## 📦 Environment Setup

We recommend using Python 3.9+ with `virtualenv` or `conda`.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/clip-fh.git
cd clip-fh
```

### 1. Create Environment and Install Dependencies

```bash
python -m venv clipfh-env
source clipfh-env/bin/activate  # or clipfh-env\Scripts\activate on Windows

pip install -r requirements.txt
```

---

## 📁 Dataset Preparation

### 🖐️ Prepare Train/Val/Test Split for 11k Dataset

```bash
python datasets/data_preprocessing/prepare_train_val_test_11k_r_l.py
```

---

## 🚀 Stage 1: Baseline Model Evaluation (Zero shot)

### 🔁 Evaluate All Models

### 🔎 Evaluate stage 1 models

```bash
python experiments/stage1_baseline_inference/eval_baseline_clip_single.py --config configs/baseline/eval_vitb16_11k_dorsal_r.yml
python experiments/stage1_baseline_inference/eval_baseline_clip_single.py --config configs/baseline/eval_rn50_11k_dorsal_r.yml
```

---

## 🔄 Stage 2: CLIP-ReID Integration (Prompt + Image Encoder Fine-tuning)

### 🧠 Train with Joint Prompt & Image Embedding

```bash
python experiments/stage2_clipreid_integration/train_stage2_joint.py --config configs/train_stage2_clip_reid/train_joint_vitb16_11k_dorsal_r.yml
python experiments/stage2_clipreid_integration/train_stage2_joint.py --config configs/train_stage2_clip_reid/train_joint_rn50_11k_dorsal_r.yml
```

*For RN50, create `train_joint_rn50_11k_dorsal_r.yml` and run similarly.*

### 🧪 Evaluate Stage 2

```bash
python experiments/stage2_clipreid_integration/eval_stage2_joint.py configs/eval_stage2_clip_reid/eval_joint_vitb16_11k_dorsal_r.yml
python experiments/stage2_clipreid_integration/eval_stage2_joint.py configs/eval_stage2_clip_reid/eval_joint_rn50_11k_dorsal_r.yml
```

*Create and use RN50 config if needed.*

---

## 🎯 Stage 3: PromptSG Fine-tuning (Prompt + Image Encoder with Semantic Guidance)

### 🏋️ Train on ViT-B/16 and RN50

```bash
python experiments/stage3_promptsg_integration/train_stage3_promptsg.py --config configs/train_stage3_promptsg/train_stage3_vitb16_11k_dorsal_r.yml
python experiments/stage3_promptsg_integration/train_stage3_promptsg.py --config configs/train_stage3_promptsg/train_stage3_rn50_11k_dorsal_r.yml
```

### ✅ Evaluate Stage 3

```bash
python experiments/stage3_promptsg_integration/eval_stage3_promptsg.py configs/eval_stage3_promptsg/eval_stage3_vitb16_11k_dorsal_r.yml
python experiments/stage3_promptsg_integration/eval_stage3_promptsg.py configs/eval_stage3_promptsg/eval_stage3_rn50_11k_dorsal_r.yml
```

---

## 📊 Plotting and Log Analysis

### 🌳 Tree Visualization

```bash
python experiments/conclusion_outputs/generate_tree.py
```
### 📋 Evaluation Log Analysis and CSV Creation

```bash
python experiments/conclusion_outputs/stage2_eval_log_analysis.py
python experiments/conclusion_outputs/stage3_eval_log_analysis.py
```

### 🧾 Training Log Analysis and CSV Creation

```bash
python experiments/conclusion_outputs/stage2_train_log_analysis.py
python experiments/conclusion_outputs/stage3_train_log_analysis.py
```


### 📈 Training Metrics Plots

```bash
python experiments/conclusion_outputs/plot_stage2_train_metrics.py
python experiments/conclusion_outputs/plot_stage3_train_metrics.py
```

---

## 🧠 Notes

* Make sure all `config` files exist before running each stage.
* For RN50 in Stage 2, create `train_joint_rn50_11k_dorsal_r.yml` and `eval_joint_rn50_11k_dorsal_r.yml` by copying the ViT config and updating `MODEL.NAME` to `RN50`.
* For future datasets (e.g., HD Hands), replicate the above structure using `*_hd_dorsal_r.yml`.

---

## 📬 Contact

For queries, suggestions, or collaboration, feel free to reach out via GitHub Issues or [babupallam@gmail.com](mailto:babupallam@gmail.com).
