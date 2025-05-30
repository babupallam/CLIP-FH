
# 📊 CLIP-FH: Log Analysis and Visualization Scripts

This folder contains post-training analysis tools for the **CLIP-FH** project, including codebase structure generation, training/evaluation log parsing, metric CSV export, and result plotting for **Stage 2** and **Stage 3**.

---

## 🗂️ Files Overview

| File                              | Description |
|-----------------------------------|-------------|
| `generate_tree.py`               | Generates the full directory and file tree of the project, excluding logs, datasets, and hidden folders. Outputs to `code_structure.txt`. |
| `stage2_train_log_analysis.py`   | Parses Stage 2 training logs for both ViT-B/16 and RN50, and saves per-epoch metrics (losses, rank@k, mAP) to CSVs. |
| `stage2_eval_log_analysis.py`    | Parses Stage 2 evaluation logs across all splits and versions, and outputs summary metrics to CSV. |
| `plot_stage2_train_metrics.py`   | Generates plots from Stage 2 training CSVs: loss vs. epoch, rank@k, mAP, and loss breakdown. |
| `stage3_train_log_analysis.py`   | Parses Stage 3 training logs (PromptSG) and extracts training + validation metrics into CSVs. |
| `stage3_eval_log_analysis.py`    | Parses Stage 3 evaluation logs and exports Rank-1, Rank-5, Rank-10, and mAP values per split and final average to CSVs. |
| `plot_stage3_train_metrics.py`   | Plots validation Rank-1, mAP curves, and final accuracy bar charts from Stage 3 CSVs. |

---

## 📁 Output Folders

- All outputs (CSV tables and plots) are saved in:
```

result\_logs/

````

- Ensure this directory exists or will be auto-created by the scripts.

---

## 🧰 How to Use the Scripts

### 1. Generate Codebase Structure

```bash
python generate_tree.py
````

* Output: `code_structure.txt`

---

### 2. Parse Stage 2 Logs

#### a. Training Logs → CSVs

```bash
python stage2_train_log_analysis.py
```

* Outputs:

  * `result_logs/stage2_vitb16_train_table.csv`
  * `result_logs/stage2_rn50_train_table.csv`

#### b. Evaluation Logs → CSVs

```bash
python stage2_eval_log_analysis.py
```

* Outputs:

  * `result_logs/stage2_vitb16_eval_table.csv`
  * `result_logs/stage2_rn50_eval_table.csv`

---

### 3. Plot Stage 2 Training Metrics

```bash
python plot_stage2_train_metrics.py
```

* Uses CSVs from `stage2_train_log_analysis.py`
* Generates:

  * Loss vs Epoch
  * Rank-1 & mAP vs Epoch
  * Final Accuracy per Version (bar plot)
  * Loss Breakdown Facets

---

### 4. Parse Stage 3 Logs

#### a. Training Logs → CSVs

```bash
python stage3_train_log_analysis.py
```

* Outputs:

  * `result_logs/stage3_vitb16_train_table.csv`
  * `result_logs/stage3_rn50_train_table.csv`

#### b. Evaluation Logs → CSVs

```bash
python stage3_eval_log_analysis.py
```

* Outputs:

  * `result_logs/stage3_vitb16_eval_table.csv`
  * `result_logs/stage3_rn50_eval_table.csv`

---

### 5. Plot Stage 3 Metrics

```bash
python plot_stage3_train_metrics.py
```

* Uses CSVs from Stage 3 training logs.
* Generates:

  * Validation Rank-1 / mAP vs Epoch
  * Final accuracy bar plots

---

## ✅ Prerequisites

Ensure the following Python packages are installed:

```bash
pip install pandas matplotlib seaborn
```

---

## 📌 Notes

* These scripts are **post-processing utilities**. You must first run the training and evaluation steps from the main CLIP-FH pipeline.
* All log files must be placed in the expected folder structure:

  * `train_logs/stage2-v*/`
  * `train_logs/stage3-v*/`
  * `eval_logs/stage2-v*/`
  * `eval_logs/stage3-v*/`

---

## 📬 Contact

For questions or contributions, contact: [babupallam@gmail.com](mailto:babupallam@gmail.com)

```

---
