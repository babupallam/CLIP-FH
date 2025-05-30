# **Baseline (Applying Direct CLIP Over Images)  Detailed Algorithmic Breakdown**

Well assume you have four main files involved for the baseline:

1. **`run_all_experiments.py`**  
2. **`run_eval_clip.py`**  
3. **`engine/baseline_inference.py`** (or similarly named)  
4. **`engine/evaluator.py`** (or a similar evaluator file)

Below is the sequence of calls and how each function operates:

---

## 1. **`run_all_experiments.py`**  
**(Entry Step)**

1. **List config files**  
   - Gathers all `.yml` baseline config files from a directory like `../configs/baseline/`.
2. **Loop** over these configs:
   1. **Spawn** a process (or direct function call) to run `run_eval_clip.py --config <config_file>`.
3. **End**: Once all configs are processed, you have results for each baseline experiment.

**Effect**: This script is basically a batch-run manager that organizes multiple evaluation runs.

---

## 2. **`run_eval_clip.py`**  
**(Core Baseline Evaluation Script)**

1. **Parse CLI Args** (with `argparse`):
   - `--config`  path to a YAML file (like `baseline_vitb16_11k_dorsal_r.yml`).

2. **Load Config** (using `yaml.safe_load`):
   - E.g.: 
     ```python
     with open(args.config, 'r') as f:
         config = yaml.safe_load(f)
     ```
   - Contains keys like:  
     ```yaml
     dataset: 11k
     aspect: dorsal_r
     model: vitb16
     ...
     ```
3. **Call** a main function, e.g. `run_evaluation(config)`:
   1. This function does the heavy lifting.  

### Inside `run_evaluation(config)`:
1. **Extract** relevant fields:
   - `model_type = config["model"]` (e.g., `'vitb16'` or `'rn50'`).
   - `dataset = config["dataset"]` (e.g., `'11k'`).
   - `aspect = config["aspect"]` (e.g., `'dorsal_r'`).
   - Possibly `variant = config["variant"]` to see if its `baseline`, `finetune`, etc.

2. **Load CLIP** (in baseline mode, no fine-tuned weights):
   ```python
   import clip
   model, preprocess = clip.load("ViT-B/16", device=device)
   # no checkpoint loaded, just zero-shot model
   ```

3. **Iterate** over the 10 query/gallery splits (if your dataset is split that way):
   - For `split in range(10):`
     - Build query_dir and gallery_dir.
     - **Load** them via `get_dataloader(query_dir)` and `get_dataloader(gallery_dir)`.

4. **Extract Features** for Query:
   1. `query_features, query_labels = extract_features(model, query_loader, device)`
      - This calls the function from `baseline_inference.py`.

5. **Extract Features** for Gallery:
   1. `gallery_features, gallery_labels = extract_features(model, gallery_loader, device)`

6. **Compute Similarity**:
   1. `sim_matrix = compute_similarity_matrix(query_features, gallery_features)`
      - Also from `baseline_inference.py` (or same file).
      - Typically `sim = query @ gallery.T` or a cos sim approach.

7. **Evaluate** ReID metrics:
   1. `metrics = evaluate_rank(sim_matrix, query_labels, gallery_labels)`
      - Found in `engine/evaluator.py`.
      - Returns e.g.: `{ "rank1": 0.72, "rank5": 0.86, "mAP": 0.79 }`.

8. **Aggregate** across splits:
   - Possibly store `metrics` in a list, then average them at the end.

9. **Print/Save** final results:
   - E.g. Rank-1, Rank-5, Rank-10, mAP average across all splits.

**End** of `run_eval_clip.py`.

---

## 3. **`engine/baseline_inference.py`**  
**(Feature Extraction + Similarity)**

This file usually has 23 key functions:

#### 3.1. `extract_features(model, dataloader, device)`

1. **Initialize** two lists: `features_list`, `labels_list`.
2. **Loop** over `dataloader`:
   1. For each `(images, labels)` batch:
      - Move `images`  `device`.
      - `with torch.no_grad():`
        - `image_features = model.encode_image(images)` or something similar.  
        - Possibly `image_features = F.normalize(image_features, dim=-1)`.
      - Append `image_features.cpu()` and `labels` to the respective lists.
3. **Concatenate** them at the end:
   ```python
   all_feats = torch.cat(features_list, dim=0)
   all_labels = torch.cat(labels_list, dim=0)
   ```
4. **Return** `(all_feats, all_labels)`.

#### 3.2. `compute_similarity_matrix(query_feats, gallery_feats)`

1. Possibly just does:
   ```python
   sim_matrix = query_feats @ gallery_feats.t()
   # or use a cos. sim approach
   return sim_matrix
   ```
2. If needed, normalizes the features first or calls `F.cosine_similarity`.

---

## 4. **`engine/evaluator.py`**  
**(ReID Metric Calculation)**

#### 4.1. `evaluate_rank(sim_matrix, q_labels, g_labels)`
1. For each query:
   - **Sort** gallery indices by descending similarity.
   - Check the rank at which the correct label appears  accumulate rank-1, rank-5, etc.
   - Also compute **mAP** using standard ReID approach (counting how quickly the correct matches appear).
2. Average across all queries  final rank-1, rank-5, rank-10, mAP.
3. Return a dictionary of metrics.

---

## Overall Algorithm Flow

1. **[run_all_experiments.py]**  Finds config files  calls `run_eval_clip.py` for each.
2. **[run_eval_clip.py]**  
   1. Parse config  
   2. Setup CLIP zero-shot model  
   3. For each query/gallery split:  
      - `extract_features(...  baseline_inference.py )`  
      - `compute_similarity_matrix(...  baseline_inference.py )`  
      - `evaluate_rank(...  evaluator.py )`  
   4. Average results  print/save
3. **[baseline_inference.py]** does the low-level feature extraction (calls CLIPs `encode_image`) and similarity.
4. **[evaluator.py]** (or relevant file) does the final ranking metrics.

---
