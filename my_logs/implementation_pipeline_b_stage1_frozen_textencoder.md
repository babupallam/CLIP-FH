# **Stage 1 Fine-Tuning + Evaluation Flow**

## **File & Folder Overview**

1. **Training** (Stage 1):
   - `train_stage1_frozen_text_vitb16_11k_dorsal_r.py`  
   - `build_dataloader.py`  
   - `make_model.py`  
   - `finetune_trainer_stage1.py`  
   - A config file, e.g., `configs/finetuning_stage1_frozen_text/train_vitb16_11k_dorsal_r.yml` (for training)

2. **Evaluation** (Stage 1):
   - `run_all_finetuned_stage1.py` (a batch-run script)
   - `run_eval_clip.py` (the universal evaluation script)
   - A config file, e.g., `configs/finetuning_stage1_frozen_text/eval_vitb16_11k_dorsal_r.yml` (for evaluation)
   - Possibly more `.yml` files for other variants

---

## **A) Stage 1 Training Workflow**

### **1) `train_stage1_frozen_text_vitb16_11k_dorsal_r.py`**

1. **Parses `--config <path>`**:
   - e.g. `configs/finetuning_stage1_frozen_text/train_vitb16_11k_dorsal_r.yml`
2. **Loads YAML** with `yaml.safe_load()`
   - Typically has keys like:
     ```yaml
     dataset: 11k
     aspect: dorsal_r
     model: vitb16
     epochs: 10
     lr: 0.0001
     batch_size: 64
     save_path: "saved_models/stage1_vitb16_11k_dorsal_r.pth"
     log_path: "logs/stage1_vitb16_11k_dorsal_r.log"
     ```
3. **Builds Dataloaders** using:
   ```python
   train_loader, _, num_classes = get_train_val_loaders(config)
   config["num_classes"] = num_classes
   ```
4. **Builds Model** by calling:
   ```python
   clip_model, classifier = build_model(config, freeze_text=True)
   ```
   - `freeze_text=True` means the CLIP text transformer’s parameters are locked.

5. **Init** the trainer:
   ```python
   trainer = FinetuneTrainerStage1(clip_model, classifier, train_loader, config, device)
   trainer.train()
   ```

**End**: Produces a trained model (image encoder + classifier) saved to `save_path`.

---

### **2) `build_dataloader.py`**

- **Function** `get_train_val_loaders(config)`:
  1. Checks `config["dataset"]` & `config["aspect"]` → builds path like `./datasets/11khands/train_val_test_split_dorsal_r`
  2. Creates `train_dir` = `.../train`, `val_dir` = `.../val`
  3. Uses `ImageFolder(...)` with transforms to load images
  4. Returns `(train_loader, val_loader, num_classes)`

**Used** by the training script to supply images for fine-tuning.

---

### **3) `make_model.py -> build_model(config, freeze_text=True)`**

1. Loads CLIP:
   ```python
   model_name = config["model"]  # "vitb16" => use "ViT-B/16"
   clip_model, _ = clip.load("ViT-B/16", device=device)
   ```
2. If `freeze_text=True`, loops over `clip_model.transformer.parameters()` and sets `p.requires_grad = False`.
3. Creates a classification head:
   ```python
   image_embed_dim = clip_model.visual.output_dim
   num_classes = config["num_classes"]
   classifier = nn.Linear(image_embed_dim, num_classes)
   ```
4. Returns `(clip_model, classifier)`.

---

### **4) `finetune_trainer_stage1.py -> FinetuneTrainerStage1`**

**`__init__`**:
- Receives `(clip_model, classifier, train_loader, config, device)`
- Extracts hyperparams (`epochs`, `lr`, `save_path`, `log_path`)
- Sets up a **CrossEntropyLoss** for classification:
  ```python
  self.criterion = nn.CrossEntropyLoss()
  ```
- Builds an optimizer that updates:
  - `clip_model.visual.parameters()` (image encoder)
  - `classifier.parameters()` (classification head)

**`train()`**:
1. **Loop** from `epoch=1` to `self.epochs`.
2. For each batch `(images, labels)` in `train_loader`:
   - `optimizer.zero_grad()`
   - `features = self.clip_model.encode_image(images)`  
     (if we use uses `no_grad()`, so effectively we only backprop from the classifier onward; partial fine-tuning if we want the entire image encoder to learn, we might remove `no_grad()`)
   - `outputs = self.classifier(features)`
   - `loss = self.criterion(outputs, labels)`
   - `loss.backward()`
   - `optimizer.step()`
   - Track accuracy, accumulate logs
3. After final epoch:
   - `torch.save(self.clip_model.state_dict(), self.save_path)`
   - Save logs to `self.log_path`.

**End**: The Stage 1 fine-tuned model is ready.

---

## **B) Stage 1 Evaluation Workflow**

### **1) `run_all_finetuned_stage1.py`**

1. Searches `configs/finetuning_stage1_frozen_text` for `.yml` files — e.g. `eval_vitb16_11k_dorsal_r.yml`.
2. For each config:
   - Runs `python experiments/run_eval_clip.py --config <that_config>`
3. This effectively performs a batch run of your new “finetuned” evaluation scenario.

---

### **2) `eval_vitb16_11k_dorsal_r.yml`** (Inside `configs/finetuning_stage1_frozen_text/`)

Sample content:
```yaml
dataset: 11k
aspect: dorsal_r
model: vitb16
variant: finetuned
batch_size: 32
num_splits: 10
save_path: "saved_models/stage1_vitb16_11k_dorsal_r.pth"
...
```
- Tells `run_eval_clip.py` that we’re evaluating a **finetuned** model vs. the baseline.

---

### **3) `run_eval_clip.py`** (Universal evaluation script)

1. **Parses** `--config`.
2. **Loads** config → sees `variant: finetuned`.
3. **Loads** the corresponding **finetuned** model checkpoint from `save_path` (the same path used in training).
   - E.g., 
     ```python
     clip_model, _ = clip.load(...)
     clip_model.load_state_dict(torch.load(config["save_path"]))
     ```
4. **Loads** query & gallery splits via your typical `get_dataloader(...)` approach.
5. **Extract Features** for query & gallery:
   - Possibly uses a function `extract_features(clip_model, data_loader, device)` 
     (similar to baseline_inference).
6. **Compute** similarity & ReID metrics across multiple splits.
7. **Print** final rank-1, rank-5, rank-10, mAP, etc.

---

## **Putting It All Together**

1. **Train** (Stage 1)  
   - `python experiments/train_stage1_frozen_text_vitb16_11k_dorsal_r.py --config configs/finetuning_stage1_frozen_text/train_vitb16_11k_dorsal_r.yml`
   - → Produces `saved_models/stage1_vitb16_11k_dorsal_r.pth`
2. **Evaluate**  
   - `python run_all_finetuned_stage1.py` → Looks at `configs/finetuning_stage1_frozen_text/*.yml`  
     - For each `eval_*.yml`, calls `run_eval_clip.py --config <that_eval_yml>`  
   - `run_eval_clip.py` sees `variant=finetuned`, loads the checkpoint → measures new performance.

---

### **Final Clarification**: “No Steps Missed”
- We have included:
  1. The training script (`train_stage1_frozen_text_vitb16_11k_dorsal_r.py`)
  2. The trainer code (`finetune_trainer_stage1.py`)
  3. The model builder (`make_model.py`)
  4. The data loaders (`build_dataloader.py`)
  5. The evaluation approach:
     - `run_all_finetuned_stage1.py` (for batch runs)
     - `eval_vitb16_11k_dorsal_r.yml` config
     - `run_eval_clip.py` logic (universal script for baseline or finetuned)
- Each part references one another to **train** or **evaluate** the Stage 1 approach: “frozen text encoder + cross-entropy on the image encoder.”
