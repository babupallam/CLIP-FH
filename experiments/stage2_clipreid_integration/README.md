**Implementation of experiments/stage2_clipreid_integration/train_stage2_joint.py**

========================  
**train_joint(cfg_path)**  
  - **Load Config**:  
    - Reads the YAML config file (`cfg_path`) via `yaml.safe_load(f)`.  
    - Stores hyperparameters like `epochs_prompt`, `epochs_image`, `lr`, and `n_ctx`.  
  - **Build log file path**:  
    - Uses `build_filename(cfg, cfg.get("epochs_image"), stage="image", ...)` to create a systematic `.log` filename and path.  
    - Calls `setup_logger(log_path)` to initialize the logger.  
  - **Setup**:  
    - Chooses `device = "cuda"` if available, else `"cpu"`.  
    - Reads `model_type` (e.g. `"ViT-B/32"`) and `stage_mode` (e.g. `"prompt_then_image"`) from `cfg`.  
    - Retrieves `epochs_prompt`, `epochs_image`, and `lr` from the config.  
  - **Load Model & Data**:  
    - Calls `load_clip_with_patch(model_type, device, freeze_all=True)` to get a `clip_model`.  
      - By default, everything in CLIP is frozen.  
    - Calls `get_train_val_loaders(cfg)` to get `train_loader`, `val_loader`, and `num_classes`.  
      - `train_loader` and `val_loader` contain images + labels, with possible query/gallery splits for ReID.  
    - Retrieves the `class_to_idx` mapping and sorts it to build a list of `classnames` in label order.  
  - **Prompt Learner Creation**:  
    - Constructs `PromptLearner(...)` using:  
      - the `classnames`,  
      - the `clip_model`,  
      - `n_ctx` (number of context tokens),  
      - `prompt_template` (e.g. `"{aspect} hand of a person"`),  
      - `device`.  
    - This module learns prompt embeddings for each class label (often a textual context).  
  - **Infer feature dimension**:  
    - Uses a **dummy forward pass** (with a random image) through `clip_model.encode_image(dummy_input)` to get the shape `[1, feat_dim]`.  
    - Stores `feat_dim` for BNNeck or ArcFace additions.  
  - **Register BNNeck & ArcFace**:  
    - `register_bnneck_and_arcface(...)` modifies the `clip_model` to include:  
      - A `bottleneck` layer (BNNeck)  
      - An `arcface` layer for classification, using `feat_dim` → `num_classes`  
  - **Optimizer & Scheduler**:  
    - Constructs an Adam optimizer for all trainable parameters (prompt learner + any unfrozen parts of `clip_model`).  
    - Uses a `CosineAnnealingLR(optimizer, T_max=epochs_prompt + epochs_image, eta_min=1e-6)` for learning-rate scheduling.  
    - Builds the combined `loss_fn` via `build_loss(cfg["loss_list"], ...)`, typically for contrastive or classification objectives.  
    - Also creates separate references to `CrossEntropyLoss()` and `TripletLoss(margin=0.3)` as `ce_loss` and `triplet_loss`.  
  - **Stage Logic**:  
    - Checks `stage_mode` to decide the training sequence:  
      1. **"prompt_then_image"** (default):  
         - Trains prompt embeddings first (prompt stage), then trains the image encoder stage.  
      2. **"prompt_only"**:  
         - Trains only the prompt learner, skipping image fine-tuning.  
      3. **"image_only"**:  
         - Skips prompt training, fine-tunes only the image encoder.  
    - **train_clipreid_prompt_stage(...)** is called if we do `prompt_only` or `prompt_then_image`.  
      - After that, optionally run the **validate_stage1_prompts(...)** check (commented out for now).  
    - **train_clipreid_image_stage(...)** is called if `image_only` or after the prompt stage in `prompt_then_image`.  
      - This stage updates the image encoder + BNNeck + ArcFace.  
  - **Command-line Execution**:  
    - If script is run directly (`__main__`), it parses `--config` argument and calls `train_joint(args.config)`.

---

**Implementation of experiments/stage2_clipreid_integration/train_clipreid_stages.py**

========================  
**train_clipreid_prompt_stage(clip_model, prompt_learner, optimizer, scheduler, train_loader, cfg, device, logger)**  
  - **Freeze Entire CLIP**:  
    - Calls `freeze_entire_clip_model(clip_model, logger.info)` so that only the prompt embeddings are learnable.  
  - **Cache Frozen Image Features**:  
    - `cache_image_features(clip_model, train_loader, device)` → extracts image embeddings once, storing them in `image_feats` and `labels`.  
    - This is more efficient for prompt tuning because we no longer do a full forward pass for each step.  
  - **Set Model Modes**:  
    - `clip_model.eval()` so it remains in inference mode.  
    - `prompt_learner.train()` is set to training mode.  
  - **Epoch Loop**: `for epoch in range(cfg["epochs_prompt"])`  
    1. **Shuffle Indices**: Creates a random permutation of the cached features.  
    2. **Mini-batch Loop**:  
       - Slices the shuffled indices in `cfg["batch_size"]` chunks.  
       - **Prompt Forward**: `prompt_learner.forward_batch(batch_labels)` to create text embeddings.  
         - Then adds `clip_model.positional_embedding`, passes them through `clip_model.transformer`, normalizes to get `text_feats`.  
       - **Contrastive Loss**:  
         - `loss_i2t` = `loss_fn(..., mode="contrastive")` comparing image→text embeddings.  
         - `loss_t2i` = similarly for text→image.  
         - Adds a small L2 reg on prompt embeddings: `0.001 * (prompt_learner.ctx**2).mean()`.  
       - **Backprop**: `optimizer.zero_grad() → loss.backward() → optimizer.step()`.  
       - Tracks running `loss` and logs parameters like `prompt_norm`, `prompt_var`, `prompt_grad`.  
    3. **Epoch summary**:  
       - Logs the average prompt loss for that epoch.  
       - *Optionally* can save a “BEST Prompt” checkpoint and do `scheduler.step()`. (Currently commented out.)  
  - **Final Prompt Model Save**:  
    - Saves `_FINAL.pth` checkpoint with `save_checkpoint(...)`.  

========================  
**train_clipreid_image_stage(clip_model, prompt_learner, optimizer, scheduler, train_loader, val_loader, cfg, device, logger, loss_fn, ce_loss, triplet_loss)**  
  - **Unfreeze CLIP Text Encoder**:  
    - Calls `unfreeze_clip_text_encoder(clip_model, logger.info)`, so now both image + text could be trainable if desired.  
  - **Optional: Freeze Prompt Learner**:  
    - If `cfg.get("freeze_prompt", True)`, calls `freeze_prompt_learner(prompt_learner, logger.info)` so prompt embeddings remain fixed.  
  - **Add Center Loss**:  
    - Creates a `CenterLoss(num_classes=cfg["num_classes"], feat_dim=feat_dim, device=device)` to refine feature centers.  
    - Also an `optimizer_center` (SGD) for center-criterion parameters.  
  - **Epoch Loop**: `for epoch in range(cfg["epochs_image"])`  
    - Sets both `prompt_learner.train()` and `clip_model.train()`.  
    - **Batch Loop** over `train_loader`:  
      - Moves `(images, labels_batch)` to `device`.  
      - **Image Features**: `clip_model.encode_image(images) → L2 normalize them`.  
      - **Prompt Forward**: Prompt embeddings from `prompt_learner.forward_batch(labels_batch)` + `clip_model.positional_embedding`, run through `clip_model.transformer`, get normalized `text_feats`.  
      - **Contrastive Loss**:  
        - `loss_i2t = loss_fn(..., mode="contrastive")`  
        - `loss_t2i = ...` similarly for text→image  
      - **BNNeck & ArcFace**:  
        - `feats_bn = clip_model.bottleneck(image_feats)` → BNNeck stage  
        - `arc_logits = clip_model.arcface(feats_bn, labels_batch)`  
        - Optionally logs `feat_norm` (mean norm of BNNeck features) and `arc_conf` (avg max softmax confidence).  
      - **Center Loss**:  
        - `center_loss_val = center_criterion(feats_bn, labels_batch)`  
      - **ID Loss**:  
        - `id_loss = ce_loss(arc_logits, labels_batch)`  
      - **Triplet Loss**:  
        - `tri_loss = triplet_loss(image_feats, labels_batch)`  
      - **Sum Loss**:  
        - `loss = id_loss + tri_loss + loss_i2t + loss_t2i + cfg["center_loss_weight"] * center_loss_val`  
      - **Backprop**:  
        - Zero both `optimizer` and `optimizer_center`.  
        - `loss.backward()`, scale center criterion grads by `(1 / center_loss_weight)`.  
        - `optimizer.step()`.  
      - Logs the batch-level stats in tqdm: `loss`, `id`, `tri`, `cen`, `feat_n`, etc.  
    - **After each epoch**:  
      - Logs current LR: `scheduler.get_last_lr()[0]`.  
      - **Run Validation**: calls `validate(clip_model, prompt_learner, val_loader, device, logger.info, cfg)`.  
        - Gets `rank1`, `mAP`, etc.  
      - **Check Best Rank-1**: if improved, saves `_BEST.pth` checkpoint.  
      - Then calls `scheduler.step()`.  
  - **Save Final Checkpoint**:  
    - After finishing all `epochs_image`, saves `_FINAL.pth` with `save_checkpoint(...)`.

---

**Implementation of experiments/stage2_clipreid_integration/save_load_models.py**

This file contains two main functions for **checkpoint management**:

========================  
**save_checkpoint(...)**  
  - Gathers everything into a dictionary:  
    - `model_state_dict`  
    - `optimizer_state_dict`  
    - `classifier_state_dict` (if `classifier` is provided)  
    - `scheduler_state_dict` (if scheduler is provided)  
    - `metadata` about the model architecture, config, hash, etc.  
    - `rng_state` (PyTorch + CUDA RNG) for reproducibility  
    - `train_loss`, `val_metrics`, etc.  
  - If `prompt_learner` is given, also saves a separate `..._prompt.pth` state_dict for that learner.  
  - Prints the tag “BEST” or “FINAL” after saving to indicate checkpoint type.  

========================  
**load_checkpoint(...)**  
  - Loads the `.pth` checkpoint from disk, maps it to the chosen `device`.  
  - Restores:  
    - `model` weights, classifier weights, optimizer states, scheduler states if present  
    - `trainable_flags` to ensure the same `.requires_grad` setup as when saved  
    - RNG states if included (both CPU and CUDA)  
    - `metadata` and `epoch` for logging.  
  - Warns if some keys are missing/unexpected.  

---

**Implementation of experiments/stage2_clipreid_integration/eval_utils.py**

This file defines **ReID-style** validation and optional **prompt-based** validation:

========================  
**validate(model, prompt_learner, val_loader, device, log, val_type="reid", batch_size=64, loss_fn=None)**  
  - **model.eval()** & `prompt_learner.eval()` if provided.  
  - **Feature Extraction**:  
    - Loops over `val_loader`, encodes images via `model.encode_image`, normalizes them.  
    - Stores them in `all_feats`, `all_labels`.  
  - **Split Query/Gallery**:  
    - Derives `query_len` from `val_loader.dataset.datasets[0]`, `gallery_len` from `.datasets[1]`.  
    - Splits `all_feats` accordingly into `query_feats`, `gallery_feats`.  
    - Similarly for labels.  
  - **Similarity Matrix**:  
    - `sim_matrix = query_feats @ gallery_feats.T`.  
  - **ReID Metrics**:  
    - For each query, sorts gallery by descending similarity, compares top ranks for matching ID.  
    - Accumulates rank positions in `cmc`, computes `average_precision_score` for each query → `aps`.  
    - **rank1** = `100*cmc[0]`, rank5 = `100*cmc[4]`, etc.  
    - **mAP** = `mean(aps)*100`.  
  - Prints final ranks and mAP in a structured format.  
  - Returns `{"rank1": ..., "rank5": ..., "rank10": ..., "mAP": ...}`.  

========================  
**validate_stage1_prompts(model, prompt_learner, val_loader, device, log=print, batch_size=64)**  
  - Similar ReID evaluation but compares **image_features** to **prompt-generated text embeddings**.  
  - Creates text embeddings for each class ID in bulk.  
  - Extracts image embeddings from `val_loader`.  
  - Builds a `sim_matrix = image_feats @ text_feats.T`.  
  - Computes rank-based results and mAP.  
  - Logs final rank1, mAP, etc.  

========================  
**validate_promptsg(model_components, val_loader, device, compose_prompt, loss_fn=None)**  
  - Another specialized validation routine for *PromptSG* approach.  
  - Involves an `inversion_model`, a `multimodal_module`, a `classifier`, etc.  
  - Logs `avg_val_loss`, `top1_accuracy`, `top5_accuracy`, `top10_accuracy`, `mAP`.  
  - This is used if you’re exploring more advanced synergy between text prompts + visual embeddings.  

---

### **Key Printed/Logged Parameters & Their Meaning**  
Throughout these stage 2 scripts, you’ll see references to:  

- **`rank1`, `rank5`, `rank10`, `mAP`**:  
  - Standard ReID metrics.  
  - `rank1` → percentage of queries whose correct match is at position 1 in the sorted gallery.  
  - `mAP` → mean Average Precision, capturing overall retrieval performance.  
- **`loss`, `id_loss`, `tri_loss`, `center_loss`, `loss_i2t`, `loss_t2i`**:  
  - Various loss components. Summed up for the total training objective.  
- **`feat_norm`, `arc_conf`**:  
  - Diagnostics in `train_clipreid_image_stage`: average norm of BNNeck features, average confidence from ArcFace classifier.  
- **`prompt_norm`, `prompt_var`, `prompt_grad`**:  
  - Diagnostics in the prompt stage, describing how learned prompt embeddings behave.  
- **`logits std`** (from the stage1 approach) can also appear if extended logging is used.  

All these logs help track training stability and model performance across epochs.

---

**Conclusion**  
In **stage2** (CLIP-ReID integration), you combine **prompt learning** + **image encoder fine-tuning** with optional BNNeck, ArcFace, and CenterLoss. The code:

1. **Freezes** or **unfreezes** relevant parts of CLIP,  
2. **Learns** prompt embeddings for each class,  
3. **Fine-tunes** the image branch with advanced ReID losses (contrastive, ArcFace, center, triplet),  
4. **Evaluates** using standard ReID metrics (rank1, rank5, rank10, mAP).  

Everything is **modular** and **checkpoint-friendly** (via `save_load_models.py`). You can **incrementally** experiment with prompt tuning alone, image tuning alone, or both sequentially.