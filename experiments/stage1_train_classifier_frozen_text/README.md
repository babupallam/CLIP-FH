## **Implementation of `experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py`**

---

### `main()`

#### 🔹 Step 1: Configuration Setup
- Uses `config.get("", default_value)` to load hyperparameters, dataset details, paths, etc.
- This enables flexibility and modular control over experiments.

#### 🔹 Step 2: Output Naming
- Calls `build_filename()` to create a unique name for logs and model checkpoints.
- Helps manage multiple runs and track performance across experiments.

#### 🔹 Step 3: Device Selection
- Chooses `cuda` if available, else `cpu`, using `torch.device(...)`.
- Ensures model runs on GPU when possible for faster training.

#### 🔹 Step 4: Load Datasets and DataLoaders
- Uses `get_train_val_loaders(config)` to return:
  - `train_loader`, `val_loader`, `num_classes`
  - Loads from dataset directory split into `train/`, `query0/`, and `gallery0/`
  
  ##### Substep: `transform()`
  - Applies `Resize(224, 224)` and `ToTensor()` to prepare image inputs for CLIP.
  - Keeps input format consistent with CLIP’s pretraining resolution.

---

### `build_model()` (called inside main)

#### 🔹 CLIP Model Creation
- Loads pre-trained CLIP model using `clip.load(model_variant)` from OpenAI’s library.
- **Why CLIP model?**
  - Used for extracting robust image features via the `visual` encoder.
  - Text encoder is **frozen** in this stage (Stage 1).

#### 🔹 Classifier Creation
- A simple linear classifier: `nn.Linear(image_embed_dim, num_classes)`
- **Why classifier?**
  - Trains to distinguish identities using CLIP image features.
  - Works with `CrossEntropyLoss` to learn discriminative features.

---

### Trainer Setup

#### 🔹 Step: Instantiate Trainer Class
- `trainer = FinetuneTrainerStage1(...)` from `engine/train_classifier_stage1.py`
- Packages the full training pipeline: data, model, optimizer, loss, and logging.

---

### `train()` Method Inside `FinetuneTrainerStage1`

---

### `__init__()` (Constructor)

#### 🔹 Assign Data & Model Components
- Stores `clip_model`, `classifier`, `train_loader`, `val_loader`, `device`, etc.

#### 🔹 Loss Functions
- `CrossEntropyLoss()` for classification.
- `TripletLoss(margin=0.3)` for embedding separation.

#### 🔹 Early Stopping Parameter
- `self.early_stop_patience` controls how many epochs to wait before early stopping if no Rank-1 improvement.

#### 🔹 Optimizer Setup
- Uses Adam optimizer with **differential learning rates**:
  - `"params": clip_model.visual.parameters(), "lr": self.lr`
    - Why? Image encoder is pretrained → needs a moderate learning rate.
  - `"params": classifier.parameters(), "lr": self.lr * 0.1`
    - Why? Classifier is small and new → needs a smaller rate to avoid overfitting.

#### 🔹 Logging Setup
- Creates log files and CSV metrics in the specified directory.

---

### `train()` (Training Loop)

#### 🔹 Start Training Mode
- Sets the model to training mode: `clip_model.train()`

#### 🔹 Epoch Loop
- For each epoch:
  - Initializes timers and accumulators for loss and accuracy.

#### 🔹 Batch Loop
- For each batch in `train_loader`:

  ##### Substep: Forward Pass
  - Move `images`, `labels` to GPU/CPU.
  - Reset gradients: `optimizer.zero_grad()`
  - Encode images: `clip_model.encode_image(images)`
  - Run classifier: `outputs = classifier(features)`

  ##### Substep: Loss Calculation
  - `ce = CrossEntropyLoss(outputs, labels)`
  - `tri = TripletLoss(features, labels)`
  - Combine: `loss = ce + tri`

  ##### Substep: Backward Pass & Update
  - `loss.backward()` for gradient computation.
  - Clip gradients using `clip_grad_norm_` to stabilize training.
  - `optimizer.step()` to update parameters.

  ##### Substep: Metric Logging
  - Compute top-k (Rank-1, 5, 10) accuracies using `outputs.topk()`.
  - Accumulate total loss and metrics for epoch reporting.

#### 🔹 Epoch End
- Compute average loss and accuracy.
- Log epoch metrics using `self.logger`.

---

### Validation

#### 🔹 Run ReID-style Validation
- Combine `query0 + gallery0` in validation loader.
- `validate()` returns:
  - Rank-1, Rank-5, Rank-10, mAP, and optionally losses.
- Uses same model and classifier to test retrieval capability.

---

### Model Checkpointing

#### 🔹 Save Best Checkpoint
- If current epoch improves `Rank-1`, save `_BEST.pth` model.
- Resets `no_improve_epochs` counter.

#### 🔹 Early Stopping
- If no improvement for `early_stop_patience`, break training loop.

---

### Final Save

#### 🔹 Save Final Checkpoint
- After training ends (early or full), save final model as `_FINAL.pth`.

---
