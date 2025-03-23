
# ✅ 1️⃣ Fine-Tuning CLIP (OpenCLIP) - Comprehensive Guide
---

### ✅ What is OpenCLIP?
OpenCLIP is an open-source implementation of CLIP with:
- Support for **different backbones** (ViT, ResNet).
- **Pretrained weights** from LAION datasets.
- **Custom training and fine-tuning** workflows.
- **More flexible configuration** and **PyTorch-friendly** codebase.

GitHub: [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

---

## ✅ Step 1: Install OpenCLIP
```bash
pip install open_clip_torch
```

---

## ✅ Step 2: Load Pretrained OpenCLIP Models
```python
import open_clip
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a pretrained OpenCLIP model (ViT-B/32)
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",   # Choose backbone (ViT-B-32, ViT-L-14, RN50, etc.)
    pretrained="laion2b_s34b_b79k",  # Choose pretrained weights
    device=device
)

# Load tokenizer for text inputs
tokenizer = open_clip.get_tokenizer("ViT-B-32")
```

---

## ✅ Step 3: Dataset Preparation
### Image Dataset (Fine-tuning with labels)
```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Dataset folder should have structure: root/class_name/images.jpg
dataset = ImageFolder(root="path/to/dataset", transform=preprocess)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### Text Data (Optional for zero-shot or multimodal fine-tuning)
```python
texts = ["a photo of a cat", "a picture of a dog"]
text_tokens = tokenizer(texts)
```

---

## ✅ Step 4: Understand the Model Structure
```python
print(model)  # Shows Vision and Text Encoders, Projections, etc.
```

Key components in OpenCLIP:

| ✅ Component       | ✅ Description                                               |
|--------------------|-------------------------------------------------------------|
| `model.visual`     | Vision encoder (ViT or ResNet)                              |
| `model.transformer`| Text encoder (Transformer)                                  |
| `model.logit_scale`| Learnable scale for contrastive logits                     |
| `model.text_projection` | Linear layer projecting text embeddings                |
| `model.visual.proj` | Linear layer projecting image embeddings                  |

---

## ✅ Step 5: Fine-Tuning Strategies  


> Fine-tuning lets you adapt CLIP to **new datasets** or **tasks** by training some or all parts of the model.

There are **different ways** to fine-tune CLIP. Each strategy gives you **control over which parts of the model** to train and which to keep frozen.  
Freezing layers means **not updating their weights** during training, which saves memory and prevents overfitting.

---

### 🔷 **Strategy 1: Full Fine-Tuning (Train Everything)**  
✅ You **train the entire CLIP model**, both **vision** and **text encoders**, and the **projection heads**.  
✅ This gives the **best flexibility**, but it’s **expensive** in terms of memory and time.  
✅ Recommended when:
- You have a **large dataset**.
- Your task is **very different** from CLIP's pretraining (e.g., medical images).

```python
# Unfreeze (train) all layers
for param in model.parameters():
    param.requires_grad = True
```

---

### 🔷 **Strategy 2: Freeze Vision Encoder (Train Text Encoder Only)**  
✅ You **freeze the image encoder**, so it **doesn't change** during training.  
✅ You **only train** the **text encoder** and projection heads.  
✅ Useful when:
- Your **image data is similar** to CLIP’s pretraining, but you have **new text prompts or labels** to fine-tune.

```python
# Freeze vision encoder
for param in model.visual.parameters():
    param.requires_grad = False

# Train text encoder
for param in model.transformer.parameters():
    param.requires_grad = True
```

---

### 🔷 **Strategy 3: Freeze Text Encoder (Train Vision Encoder Only)**  
✅ You **freeze the text encoder** and **train** only the **vision encoder** and projection heads.  
✅ Useful when:
- You have **specialized image data**, but you’re **happy with the existing text representations**.

```python
# Train vision encoder
for param in model.visual.parameters():
    param.requires_grad = True

# Freeze text encoder
for param in model.transformer.parameters():
    param.requires_grad = False
```

---

### 🔷 **Strategy 4: Freeze All Except Projection Heads**  
✅ You **freeze everything**, except the **projection layers** (and optionally `logit_scale`).  
✅ Only the **linear layers** that map image/text features to the shared space are trained.  
✅ Fast and lightweight fine-tuning, useful when:
- You have **limited data**.
- You only want to **adapt the similarity space** for a new task.

```python
# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Train projection layers
model.visual.proj.requires_grad = True
model.text_projection.requires_grad = True
model.logit_scale.requires_grad = True
```

---

### 🔷 **Strategy 5: Unfreeze Last N Layers (Partially Train Vision/Text)**  
✅ You freeze **most** of the model but **unfreeze the last few transformer blocks** to allow **some adaptation**.  
✅ Less training time and memory than full fine-tuning.  
✅ Useful when:
- You want to **preserve CLIP's knowledge** but make **small adjustments** for your data.

```python
# Unfreeze the last 2 layers of the vision transformer
num_layers = len(model.visual.transformer.resblocks)

for i, block in enumerate(model.visual.transformer.resblocks):
    requires_grad = (i >= num_layers - 2)  # last 2 layers
    for param in block.parameters():
        param.requires_grad = requires_grad
```

You can do the same for **text encoder**:
```python
num_layers = len(model.transformer.resblocks)

for i, block in enumerate(model.transformer.resblocks):
    requires_grad = (i >= num_layers - 2)  # last 2 layers
    for param in block.parameters():
        param.requires_grad = requires_grad
```

---

### 🔷 **Strategy 6: Prompt Tuning (Train Only Learnable Prompts)**  
✅ Instead of changing the text encoder, you add **learnable prompt embeddings** to the input tokens.  
✅ You **freeze the entire CLIP model**, but **train prompt vectors**.  
✅ Efficient! Few parameters are trained.  
✅ Useful for **text-to-image retrieval** or **zero-shot tasks**.  
✅ Requires **manual implementation** in OpenCLIP.

Simple idea:
```python
# Create a learnable prompt embedding
prompt_embedding = nn.Parameter(torch.randn(1, prompt_length, embed_dim)).to(device)

# Concatenate prompt_embedding with tokenized text embeddings
# Then pass into the frozen text encoder...
```

---

### 🔷 **Strategy 7: Adapter Layers (LoRA / Lightweight Fine-Tuning)**  
✅ Insert **small adapter modules** inside the transformer blocks.  
✅ You **freeze most of CLIP**, but train **small new layers**.  
✅ Great for **resource-constrained** environments.  
✅ Requires **modifying OpenCLIP model structure**.

Concept:
- Add a **low-rank adapter** after attention or MLP layers.
- Only the **adapter weights** are updated.

---

### 🔷 **Strategy 8: Differential Learning Rates**  
✅ Use **different learning rates** for different parts of the model:  
- Higher LR for **new layers** (projections/classifiers).  
- Lower LR for **frozen or partially frozen layers**.  
✅ Balances **stability** and **flexibility**.

Example:
```python
optimizer = torch.optim.AdamW([
    {'params': model.visual.proj, 'lr': 1e-4},
    {'params': model.text_projection, 'lr': 1e-4},
    {'params': model.visual.parameters(), 'lr': 1e-5}
])
```

---

### 🔷 **Strategy 9: Train Only Logit Scale**  
✅ You **freeze everything** except `logit_scale`, which controls the sharpness of similarity scores.  
✅ Very **lightweight** but allows **fine-grained control** over similarity during retrieval/classification.

```python
for param in model.parameters():
    param.requires_grad = False

model.logit_scale.requires_grad = True
```

---

## ✅ Summary Table of Fine-Tuning Strategies

| **Strategy**                               | **Trainable Components**                | **When to Use**                                |
|--------------------------------------------|-----------------------------------------|------------------------------------------------|
| Full Fine-Tune                             | Everything                              | Large datasets, major task/domain shift       |
| Freeze Vision Encoder                      | Text encoder + projections              | New text prompts, labels                      |
| Freeze Text Encoder                        | Vision encoder + projections            | New image domain                              |
| Freeze All Except Projection Heads         | Only projections + logit scale          | Limited data, simple adaptation               |
| Unfreeze Last N Layers                     | Last N layers in vision/text transformer| Efficient partial fine-tuning                 |
| Prompt Tuning                              | Only prompt embeddings                  | Efficient zero-shot or prompt-based tuning    |
| Adapter Layers (LoRA)                      | Adapter modules in transformer blocks   | Low-resource fine-tuning                      |
| Differential Learning Rates                | Custom per-layer learning rates         | Control learning stability and flexibility    |
| Train Only Logit Scale                     | logit_scale                             | Fine control over similarity calculation      |

---

✅ **You can combine strategies** (e.g., **prompt tuning + logit_scale**, or **projection heads + last N layers**).  
✅ The **right strategy** depends on:
- **Dataset size**
- **Task type** (classification, retrieval, etc.)
- **Hardware constraints**

---

---

## ✅ Step 6: Replace Projection Layers (Optional)
```python
import torch.nn as nn

# Replace Image Projection Layer (Example: New Dim = 256)
new_proj_dim = 256
model.visual.proj = nn.Parameter(torch.randn(new_proj_dim, model.visual.proj.shape[1]) * 0.02).to(device)

# Replace Text Projection Layer
model.text_projection = nn.Parameter(torch.randn(new_proj_dim, model.text_projection.shape[1]) * 0.02).to(device)
```

---

## ✅ Step 7: Optimizer Setup
```python
import torch.optim as optim

# Collect all trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]

# Choose optimizer
optimizer = optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.01)
```

---

## ✅ Step 8: Loss Function (Contrastive / Classification)
### 🔹 Contrastive Learning (Image ↔ Text Similarity)
```python
import torch.nn.functional as F

def clip_contrastive_loss(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    ground_truth = torch.arange(len(logits_per_image)).to(device)
    
    loss_i = F.cross_entropy(logits_per_image, ground_truth)
    loss_t = F.cross_entropy(logits_per_text, ground_truth)
    
    return (loss_i + loss_t) / 2
```

### 🔹 Classification Head (Custom Tasks)
You can add a classifier on top of the image embeddings:
```python
classifier = nn.Linear(model.visual.proj.shape[0], num_classes).to(device)

# Example forward pass
image_features = model.encode_image(images)
logits = classifier(image_features)
loss = F.cross_entropy(logits, labels)
```

---

## ✅ Step 9: Training Loop
```python
epochs = 5

for epoch in range(epochs):
    model.train()
    for images, labels in dataloader:
        images = images.to(device)
        
        # Forward pass
        image_features = model.encode_image(images)
        
        # Normalize (unit length)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Example: Classification
        logits = classifier(image_features)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

---

## ✅ Step 10: Save / Load Fine-Tuned Model
```python
# Save model checkpoint
torch.save(model.state_dict(), "finetuned_openclip.pth")

# Load model checkpoint later
model.load_state_dict(torch.load("finetuned_openclip.pth"))
model.eval()
```

---

## ✅ Step 11: Advanced Fine-Tuning Strategies (Optional)
### 🔹 Prompt Tuning
- Keep the model frozen
- Learn a **prompt embedding** that adapts to your task
- OpenCLIP doesn’t provide built-in prompt tuning, but you can:
  - Manually add learnable tokens to the input sequence
  - Optimize only those prompt embeddings

### 🔹 Adapter Layers (Low-Rank Adaptation - LoRA)
- Add small trainable layers (adapters) inside the frozen encoder blocks
- Efficient fine-tuning with minimal parameters
- Requires custom modification of the OpenCLIP model structure

---

## ✅ Step 12: Evaluation & Inference
```python
model.eval()

# Inference with images
with torch.no_grad():
    image_features = model.encode_image(images)
    logits = classifier(image_features)
    predictions = torch.argmax(logits, dim=-1)
```

---

## ✅ Summary: Fine-Tuning Options in OpenCLIP

| ✅ Fine-Tuning Approach | ✅ Action                                                  |
|-------------------------|------------------------------------------------------------|
| Full Fine-Tune          | Unfreeze all layers                                        |
| Freeze + Classifier     | Freeze encoder, fine-tune classifier on embeddings         |
| Unfreeze Last N Layers  | Fine-tune final transformer blocks only                    |
| Projection Heads        | Replace / fine-tune projection layers only                 |
| Prompt Tuning           | Learn task-specific prompts (text encoder focus)           |
| LoRA / Adapters         | Insert lightweight adapter layers in vision/text encoders  |

---

## ✅ Visual Summary of Workflow
```
DATASET --> PREPROCESS --> ENCODER (VISUAL/TEXT) --> PROJECTION --> LOSS --> OPTIMIZER --> TRAIN!
```

---

✅ **Comprehensive OpenCLIP Fine-Tuning Cheat Sheet Complete!**  
Let me know if you want:
- A **working code template**  
- **LoRA adapters**  
- **Prompt tuning implementation**  
- **Advanced optimizer schedules (SGD, Cosine Annealing, etc.)**

Ready for the next one? 😄