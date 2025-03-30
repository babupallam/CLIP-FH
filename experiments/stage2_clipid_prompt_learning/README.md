
## 🔄 PromptLearner: Step-by-Step Algorithm

### 1. Initialization (`__init__`)
**Purpose**: Set up learnable class-specific prompts to interface with CLIP's text encoder.

---

#### 1.1 Store Input Parameters
- `classnames`: List of class names (e.g., `["dorsal_hand", "palmar_hand"]`)
- `n_ctx`: Number of context (learnable) tokens
- `ctx_init`: Optional initialization string for context tokens
- `prompt_template`: Template string for prompts (e.g., `"A photo of a {}."`)
- `device`: Target computation device (e.g., `"cuda"`)

---

#### 1.2 Extract and Store CLIP Model Components
- 1.2.1 `dtype`: Extract data type from `clip_model.token_embedding`
- 1.2.2 `ctx_dim`: Set context embedding dimension from `clip_model.ln_final.weight`
- 1.2.3 Store CLIP subcomponents:
  - `self.token_embedding`
  - `self.positional_embedding`
  - `self.context_length`
  - `self.tokenizer` (i.e., `clip.tokenize`)

---

#### 1.3 Generate Class-Specific Prompts
- 1.3.1 Format `prompt_template` with each class name (underscore replaced with space)
- 1.3.2 Tokenize prompts using `clip.tokenize` → shape: `(num_classes, context_length)`
- 1.3.3 Register tokenized prompts as buffer: `self.tokenized_prompts`

---

#### 1.4 Initialize Learnable Context Vectors
- 1.4.1 **If `ctx_init` is provided**:
  - Tokenize `ctx_init`
  - Extract embeddings from positions `[1 : 1 + n_ctx]`
  - Assert that `init_embedding.shape[0] == n_ctx`
- 1.4.2 **Else**:
  - Create random embeddings of shape `(n_ctx, ctx_dim)` using uniform distribution in `[-0.02, 0.02]`
- 1.4.3 Register `ctx_vectors` as trainable parameter: `self.ctx`

---

#### 1.5 Identify Prompt Prefix Length
- 1.5.1 Find the index of the first `"a"` token in the prompt (usually `[SOS] "a"`)
- 1.5.2 Store this index in `self.prefix_len` (used for prompt reconstruction)

---

### 2. Forward Pass (`forward`)
**Purpose**: Construct the final embedded prompts by inserting the learnable context tokens into the fixed template structure.

---

#### 2.1 Expand Context Embeddings for All Classes
- 2.1.1 Expand `self.ctx` from `(n_ctx, dim)` → `(n_cls, n_ctx, dim)` using broadcasting

---

#### 2.2 Retrieve Token Embeddings
- 2.2.1 Apply `self.token_embedding` to `self.tokenized_prompts`
- 2.2.2 Resulting shape: `(n_cls, context_length, dim)`

---

#### 2.3 Separate Prompt Components
- 2.3.1 Extract `prefix`: tokens before the learnable context (e.g., `[SOS]`) → `[:, :1, :]`
- 2.3.2 Extract `suffix`: tokens after the learnable context → `[:, 1 + n_ctx :, :]`

---

#### 2.4 Concatenate Prompt Embeddings
- 2.4.1 Rebuild the full prompt as: `[prefix] + [learnable context] + [suffix]`
- 2.4.2 Final shape: `(n_cls, context_length, dim)`

---

#### 2.5 Return Output
- 2.5.1 Return the constructed `prompts_embedded` tensor

---

### ✅ Final Output
- **Shape**: `(num_classes, context_length, embed_dim)`
- **Description**: Embedded prompts with learnable context, ready for CLIP text encoder

---

***
***
***

## 📄 `load_clip_with_patch`: Step-by-Step Algorithm

### 1. Function Purpose
**Goal**: Load a specific CLIP model variant and optionally freeze all its parameters.

---

### 2. Define Function

#### 2.1 Signature  
```python
def load_clip_with_patch(model_type, device, freeze_all=True):
```
- `model_type`: A string identifier like `'vitb16'`, `'rn50'`, etc.
- `device`: Torch device string (e.g., `"cuda"`, `"cpu"`)
- `freeze_all`: Whether to freeze all model parameters (default: `True`)

---

### 3. Define Model Name Mapping

#### 3.1 Create `model_map` Dictionary
- Maps short-form model types to CLIP's full model names:
  ```python
  "vitb16"   → "ViT-B/16"  
  "vitb32"   → "ViT-B/32"  
  "rn50"     → "RN50"  
  "rn101"    → "RN101"  
  "rn50x4"   → "RN50x4"  
  "rn50x16"  → "RN50x16"  
  "rn50x64"  → "RN50x64"
  ```

---

### 4. Validate Input and Resolve Model Name

#### 4.1 Normalize `model_type`
- Convert `model_type` to lowercase using `.lower()`

#### 4.2 Lookup Model Name
- Fetch from `model_map`:
  ```python
  model_name = model_map.get(model_type.lower())
  ```

#### 4.3 Raise Error if Model Type is Invalid
- If `model_name is None`, raise a `ValueError`:
  ```python
  raise ValueError(f"❌ Unknown model type: {model_type}")
  ```

---

### 5. Load CLIP Model

#### 5.1 Call `clip.load(model_name, device=device)`
- Loads both:
  - `model`: the CLIP model object
  - `_`: the CLIP preprocessing transform (often unused but returned)

---

### 6. Optionally Freeze Parameters

#### 6.1 Check `freeze_all` Flag
- If `True`, iterate over all model parameters and disable gradient computation:
  ```python
  for param in model.parameters():
      param.requires_grad = False
  ```

---

### 7. Return Output

#### 7.1 Return Model and Preprocessing Object
- Output: `(model, _)`

---

### ✅ Final Output
- **Returns**:
  - `model`: The loaded CLIP model (`torch.nn.Module`)
  - `_`: The preprocessing function (e.g., for image transforms)

- **Use Case**: Integrate CLIP into training pipelines with optional parameter freezing.

---


***
***
***

