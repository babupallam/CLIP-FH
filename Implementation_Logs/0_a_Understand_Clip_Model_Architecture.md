# UNDERSTANDING CLIP MODEL

## FULL CLIP MODEL

The **CLIP model** has **two separate parts**, each built with a **transformer architecture**:

### 1. **Image Encoder (Vision Transformer - ViT)**  
- **Purpose**: Converts an **input image** into a **512-dimensional feature vector** (embedding) that represents the image’s content.  
- **How it works**:  
  - The image is divided into **small patches**, like cutting an image into square pieces.  
  - Each patch is **flattened** and **converted into a vector** (numbers that represent pixel information).  
  - These vectors are treated like words in a sentence and passed through a **transformer**.  
- **Transformer in CLIP**:  
  - It’s an **encoder-only transformer** (no decoder).  
  - It consists of **12 transformer layers**, also called **transformer blocks**.  
  - Each layer has two main components:  
    1. **Multi-Head Self-Attention** – Allows the model to focus on different parts of the image simultaneously.  
    2. **Feed-Forward Neural Network (MLP)** – Processes and transforms the attended features.  
  - The **layer** in a transformer means **one set of attention + feed-forward + normalization operations**.  
- After passing through all layers, the **final output** is a **512-dim image embedding**, using a **projection layer**.


### 2. **Text Encoder (Text Transformer)**  
- **Purpose**: Converts a **text sentence** (like “a photo of a hand”) into a **512-dimensional vector** that captures the meaning of the text.  
- **How it works**:  
  - The text is split into **tokens** (like words or sub-words).  
  - Each token is **embedded** into a vector using the **token embedding layer**.  
  - A **positional embedding** is added to tell the model the order of the tokens.  
  - These embeddings go through a **transformer**, specifically an **encoder-only transformer**.  
- **Transformer here**:  
  - Has **12 transformer layers (blocks)**, similar to the image encoder but with different dimensions.  
  - Each layer has **self-attention** and **feed-forward networks**, allowing the model to understand the context and relationships between tokens.  
- After processing, the **text features** are passed through a **projection layer**, giving a **512-dim text embedding**.

### ✅ **Is it Encoder or Decoder?**  
- Both **image** and **text** parts use **encoder-only transformers**.  
- There’s **no decoder** here like in sequence-to-sequence models (e.g., machine translation).  
- **Encoder-only transformers** focus on understanding the **input**, while **decoders** generate **output sequences**, which CLIP doesn’t do.


### ✅ **What is a Transformer Layer?**  
- A **transformer layer** (or block) is a **building unit** inside the transformer.  
- Each one has:  
  1. **Multi-Head Self-Attention**: Looks at different parts of the input to find important information.  
  2. **Feed-Forward Network (MLP)**: Processes information for better representation.  
  3. **Layer Normalization** and **Residual Connections**: Help stabilize and speed up learning.

In CLIP’s ViT-B/32:  
- There are **12 layers** in the **image encoder** and **12 layers** in the **text encoder**.  
- The number of **layers** determines **how deep** the network is (how many times information is transformed and refined).

### ✅ **How Do Image and Text Embeddings Work Together?**  
- After both encoders process the image and text, they produce **512-dim vectors**.  
- These vectors are **normalized** (scaled to length 1).  
- The model measures **similarity** between them using **cosine similarity**, scaled by a **logit scale parameter**.  
- Higher similarity means the image and text are likely describing the **same thing**.

``` 
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python .\0_understand_clip_model_architecture.py
Using device: cpu

================ FULL CLIP MODEL ================

CLIP(
  (visual): VisionTransformer(
    (conv1): Conv2d(3, 768, kernel_size=(32, 32), stride=(32, 32), bias=False)
    (ln_pre): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (transformer): Transformer(
      (resblocks): Sequential(
        (0): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (1): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (2): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (3): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (4): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (5): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (6): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (7): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (8): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (9): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (10): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (11): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
          )
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (ln_post): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (transformer): Transformer(
    (resblocks): Sequential(
      (0): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (1): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (2): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (3): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (4): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (5): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (6): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (7): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (8): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (9): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (10): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (11): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (token_embedding): Embedding(49408, 512)
  (ln_final): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)

```
<hr style="height:10px; background-color:red; border:none;">

## IMAGE ENCODER 

The **Image Encoder** in CLIP is a **Vision Transformer (ViT)**.  
It converts an **input image** into a **512-dimensional vector** (also called an **embedding**) that captures the **visual meaning** of the image.


### ✅ **How Does It Work?**

1. **Patch Extraction & Embedding**  
   - An image (e.g., 224x224 pixels) is **split into small square patches** (each 32x32 pixels if using ViT-B/32).  
   - Each patch is **flattened into a vector** and passed through a **linear projection (conv1)** that turns it into a **768-dimensional embedding** (vector of numbers).  
   - These embeddings are like **words in a sentence** for text—each representing part of the image.  
   
2. **Positional Information (Not Shown Here)**  
   - A **positional embedding** is added to each patch embedding so the model knows **where each patch belongs** in the original image.  
   - This is crucial because transformers do not inherently understand order or position.


### ✅ **Transformer Block (What Happens Next?)**

After patch embeddings are prepared, they are passed into the **Transformer**.  
- CLIP uses an **Encoder-Only Transformer**—there’s no decoder because we only need to **encode the image**, not generate anything.  
- The transformer is made up of **12 repeated layers**, called **Residual Attention Blocks**.


### ✅ **What Is a Transformer Layer?**
Each **layer** (or block) includes:
1. **Multi-Head Self-Attention**  
   - The model learns to **focus on important patches**, considering relationships between all patches.  
   - It lets the model **understand the entire image** by seeing how parts of it relate to each other.
   
2. **Feed-Forward Neural Network (MLP)**  
   - After attention, the information is processed and transformed further through **two linear layers with activation functions** (QuickGELU here).  
   - This helps the model **build complex representations**.

3. **Layer Normalization and Residual Connections**  
   - These stabilize training and help the model **learn efficiently**.


### ✅ **Layer Details (ViT-B/32 Image Encoder)**  
- Each layer works with **768-dimensional vectors** (size of each patch embedding).  
- There are **12 layers**, each one making the representation **deeper and richer**.  
- By the end, the model has a **global understanding** of the image.


### ✅ **Post-Transformer Layer**  
After the transformer layers:
- A **LayerNorm (ln_post)** is applied to clean up the output.  
- Then, a **projection layer (not shown here)** converts it into a **512-dimensional embedding**, which CLIP uses for matching with text.

### ✅ **Is This an Encoder or Decoder?**
- This is **an encoder-only transformer**, like BERT (not GPT).  
- It **analyzes and encodes** input information.  
- There is **no decoder** because CLIP does not generate output sequences—it **compares images and text**.


### ✅ **What Makes It Special?**
- Unlike CNNs, which look at local areas, the **Vision Transformer** can **globally attend** to any part of the image, allowing it to **capture complex relationships** between different regions.
- CLIP’s image encoder **outputs a vector** that represents the **entire image**, making it easy to compare to text.


### ✅ **Simple Summary (TL;DR)**  
- CLIP’s **Image Encoder** is a **Vision Transformer (ViT-B/32)**.  
- It **splits images into patches**, **embeds them**, and **processes them with 12 transformer layers**.  
- Each layer includes **self-attention** and **MLPs** to **refine the image representation**.  
- The final result is a **512-dimensional embedding** of the image.  
- It’s **encoder-only**, with **no decoder**, because CLIP only **encodes images** for comparison with text.


``` 

================ IMAGE ENCODER =================

VisionTransformer(
  (conv1): Conv2d(3, 768, kernel_size=(32, 32), stride=(32, 32), bias=False)
  (ln_pre): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (transformer): Transformer(
    (resblocks): Sequential(
      (0): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (1): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (2): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (3): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (4): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (5): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (6): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (7): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (8): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (9): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (10): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (11): ResidualAttentionBlock(
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (ln_post): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)


```
<hr style="height:10px; background-color:red; border:none;">

### IMAGE TRANSFORMER BLOCKS (A Transformer Layer)

The **Image Transformer Blocks** are the **core processing units** of CLIP’s Vision Transformer (ViT).  
They help **understand the image** by analyzing the relationships between different parts (patches) of the image.

#### ✅ **What Is a Transformer Block?**
A **Transformer Block** is a **self-contained module** that processes data in a **step-by-step fashion**, making it easier for the model to **learn complex features** from the input.  
In CLIP’s Image Encoder, there are **12 of these blocks**, working **one after another**, like layers in a deep neural network.


#### ✅ **Structure of Each Block (ResidualAttentionBlock)**
Each block consists of **3 main parts**, repeated in every layer:

##### 1. **Multi-Head Self-Attention (attn)**  
- **What it does**: Helps the model **focus on important patches** in the image by **looking at all patches at once**, rather than one at a time.  
- **Multi-Head** means it does this in **parallel**, from different perspectives.  
- **Why it’s important**: It allows the model to **learn relationships** between different regions in the image (e.g., the relationship between a hand and its fingers).


##### 2. **Layer Normalization (ln_1 and ln_2)**  
- **What it does**: Normalizes the input values to **keep them stable** as they flow through the model.  
- It’s applied **before** the attention and **before** the feed-forward network.  
- Helps make the training **faster and more stable**.


##### 3. **Feed-Forward Network (mlp)**  
- **What it does**: Transforms and processes the output from the attention step.  
- It’s a **2-layer MLP (Multi-Layer Perceptron)** that increases the feature dimension from **768 to 3072**, applies **QuickGELU activation**, and then reduces it back to **768**.  
- Think of it as **refining the attention output** and adding **non-linear transformations**.


#### ✅ **What Does "Residual" Mean in ResidualAttentionBlock?**  
- **Residual connections** add the **input** of each layer back to its **output**.  
- This makes it easier to **train deep networks**, prevents **vanishing gradients**, and allows the model to **reuse information**.  
- You can think of it as giving the model the option to **"skip" a layer** if it doesn't need to change much.

#### ✅ **How Many Blocks?**
- There are **12 identical blocks**, numbered from **Block 0** to **Block 11**.  
- Each block refines the image representation **a little more** at each step.

#### ✅ **Details from the Code (What’s Inside Each Block?)**
##### Example (From Block 0):  
```python
(attn): MultiheadAttention(768 → 768)  # Self-attention across image patches
(ln_1): LayerNorm(768)                 # Normalizes input before attention
(mlp): 
    (c_fc): Linear(768 → 3072)         # Expands dimension
    (gelu): QuickGELU()                # Activation function
    (c_proj): Linear(3072 → 768)       # Compresses back to original size
(ln_2): LayerNorm(768)                 # Normalizes before output
```

This structure repeats in **all 12 blocks**.

#### ✅ **What’s Special About These Blocks in Vision Transformers?**
- They **process an entire image globally**, rather than small regions like CNNs.  
- They can **learn relationships** across the **whole image** (global attention), which is powerful for recognizing complex objects (like a dorsal hand with fine features).

#### ✅ **What Happens After These 12 Blocks?**
- The **final output** of Block 11 is **layer-normalized** again (`ln_post`).  
- Then it’s passed to a **projection layer** that converts it into a **512-dimensional vector**, which CLIP uses for comparing with text.

#### ✅ **Is There an Encoder or Decoder?**
- This is an **encoder-only** transformer.  
- It **encodes** the image into a **feature vector**—it does **not generate sequences**, so there’s **no decoder**.

``` 
======= IMAGE TRANSFORMER BLOCKS =======

Block 0: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Block 1: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Block 2: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Block 3: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Block 4: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Block 5: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Block 6: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Block 7: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Block 8: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Block 9: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Block 10: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Block 11: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  )
  (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=768, out_features=3072, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  )
  (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)

```
<hr style="height:10px; background-color:red; border:none;">

### IMAGE FINAL PROJECTION LAYER (proj)

After the image passes through all **12 transformer blocks**, the output is still in a **768-dimensional space** (for ViT-B/32).  
But we need to **map it** into a **shared embedding space** where it can be **compared to text embeddings** (which are 512-dimensional).

That’s where the **projection layer** comes in.


#### ✅ **What Is the Image Final Projection Layer (proj)?**

- It’s a **Linear Layer (Fully Connected Layer)**  
- **Input dimension**: 768  
- **Output dimension**: 512  
- This layer is named `proj` in the code:  
  ```python
  visual.proj
  ```


#### ✅ **What Does It Do?**

1. It **projects the 768-dimensional image embeddings** (output of the Vision Transformer)  
2. Into a **512-dimensional embedding space**, the **same space used by the text embeddings**.  
3. This makes it possible to **compare images and text** by calculating **cosine similarity** between their embeddings.


#### ✅ **Why Is This Important?**

- Without this projection, the image embeddings and text embeddings would be **incompatible**.  
- Projection ensures they **live in the same space**, enabling **contrastive learning** and **cross-modal retrieval** (e.g., finding an image from a text query).


#### ✅ **What’s Inside the Projection Layer?**

- It’s just a **weight matrix** (parameter), often visualized as a **tensor of numbers**.  
- In your output, this tensor looks like this:
  ```text
  tensor([
    [-2.6264e-03,  5.0962e-05,  2.7496e-02,  ..., -1.0025e-02, -1.2222e-02,  5.8403e-03],
    [-1.9852e-02,  7.1182e-03,  8.9788e-04,  ...,  1.1528e-02, -1.9485e-02, -8.0185e-03],
    ...
  ], requires_grad=True)
  ```
- These numbers represent the **learned transformation** from **768-dim space** to **512-dim space**.


#### ✅ **How Does It Work in a Forward Pass?**

**Step-by-step (simple view):**  
1. Image is passed through the Vision Transformer → produces a **768-dim embedding**.  
2. That embedding is multiplied by the **projection matrix (`proj`)** → produces a **512-dim embedding**.  
3. The 512-dim vector is **normalized** (unit length) so it can be compared to **text embeddings**.


#### ✅ **What Happens After Projection?**

1. Both image and text embeddings are **normalized**:  
   ```python
   image_features = image_features / image_features.norm(dim=-1, keepdim=True)
   ```
2. Then, they are **compared** using **cosine similarity**, scaled by `logit_scale`.  
   ```python
   logits_per_image = logit_scale * image_features @ text_features.t()
   ```

```
 
======= IMAGE FINAL PROJECTION LAYER (proj) =======

Parameter containing:
tensor([[-2.6264e-03,  5.0962e-05,  2.7496e-02,  ..., -1.0025e-02,
         -1.2222e-02,  5.8403e-03],
        [-1.9852e-02,  7.1182e-03,  8.9788e-04,  ...,  1.1528e-02,
         -1.9485e-02, -8.0185e-03],
        [-8.6288e-03,  1.9226e-03, -2.1725e-03,  ...,  3.9330e-03,
         -1.1269e-02,  1.5345e-03],
        ...,
        [-1.1993e-02,  1.2955e-02,  2.5848e-02,  ..., -9.8038e-03,
         -4.2076e-03,  1.5211e-04],
        [-1.2871e-02, -9.5673e-03, -1.0826e-02,  ..., -7.0610e-03,
         -4.3182e-03, -4.9353e-04],
        [-4.4098e-03,  3.3588e-03, -1.2054e-02,  ...,  6.1073e-03,
          3.9940e-03, -3.0861e-03]], requires_grad=True)


```
<hr style="height:10px; background-color:red; border:none;">

## TEXT ENCODER

``` 
================ TEXT ENCODER ================

```
<hr style="height:10px; background-color:red; border:none;">

### TEXT TRANSFORMER BLOCKS


#### ✅ What Is the Text Transformer?

The **Text Transformer** is the part of CLIP that processes **text data**.  
It converts a sequence of **tokens** (words or subwords) into a **512-dimensional embedding vector**.  
This embedding is then compared to the image embedding for similarity.

#### ✅ How Does It Work?

The **Text Transformer** is made of **12 Residual Attention Blocks**.  
Each block helps the model understand the **context and relationships** between words in a sentence.

#### ✅ What Happens Inside Each Block?

Each **ResidualAttentionBlock** has three important parts:

1. **Multihead Self-Attention (attn):**  
   - Allows the model to **focus on different words** at the same time.  
   - It helps the model understand **which words matter most** for a given word.

2. **Layer Normalization (ln_1 and ln_2):**  
   - Normalizes the data for **stable and efficient training**.  
   - Keeps things balanced as they pass through the network.

3. **MLP (Feedforward Network):**  
   - Makes the model **learn more complex patterns**.  
   - It expands the data to a **larger space (2048 dimensions)** and then reduces it back to **512 dimensions**.


#### ✅ What's Special About CLIP's Text Transformer?

- It has **12 blocks** (layers), each adding more **context and understanding**.  
- The **embedding size is 512**, which matches the **image embedding size** after projection.  
- **QuickGELU** is used as the activation function, which is a **faster version of GELU**.


#### ✅ Why Do We Need These Blocks?

These **blocks** help CLIP **understand the meaning** of sentences.  
For example:  
> "A dog playing in the park."  
Each block processes and refines this sentence’s **representation**, capturing **relationships and meaning**.


#### ✅ Text Transformer = Encoder-Only  
- It’s like **BERT**, not like an encoder-decoder model.  
- It **only encodes the input text**, no decoder is involved.  
- This is different from models like **GPT**, which are decoder-based.


#### ✅ What's the Difference from the Image Transformer?

| Feature             | Image Transformer               | Text Transformer                |
|---------------------|---------------------------------|---------------------------------|
| Embedding Size      | 768                             | 512                             |
| Feedforward Dim     | 3072                            | 2048                            |
| Data                | Image Patches                   | Text Tokens                    |
| Purpose             | Understand **visual structure** | Understand **sentence meaning** |

``` 

======= TEXT TRANSFORMER BLOCKS =======

Block 0: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
Block 1: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
Block 2: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
Block 3: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
Block 4: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
Block 5: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
Block 6: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
Block 7: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
Block 8: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
Block 9: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
Block 10: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)
Block 11: ResidualAttentionBlock(
  (attn): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
  )
  (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (mlp): Sequential(
    (c_fc): Linear(in_features=512, out_features=2048, bias=True)
    (gelu): QuickGELU()
    (c_proj): Linear(in_features=2048, out_features=512, bias=True)
  )
  (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
)

```
<hr style="height:10px; background-color:red; border:none;">


### Text Token Embedding & Positional Embedding

#### ✅ What Is Text Token Embedding?

- The **Text Token Embedding** layer converts each **text token** (word or subword) into a **vector** of numbers.  
- In CLIP (ViT-B/32), each token is mapped to a **512-dimensional vector**.  
- These vectors allow the model to **understand words mathematically**, so they can be processed by the transformer.

> **Example**  
> The word "**dog**" becomes a **512-dim vector** (just numbers that represent its meaning).


#### ✅ What Is Positional Embedding?

- Transformers **don’t know** the order of tokens by default.  
- The **Positional Embedding** gives each token a **sense of its position** in the sentence.  
- It’s like **labeling**:  
  - "First word,"  
  - "Second word,"  
  - and so on...  
- CLIP adds these **positional vectors** to the **token embeddings**, so the model knows **where each token appears**.

> **Example**  
> "A dog plays"  
> - "A" gets **position 0**  
> - "dog" gets **position 1**  
> - "plays" gets **position 2**

#### ✅ How They Work Together

1. **Token Embedding**: Converts words into vectors.  
2. **Positional Embedding**: Adds position info to these vectors.  
3. These two are **added together** and passed into the **Text Transformer** for deeper understanding.

#### ✅ Key Details in CLIP (ViT-B/32)

| Component             | Shape / Size        | What It Does                          |
|-----------------------|---------------------|--------------------------------------|
| **Token Embedding**   | `[49408, 512]`      | 49,408 tokens mapped to 512-dim vectors (learned from text data).  
| **Positional Embedding** | `[77, 512]`     | 77 positions mapped to 512-dim vectors (77 is the max context length in CLIP).

#### ✅ Why Is This Important?

- It gives CLIP the ability to **understand language structure**.  
- Without **Positional Embeddings**, CLIP would **not** know if "dog bites man" is different from "man bites dog".

``` 


================ TEXT TOKEN & POSITIONAL EMBEDDING ================

Text Token Embedding Layer (clip_model.token_embedding):
Embedding Weight Shape (vocab_size x embed_dim): torch.Size([49408, 512])
Example Tensor (first 5 tokens):
tensor([[-0.0039, -0.0063,  0.0074,  ..., -0.0107, -0.0228, -0.0109],
        [-0.0261,  0.0088, -0.0117,  ..., -0.0120, -0.0241, -0.0219],
        [-0.0196, -0.0067, -0.0091,  ...,  0.0046, -0.0207, -0.0087],
        [-0.0118,  0.0118, -0.0134,  ..., -0.0049,  0.0004, -0.0169],
        [-0.0068, -0.0024, -0.0100,  ...,  0.0033, -0.0236,  0.0141]],
       grad_fn=<SliceBackward0>)

Text Positional Embedding (clip_model.positional_embedding):
Shape (context_length x embed_dim): torch.Size([77, 512])
Example Tensor (first 5 positions):
tensor([[-1.4361e-03,  1.9820e-04, -4.1244e-03,  ..., -3.6505e-03,
         -3.4988e-03,  1.5722e-04],
        [ 3.5358e-04,  2.0378e-03,  1.1401e-03,  ..., -9.8163e-04,
          2.5043e-03,  7.1525e-04],
        [-3.6177e-04,  1.5840e-03,  1.8298e-04,  ...,  3.2474e-05,
         -5.8654e-03,  2.4234e-03],
        [-5.6016e-04,  1.4166e-03, -2.0413e-03,  ..., -2.3383e-03,
         -8.2191e-03,  2.3327e-04],
        [ 1.8524e-03,  9.0577e-04, -6.2706e-04,  ..., -4.0008e-03,
         -7.0807e-04,  1.7565e-03]], grad_fn=<SliceBackward0>)

✅ Token Embedding maps each word/token into a 512-dim vector (ViT-B/32).
✅ Positional Embedding adds sequential order information into the transformer input.


```
<hr style="height:10px; background-color:red; border:none;">


## PROJECTION & SCALE

### ✅ What Are Projection Layers?

- Both **image** and **text** embeddings go through **projection layers**.  
- These are simple **Linear layers (fully connected layers)** that **transform** the embeddings into a **shared space** (same dimension).  
- This shared space allows CLIP to **compare images and texts directly**, even though they come from **different encoders** (Vision Transformer and Text Transformer).


### ✅ What Happens in PROJECTION?

| Component           | Purpose                                                     | Output Dimension |
|---------------------|-------------------------------------------------------------|------------------|
| **visual.proj**     | Projects the **image** embedding to the shared space.       | `[768 -> 512]`   |
| **text_projection** | Projects the **text** embedding to the shared space.        | `[512 -> 512]`   |

- The **image features** (dimension **768**) are projected down to **512** using `visual.proj`.  
- The **text features** are already **512-dim**, so `text_projection` keeps them aligned in the same space.


### ✅ Why Projection Matters

- Without **projection**, the model wouldn’t be able to **align** or **compare** image and text features.  
- After projection, both modalities are in the **same 512-dimensional space**.  
- This makes **cross-modal matching** (image ↔ text) possible.


### ✅ What Is Logit Scale?

- `logit_scale` is a **learnable scalar parameter**.  
- It’s used to **scale** the **cosine similarity scores** between image and text embeddings.  
- Scaling these scores makes the **contrastive learning** more effective during training.

> **How it works:**  
> - Similarity between image and text is computed (cosine similarity).  
> - Then it’s **multiplied** by `logit_scale` (starts around `exp(4.6052) = 100`).  
> - This makes the **scores sharper**, helping the model focus on the **correct matches**.


### ✅ Simple End-to-End Flow for Projection & Scale

1. **Image** → Encoder → `visual.proj` → Normalize → **512-dim embedding**  
2. **Text** → Encoder → `text_projection` → Normalize → **512-dim embedding**  
3. **Cosine Similarity** between both embeddings.  
4. Multiply by **logit_scale** to get **final logits**.  
5. These logits are used for **contrastive loss** in training or **retrieval** in inference.


### ✅ Key Parameters in CLIP (ViT-B/32)

| Parameter                | Shape          | Description                           |
|--------------------------|----------------|---------------------------------------|
| **visual.proj**          | `[512, 768]`   | Projects image features to 512-dim.  
| **text_projection**      | `[512, 512]`   | Projects text features to 512-dim.  
| **logit_scale**          | `scalar`       | Scales similarity scores.


### ✅ Why Is This Important?

- **Projection** ensures **alignment** between vision and language.  
- **Logit scaling** improves the **separation** between positive and negative matches.  
- Both are critical for CLIP’s **zero-shot learning** and **retrieval** tasks.

``` 
================ PROJECTION & SCALE ================

Image Projection (visual.proj): Parameter containing:
tensor([[-2.6264e-03,  5.0962e-05,  2.7496e-02,  ..., -1.0025e-02,
         -1.2222e-02,  5.8403e-03],
        [-1.9852e-02,  7.1182e-03,  8.9788e-04,  ...,  1.1528e-02,
         -1.9485e-02, -8.0185e-03],
        [-8.6288e-03,  1.9226e-03, -2.1725e-03,  ...,  3.9330e-03,
         -1.1269e-02,  1.5345e-03],
        ...,
        [-1.1993e-02,  1.2955e-02,  2.5848e-02,  ..., -9.8038e-03,
         -4.2076e-03,  1.5211e-04],
        [-1.2871e-02, -9.5673e-03, -1.0826e-02,  ..., -7.0610e-03,
         -4.3182e-03, -4.9353e-04],
        [-4.4098e-03,  3.3588e-03, -1.2054e-02,  ...,  6.1073e-03,
          3.9940e-03, -3.0861e-03]], requires_grad=True)
Text Projection (clip_model.text_projection): Parameter containing:
tensor([[-0.0104,  0.0142, -0.0084,  ..., -0.0069, -0.0125,  0.0012],
        [ 0.0054,  0.0013, -0.0036,  ...,  0.0026,  0.0136, -0.0201],
        [ 0.0029,  0.0031,  0.0182,  ...,  0.0034,  0.0052, -0.0063],
        ...,
        [ 0.0094,  0.0306,  0.0135,  ...,  0.0160,  0.0014, -0.0110],
        [-0.0113,  0.0047,  0.0017,  ..., -0.0043, -0.0187, -0.0049],
        [ 0.0076, -0.0067,  0.0112,  ...,  0.0036, -0.0038,  0.0170]],
       requires_grad=True)
Logit Scale: Parameter containing:
tensor(4.6052, requires_grad=True)

```
<hr style="height:10px; background-color:red; border:none;">


## Summary of Dimensions and Parameters
``` 

================ PARAMETER COUNT =================


Total parameters: 151,277,313
Trainable parameters: 151,277,313

Image Encoder Parameters:

Total parameters: 87,849,216
Trainable parameters: 87,849,216

Text Encoder Parameters:

Total parameters: 37,828,608
Trainable parameters: 37,828,608


```
<hr style="height:10px; background-color:red; border:none;">



## Forward Pass Walkthrough (End-to-End Data Flow)
### ✅ What Is a Forward Pass?

- The **forward pass** describes how data moves **through the entire model**, from **input to output**.
- In CLIP, this happens **independently** for images and texts—two separate pipelines that **align at the end** for similarity comparison.


### IMAGE PIPELINE 

#### ✅ IMAGE PIPELINE (End-to-End)

##### 🔸 Step 1: Input Image  
- The input is a **raw image** (RGB format).

##### 🔸 Step 2: Patch Embedding (conv1)
- The image is **split into patches** using `conv1`.  
- Each patch is **embedded** into a **768-dim vector**.  
- **Output shape**: `torch.Size([1, 768, 7, 7])`  
  - `1`: batch size  
  - `768`: embedding dim  
  - `7 x 7`: number of patches

##### 🔸 Step 3: Transformer Encoder  
- The patch embeddings are **flattened** and passed into the **Vision Transformer**.  
- It applies **self-attention** and **MLP layers** repeatedly across **12 transformer blocks**.  
- Produces a **pooled feature vector** for the image.

##### 🔸 Step 4: Layer Normalization (`ln_post`)  
- Applies **normalization** to stabilize and improve learning.

##### 🔸 Step 5: Final Projection (`proj`)  
- Projects the **image embedding** (768-dim) into a **512-dim vector** (shared space with text).

##### 🔸 Step 6: Normalization  
- The 512-dim vector is **L2-normalized** to ensure **unit length**.  
  (This is important for cosine similarity.)  
- **Output shape**: `torch.Size([1, 512])`  
  (This is the **final image feature** used in similarity comparison.)

``` 


================ CLIP FORWARD PASS WALKTHROUGH ================

---- IMAGE PIPELINE ----
Patch Embeddings (conv1) Shape: torch.Size([1, 768, 7, 7])
Encoded Image Features (before projection): torch.Size([1, 512])
Normalized Image Features: torch.Size([1, 512])

```
<hr style="height:10px; background-color:red; border:none;">

### TEXT PIPELINE


#### ✅ TEXT PIPELINE (End-to-End)


##### 🔸 Step 1: Input Text  
- The input is a **sequence of text tokens** (IDs from the vocabulary).

##### 🔸 Step 2: Token Embedding (`token_embedding`)  
- Each token is mapped to a **512-dim vector**.  
  (Shape: `[sequence_length, 512]`)

##### 🔸 Step 3: Add Positional Embedding  
- Adds **positional information** to the token embeddings.  
  Helps the model understand **word order**.

##### 🔸 Step 4: Transformer Encoder  
- The embeddings go through **12 transformer blocks**, using **self-attention** and **MLPs** to capture context.

##### 🔸 Step 5: Layer Normalization (`ln_final`)  
- Normalizes the output after all transformer layers.

##### 🔸 Step 6: Final Projection (`text_projection`)  
- Projects the text feature (512-dim) into the **shared 512-dim space**.

##### 🔸 Step 7: Normalization  
- The text feature vector is **L2-normalized** to ensure **unit length**.  
- **Output shape**: `torch.Size([1, 512])`  
  (This is the **final text feature** used in similarity comparison.)

#### ✅ ALIGNMENT & COMPARISON (Image ↔ Text)
- 
- Both **image** and **text** vectors are now **512-dimensional normalized embeddings**.  
- These embeddings represent the **features** of the image and text in the **same space**, so they can be directly compared.


##### ✅ What is Cosine Similarity?
- **Cosine similarity** measures **how similar two vectors are**, based on the **angle** between them.  
- It ignores **magnitude** and focuses on **direction**.  
- Formula:  
  `cosine_similarity = (A · B) / (||A|| * ||B||)`  
  - `A · B`: dot product of vectors  
  - `||A||` and `||B||`: magnitudes (lengths), but since embeddings are **normalized**, these are `1`.  
- So, it becomes just the **dot product** of the two normalized vectors.  
- A similarity of `1` means **perfect alignment** (very similar), and `-1` means **opposite**.


##### ✅ How It’s Done in CLIP:
1. After projection, both image and text embeddings are **L2-normalized** (unit length).  
2. Compute **dot product** (cosine similarity) between them.  
3. Multiply the result by a **learnable parameter** called `logit_scale`.  
   - `logit_scale` controls the **sharpness** of the similarity distribution.  
   - A higher `logit_scale` makes the model **more confident** in its predictions.

##### ✅ Result:
- **Higher cosine similarity** (closer to `1`) means a **better match** between image and text.  
- This value is used in **contrastive learning** to **align** matching image-text pairs and **separate** unrelated ones.

#### ✅ Parameter Freezing (Why?)  
- During fine-tuning or training, you often **freeze** parts of the model to **save computation** or **preserve learned features**.  
- In your output, many components are **frozen**, including:  
  - **class_embedding**  
  - **positional_embedding**  
  - **projections**  
  - Layers of both **vision** and **text** transformers  
- This suggests the model is in **inference mode** or **fine-tuning with select layers**.


``` 

---- TEXT PIPELINE ----
Encoded Text Features (before projection): torch.Size([1, 512])
Normalized Text Features: torch.Size([1, 512])

✅ Both pipelines produce normalized embeddings in the same space.

Freezing: class_embedding
Freezing: positional_embedding
Freezing: proj
Freezing: conv1.weight
Freezing: ln_pre.weight
Freezing: ln_pre.bias
Freezing: transformer.resblocks.0.attn.in_proj_weight
Freezing: transformer.resblocks.0.attn.in_proj_bias
Freezing: transformer.resblocks.0.attn.out_proj.weight
Freezing: transformer.resblocks.0.attn.out_proj.bias
Freezing: transformer.resblocks.0.ln_1.weight
Freezing: transformer.resblocks.0.ln_1.bias
Freezing: transformer.resblocks.0.mlp.c_fc.weight
Freezing: transformer.resblocks.0.mlp.c_fc.bias
Freezing: transformer.resblocks.0.mlp.c_proj.weight
Freezing: transformer.resblocks.0.mlp.c_proj.bias
Freezing: transformer.resblocks.0.ln_2.weight
Freezing: transformer.resblocks.0.ln_2.bias
Freezing: transformer.resblocks.1.attn.in_proj_weight
Freezing: transformer.resblocks.1.attn.in_proj_bias
Freezing: transformer.resblocks.1.attn.out_proj.weight
Freezing: transformer.resblocks.1.attn.out_proj.bias
Freezing: transformer.resblocks.1.ln_1.weight
Freezing: transformer.resblocks.1.ln_1.bias
Freezing: transformer.resblocks.1.mlp.c_fc.weight
Freezing: transformer.resblocks.1.mlp.c_fc.bias
Freezing: transformer.resblocks.1.mlp.c_proj.weight
Freezing: transformer.resblocks.1.mlp.c_proj.bias
Freezing: transformer.resblocks.1.ln_2.weight
Freezing: transformer.resblocks.1.ln_2.bias
Freezing: transformer.resblocks.2.attn.in_proj_weight
Freezing: transformer.resblocks.2.attn.in_proj_bias
Freezing: transformer.resblocks.2.attn.out_proj.weight
Freezing: transformer.resblocks.2.attn.out_proj.bias
Freezing: transformer.resblocks.2.ln_1.weight
Freezing: transformer.resblocks.2.ln_1.bias
Freezing: transformer.resblocks.2.mlp.c_fc.weight
Freezing: transformer.resblocks.2.mlp.c_fc.bias
Freezing: transformer.resblocks.2.mlp.c_proj.weight
Freezing: transformer.resblocks.2.mlp.c_proj.bias
Freezing: transformer.resblocks.2.ln_2.weight
Freezing: transformer.resblocks.2.ln_2.bias
Freezing: transformer.resblocks.3.attn.in_proj_weight
Freezing: transformer.resblocks.3.attn.in_proj_bias
Freezing: transformer.resblocks.3.attn.out_proj.weight
Freezing: transformer.resblocks.3.attn.out_proj.bias
Freezing: transformer.resblocks.3.ln_1.weight
Freezing: transformer.resblocks.3.ln_1.bias
Freezing: transformer.resblocks.3.mlp.c_fc.weight
Freezing: transformer.resblocks.3.mlp.c_fc.bias
Freezing: transformer.resblocks.3.mlp.c_proj.weight
Freezing: transformer.resblocks.3.mlp.c_proj.bias
Freezing: transformer.resblocks.3.ln_2.weight
Freezing: transformer.resblocks.3.ln_2.bias
Freezing: transformer.resblocks.4.attn.in_proj_weight
Freezing: transformer.resblocks.4.attn.in_proj_bias
Freezing: transformer.resblocks.4.attn.out_proj.weight
Freezing: transformer.resblocks.4.attn.out_proj.bias
Freezing: transformer.resblocks.4.ln_1.weight
Freezing: transformer.resblocks.4.ln_1.bias
Freezing: transformer.resblocks.4.mlp.c_fc.weight
Freezing: transformer.resblocks.4.mlp.c_fc.bias
Freezing: transformer.resblocks.4.mlp.c_proj.weight
Freezing: transformer.resblocks.4.mlp.c_proj.bias
Freezing: transformer.resblocks.4.ln_2.weight
Freezing: transformer.resblocks.4.ln_2.bias
Freezing: transformer.resblocks.5.attn.in_proj_weight
Freezing: transformer.resblocks.5.attn.in_proj_bias
Freezing: transformer.resblocks.5.attn.out_proj.weight
Freezing: transformer.resblocks.5.attn.out_proj.bias
Freezing: transformer.resblocks.5.ln_1.weight
Freezing: transformer.resblocks.5.ln_1.bias
Freezing: transformer.resblocks.5.mlp.c_fc.weight
Freezing: transformer.resblocks.5.mlp.c_fc.bias
Freezing: transformer.resblocks.5.mlp.c_proj.weight
Freezing: transformer.resblocks.5.mlp.c_proj.bias
Freezing: transformer.resblocks.5.ln_2.weight
Freezing: transformer.resblocks.5.ln_2.bias
Freezing: transformer.resblocks.6.attn.in_proj_weight
Freezing: transformer.resblocks.6.attn.in_proj_bias
Freezing: transformer.resblocks.6.attn.out_proj.weight
Freezing: transformer.resblocks.6.attn.out_proj.bias
Freezing: transformer.resblocks.6.ln_1.weight
Freezing: transformer.resblocks.6.ln_1.bias
Freezing: transformer.resblocks.6.mlp.c_fc.weight
Freezing: transformer.resblocks.6.mlp.c_fc.bias
Freezing: transformer.resblocks.6.mlp.c_proj.weight
Freezing: transformer.resblocks.6.mlp.c_proj.bias
Freezing: transformer.resblocks.6.ln_2.weight
Freezing: transformer.resblocks.6.ln_2.bias
Freezing: transformer.resblocks.7.attn.in_proj_weight
Freezing: transformer.resblocks.7.attn.in_proj_bias
Freezing: transformer.resblocks.7.attn.out_proj.weight
Freezing: transformer.resblocks.7.attn.out_proj.bias
Freezing: transformer.resblocks.7.ln_1.weight
Freezing: transformer.resblocks.7.ln_1.bias
Freezing: transformer.resblocks.7.mlp.c_fc.weight
Freezing: transformer.resblocks.7.mlp.c_fc.bias
Freezing: transformer.resblocks.7.mlp.c_proj.weight
Freezing: transformer.resblocks.7.mlp.c_proj.bias
Freezing: transformer.resblocks.7.ln_2.weight
Freezing: transformer.resblocks.7.ln_2.bias
Freezing: transformer.resblocks.8.attn.in_proj_weight
Freezing: transformer.resblocks.8.attn.in_proj_bias
Freezing: transformer.resblocks.8.attn.out_proj.weight
Freezing: transformer.resblocks.8.attn.out_proj.bias
Freezing: transformer.resblocks.8.ln_1.weight
Freezing: transformer.resblocks.8.ln_1.bias
Freezing: transformer.resblocks.8.mlp.c_fc.weight
Freezing: transformer.resblocks.8.mlp.c_fc.bias
Freezing: transformer.resblocks.8.mlp.c_proj.weight
Freezing: transformer.resblocks.8.mlp.c_proj.bias
Freezing: transformer.resblocks.8.ln_2.weight
Freezing: transformer.resblocks.8.ln_2.bias
Freezing: transformer.resblocks.9.attn.in_proj_weight
Freezing: transformer.resblocks.9.attn.in_proj_bias
Freezing: transformer.resblocks.9.attn.out_proj.weight
Freezing: transformer.resblocks.9.attn.out_proj.bias
Freezing: transformer.resblocks.9.ln_1.weight
Freezing: transformer.resblocks.9.ln_1.bias
Freezing: transformer.resblocks.9.mlp.c_fc.weight
Freezing: transformer.resblocks.9.mlp.c_fc.bias
Freezing: transformer.resblocks.9.mlp.c_proj.weight
Freezing: transformer.resblocks.9.mlp.c_proj.bias
Freezing: transformer.resblocks.9.ln_2.weight
Freezing: transformer.resblocks.9.ln_2.bias
Freezing: transformer.resblocks.10.attn.in_proj_weight
Freezing: transformer.resblocks.10.attn.in_proj_bias
Freezing: transformer.resblocks.10.attn.out_proj.weight
Freezing: transformer.resblocks.10.attn.out_proj.bias
Freezing: transformer.resblocks.10.ln_1.weight
Freezing: transformer.resblocks.10.ln_1.bias
Freezing: transformer.resblocks.10.mlp.c_fc.weight
Freezing: transformer.resblocks.10.mlp.c_fc.bias
Freezing: transformer.resblocks.10.mlp.c_proj.weight
Freezing: transformer.resblocks.10.mlp.c_proj.bias
Freezing: transformer.resblocks.10.ln_2.weight
Freezing: transformer.resblocks.10.ln_2.bias
Freezing: transformer.resblocks.11.attn.in_proj_weight
Freezing: transformer.resblocks.11.attn.in_proj_bias
Freezing: transformer.resblocks.11.attn.out_proj.weight
Freezing: transformer.resblocks.11.attn.out_proj.bias
Freezing: transformer.resblocks.11.ln_1.weight
Freezing: transformer.resblocks.11.ln_1.bias
Freezing: transformer.resblocks.11.mlp.c_fc.weight
Freezing: transformer.resblocks.11.mlp.c_fc.bias
Freezing: transformer.resblocks.11.mlp.c_proj.weight
Freezing: transformer.resblocks.11.mlp.c_proj.bias
Freezing: transformer.resblocks.11.ln_2.weight
Freezing: transformer.resblocks.11.ln_2.bias
Freezing: ln_post.weight
Freezing: ln_post.bias
```
<hr style="height:10px; background-color:red; border:none;">



## FREEZE / UNFREEZE Example
``` 

======= UNFREEZE LAST 2 IMAGE TRANSFORMER BLOCKS =======

Unfreezing: transformer.resblocks.10.attn.in_proj_weight
Unfreezing: transformer.resblocks.10.attn.in_proj_bias
Unfreezing: transformer.resblocks.10.attn.out_proj.weight
Unfreezing: transformer.resblocks.10.attn.out_proj.bias
Unfreezing: transformer.resblocks.10.ln_1.weight
Unfreezing: transformer.resblocks.10.ln_1.bias
Unfreezing: transformer.resblocks.10.mlp.c_fc.weight
Unfreezing: transformer.resblocks.10.mlp.c_fc.bias
Unfreezing: transformer.resblocks.10.mlp.c_proj.weight
Unfreezing: transformer.resblocks.10.mlp.c_proj.bias
Unfreezing: transformer.resblocks.10.ln_2.weight
Unfreezing: transformer.resblocks.10.ln_2.bias
Unfreezing: transformer.resblocks.11.attn.in_proj_weight
Unfreezing: transformer.resblocks.11.attn.in_proj_bias
Unfreezing: transformer.resblocks.11.attn.out_proj.weight
Unfreezing: transformer.resblocks.11.attn.out_proj.bias
Unfreezing: transformer.resblocks.11.ln_1.weight
Unfreezing: transformer.resblocks.11.ln_1.bias
Unfreezing: transformer.resblocks.11.mlp.c_fc.weight
Unfreezing: transformer.resblocks.11.mlp.c_fc.bias
Unfreezing: transformer.resblocks.11.mlp.c_proj.weight
Unfreezing: transformer.resblocks.11.mlp.c_proj.bias
Unfreezing: transformer.resblocks.11.ln_2.weight
Unfreezing: transformer.resblocks.11.ln_2.bias

```
<hr style="height:10px; background-color:red; border:none;">


## Replace Layers Example

### ✅ REPLACED IMAGE PROJECTION LAYER

- The **projection layer** is the final transformation applied to the **image embeddings** before comparison with text embeddings.
- It **maps** the extracted image features to a new space where they can be aligned with text features.
- **Replacing this layer** allows for:
  - **Different embedding dimensions** (e.g., reducing from `512` to `256`).
  - **Task-specific adaptation**, such as fine-tuning for a different dataset or objective.

### ✅ Why Replace the Projection Layer?
1. **Adjusting Feature Space**: If the target task requires a **different embedding size**, replacing the projection layer helps adapt the model.
2. **Fine-Tuning CLIP for a New Task**: Instead of using CLIP's default projection, you can learn a custom projection for **better performance on specific data**.
3. **Domain Adaptation**: For tasks like **hand recognition** (HandCLIP), a specialized projection might work better than CLIP’s general-purpose one.

### ✅ How It’s Done:

```python
import torch.nn as nn

# Get the current projection size of the CLIP model
original_proj_dim = visual_encoder.proj.shape[0]  # Typically 512 for ViT-B/32

# Define a new projection layer (changing output dimension to 256)
new_proj = torch.nn.Parameter(torch.randn(256, original_proj_dim) * 0.02).to(device)

# Assign the new projection
visual_encoder.proj = new_proj

# Print confirmation
print("\n======= REPLACED IMAGE PROJECTION LAYER =======\n")
print(visual_encoder.proj)
```

### ✅ What Happens After Replacement?
- Instead of using CLIP’s **default 512-dimensional projection**, the model will now output **256-dimensional image embeddings**.
- These **new embeddings** will still be **compared** with text embeddings, but might require **adjustments** in text projection for proper alignment.

``` 

======= REPLACED IMAGE PROJECTION LAYER =======

Parameter containing:
tensor([[-0.0202, -0.0051,  0.0044,  ..., -0.0046,  0.0057, -0.0377],
        [ 0.0041, -0.0056,  0.0104,  ..., -0.0252,  0.0065, -0.0012],
        [-0.0213, -0.0111,  0.0102,  ..., -0.0081, -0.0034, -0.0006],
        ...,
        [ 0.0418,  0.0019, -0.0089,  ..., -0.0201, -0.0130, -0.0185],
        [-0.0050,  0.0110, -0.0285,  ..., -0.0220, -0.0179, -0.0168],
        [ 0.0080,  0.0046, -0.0142,  ...,  0.0143,  0.0151, -0.0260]],
       requires_grad=True)

```
<hr style="height:10px; background-color:red; border:none;">

## Logit Scaling Operation (Similarity Computation)

### ✅ What Is Happening Here?

- After both **image** and **text** embeddings are **normalized**, CLIP computes their **similarity**.
- This **similarity** is calculated as the **cosine similarity** between the two vectors.
- The **logit_scale** parameter is used to **sharpen** or **scale** these similarity scores, making the contrastive learning process more effective.


### ✅ Key Terms:

1. **Cosine Similarity**  
   - Measures the **angle** between two vectors.
   - Ranges from **-1** (opposite direction) to **1** (same direction).  
   - In CLIP, cosine similarity works because both image and text vectors are **normalized** to **unit length**, making the **dot product** equal to the cosine similarity.

2. **logit_scale**  
   - A **learnable scalar** that multiplies the cosine similarity.
   - Controls how **confident** the model is about similarity.  
   - Initialized in CLIP as `logit_scale = exp(4.6052) ≈ 100.0`  
   - A **higher** value makes the softmax **sharper**, which leads to **stronger contrast** between matching and non-matching pairs.


### ✅ Example from Output:

```
logit_scale (exp): 100.0000

Logits Per Image Shape: torch.Size([1, 1])

Logits Per Text Shape: torch.Size([1, 1])
```

- **logit_scale** is `100.0`, applied after cosine similarity.
- The **logits** are the scaled similarity scores between **each image** and **each text** embedding.
- Shape `[1, 1]` means **1 image compared to 1 text** (batch of 1).


### ✅ How It Works (Simplified Steps):

1. **Normalize** both image and text embeddings:  
   ```
   image_features = image_features / image_features.norm(dim=-1, keepdim=True)
   text_features = text_features / text_features.norm(dim=-1, keepdim=True)
   ```

2. **Compute Cosine Similarity**:  
   ```
   similarity = image_features @ text_features.T
   ```

3. **Scale Similarity by logit_scale**:  
   ```
   logits = logit_scale * similarity
   ```

4. **Resulting logits** are passed to **softmax** (in training) or **directly used** for comparison (in inference).


### ✅ Why It Matters:
✔ **logit_scale** makes it easier for CLIP to **separate positive pairs** (correct matches) from **negative pairs**.  
✔ It **amplifies differences** in similarity scores, making **learning** and **retrieval** more effective.  
✔ During **fine-tuning**, `logit_scale` is often left **trainable**, so it can adjust for different datasets.

``` 

================ LOGIT SCALING OPERATION ================

logit_scale (exp): 100.0000

Logits Per Image Shape: torch.Size([1, 1])

Logits Per Text Shape: torch.Size([1, 1])

✅ Similarity logits computed using logit_scale and normalized embeddings.


```
<hr style="height:10px; background-color:red; border:none;">

##  Text and Image Output Embedding Dimension Alignment
``` 

================ TEXT AND IMAGE OUTPUT EMBEDDING ALIGNMENT ================

Image Output Embedding Dimension (proj): 256
Text Output Embedding Dimension (text_projection): 512

⚠️ Embeddings have different dimensions. Alignment needed!

Image Feature Norms (should be 1): tensor([1.])
Text Feature Norms (should be 1): tensor([1.])

✅ Embeddings normalized to unit length before similarity comparison.

```
<hr style="height:10px; background-color:red; border:none;">

## Differences Between Vision and Text Transformers
``` 
================ DIFFERENCES BETWEEN VISION AND TEXT TRANSFORMERS ================

Vision Transformer (ViT-B/32):
- Patch Embedding Dim: 768
- Patch Size: (32, 32)
- Number of Transformer Blocks: 12
- Hidden Layer Width: 768

Text Transformer:
- Token Embedding Dim: 512
- Context Length (max tokens): 77
- Number of Transformer Blocks: 12
- Hidden Layer Width: 512

✅ Vision and Text Transformers are specialized:
- Vision Transformer: Wider, operates on spatial patches (images).
- Text Transformer: Narrower, processes sequences of tokens (text).


```
<hr style="height:10px; background-color:red; border:none;">

## Visualization of Embeddings (Text and Image)
``` 
================ VISUALIZATION OF EMBEDDINGS ================


✅ Embeddings visualized in 2D PCA space.

```
<hr style="height:10px; background-color:red; border:none;">

## Comparison with ResNet-based CLIP Encoder
``` 


================ COMPARISON: ViT vs ResNet CLIP ENCODERS ================

ViT-B/32 Vision Encoder:
- Type: Vision Transformer
- Patch size: 32x32
- Embedding dimension: 768
- Number of transformer layers: 12

ResNet-50 Vision Encoder:
- Type: ResNet-50 backbone
- Layers: conv1, layer1-4, attention_pool
- Embedding dimension (final): 1024
- ResNet Blocks: Sequential(
  (0): Bottleneck(
    (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
    (downsample): Sequential(
      (-1): AvgPool2d(kernel_size=1, stride=1, padding=0)
      (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (1): Bottleneck(
    (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
  (2): Bottleneck(
    (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
), Sequential(
  (0): Bottleneck(
    (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
    (downsample): Sequential(
      (-1): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (1): Bottleneck(
    (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
  (2): Bottleneck(
    (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
  (3): Bottleneck(
    (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
), Sequential(
  (0): Bottleneck(
    (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
    (downsample): Sequential(
      (-1): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (1): Bottleneck(
    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
  (2): Bottleneck(
    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
  (3): Bottleneck(
    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
  (4): Bottleneck(
    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
  (5): Bottleneck(
    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
), Sequential(
  (0): Bottleneck(
    (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
    (downsample): Sequential(
      (-1): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (1): Bottleneck(
    (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
  (2): Bottleneck(
    (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu1): ReLU(inplace=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu2): ReLU(inplace=True)
    (avgpool): Identity()
    (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu3): ReLU(inplace=True)
  )
)

✅ ViT encodes the image by splitting it into fixed-size patches processed by transformers.
✅ ResNet processes the entire image hierarchically using convolutional blocks and global attention pooling.


```
<hr style="height:10px; background-color:red; border:none;">


## Explanation and Use of QuickGELU Activation
``` 
================ EXPLANATION: QuickGELU vs GELU ================


✅ QuickGELU approximates GELU for faster computation while maintaining performance.


```
<hr style="height:10px; background-color:red; border:none;">



```

✅ CLIP Decomposition Complete. You can now fine-tune specific parts!



```