# How Re-identification Work?

### **1. The Overall Goal**

Re-Identification means that, given an image of a person (or hand, in your case) in a **Query** set, we want to figure out whether there is a matching image of the **same** person in the **Gallery** set. In your specific example:

- **Query set (`query0`)**: Contains **multiple images** for each identity (e.g., `0001051` has several images).
- **Gallery set (`gallery0`)**: Contains only **one image** per identity (e.g., `0001051` has just one image).

---

### **2. CLIP’s Role**

OpenAI’s CLIP model has **two main encoders**:

1. **Image Encoder** – This converts images into a numerical feature vector (embedding).
2. **Text Encoder** – This converts text prompts into a numerical feature vector (same dimensional space as the image embeddings).

**In your code**, only the **image encoder** is being used.  
- You do **not** call `model.encode_text(...)`.
- You **do** call `model.encode_image(...)` for each image in both query and gallery.

---

### **3. Feature Extraction**

When you run:
```python
query_feats, query_labels = extract_features(model, query_loader, device)
gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
```
Each image in the query set is fed through the **image encoder** to produce a **feature vector**. The same happens for the gallery images.

- If an identity (e.g., `0001051`) has 5 images in the query set, you’ll get **5 feature vectors** for that identity.  
- If the gallery folder has 1 image for the same identity, you’ll get **1 feature vector** for that identity in the gallery.

All of these are combined into two big tensors:  
- `query_feats` = \[Number of total query images\] x \[embedding dimension\]  
- `gallery_feats` = \[Number of gallery images\] x \[embedding dimension\]

---

### **4. Computing Cosine Similarity**

Next, the code does:
```python
sim_matrix = torch.matmul(query_feats, gallery_feats.T)
```
Because your features are normalized (L2-normalized) by `normalize(features, dim=1)`, this **dot product** effectively calculates the **cosine similarity** between every query image and every gallery image.

- **Rows** in `sim_matrix` = each query image  
- **Columns** in `sim_matrix` = each gallery image  

So if you have 5 query images for `0001051` and 1 gallery image for `0001051`, you end up with **5 similarity scores** (one per query image) against that single gallery image. You also get scores against all the **other** gallery images in the dataset.

---

### **5. How ReID Decides the Match**

With the similarity scores, your evaluation code (`evaluate_rank`) checks:

1. For each **query image**, which gallery identity has the **highest cosine similarity**?  
2. Is that highest similarity from the **correct** identity’s gallery image?  
3. Based on how many correct matches you get, it calculates **mAP** (mean Average Precision), **CMC** (Cumulative Matching Characteristic), and so on.

Even though you have **multiple query images** for a single identity, the system treats them **independently** during similarity computation (each has its own row in the matrix). In the final evaluation, the metrics still recognize they belong to the same person, so you’ll see if they match the correct gallery entry.

---

### **6. Involving the Text Encoder (Optional Context)**

- **Text Encoder** in CLIP: 
  - If you had a caption or textual prompt describing the hand or person (e.g., “A photo of ID 0001051’s dorsal hand”), you could encode that text with `model.encode_text(...)`.  
  - Then you could compare image embeddings to text embeddings in the **same** vector space. 

**However**, in your ReID workflow, **you are not using any text descriptions**, so the **text encoder is not used**. Everything stays image-to-image.

---

### **7. Simple Example**

1. **Query**: Identity `0001051` has 3 images.
2. **Gallery**: Identity `0001051` has 1 image.
3. Each image goes through the CLIP image encoder → returns a feature vector.
   - Query might get 3 vectors: \[Q1, Q2, Q3\]
   - Gallery might get 1 vector: \[G\]

4. The similarity matrix rows = \[Q1, Q2, Q3\], columns = \[G\].
   - So we get 3 similarity scores:  
     - `CosSim(Q1, G)`  
     - `CosSim(Q2, G)`  
     - `CosSim(Q3, G)`
5. If they’re sufficiently high, it indicates the query images match the single gallery image of `0001051`.

That’s how the algorithm confirms who’s who in the dataset, all via **cosine similarity** of image embeddings.

---

**In short:**

1. **Image encoder** transforms each image → feature vector.  
2. **Cosine similarity** is computed for every query image vs. every gallery image.  
3. **Evaluation** checks if the top-matching gallery image is from the correct identity.  
4. **Text encoder** is not used since we’re only dealing with images, not text queries.


***
***
***
***



