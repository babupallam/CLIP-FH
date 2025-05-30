#  **Why Distractors Are Present Only in the HD Dataset: A Comprehensive Analysis**

---

##  **Section 1: Introduction  Understanding the Dataset Context**

In hand-based biometric recognition research, dataset construction plays a pivotal role in evaluating the robustness and generalizability of models. When reviewing datasets like **11k Hands** and **HD (Hong Kong Polytechnic University Hand Dorsal)**, one finds a key design difference:

> Only the **HD dataset** includes an additional set of **213 distractor identities** in its gallery set, whereas the **11k Hands** dataset does not.

Understanding **why these distractors are added**, **how they are selected**, and **what role they serve** is crucial to interpreting evaluation results and designing effective ReID models.

---

###  **What Are "Distractor Identities"?**

In the context of Re-Identification (ReID) and biometric search systems:

- **Query set** contains images of identities you're trying to find.
- **Gallery set** is your search database  it contains one or more images of known identities.
- **Distractor identities** are additional images in the **gallery** that **do not** belong to the test set of known identities.

They act as **decoys**  visually similar samples that increase the difficulty of retrieval. The model must correctly **ignore these distractors** and still retrieve the correct match from the gallery.

---

###  **Sources of Data: 11k vs HD**

| Dataset     | Total IDs | Train/Test Split | Distractors |
|-------------|-----------|------------------|-------------|
| 11k Hands   | 190       | Even split (e.g., 71/72 IDs) |  No distractors used |
| HD Dataset  | 502       | 251 train / 251 test         |  213 distractor identities added to gallery |

This distinction is explicitly explained in the MBA-Net paper, and it aligns with earlier benchmark setups like [2] that they follow.

---

###  **How Are Distractors Added in HD?**

From the paper:
> *"The HD dataset has additional images of 213 subjects, which lack clarity or do not have second minor knuckle patterns, and are added to the HD gallery."*

This means:

- These 213 distractors are **not included in training or query**
- They are **real identities**, but their images are considered **noisy**, **blurred**, or **biometrically incomplete**
- They are placed **only in the gallery**, to **act as hard negatives**

In practice, one image from each test identity is used in the gallery. Then, **213 more identities** (with potentially confusing or similar hand images) are added to that same gallery.

---

###  **Purpose and Motivation Behind Adding Distractors**

The reason for adding distractors  and only to the HD dataset  is rooted in the **real-world simulation of biometric systems**:

1. **Real-World Search Has Distractors:**
   In biometric verification (e.g., border control, forensics), the gallery is rarely clean. It includes **people not in your target query set**, and your system must filter them out.

2. **HD Dataset Enables It:**
   The HD dataset includes these 213 extra identities **explicitly for this purpose**, labeled as low-quality or incomplete  the 11k dataset does not have such "extra" unused identities.

3. **Evaluating Robustness:**
   The goal is to test:
   > Can the model still retrieve the correct identity from the gallery, even when unrelated and visually similar hands are present?

   Its a **harder** and more **realistic** challenge.

---

###  **Why This Is NOT Done for 11k Dataset**

The 11k Hands dataset:
- Is smaller (143151 identities per subset)
- Lacks a clear distinction between "valid" and "distractor" identities
- Every identity is either used for training or testing
- No leftover "junk" identities are available to act as clean distractors

Hence, the evaluation for 11k is **simpler**  there are no distractors in its gallery.

---

 **In summary for Section 1:**

> **Distractors are present only in the HD dataset** because:
>
> - The dataset **includes additional images of 213 real but low-quality identities**
> - These were explicitly added to the gallery to simulate **real-world noise**
> - They increase the **difficulty of the retrieval task**
> - This helps test the model's ability to **avoid false positives** in complex scenarios
> - The 11k dataset lacks such extra identities and is therefore evaluated on cleaner gallery-query splits

---

#  **Section 2: How Distractors Affect Evaluation and Why Handling Them Correctly Is Critical**

Having understood why distractors are present in the HD dataset, we now need to explore **how they influence evaluation**, especially metrics like **Rank-k** and **mean Average Precision (mAP)**. This section will also explain **why improper handling can silently break your evaluation** and lead to misleading results  like `Rank-1 = 0.0`.

---

##  **Quick Recap of ReID Evaluation Metrics**

Before diving into the impact of distractors, let's revisit the **two primary ReID metrics**:

###  **Rank-k Accuracy**
- Measures how often the correct match appears in the **top-k retrieved gallery results** for each query.
- Rank-1 means the **top match is correct**.
- If the correct identity is **not in the gallery** or incorrectly labeled, **Rank-1 will be 0**.

###  **mean Average Precision (mAP)**
- Measures **how well the model ranks** all relevant matches (precision-recall curve).
- Affected by the number of **true positives**, **false positives**, and **ranking order**.
- If the correct label doesnt align between query and gallery, or is overwhelmed by distractors, **mAP drops significantly**.

---

##  **How Distractors Change Evaluation**

### 1 They **inflate the gallery size**
- Gallery goes from 251 (1 per ID) to **464** (251 test IDs + 213 distractors).
- This makes matching **harder**, especially if distractors look similar.

### 2 They **dont have matching queries**
- None of the 213 distractors appear in the query set.
- But if these distractors are given **class labels**, they will be **evaluated as if they are valid match candidates**.

### 3 They **can be retrieved instead of the correct identity**
- If a distractor image is visually closer to the query than the true gallery match, it may appear in the top-k.
- Since it **has a different label**, it counts as a **false positive**  reducing Rank-1 and mAP.

---

##  **What Goes Wrong in Code (Common Mistakes)**

###  Problem: Distractors Are Treated as Labeled Classes
When you use PyTorchs `ImageFolder`, it assigns a **numeric label** to **every folder**:
```python
gallery_dataset = ImageFolder("gallery0")
```

This results in:
- Real IDs like `448  label 0`, `449  label 1`, etc.
- Distractors like `1001  label 251`, `1002  label 252`, etc.

But your query dataset only uses labels `0250`.  
So **during evaluation**, your system might compare:
```python
query_label = 0   # (ID 448)
gallery_label = 251  # (ID 1001)
```
and **assume it's a valid negative**, even if the model retrieved the correct person.

###  Problem: Label Mismatch
Even worse, if the distractors are added **before** real IDs (alphabetically), they shift the label mapping:
- `1001  label 0`
- `448  label 213`

Now your query expects label 0, but it's assigned to a distractor  **Rank-1 = 0, even for perfect retrieval**.

---

##  Whats the Correct Way to Handle Distractors?

| Correct Handling                           | Why It Matters                                        |
|--------------------------------------------|-------------------------------------------------------|
| Include distractor images in similarity matrix | So they act as hard negatives                        |
| Exclude distractor labels from evaluation    | So the model isnt penalized for failing to match them |
| Ensure label mapping is **aligned** between query and gallery | So that correct IDs match numerically                 |
| Use folder names (IDs) instead of numeric labels | Avoids misalignment caused by `ImageFolder` sorting   |

---

##  Example: Effect on Evaluation

Lets say:
- Query image: `ID 448`  label `0`
- Gallery images:
  - Correct match (ID 448)  label `0`
  - Distractor (ID 1001)  label `1` or `251` (depends on sorting)

### Scenario A: Labels aligned  perfect match
- Model retrieves correct image (label 0) first
-  Rank-1 = 1.0

### Scenario B: Label mismatch or distractor overlap
- Model retrieves correct image, but label is `213` (due to misalignment)
-  Rank-1 = 0, even though it was right visually

---

##  Summary of Section 2

> Distractors are **meant to challenge the model**, but they must be **handled with care** in evaluation.

If not:
- They corrupt your label mapping
- They get treated as valid identities
- And you get misleading results like `Rank-1 = 0`, `mAP  0.007`

The key takeaway:
> Distractors should **influence ranking (hard negatives)** but not be **considered valid class labels** during evaluation.

---

#  **Section 3: How the Distractor Handling in Code Affected Your Baseline Output  A Step-by-Step Example**

In this section, well analyze how the **code you used to handle HD dataset splitting and evaluation** affected the results in the baseline ReID experiment  particularly focusing on:

- Which files were involved
- Where the mistakes were introduced
- How the distractor setup silently led to poor performance

Well walk through this using **Split 0 (gallery0 & query0)** of the **HD dorsal dataset**.

---

##  The Goal

You want to evaluate how well CLIP (e.g., ViT-B/16) performs on **gallery0 vs query0**, using cosine similarity between embeddings.

Expected result:  
> If CLIP can correctly match the same identities (e.g., ID `448`) across query and gallery, you should see decent **Rank-1** and **mAP** scores.

---

##  The Result You Got

From your `.log` files:
```
Rank-1: 0.0000
Rank-5: 0.0003
mAP   : 0.0074
```

This means:
> Your model **never retrieved the correct top-1 match**, and barely retrieved correct identities even in top-10.

Thats **not a CLIP problem**  its a **data-label mismatch** problem.

---

##  Step-by-Step Breakdown Using Your Files

---

###  1. **`prepare_train_val_test_hd.py`  The Splitter**

What it did:

 For each test identity (`448698`):
- Selected one image  `gallery0/448/`
- Remaining images  `query0/448/`

 **Also added 213 more folders** to `gallery0` from extra identities (e.g., IDs `502712`  `10011212`, etc.)

These folders looked like:
```
gallery0/
   448/
   449/
  ...
   1001/
   1002/
  ...
```

Thats where **distractors entered**.

---

###  2. **`datasets/build_dataloader.py`  The DataLoader**

You used:
```python
dataset = ImageFolder(root=data_dir, transform=...)
```

 **Problem here:**
- `ImageFolder` auto-assigns numeric labels in folder name order
- So:
  - If `gallery0` has folders `1001`, `448`, `449`, ...
  - It may assign:
    - `1001  label 0`
    - `448  label 1`
    - `449  label 2`
    - ...

But:
- In `query0`, the folders are only `448698`, so:
  - `448  label 0`
  - `449  label 1`
  - ...

 So even if the model retrieves image from `gallery0/448/`, the label might be `1`, not `0`  **label mismatch!**

---

###  3. **`engine/baseline_inference.py`  Feature Extraction**

This file uses:
```python
features = model.encode_image(images)
```
 Embeddings are fine.

But then, it saves:
```python
return torch.cat(all_features), torch.cat(all_labels)
```
And those labels come directly from `ImageFolder`  meaning:

> The **numerical label** attached to each image may **not correspond to the real identity**, if gallery and query were mapped differently.

---

###  4. **`engine/evaluator.py`  Evaluation**

This uses:
```python
metrics = evaluate_rank(sim_matrix, query_labels, gallery_labels)
```

It assumes:
> `query_label == gallery_label  correct match`

 But thats broken due to different label orders!

So:
- Even if CLIP retrieves the correct image visually (same ID, like `448`)
- The numeric labels are off (e.g., `query: 0`, `gallery: 1`)
-  So it's counted as a **wrong match**

 This explains:
- `Rank-1 = 0`
- `mAP = 0.007`

Even though CLIP may have found the right image, **evaluation logic failed** due to label mismatch.

---

##  What You Should Do

| Step | Fix |
|------|-----|
|  Ensure same `class_to_idx` in query & gallery | Force both to use the same label mapping |
|  Exclude distractor labels from `class_to_idx` | Add them to `gallery`, but not to label space |
|  Validate label mapping during inference | Print a few sample `image, label` from both sets |

---

##  Summary of What Happened

> Distractors were added (correctly) to increase difficulty,  
> But PyTorchs `ImageFolder` mapped their folders **before the real IDs**,  
> Leading to **label shifts**,  
> So your evaluator compared the **wrong labels**,  
> Resulting in **Rank-1 = 0**, even when the visual match was right.

---

