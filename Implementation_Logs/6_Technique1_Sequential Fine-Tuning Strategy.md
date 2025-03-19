## ✅ Phase 1: **Sequential Fine-Tuning Strategy**
We will fine-tune HandCLIP **step-by-step**, moving from **dorsal right** → **dorsal left** → **palmar right** → **palmar left**, reusing the previous fine-tuned model at each stage.

---

### ✅ Step 1: **Prepare Your Dataset Splits**
Make sure the following folders are ready for each aspect:
```
./11k/train_val_test_split_dorsal_r/
├── train/
├── val/

./11k/train_val_test_split_dorsal_l/
├── train/
├── val/

./11k/train_val_test_split_palmar_r/
├── train/
├── val/

./11k/train_val_test_split_palmar_l/
├── train/
├── val/
```
➡️ You'll fine-tune on `train` and validate on `val` in each step.

---

### ✅ Step 2: **Fine-Tune on Dorsal Right (Stage 1)**  
- Train your **HandCLIP model** on `train_val_test_split_dorsal_r/train`  
- Validate on `train_val_test_split_dorsal_r/val`  
- Save the **fine-tuned model checkpoint** as:
  ```
  handclip_finetuned_dorsal_r.pth
  ```

---

### ✅ Step 3: **Fine-Tune on Dorsal Left (Stage 2)**  
- Load **handclip_finetuned_dorsal_r.pth**  
- Fine-tune on `train_val_test_split_dorsal_l/train`  
- Validate on `train_val_test_split_dorsal_l/val`  
- Save as:
  ```
  handclip_finetuned_dorsal_rl.pth
  ```

---

### ✅ Step 4: **Fine-Tune on Palmar Right (Stage 3)**  
- Load **handclip_finetuned_dorsal_rl.pth**  
- Fine-tune on `train_val_test_split_palmar_r/train`  
- Validate on `train_val_test_split_palmar_r/val`  
- Save as:
  ```
  handclip_finetuned_dorsal_rlp.pth
  ```

---

### ✅ Step 5: **Fine-Tune on Palmar Left (Stage 4)**  
- Load **handclip_finetuned_dorsal_rlp.pth**  
- Fine-tune on `train_val_test_split_palmar_l/train`  
- Validate on `train_val_test_split_palmar_l/val`  
- Save as:
  ```
  handclip_finetuned_final_all_views.pth
  ```

---

## ✅ Phase 2: **Post Fine-Tuning Evaluation**
Now you have a **unified HandCLIP model** fine-tuned across **all hand aspects**.

---

### ✅ Step 6: **Evaluate on Query & Gallery Splits**
- Evaluate on:
  - `queryX` + `galleryX`  
  - `queryX` + `galleryX_all`  
- Run evaluations for **each aspect**, check **Rank-1** and **mAP**.  
- Verify **cross-aspect matching** improvements.

---

### ✅ Step 7: **Optional: Evaluate on the `test` Set**
- Evaluate in **ReID mode**, **not classification**, since test IDs are unseen.  
- Use a **query-gallery split**, even for test identities.

---

## ✅ Phase 3: **Optional Improvements After Sequential Fine-Tuning**
Once you have **sequential fine-tuning** working, you can consider further refinements.

---

### ➡️ **1. Unfreeze More Layers**  
- During each fine-tuning step, gradually **unfreeze more layers** in CLIP.  
- Early stages: Fine-tune only **classification head + some encoder layers**  
- Later stages: Unfreeze **more encoder blocks**.

---

### ➡️ **2. Contrastive Loss Fine-Tuning (Optional Later)**  
- After sequential fine-tuning, fine-tune the final model using **image-text contrastive learning**.  
- Prepare **aspect-specific prompts**, e.g.:  
  - `"A dorsal right hand of person ID XXXX"`

---

### ➡️ **3. Hard Negative Mining**
- Identify difficult pairs (false positives/negatives).  
- Use **hard samples** to fine-tune the model for **better discrimination**.

---

### ➡️ **4. Data Augmentation & Regularization**
- Apply **augmentation** during fine-tuning (random crops, rotations).  
- Helps with **generalization** to unseen views and hands.

---

### ➡️ **5. Post-Processing with Reranking (Optional)**
- Apply **re-ranking algorithms** on similarity matrices to **boost mAP and Rank-1**.  
- Example: k-reciprocal re-ranking.

---

## ✅ Phase 4: **Deploy & Evaluate Final Model**
Once everything is complete:
- Save final model:  
  ```
  handclip_finetuned_final_all_views.pth
  ```
- Evaluate on **all query-gallery splits**, record **Rank-1/mAP**.  
- Compare with your **baseline CLIP results**.

---

## ✅ File Naming Suggestions
| **Stage**                     | **Checkpoint Filename**                          |
|-------------------------------|--------------------------------------------------|
| After Dorsal Right            | `handclip_finetuned_dorsal_r.pth`               |
| After Dorsal Right + Left     | `handclip_finetuned_dorsal_rl.pth`              |
| After Dorsal + Palmar Right   | `handclip_finetuned_dorsal_rlp.pth`             |
| After Dorsal + Palmar + Left  | `handclip_finetuned_final_all_views.pth`        |

---
