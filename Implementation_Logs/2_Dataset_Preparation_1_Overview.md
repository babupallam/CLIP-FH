# 📁 Dataset: **11k Hands Dataset**

## ✨ Purpose
This document describes the preprocessing steps for the **11k Hands Dataset**, which are essential to prepare the data for training, validation, testing, and evaluating the HandCLIP model. This preparation ensures the dataset is structured correctly for **biometric recognition** and **Re-Identification (Re-ID)** tasks.

---

## 📝 Dataset Overview
- **Dataset Name**: 11k Hands Dataset  
- **Data Location**: `./11k`  
- **Components**:  
  - Hand images categorized by **aspect** (Dorsal/Palmar) and **side** (Left/Right)  
  - Metadata CSV file: `HandInfo.csv`  
    - `id`: Subject identifier  
    - `aspectOfHand`: View (dorsal/palmar, left/right)  
    - `accessories`: Indicates if accessories are present (excluded in processing)  
    - `imageName`: Name of the image file  

---

## ⚙️ Preprocessing Pipeline Overview
The Python script **`prepare_train_val_test_11k_r_l.py`** performs the following tasks:

---

### ✅ Step 1: Read Dataset and Metadata
- Loads the **11k Hands Dataset** and the `HandInfo.csv` metadata file.
- Filters out images with **accessories** (`accessories == 0`).

---

### ✅ Step 2: Create Directory Structure
For each hand aspect and side (Dorsal Right/Left, Palmar Right/Left), it creates the following directories:
```
train_val_test_split_dorsal_r/
├── train_all/
├── train/
├── val/
├── test/
├── query0/ .. query9/
├── gallery0/ .. gallery9/
├── gallery0_all/ .. gallery9_all/
```

---

### ✅ Step 3: Split Into Train, Validation, and Test Sets
- **train_all**: Contains **both** training and validation samples (used to create train/val splits).
- **train**: N-1 samples from the first half of identities.
- **val**: 1 sample per identity (randomly selected).
- **test**: Contains all images from the **second half** of identities.

---

### ✅ Step 4: Query & Gallery Split (Re-ID Setup)
- Performs **10 Monte Carlo runs (N=10)**.
- For each run:
  - Randomly selects **one image per identity** for the **gallery**.
  - Remaining images are used as **queries**.
- Separate query and gallery folders are generated for:
  - Dorsal Right/Left  
  - Palmar Right/Left  

---

### ✅ Step 5: Generate `gallery_all` Folders (Cross-Aspect Evaluation)
- Combines gallery folders from different aspects into `gallery_all`.
- Unique **ID offsets** ensure identities from different views do not collide:
  - +11,000,000 for dorsal left
  - +21,000,000 for palmar right
  - +31,000,000 for palmar left
- Enables **cross-aspect** evaluation (e.g., Dorsal Right query to Palmar Left gallery).

---

## 📂 Output Folder Structure
```
11k/
├── Hands/                        # Raw images
├── train_val_test_split_dorsal_r/
│   ├── train_all/                # Train + Val combined
│   ├── train/                    # Train subset
│   ├── val/                      # Validation subset
│   ├── test/                     # Test identities (not in train/val)
│   ├── query0/ .. query9/        # Queries for Re-ID evaluations
│   ├── gallery0/ .. gallery9/    # Galleries for Re-ID evaluations
│   └── gallery0_all/ .. gallery9_all/ # Combined galleries for cross-aspect testing
├── train_val_test_split_dorsal_l/ (same structure)
├── train_val_test_split_palmar_r/ (same structure)
├── train_val_test_split_palmar_l/ (same structure)
```

---

## ✅ Why This Is Necessary
1. **Consistent Splitting**: Ensures proper separation of **train/val/test** sets to avoid data leakage.
2. **Re-ID Task Readiness**: Prepares **query/gallery splits** needed for **Re-ID model training and evaluation**.
3. **Cross-Aspect Testing**: `gallery_all` allows **cross-view evaluation**, testing the model’s ability to match across hand views.
4. **Reproducibility**: Monte Carlo runs provide **robust performance metrics** by averaging results over multiple splits.

---

## 🔧 Files Involved
| File Name                         | Description                                                |
|---------------------------------- |------------------------------------------------------------|
| `prepare_train_val_test_11k_r_l.py` | Main script for dataset preparation (this document explains its steps). |
| `HandInfo.csv`                   | Metadata file with details for each image in the dataset.  |
| `Hands/`                         | Original images directory (dorsal/palmar views).           |

---

## 📝 Key Points and Observations
- **Accessories are excluded** to ensure consistent biometric features.
- **Random sampling** is used to ensure unbiased **train/val/test** splits.
- **ID offsets** in `gallery_all` prevent conflicts and ensure **unique identity mapping** across different hand aspects.
- This step is **critical** to enable the **HandCLIP fine-tuning** and **evaluation pipeline**.

---

## ✅ Next Steps After Dataset Preparation
1. **Feature Extraction**  
   ➡️ Use the **CLIP model** to extract image features from query and gallery sets.

2. **HandCLIP Fine-Tuning**  
   ➡️ Train the **HandCLIP model** using the `train` and `val` sets.

3. **Re-ID Evaluation**  
   ➡️ Perform evaluation using the `query` and `gallery` folders.

---

Let me know if you'd like to create more documentation files for the next steps!