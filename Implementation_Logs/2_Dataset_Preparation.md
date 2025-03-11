# Dataset Preparation

## Dataset
**11k Hands Dataset**  
- Contains images of hand dorsals and palmars for both left and right hands.
- Includes metadata about each sample (e.g., hand aspect, presence of accessories).

## Objective
Prepare the 11k Hands dataset by:
1. Splitting it into **Train**, **Validation**, and **Test** sets.
2. Creating **Query** and **Gallery** sets for Re-Identification (Re-ID) evaluation.
3. Performing **Monte Carlo runs** for robust testing.

---

## Step 1: Dataset Structure Setup

### Input:
- Root data path: `./11k`
- Subdirectories:
  - `Hands/`: Contains all hand images.
  - `HandInfo.csv`: Metadata for each image.

### Output Directories Created:
- `train_val_test_split_dorsal_r` (right dorsal hand images)
- `train_val_test_split_dorsal_l` (left dorsal hand images)
- `train_val_test_split_palmar_r` (right palmar hand images)
- `train_val_test_split_palmar_l` (left palmar hand images)

Each output directory contains:
	train_val_test_split_<aspect>/ ├── train_all/ ├── train/ ├── val/ ├── test/


---

## Step 2: Train, Val, Test Splits

### Split Logic:
- **Train + Val** (aka `train_all`): First half of the identities.
- **Test**: Second half of the identities.

Within **train_all**:
- One random sample per identity is moved to the **Validation** set (`val`).
- Remaining samples are placed in **Train** (`train`).

### Selection Criteria:
- Exclude images with accessories (i.e., `accessories == 0`).
- Split based on `aspectOfHand` and `id`.

---

## Step 3: Query and Gallery Splits for Test Data (Re-ID)

### Purpose:
- Evaluate Re-ID performance using **Query** and **Gallery** sets.
- **Gallery**: One image per identity.
- **Query**: All other images.

### Method:
- Randomly pick one image as **Gallery**.
- Remaining images are **Query**.
- Repeat this **10 times** (Monte Carlo runs):
  - `query0` / `gallery0`
  - `query1` / `gallery1`
  - ...
  - `query9` / `gallery9`

---

## Step 4: Gallery All

### Purpose:
- Create **gallery_all** to combine all four aspects (dorsal_r, dorsal_l, palmar_r, palmar_l) in one directory for broader evaluation.

### ID Adjustments (to ensure uniqueness across aspects):
| Aspect       | Offset      |
|--------------|-------------|
| Dorsal Right | +0          |
| Dorsal Left  | +11,000,000 |
| Palmar Right | +21,000,000 |
| Palmar Left  | +31,000,000 |

### Example:
- If `id = 1001` in dorsal left, it becomes `11001001` in `gallery_all`.

---

## How the Code Works

### Dependencies
- `csv`: Read metadata from `HandInfo.csv`
- `os` and `shutil`: Create directories and copy files.
- `numpy`: Random selection for validation and query/gallery splits.

### Major Sections
1. **Directory Creation**: Ensures proper folder structure before copying images.
2. **Train/Val/Test Split**:
   - `train_all`: Identities <= threshold.
   - `test`: Identities > threshold.
3. **Query/Gallery Splits**:
   - Randomly selects one gallery image per identity.
4. **Gallery_All**:
   - Combines gallery images with unique IDs.

---

## Observations
- Dataset imbalance is mitigated by ensuring a validation image for each identity in `train_all`.
- Excludes images with accessories for consistency in recognition tasks.
- Monte Carlo runs ensure robust evaluation by randomizing query/gallery splits.

---

## Things to Do (Optional Enhancements)
- Verify the balance in train/val splits across identities.
- Automate augmentation during preprocessing.
- Log progress and errors during dataset preparation.

---

## Dataset Preparation Summary
| Operation         | Description                                   |
|-------------------|-----------------------------------------------|
| **Train Split**   | First half identities, excluding val samples  |
| **Val Split**     | 1 random image per identity from train_all    |
| **Test Split**    | Second half identities                        |
| **Query/Gallery** | Randomly select gallery images; 10 runs       |
| **Gallery All**   | Combine galleries with unique ID mapping      |

---

## Next Steps
- Augment train dataset (rotation, scale, lighting variations).
- Train baseline CLIP model on `train_all`.
- Evaluate baseline performance on query/gallery splits.

