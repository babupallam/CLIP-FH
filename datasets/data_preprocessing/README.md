
## Databases

## prepare_11k.py 

```
# From your project root:
python datasets/data_preprocessing/prepare_11k.py --data_path ./datasets/11khands --aspects "dorsal right" "dorsal left"

#Just dorsal right:
python datasets/data_preprocessing/prepare_11k.py --aspects "dorsal right"

# All aspects:
python datasets/data_preprocessing/prepare_11k.py --aspects "dorsal right" "dorsal left" "palmar right" "palmar left"


```


## prepare_hd.py

```
python datasets/data_preprocessing/prepare_hd.py \
    --data_path ./datasets/HD/Original\ Images \
    --runs 10 \
    --random_gallery

```


 **Explanation of the Arguments**

| Argument             | Value                             | Explanation                                                                                     |
|----------------------|-----------------------------------|-------------------------------------------------------------------------------------------------|
| `--data_path`        | `./datasets/HD/Original Images`   | This should point to the **root folder** where you have `1-501/` and `502-712/` inside HD.     |
| `--runs`             | `10`                             | Number of **query/gallery Monte Carlo runs** (like in your original script).                    |
| `--random_gallery`   | *flag* (True when present)        | Randomly selects **gallery images** from the test set (same as in your original code behavior). |

---

 **Directory Expectation Before Running**
```
datasets/
 HD/
     Original Images/
         1-501/               # Contains training/test images with IDs 1-501
         502-712/             # Additional gallery identities (211 subjects)
```

---

 **What This Command Will Do**
1. Split `1-501` images into:
   - `train_all` (IDs 1-447)
   - `test` (IDs 448-501)
2. From `train_all`, split into `train` and `val` (random val sample per subject).
3. Perform `runs` number of query/gallery splits:
   - For each, random gallery image + others as query.
   - Adds extra subjects from `502-712` into gallery.
4. Save everything into:
   ```
   datasets/HD/Original Images/train_val_test_split/
    train_all/
    train/
    val/
    test/
    query0/ ... query9/
    gallery0/ ... gallery9/
   ```

---
 **Optional Variations**
- If you do NOT want random gallery selection (and instead just include all images in gallery):
  ```bash
  python datasets/data_preprocessing/prepare_hd.py --data_path ./datasets/HD/Original\ Images --runs 10
  ```
  (Exclude `--random_gallery` flag.)

---


# validate_dataset_splits.py

- its for checking the splitting output when we do it with prepare_11k.py and prepare_hd.py


# Note (babu)
    - we need to change the three files: prepare_11k.py, prepare_hd.py, and alidate_dataset_splits.py,
        according to the original once...
