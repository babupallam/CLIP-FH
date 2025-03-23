# Base Line Evaluation

## Output of 2_baseline_clip_evaluation_test.py

```
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python .\2_baseline_clip_evaluation.py
Loading CLIP model...
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 71/71 [00:09<00:00,  7.19it/s] 
Extracting features from: ./11k/train_val_test_split_dorsal_r/query0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 71/71 [02:36<00:00,  2.20s/it] 

Computing similarity between query and gallery images...

Evaluating performance...
Rank-1 Accuracy: 0.7209
Mean Average Precision (mAP): 0.7895

Baseline CLIP evaluation completed ✅

```
<hr style="height:10px; background-color:red; border:none;">

## Output of 2_baseline_clip_evaluation_dorsal_r.py

```
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python .\2_baseline_clip_evaluation.py
Loading CLIP model...

Running split 0...
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery0
Extracting features from: ./11k/train_val_test_split_dorsal_r/query0
Computing similarity between query and gallery images...
Evaluating performance...
Rank-1 Accuracy: 0.7209
Mean Average Precision (mAP): 0.7895

Running split 1...
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery1
Extracting features from: ./11k/train_val_test_split_dorsal_r/query1
Computing similarity between query and gallery images...
Evaluating performance...
Rank-1 Accuracy: 0.7662
Mean Average Precision (mAP): 0.8239

Running split 2...
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery2
Extracting features from: ./11k/train_val_test_split_dorsal_r/query2
Computing similarity between query and gallery images...
Evaluating performance...
Rank-1 Accuracy: 0.7384
Mean Average Precision (mAP): 0.8075

Running split 3...
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery3
Extracting features from: ./11k/train_val_test_split_dorsal_r/query3
Computing similarity between query and gallery images...
Evaluating performance...
Rank-1 Accuracy: 0.7250
Mean Average Precision (mAP): 0.7938

Running split 4...
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery4
Extracting features from: ./11k/train_val_test_split_dorsal_r/query4
Computing similarity between query and gallery images...
Evaluating performance...
Rank-1 Accuracy: 0.7209
Mean Average Precision (mAP): 0.7917

Running split 5...
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery5
Extracting features from: ./11k/train_val_test_split_dorsal_r/query5
Computing similarity between query and gallery images...
Evaluating performance...
Rank-1 Accuracy: 0.6941
Mean Average Precision (mAP): 0.7685

Running split 6...
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery6
Extracting features from: ./11k/train_val_test_split_dorsal_r/query6
Computing similarity between query and gallery images...
Evaluating performance...
Rank-1 Accuracy: 0.7065
Mean Average Precision (mAP): 0.7826

Running split 7...
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery7
Extracting features from: ./11k/train_val_test_split_dorsal_r/query7
Computing similarity between query and gallery images...
Evaluating performance...
Rank-1 Accuracy: 0.7291
Mean Average Precision (mAP): 0.8043

Running split 8...
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery8
Extracting features from: ./11k/train_val_test_split_dorsal_r/query8
Computing similarity between query and gallery images...
Evaluating performance...
Rank-1 Accuracy: 0.7199
Mean Average Precision (mAP): 0.7934

Running split 9...
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery9
Extracting features from: ./11k/train_val_test_split_dorsal_r/query9
Computing similarity between query and gallery images...
Evaluating performance...
Rank-1 Accuracy: 0.7230
Mean Average Precision (mAP): 0.7852

✅ Baseline CLIP Evaluation Across All Splits Completed!
Average Rank-1 Accuracy: 0.7244
Average Mean Average Precision (mAP): 0.7940

```
<hr style="height:10px; background-color:red; border:none;">

## Output of 2_baseline_clip_evaluation_all_vitb32.py

```
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python .\2_c_baseline_clip_evaluation_all.py
Loading CLIP model...

🚀 Evaluating Dorsal Right view...

🔹 Split 0 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_r\gallery0
Extracting query features from: ./11k\train_val_test_split_dorsal_r\query0
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7209
Mean Average Precision (mAP): 0.7895

🔹 Split 1 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_r\gallery1
Extracting query features from: ./11k\train_val_test_split_dorsal_r\query1
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7662
Mean Average Precision (mAP): 0.8239

🔹 Split 2 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_r\gallery2
Extracting query features from: ./11k\train_val_test_split_dorsal_r\query2
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7384
Mean Average Precision (mAP): 0.8075

🔹 Split 3 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_r\gallery3
Extracting query features from: ./11k\train_val_test_split_dorsal_r\query3
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7250
Mean Average Precision (mAP): 0.7938

🔹 Split 4 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_r\gallery4
Extracting query features from: ./11k\train_val_test_split_dorsal_r\query4
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7209
Mean Average Precision (mAP): 0.7917

🔹 Split 5 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_r\gallery5
Extracting query features from: ./11k\train_val_test_split_dorsal_r\query5
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6941
Mean Average Precision (mAP): 0.7685

🔹 Split 6 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_r\gallery6
Extracting query features from: ./11k\train_val_test_split_dorsal_r\query6
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7065
Mean Average Precision (mAP): 0.7826

🔹 Split 7 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_r\gallery7
Extracting query features from: ./11k\train_val_test_split_dorsal_r\query7
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7291
Mean Average Precision (mAP): 0.8043

🔹 Split 8 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_r\gallery8
Extracting query features from: ./11k\train_val_test_split_dorsal_r\query8
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7199
Mean Average Precision (mAP): 0.7934

🔹 Split 9 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_r\gallery9
Extracting query features from: ./11k\train_val_test_split_dorsal_r\query9
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7230
Mean Average Precision (mAP): 0.7852

✅ Finished evaluation on Dorsal Right
✅ Average Rank-1 Accuracy for Dorsal Right: 0.7244
✅ Average mAP for Dorsal Right: 0.7940

🚀 Evaluating Dorsal Left view...

🔹 Split 0 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_l\gallery0
Extracting query features from: ./11k\train_val_test_split_dorsal_l\query0
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7318
Mean Average Precision (mAP): 0.8041

🔹 Split 1 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_l\gallery1
Extracting query features from: ./11k\train_val_test_split_dorsal_l\query1
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7409
Mean Average Precision (mAP): 0.8054

🔹 Split 2 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_l\gallery2
Extracting query features from: ./11k\train_val_test_split_dorsal_l\query2
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7672
Mean Average Precision (mAP): 0.8292

🔹 Split 3 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_l\gallery3
Extracting query features from: ./11k\train_val_test_split_dorsal_l\query3
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7298
Mean Average Precision (mAP): 0.8008

🔹 Split 4 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_l\gallery4
Extracting query features from: ./11k\train_val_test_split_dorsal_l\query4
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7824
Mean Average Precision (mAP): 0.8495

🔹 Split 5 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_l\gallery5
Extracting query features from: ./11k\train_val_test_split_dorsal_l\query5
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7358
Mean Average Precision (mAP): 0.8084

🔹 Split 6 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_l\gallery6
Extracting query features from: ./11k\train_val_test_split_dorsal_l\query6
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7449
Mean Average Precision (mAP): 0.8046

🔹 Split 7 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_l\gallery7
Extracting query features from: ./11k\train_val_test_split_dorsal_l\query7
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7409
Mean Average Precision (mAP): 0.8021

🔹 Split 8 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_l\gallery8
Extracting query features from: ./11k\train_val_test_split_dorsal_l\query8
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7166
Mean Average Precision (mAP): 0.7884

🔹 Split 9 🔹
Extracting gallery features from: ./11k\train_val_test_split_dorsal_l\gallery9
Extracting query features from: ./11k\train_val_test_split_dorsal_l\query9
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.7551
Mean Average Precision (mAP): 0.8036

✅ Finished evaluation on Dorsal Left
✅ Average Rank-1 Accuracy for Dorsal Left: 0.7445
✅ Average mAP for Dorsal Left: 0.8096

🚀 Evaluating Palmar Right view...

🔹 Split 0 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_r\gallery0
Extracting query features from: ./11k\train_val_test_split_palmar_r\query0
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6652
Mean Average Precision (mAP): 0.7338

🔹 Split 1 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_r\gallery1
Extracting query features from: ./11k\train_val_test_split_palmar_r\query1
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6358
Mean Average Precision (mAP): 0.7187

🔹 Split 2 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_r\gallery2
Extracting query features from: ./11k\train_val_test_split_palmar_r\query2
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6467
Mean Average Precision (mAP): 0.7197

🔹 Split 3 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_r\gallery3
Extracting query features from: ./11k\train_val_test_split_palmar_r\query3
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6609
Mean Average Precision (mAP): 0.7289

🔹 Split 4 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_r\gallery4
Extracting query features from: ./11k\train_val_test_split_palmar_r\query4
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6587
Mean Average Precision (mAP): 0.7371

🔹 Split 5 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_r\gallery5
Extracting query features from: ./11k\train_val_test_split_palmar_r\query5
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6249
Mean Average Precision (mAP): 0.7109

🔹 Split 6 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_r\gallery6
Extracting query features from: ./11k\train_val_test_split_palmar_r\query6
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6390
Mean Average Precision (mAP): 0.7184

🔹 Split 7 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_r\gallery7
Extracting query features from: ./11k\train_val_test_split_palmar_r\query7
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6249
Mean Average Precision (mAP): 0.7105

🔹 Split 8 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_r\gallery8
Extracting query features from: ./11k\train_val_test_split_palmar_r\query8
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6510
Mean Average Precision (mAP): 0.7246

🔹 Split 9 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_r\gallery9
Extracting query features from: ./11k\train_val_test_split_palmar_r\query9
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6074
Mean Average Precision (mAP): 0.6846

✅ Finished evaluation on Palmar Right
✅ Average Rank-1 Accuracy for Palmar Right: 0.6414
✅ Average mAP for Palmar Right: 0.7187

🚀 Evaluating Palmar Left view...

🔹 Split 0 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_l\gallery0
Extracting query features from: ./11k\train_val_test_split_palmar_l\query0
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6561
Mean Average Precision (mAP): 0.7316

🔹 Split 1 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_l\gallery1
Extracting query features from: ./11k\train_val_test_split_palmar_l\query1
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6319
Mean Average Precision (mAP): 0.7034

🔹 Split 2 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_l\gallery2
Extracting query features from: ./11k\train_val_test_split_palmar_l\query2
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6224
Mean Average Precision (mAP): 0.7076

🔹 Split 3 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_l\gallery3
Extracting query features from: ./11k\train_val_test_split_palmar_l\query3
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6582
Mean Average Precision (mAP): 0.7288

🔹 Split 4 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_l\gallery4
Extracting query features from: ./11k\train_val_test_split_palmar_l\query4
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6656
Mean Average Precision (mAP): 0.7463

🔹 Split 5 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_l\gallery5
Extracting query features from: ./11k\train_val_test_split_palmar_l\query5
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6804
Mean Average Precision (mAP): 0.7584

🔹 Split 6 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_l\gallery6
Extracting query features from: ./11k\train_val_test_split_palmar_l\query6
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6150
Mean Average Precision (mAP): 0.6946

🔹 Split 7 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_l\gallery7
Extracting query features from: ./11k\train_val_test_split_palmar_l\query7
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6192
Mean Average Precision (mAP): 0.6986

🔹 Split 8 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_l\gallery8
Extracting query features from: ./11k\train_val_test_split_palmar_l\query8
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6224
Mean Average Precision (mAP): 0.7162

🔹 Split 9 🔹
Extracting gallery features from: ./11k\train_val_test_split_palmar_l\gallery9
Extracting query features from: ./11k\train_val_test_split_palmar_l\query9
Computing cosine similarity...
Evaluating...
Rank-1 Accuracy: 0.6466
Mean Average Precision (mAP): 0.7139

✅ Finished evaluation on Palmar Left
✅ Average Rank-1 Accuracy for Palmar Left: 0.6418
✅ Average mAP for Palmar Left: 0.7199

🎉 All views evaluated!
Dorsal Right: Rank-1 = 0.7244, mAP = 0.7940
Dorsal Left: Rank-1 = 0.7445, mAP = 0.8096
Palmar Right: Rank-1 = 0.6414, mAP = 0.7187
Palmar Left: Rank-1 = 0.6418, mAP = 0.7199

```
<hr style="height:10px; background-color:red; border:none;">


## Output of 3_a_handclip_finetune_dorsal_r_train
```
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python .\3_a_handclip_finetune_dorsal_r_train.py     
Using device: cpu
[Epoch 1/5] Train Loss: 4.6794, Train Acc: 0.1989
[Epoch 1/5] Val Acc: 0.2500
✅ Best model saved with val_acc: 0.2500
[Epoch 2/5] Train Loss: 3.3065, Train Acc: 0.7258
[Epoch 2/5] Val Acc: 0.7222
✅ Best model saved with val_acc: 0.7222
[Epoch 3/5] Train Loss: 2.1709, Train Acc: 0.9461
[Epoch 3/5] Val Acc: 0.8750
✅ Best model saved with val_acc: 0.8750
[Epoch 4/5] Train Loss: 1.4562, Train Acc: 0.9921
[Epoch 4/5] Val Acc: 0.9028
✅ Best model saved with val_acc: 0.9028
[Epoch 5/5] Train Loss: 1.0357, Train Acc: 0.9989
[Epoch 5/5] Val Acc: 0.9167
✅ Best model saved with val_acc: 0.9167

Training complete.
Best validation accuracy: 0.9166666666666666

```
<hr style="height:10px; background-color:red; border:none;">

## Output of 3_a_handclip_finetune_dorsal_r_eval_multi.py
```
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python .\3_a_handclip_finetune_dorsal_r_eval_multi.py
Using device: cpu
Loaded model from ./models/handclip_finetuned_model_dorsal_r.pth
[Split 0] Rank-1: 0.8733, mAP: 0.9096
[Split 1] Rank-1: 0.9032, mAP: 0.9348
[Split 2] Rank-1: 0.9022, mAP: 0.9296
[Split 3] Rank-1: 0.9197, mAP: 0.9417
[Split 4] Rank-1: 0.9279, mAP: 0.9502
[Split 5] Rank-1: 0.8774, mAP: 0.9134
[Split 6] Rank-1: 0.8980, mAP: 0.9228
[Split 7] Rank-1: 0.8970, mAP: 0.9312
[Split 8] Rank-1: 0.9022, mAP: 0.9286
[Split 9] Rank-1: 0.8898, mAP: 0.9247

=== Final Results over 10 splits ===
Mean Rank-1: 0.8991
Mean mAP:    0.9287

```
<hr style="height:10px; background-color:red; border:none;">

## Output of 3_a_handclip_finetune_dorsal_r_eval_multi_gallery_all.py

###  Cross-Aspect Re-ID
- Evaluate each query split (query0 to query9) against the galleryX_all folders.
- These galleryX_all folders likely contain combined galleries from different aspects or entire datasets, simulating a cross-aspect Re-ID task.
- Compute Rank-1 and mAP for each query/gallery_all pair, then average across all splits.

```
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python 3_a_handclip_finetune_dorsal_r_eval_multi_gallery_all.py
Using device: cpu
Loaded model from ./models/handclip_finetuned_dorsal_r.pth
[Split 0] Rank-1: 0.7508, mAP: 0.8005
[Split 1] Rank-1: 0.8084, mAP: 0.8398
[Split 2] Rank-1: 0.7806, mAP: 0.8146
[Split 3] Rank-1: 0.8177, mAP: 0.8416
[Split 4] Rank-1: 0.8167, mAP: 0.8417
[Split 5] Rank-1: 0.7642, mAP: 0.8049
[Split 6] Rank-1: 0.7549, mAP: 0.8045
[Split 7] Rank-1: 0.7817, mAP: 0.8195
[Split 8] Rank-1: 0.7806, mAP: 0.8194
[Split 9] Rank-1: 0.7817, mAP: 0.8136

=== Final Cross-Aspect Results over 10 splits ===
Mean Rank-1: 0.7837
Mean mAP:    0.8200

```
### Observation
 - Good result for cross-aspect Re-ID, though the training was aspect-specific (dorsal_r).
 - Results
   - Rank-1 (78%): Out of 100 searches, 78 times the first suggestion is the right person.
   - mAP (82%): Across all searches, the system is usually very accurate in ranking the correct people close to the top, not leaving them far down the list.
 
<hr style="height:10px; background-color:red; border:none;">

