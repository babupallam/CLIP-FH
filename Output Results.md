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
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python .\3_a_handclip_finetune_train.py
Using device: cpu
[Epoch 1/5] Train Loss: 4.5602, Train Acc: 0.2225
[Epoch 1/5] Val Acc: 0.2361
✅ Best model saved with val_acc: 0.2361
[Epoch 2/5] Train Loss: 3.1768, Train Acc: 0.7000
[Epoch 2/5] Val Acc: 0.6667
✅ Best model saved with val_acc: 0.6667
[Epoch 3/5] Train Loss: 2.0368, Train Acc: 0.9483
[Epoch 3/5] Val Acc: 0.8889
✅ Best model saved with val_acc: 0.8889
[Epoch 4/5] Train Loss: 1.3493, Train Acc: 0.9888
[Epoch 4/5] Val Acc: 0.9167
✅ Best model saved with val_acc: 0.9167
[Epoch 5/5] Train Loss: 0.9520, Train Acc: 0.9989
[Epoch 5/5] Val Acc: 0.9167
Training completed. Best validation accuracy: 0.9166666666666666

```

<hr style="height:10px; background-color:red; border:none;">

## Output of 3_a_handclip_finetune_dorsal_r_evaluate_q_g.py (testing with one split)
```
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python .\3_a_handclip_finetune_evaluate.py
Using device: cpu

Loading CLIP model...
Extracting features from: ./11k/train_val_test_split_dorsal_r/query0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [01:55<00:00,  3.73s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery0
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.69s/it] 

Computing similarity between query and gallery images...

Evaluating performance...
Rank-1 Accuracy: 0.9156
Mean Average Precision (mAP): 0.9433

HandCLIP evaluation completed ✅

```

<hr style="height:10px; background-color:red; border:none;">

## Output of 3_a_handclip_finetune_dorsal_r_evaluate_q_g.py 
```
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python .\3_a_handclip_finetune_evaluate.py
Using device: cpu

========== Evaluating Split 0 ==========

Extracting features from: ./11k/train_val_test_split_dorsal_r\query0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [01:52<00:00,  3.64s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r\gallery0
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.67s/it] 

Computing similarity between query and gallery images...

[Split 0] Rank-1 Accuracy: 0.9156
[Split 0] Mean Average Precision (mAP): 0.9433

========== Evaluating Split 1 ==========

Extracting features from: ./11k/train_val_test_split_dorsal_r\query1
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:21<00:00,  4.55s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r\gallery1
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:13<00:00,  4.60s/it] 

Computing similarity between query and gallery images...

[Split 1] Rank-1 Accuracy: 0.9475
[Split 1] Mean Average Precision (mAP): 0.9669

========== Evaluating Split 2 ==========

Extracting features from: ./11k/train_val_test_split_dorsal_r\query2
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:08<00:00,  4.13s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r\gallery2
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:06<00:00,  2.30s/it] 

Computing similarity between query and gallery images...

[Split 2] Rank-1 Accuracy: 0.9331
[Split 2] Mean Average Precision (mAP): 0.9544

========== Evaluating Split 3 ==========

Extracting features from: ./11k/train_val_test_split_dorsal_r\query3
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [01:37<00:00,  3.14s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r\gallery3
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:07<00:00,  2.35s/it] 

Computing similarity between query and gallery images...

[Split 3] Rank-1 Accuracy: 0.9361
[Split 3] Mean Average Precision (mAP): 0.9583

========== Evaluating Split 4 ==========

Extracting features from: ./11k/train_val_test_split_dorsal_r\query4
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:17<00:00,  4.45s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r\gallery4
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:14<00:00,  4.78s/it] 

Computing similarity between query and gallery images...

[Split 4] Rank-1 Accuracy: 0.9506
[Split 4] Mean Average Precision (mAP): 0.9674

========== Evaluating Split 5 ==========

Extracting features from: ./11k/train_val_test_split_dorsal_r\query5
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:36<00:00,  5.06s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r\gallery5
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:12<00:00,  4.25s/it] 

Computing similarity between query and gallery images...

[Split 5] Rank-1 Accuracy: 0.9135
[Split 5] Mean Average Precision (mAP): 0.9409

========== Evaluating Split 6 ==========

Extracting features from: ./11k/train_val_test_split_dorsal_r\query6
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:13<00:00,  4.29s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r\gallery6
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:09<00:00,  3.14s/it] 

Computing similarity between query and gallery images...

[Split 6] Rank-1 Accuracy: 0.9258
[Split 6] Mean Average Precision (mAP): 0.9492

========== Evaluating Split 7 ==========

Extracting features from: ./11k/train_val_test_split_dorsal_r\query7
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:39<00:00,  5.15s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r\gallery7
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:10<00:00,  3.41s/it] 

Computing similarity between query and gallery images...

[Split 7] Rank-1 Accuracy: 0.9372
[Split 7] Mean Average Precision (mAP): 0.9594

========== Evaluating Split 8 ==========

Extracting features from: ./11k/train_val_test_split_dorsal_r\query8
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:26<00:00,  4.73s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r\gallery8
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.70s/it] 

Computing similarity between query and gallery images...

[Split 8] Rank-1 Accuracy: 0.9083
[Split 8] Mean Average Precision (mAP): 0.9398

========== Evaluating Split 9 ==========

Extracting features from: ./11k/train_val_test_split_dorsal_r\query9
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:10<00:00,  4.20s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r\gallery9
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:07<00:00,  2.51s/it] 

Computing similarity between query and gallery images...

[Split 9] Rank-1 Accuracy: 0.9104
[Split 9] Mean Average Precision (mAP): 0.9404

========== Final Averaged Results Across All Splits ==========

Split 0: Rank-1 Accuracy = 0.9156, mAP = 0.9433
Split 1: Rank-1 Accuracy = 0.9475, mAP = 0.9669
Split 2: Rank-1 Accuracy = 0.9331, mAP = 0.9544
Split 3: Rank-1 Accuracy = 0.9361, mAP = 0.9583
Split 4: Rank-1 Accuracy = 0.9506, mAP = 0.9674
Split 5: Rank-1 Accuracy = 0.9135, mAP = 0.9409
Split 6: Rank-1 Accuracy = 0.9258, mAP = 0.9492
Split 7: Rank-1 Accuracy = 0.9372, mAP = 0.9594
Split 8: Rank-1 Accuracy = 0.9083, mAP = 0.9398
Split 9: Rank-1 Accuracy = 0.9104, mAP = 0.9404

Average Rank-1 Accuracy: 0.9278
Average Mean Average Precision (mAP): 0.9520

HandCLIP multi-split evaluation completed ✅

```
## Output of 3_a_handclip_finetune_dorsal_r_evaluate_test.py
- the result is law These are very low accuracies, suggesting major problems in the generalization or evaluation.
- test/ identities are different from train, classification won't work
- test is completely different from the train set
- so this answer is quite convincible
```
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python .\3_a_handclip_finetune_dorsal_r_evaluate_test.py
Using device: cpu
Evaluating on Test Set: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33/33 [02:01<00:00,  3.68s/it]
✅ Classification Evaluation Completed on Test Set
Top-1 Accuracy: 0.0144
Top-5 Accuracy: 0.0729

```
## Output of 3_a_handclip_finetune_dorsal_r_evaluate_gallery_all.py
```
(.venv) PS C:\Users\Girija\OneDrive - De Montfort University\MSC PROJECT\BABU PALLAM\HandCLIP\HandCLIP> python .\3_a_handclip_finetune_dorsal_r_evaluate_gallery_all.py
Using device: cpu

Evaluating: Query=./11k/train_val_test_split_dorsal_r/query0, Gallery=./11k/train_val_test_split_dorsal_r/gallery0_all
Extracting features from: ./11k/train_val_test_split_dorsal_r/query0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:29<00:00,  4.82s/it]
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery0_all: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:41<00:00,  4.16s/it]
✅ [Query 0] Rank-1 Accuracy: 0.6601, mAP: 0.7422

Evaluating: Query=./11k/train_val_test_split_dorsal_r/query1, Gallery=./11k/train_val_test_split_dorsal_r/gallery1_all
Extracting features from: ./11k/train_val_test_split_dorsal_r/query1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:13<00:00,  4.30s/it]
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery1_all: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.80s/it]
✅ [Query 1] Rank-1 Accuracy: 0.7199, mAP: 0.7835

Evaluating: Query=./11k/train_val_test_split_dorsal_r/query2, Gallery=./11k/train_val_test_split_dorsal_r/gallery2_all
Extracting features from: ./11k/train_val_test_split_dorsal_r/query2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:22<00:00,  4.59s/it]
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery2_all: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:38<00:00,  3.88s/it]
✅ [Query 2] Rank-1 Accuracy: 0.7127, mAP: 0.7767

Evaluating: Query=./11k/train_val_test_split_dorsal_r/query3, Gallery=./11k/train_val_test_split_dorsal_r/gallery3_all
Extracting features from: ./11k/train_val_test_split_dorsal_r/query3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [01:57<00:00,  3.79s/it] 
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery3_all: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:36<00:00,  3.68s/it]
✅ [Query 3] Rank-1 Accuracy: 0.6601, mAP: 0.7429

Evaluating: Query=./11k/train_val_test_split_dorsal_r/query4, Gallery=./11k/train_val_test_split_dorsal_r/gallery4_all
Extracting features from: ./11k/train_val_test_split_dorsal_r/query4: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:16<00:00,  4.42s/it]
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery4_all: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:41<00:00,  4.11s/it]
✅ [Query 4] Rank-1 Accuracy: 0.6797, mAP: 0.7530

Evaluating: Query=./11k/train_val_test_split_dorsal_r/query5, Gallery=./11k/train_val_test_split_dorsal_r/gallery5_all
Extracting features from: ./11k/train_val_test_split_dorsal_r/query5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:07<00:00,  4.10s/it]
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery5_all: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:42<00:00,  4.20s/it]
✅ [Query 5] Rank-1 Accuracy: 0.6385, mAP: 0.7173

Evaluating: Query=./11k/train_val_test_split_dorsal_r/query6, Gallery=./11k/train_val_test_split_dorsal_r/gallery6_all
Extracting features from: ./11k/train_val_test_split_dorsal_r/query6: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [02:02<00:00,  3.95s/it]
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery6_all: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:33<00:00,  3.38s/it]
✅ [Query 6] Rank-1 Accuracy: 0.6601, mAP: 0.7398

Evaluating: Query=./11k/train_val_test_split_dorsal_r/query7, Gallery=./11k/train_val_test_split_dorsal_r/gallery7_all
Extracting features from: ./11k/train_val_test_split_dorsal_r/query7: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [01:53<00:00,  3.65s/it]
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery7_all: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.78s/it]
✅ [Query 7] Rank-1 Accuracy: 0.6746, mAP: 0.7574

Evaluating: Query=./11k/train_val_test_split_dorsal_r/query8, Gallery=./11k/train_val_test_split_dorsal_r/gallery8_all
Extracting features from: ./11k/train_val_test_split_dorsal_r/query8: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [01:43<00:00,  3.33s/it]
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery8_all: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:28<00:00,  2.83s/it]
✅ [Query 8] Rank-1 Accuracy: 0.6838, mAP: 0.7568

Evaluating: Query=./11k/train_val_test_split_dorsal_r/query9, Gallery=./11k/train_val_test_split_dorsal_r/gallery9_all
Extracting features from: ./11k/train_val_test_split_dorsal_r/query9: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31/31 [01:33<00:00,  3.02s/it]
Extracting features from: ./11k/train_val_test_split_dorsal_r/gallery9_all: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:28<00:00,  2.85s/it]
✅ [Query 9] Rank-1 Accuracy: 0.6880, mAP: 0.7546

✅ Final Evaluation on `galleryX_all` (Dorsal Right)
Average Rank-1 Accuracy: 0.6778
Average Mean Average Precision (mAP): 0.7524

```

<hr style="height:20px; background-color:red; border:none;">


# Analysis of the above three Results

## 📝 1. **Analysis of Your Three Evaluations**
| **Evaluation Type**                                           | **Dataset Used**                     | **What You Did**                               | **Results**                          |
|---------------------------------------------------------------|-------------------------------------|------------------------------------------------|-------------------------------------|
| **(A) `evaluate_q_g.py`**                                    | `queryX` + `galleryX`               | Classic ReID Setup - Query & Gallery **same aspect** (dorsal_r) | High Accuracy, Good mAP (Avg Rank-1 ~92.78%) |
| **(B) `evaluate_test.py`**                                   | `test`                              | Classification on **unseen identities** | Very Low Accuracy (Top-1 ~1.4%) |
| **(C) `evaluate_gallery_all.py`**                            | `queryX` + `galleryX_all`           | Cross-aspect gallery (dorsal + dorsal + palmar + palmar) | Moderate Accuracy (Avg Rank-1 ~67.78%) |

---

## ✅ 2. **Detailed Analysis & What’s Happening**

### ➡️ **A. Evaluation on Query-Gallery Splits (`evaluate_q_g.py`)**
#### ✔️ **What You Did**:
- Ran 10-fold Monte Carlo splits of query/gallery (`queryX` vs. `galleryX`)  
- Query & Gallery are from the **same aspect (dorsal_r)**  
- Dataset split to ensure **intra-aspect matching**, identities **seen during training**

#### 📈 **Results**:
- Rank-1 Accuracy **ranges from ~90.83% to 95.06%**  
- Average Rank-1 Accuracy: **92.78%**  
- mAP: **95.20%**

#### 🎯 **Conclusion**:
- **Fine-tuning was effective** on this aspect-specific split.  
- Queries are **close to training distribution**, so the model **performs very well**.

---

### ➡️ **B. Evaluation on `test` Data (`evaluate_test.py`)**
#### ✔️ **What You Did**:
- Evaluated the model in **classification mode** on the `test` set.  
- `test` set identities are **completely different** from train/val identities.  
- No **transfer of knowledge** to these new identities.

#### 📉 **Results**:
- Top-1 Accuracy: **1.44%**  
- Top-5 Accuracy: **7.29%**

#### 🎯 **Conclusion**:
- The model **fails** on unseen identities in a **closed-set classification** setting.  
- This is **expected** because:
  - Classification assumes **fixed classes**, but **test identities are new**.  
  - You should **not** evaluate **classification accuracy** for unseen identities.

#### ✅ **What Works Better Instead**:
- Use **ReID methods** (Query vs. Gallery) for **open-set identification**.  
- **Similarity-based matching**, not classification.

---

### ➡️ **C. Evaluation on `galleryX_all` (`evaluate_gallery_all.py`)**
#### ✔️ **What You Did**:
- Evaluated queries from `queryX` (dorsal_r) against **galleryX_all**  
- `galleryX_all` contains:
  - Dorsal Left  
  - Palmar Right  
  - Palmar Left  
- Galleries are **cross-aspect**; IDs made **unique** by adding offsets.

#### 📉 **Results**:
- Rank-1 Accuracy: **~66% to 71%**  
- mAP: **~71% to 78%**  
- Average Rank-1: **67.78%**  
- Average mAP: **75.24%**

#### 🎯 **Conclusion**:
- **Cross-aspect matching is harder**, especially since you fine-tuned on **dorsal_r only**.  
- The model struggles to match dorsal queries to palmar or different dorsal aspects.  
- **Different viewpoints** and **appearance variations** make this task challenging.

---

## ✅ 3. **Summary of Model Behavior**
| **Evaluation Type**                  | **Works Well?**  | **Reason**                                           |
|--------------------------------------|------------------|------------------------------------------------------|
| `queryX` vs `galleryX` (same aspect) | ✅ Yes           | Same aspect, trained on these identities/features   |
| `test` classification (unseen IDs)   | ❌ No            | New identities; classifier isn’t trained on them    |
| `queryX` vs `galleryX_all` (cross-aspect) | 🔶 Moderate | No fine-tuning across different aspects. Appearance shift affects accuracy |

---
<hr style="height:20px; background-color:red; border:none;">

# Further  Fine-Tuning
based on: 6_Technique1_Sequential Fine-Tuning Strategy.md



