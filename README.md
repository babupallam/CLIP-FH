# HandCLIP

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

## 