# CLIP-FH: Fine-Tuning CLIP for Hand-Based Identity Matching


## To Run evaluation of Models
```angular2html
python experiments/run_all_experiments.py

```

python experiments/run_all_finetuned_stage1.py



## To Run Stage 1 Fine-Tuning:
```angular2html
python experiments/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_vitb16_11k_dorsal_r.yml

```

## To Run Stage 2 Fine-Tuning:

``` 
python experiments/train_stage2_loss_variants_vitb16_11k_dorsal_r.py --config configs/train_stage2_loss_variants/train_vitb16_11k_dorsal_r_ce_triplet.yml
python experiments/train_stage2_loss_variants_vitb16_11k_dorsal_r.py --config configs/train_stage2_loss_variants/train_vitb16_11k_dorsal_r_ce_triplet_center.yml

```
