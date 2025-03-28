# CLIP-FH: Fine-Tuning CLIP for Hand-Based Identity Matching


## stage0_baseline_model_evaluation

```angular2html
python experiments/stage0_baseline_inference/eval_baseline_clip.py

```


## To Run Stage 1 Fine-Tuning:
```angular2html
python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_vitb16_11k_dorsal_r.yml
python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_vitb16_11k_dorsal_l.yml
python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_vitb16_11k_palmar_r.yml
python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_vitb16_11k_palmar_l.yml
python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_vitb16_hd_dorsal_r.yml

python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_rn50_11k_dorsal_r.yml
python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_rn50_11k_dorsal_l.yml
python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_rn50_11k_palmar_r.yml
python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_rn50_11k_palmar_l.yml
python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_rn50_hd_dorsal_r.yml

```

## To Run Stage 1 Evaluation: all together...

```angular2html

python .\experiments\stage1_train_classifier_frozen_text\eval_stage1_frozen_text.py              

```

## To Run Stage 1 Fine-Tuning:
```angular2html
python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text_continue.py --config configs/train_stage1_frozen_text/continue_train_vitb16_11k_dorsal_r.yml

```

***
***
***

## To Run Stage 2 Fine-Tuning:

``` 
python experiments/train_stage2_loss_variants_vitb16_11k_dorsal_r.py --config configs/train_stage2_loss_variants/train_vitb16_11k_dorsal_r_ce_triplet.yml
python experiments/train_stage2_loss_variants_vitb16_11k_dorsal_r.py --config configs/train_stage2_loss_variants/train_vitb16_11k_dorsal_r_ce_triplet_center.yml

python experiments/train_stage3a_prompt_learn_vitb16_11k_dorsal_r.py --config configs/train_stage3_clipreid/train_prompt_vitb16_11k_dorsal_r.yml
python experiments/train_stage3b_img_encoder_vitb16_11k_dorsal_r.py --config configs/train_stage3_clipreid/train_img_encoder_vitb16_11k_dorsal_r.yml

```

