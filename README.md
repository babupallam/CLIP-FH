# CLIP-FH: Fine-Tuning CLIP for Hand-Based Identity Matching


## Data preprocessing
```
python .\datasets\data_preprocessing\prepare_train_val_test_11k_r_l.py
python .\datasets\data_preprocessing\prepare_train_val_test_hd.py     
```

## stage0_baseline_model_evaluation

### for all
```angular2html
python experiments/stage0_baseline_inference/eval_baseline_clip.py

```

### for single config

````angular2html
python experiments/stage0_baseline_inference/eval_baseline_clip_single.py --config configs/baseline/eval_vitb16_11k_dorsal_r.yml

````

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
## To Run Stage 1 Evaluation: Single model evaluation
```angular2html

python experiments/stage1_train_classifier_frozen_text/eval_stage1_frozen_text_single.py configs/eval_stage1_frozen_text/eval_vitb16_11k_dorsal_r.yml


```

## To Run Stage 1 Fine-Tuning continuation:
```angular2html
python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text_continue.py --config configs/train_stage1_frozen_text/continue_train_vitb16_11k_dorsal_r.yml

```

***
***
***

## To Run Stage 2 stage 1: Training

``` 
python experiments/stage2_clipid_prompt_learning/train_stage2a_prompt_learn.py --config configs/train_stage2_clip_reid/train_stage2a_vitb16_11k_dorsal_r.yml

```
## Validation of Stage 2 Stage 1

```angular2html
python experiments/stage2_clipid_prompt_learning/validate_stage2a_prompt_model.py --config configs/validate_stage2_clip_reid/validate_stage2a_vitb16_11k_dorsal_r.yml

```

## To Run Stage 2 stage 2: Training

``` 
python experiments/stage2_clipid_prompt_learning/train_stage2b_finetune_image_encoder.py --config configs/train_stage2_clip_reid/train_stage2b_vitb16_11k_dorsal.yml

```


## To run Stage 2 Evaluation

```angular2html
python experiments/stage2_clipid_prompt_learning/eval_stage2b_finetune_image_encoder.py configs/eval_stage2_clip_reid/eval_vitb16_11k_dorsal_r.yml

```


# To run stage 3 training

python experiments/stage3_promptsg_integration/train_stage3_promptsg.py --config configs/train_stage3_promptsg/train_stage3_vitb16_11k_dorsal_r.yml






***
***
***
#TO BE EXICUTED
```angular2html
python experiments/stage0_baseline_inference/eval_baseline_clip_single.py --config configs/baseline/eval_vitb16_11k_dorsal_r.yml

python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_vitb16_11k_dorsal_r.yml

python experiments/stage1_train_classifier_frozen_text/eval_stage1_frozen_text_single.py configs/eval_stage1_frozen_text/eval_vitb16_11k_dorsal_r.yml

python experiments/stage2_clipid_prompt_learning/train_stage2a_prompt_learn.py --config configs/train_stage2_clip_reid/train_stage2a_vitb16_11k_dorsal_r.yml

python experiments/stage2_clipid_prompt_learning/train_stage2b_finetune_image_encoder.py --config configs/train_stage2_clip_reid/train_stage2b_vitb16_11k_dorsal.yml


```