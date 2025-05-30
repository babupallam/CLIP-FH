
experiments/stage2_clipreid_integration/ANLAYSIS REPORT.md
============================================================


### **stage 2-v1  ViT-B/16, 11k Hands (dorsal_r)**  
Prompt + image encoder fine-tuning with ReID-style evaluation. Prompt tuning stage learns class-specific [CTX] embeddings (n_ctx=12) with templated prompt `"A photo of {}'s {aspect} hand for identification."`.  
Image encoder is fully unfrozen in Stage 2 with BNNeck, ArcFace, and SupCon+ID+Triplet+CenterLoss.  
Optimizer: Adam (lr=1e-4, weight_decay=5e-4), center_lr=0.5, center_loss_weight=0.0005.  
Scheduler: CosineAnnealingLR. Prompt is frozen in Stage 2.  
Evaluation on 10 splits (query/gallery) using Rank-1/5/10 and mAP.  
Training: epochs_prompt=30, epochs_image=30, early_stop_patience=5, batch_size=32.  
All components are modular and YAML-configurable. Total trainable params 149M.  
ArcFace duplicate param issue resolved.  

Command: python experiments/stage2_clipreid_integration/train_stage2_joint.py --config configs/train_stage2_clip_reid/train_joint_vitb16_11k_dorsal_r.yml
Final results: 
Rank-1 Accuracy : 62.11%
Rank-5 Accuracy : 82.66%
Rank-10 Accuracy: 88.80%
Mean AP         : 71.35%
 

---

### **stage 2-v1  RN50, 11k Hands (dorsal_r)**  
Same dual-stage pipeline as v1-ViT. Prompt learner uses [CTX] (n_ctx=12) with the same template prompt.  
Stage 1 trains class prompts with frozen image encoder. Stage 2 unfreezes entire RN50 visual encoder for supervised contrastive + identity losses.  
Uses BNNeck, ArcFace, SupCon+ID+Triplet+CenterLoss.  
Optimizer: Adam (lr=1e-4, weight_decay=5e-4), center_lr=0.5, center_loss_weight=0.0005.  
Scheduler: CosineAnnealingLR. Prompt is frozen in Stage 2.  
Training config: epochs_prompt=30, epochs_image=30, early_stop_patience=5, batch_size=32.  
Evaluation across 10 ReID-style splits.  
All components modular via YAML. Ready for comparative benchmarking with ViT-B/16.

Command: python experiments/stage2_clipreid_integration/train_stage2_joint.py --config .\configs\train_stage2_clip_reid\train_joint_rn50_11k_dorsal_r.yml
Final results: 
Rank-1 Accuracy : 57.64% 
Rank-5 Accuracy : 77.74% 
Rank-10 Accuracy: 85.09% 
Mean AP         : 67.07%

---
****
****
****
****

### **stage 2-v2  ViT-B/16, 11k Hands (dorsal_r)**  
Improved version of v1.  
Prompt learner uses `[CTX]` (n_ctx=12) with **diverse prompt templates** sampled randomly from a list.  
Stage 1 trains class prompts with frozen image encoder.  
Stage 2 unfreezes entire ViT-B/16 image encoder and applies:  
 BNNeck, ArcFace, SupCon + ID + Triplet + CenterLoss.  
 Feature normalization after projection (image + text).  
 Prompt is **frozen** in Stage 2.  
 Optimizer: **AdamW** with separate LR & weight decay for prompt, image, text.  
 Scheduler: **CosineAnnealingLR**, stepped **per batch**.  
All components modular via YAML. Fully compatible with v1 benchmarking setup.  
ReID-style evaluation across 10 splits (query/gallery).  

**YAML Hyperparameters:**
```yaml
n_ctx: 12
prompt_template_list:
  - "A photo of a X X X X hand."
  - "A close-up of the {} of a person."
  - "{} hand image."
  - "An image showing a {}."

lr_prompt: 0.0001
lr_visual: 0.00001
lr_text: 0.000005

weight_decay_prompt: 0.0005
weight_decay_visual: 0.0005
weight_decay_text: 0.0005

center_loss_weight: 0.0005
center_lr: 0.5

epochs_prompt: 30
epochs_image: 30
early_stop_patience: 5
batch_size: 32
```

**Command:** python experiments/stage2_clipreid_integration/train_stage2_joint.py --config configs/train_stage2_clip_reid/train_joint_vitb16_11k_dorsal_r.yml
Final results: 
Rank-1 Accuracy : 75.90%
Rank-5 Accuracy : 92.42%
Rank-10 Accuracy: 95.59%
Mean AP         : 83.19%


---

### **stage 2-v2  RN50, 11k Hands (dorsal_r)**  
Same pipeline and improvements as v2-ViT.  
Prompt learner and encoder settings identical except for LR tuning.  
Stage 2 unfreezes entire RN50 visual encoder with projection-aware normalization and prompt freezing.  
Evaluated across 10 ReID-style splits.

**YAML Hyperparameters:**
```yaml
n_ctx: 12
prompt_template_list:
  - "A photo of a X X X X hand."
  - "A close-up of the {} of a person."
  - "{} hand image."
  - "An image showing a {}."

lr_prompt: 0.0001
lr_visual: 0.0001
lr_text: 0.000005

weight_decay_prompt: 0.0005
weight_decay_visual: 0.0005
weight_decay_text: 0.0005

center_loss_weight: 0.0005
center_lr: 0.5

epochs_prompt: 30
epochs_image: 30
early_stop_patience: 5
batch_size: 32
```

**Command:** python experiments/stage2_clipreid_integration/train_stage2_joint.py --config configs/train_stage2_clip_reid/train_joint_rn50_11k_dorsal_r.yml
Final results: 
Rank-1 Accuracy : 46.09%
Rank-5 Accuracy : 71.13%
Rank-10 Accuracy: 80.10%
Mean AP         : 57.59%



*****
*****
*****

### **stage 2-v3  ViT-B/16, 11k Hands (dorsal_r)**  
Further enhanced version of v2.  
Prompt learner still uses `[CTX]` (n_ctx=12) with **diverse prompt templates**.  
Stage 1 trains class prompts with frozen image encoder.  
Stage 2 unfreezes entire ViT-B/16 image encoder and applies:  
 BNNeck, ArcFace, SupCon + ID + Triplet + CenterLoss.  
 **Feature normalization after projection** (image + text).  
 **Prompt is frozen** in Stage 2.  
 **AdamW optimizer** with LR/weight_decay split by component.  
 **OneCycleLR scheduler** (replaces CosineAnnealingLR), stepped **per batch**.

All components remain modular via YAML. Same benchmarking compatibility as v1/v2.  
Evaluation performed over 10 ReID-style splits (query/gallery).

**YAML Hyperparameters (unchanged from v2):**
```yaml
n_ctx: 12
prompt_template_list:
  - "A photo of a X X X X hand."
  - "A close-up of the {} of a person."
  - "{} hand image."
  - "An image showing a {}."

lr_prompt: 0.0001
lr_visual: 0.00001
lr_text: 0.000005

weight_decay_prompt: 0.0005
weight_decay_visual: 0.0005
weight_decay_text: 0.0005

center_loss_weight: 0.0005
center_lr: 0.5

epochs_prompt: 30
epochs_image: 30
early_stop_patience: 5
batch_size: 32

loss_use_supcon: true
loss_use_arcface: true
loss_use_triplet: true
loss_use_center: true
loss_use_id: true

```

**Command:**  
```bash
python experiments/stage2_clipreid_integration/train_stage2_joint.py --config configs/train_stage2_clip_reid/train_joint_vitb16_11k_dorsal_r.yml
```

**Final results:**  
Rank-1 Accuracy : 84.82%
Rank-5 Accuracy : 96.40%
Rank-10 Accuracy: 98.25%
Mean AP         : 89.91%

---

### **stage 2-v3  RN50, 11k Hands (dorsal_r)**  
Same pipeline and improvements as v3-ViT.  
Prompt learner and training stages are identical except for `lr_visual`.  
Stage 2 unfreezes entire RN50 image encoder and applies:  
 BNNeck, ArcFace, SupCon + ID + Triplet + CenterLoss.  
 **Feature normalization after projection** (image + text).  
 **Prompt is frozen** in Stage 2.  
 **AdamW optimizer** with component-wise LR/WD.  
 **OneCycleLR scheduler** (stepped per batch, replaces CosineAnnealingLR).

All config elements unchanged from v2 other than scheduling.

**YAML Hyperparameters (unchanged from v2):**
```yaml
n_ctx: 12
prompt_template_list:
  - "A photo of a X X X X hand."
  - "A close-up of the {} of a person."
  - "{} hand image."
  - "An image showing a {}."

lr_prompt: 0.0001
lr_visual: 0.0005
lr_text: 0.000005

weight_decay_prompt: 0.0005
weight_decay_visual: 0.0005
weight_decay_text: 0.0005

center_loss_weight: 0.0005
center_lr: 0.5

epochs_prompt: 30
epochs_image: 30
early_stop_patience: 5
batch_size: 32

loss_use_supcon: true
loss_use_arcface: true
loss_use_triplet: true
loss_use_center: true
loss_use_id: true

```

**Command:**  
```bash
python experiments/stage2_clipreid_integration/train_stage2_joint.py --config configs/train_stage2_clip_reid/train_joint_rn50_11k_dorsal_r.yml
```

**Final results:**  
Rank-1 Accuracy : 53.37%
Rank-5 Accuracy : 73.68%
Rank-10 Accuracy: 82.52%
Mean AP         : 62.85%




****
****
****
### Stage 2-v4 vitb16 

settings changed:
loss_use_triplet: false
loss_use_center: false


command:
python experiments/stage2_clipreid_integration/train_stage2_joint.py --config configs/train_stage2_clip_reid/train_joint_vitb16_11k_dorsal_r.yml
Final Result:
Rank-1 Accuracy : 74.07%
Rank-5 Accuracy : 91.78%
Rank-10 Accuracy: 96.27%
Mean AP         : 81.93%




### Stage 2-v4 rn50

setting changed: 
loss_use_triplet: false
loss_use_center: false


command:
python experiments/stage2_clipreid_integration/train_stage2_joint.py --config configs/train_stage2_clip_reid/train_joint_rn50_11k_dorsal_r.yml
Final Result:
Rank-1 Accuracy : 50.82%
Rank-5 Accuracy : 72.91%
Rank-10 Accuracy: 82.19%
Mean AP         : 61.34%


Performace is law so rollback:

****
****
****
v5 
vitb and rn50
Fine-Tune ArcFace (critical) --- reduced the values which has been used before 30 and 0.5
arcface_scale: 20
arcface_margin: 0.3

loss_use_id: true
loss_use_arcface: true
loss_use_supcon: true
loss_use_triplet: true
loss_use_center: true


Final Results:vitb16
Rank-1 Accuracy : 88.18%
Rank-5 Accuracy : 97.32%
Rank-10 Accuracy: 98.58%
Mean AP         : 92.20%


Final Results:rn50
Rank-1 Accuracy : 53.74%
Rank-5 Accuracy : 75.44%
Rank-10 Accuracy: 84.34%
Mean AP         : 63.69%



*****
*****
*****
v6 -- vitb16 and rn50

unfreeze Partial Image Encoder (Layer-wise Fine-Tuning)
Control number of unfrozen layers from YAML config (unfreeze_blocks).
unfreeze_blocks: 2   # Unfreeze last 2 blocks of visual transformer (default: 0 = frozen)


Final Results: vitb16
Rank-1 Accuracy : 76.08%
Rank-5 Accuracy : 91.70%
Rank-10 Accuracy: 95.75%
Mean AP         : 83.03%


Final Results: rn50

Rank-1 Accuracy : 45.09%
Rank-5 Accuracy : 70.30%
Rank-10 Accuracy: 80.19%
Mean AP         : 56.47%



****
****
****
v7

rn50:

lr_visual: 0.0005  -- reduced
arcface_scale: 25
arcface_margin: 0.35
unfreeze_blocks = 2 -- 

Rank-1 Accuracy : 51.62%
Rank-5 Accuracy : 76.08%
Rank-10 Accuracy: 84.69%
Mean AP         : 62.63%


=======================
v8:

Component	Change/Keep	Reason
arcface_scale Increase to 30	Stronger decision boundaries; improves class margin
arcface_margin	Increase to 0.35	Helps class separation and reduces feature overlap
loss_use_center  Turn back ON	Encourages intra-class compactness; improves generalization
loss_use_triplet Keep	Works well with SupCon for hard samples
unfreeze_blocks	 Keep at 4	Seems effective for deeper feature tuning
lr_visual	 Keep at 0.0001	Safe learning rate for RN50 full or partial fine-tuning
scheduler	CosineAnnealingLR

Rank-1 Accuracy : 52.97%
Rank-5 Accuracy : 78.04%
Rank-10 Accuracy: 85.97%
Mean AP         : 64.30%

=================================

*****
******

v9: for both
Change	Value	Reason
arcface_scale	35	Further separation between classes.
arcface_margin	0.4	Stronger intra-class compactness.
center_loss_weight	0.0003	Prevent over-scaling center gradients.
scheduler	CosineAnnealingLR
loss_use_triplet: false


for rn50 only, the following will aslo change: 

lr_visual: 0.00007 



vitb16:

Rank-1 Accuracy : 28.88%
Rank-5 Accuracy : 70.13%
Rank-10 Accuracy: 86.14%
Mean AP         : 47.23%




rn50:
Rank-1 Accuracy : 59.84%
Rank-5 Accuracy : 82.47%
Rank-10 Accuracy: 89.04%
Mean AP         : 69.80%

****
****
*****
### v10

for rn50

center_loss_weight	0.0003 (optional tweak)	May slightly stabilize gradient variance
arcface_scale	Keep at 30	Working well
arcface_margin	Keep at 0.4	Best margin so far
lr_visual	Keep at 0.00007	Perfect balance of stability and learning
loss_use_triplet	true is fine	No clear need for triplet due to SupCon+Center
prompt	 Keep frozen	Learned prompts are generalizing well
unfreeze_blocks	 Keep at 4	Full tuning proves optimal

Rank-1 Accuracy : 51.52%
Rank-5 Accuracy : 74.93%
Rank-10 Accuracy: 83.98%
Mean AP         : 62.26%



===============================
===============================
===============================

# Build detailed remarks dictionaries including all points
vit_remarks_full = {
    "v1": ("Baseline dualstage pipeline: classspecific [CTX]12 prompt, templated string, "
           "full ViTB/16 image encoder unfrozen with BNNeck + ArcFace + SupCon + Triplet + Center; "
           "Adam (lr 1e4, wd 5e4), center_lr 0.5, center_loss_weight 0.0005; CosineAnnealingLR; "
           "prompt frozen in Stage2; eval on 10 splits; ArcFace duplicateparam bug fixed."),
    "v2": ("Added diverse prompt_template_list; kept prompt frozen; enabled projection norm; "
           "switched optimizer to AdamW with componentwise LR/WD (vis 1e5, prompt 1e4); "
           "CosineAnnealingLR stepped per batch; all five losses retained."),
    "v3": ("Scheduler changed to OneCycleLR (perbatch) while retaining v2 template diversity, AdamW, "
           "feature normalisation & losses  yielded best ViT performance."),
    "v4": ("Ablation: disabled Triplet & Center losses (ArcFace + SupCon + ID only)  performance dropped."),
    "v5": ("Restored Triplet & Center; softened ArcFace (scale 20, margin 0.3)  new peak (R1 88.18, mAP 92.2)."),
    "v6": ("Partial finetuning: set unfreeze_blocks=2 (last two ViT transformer blocks) while keeping v5 losses "
           "and ArcFace settings."),
    "v7": "No ViTB/16 experiment (RN50exclusive tweaks).",
    "v8": "No ViTB/16 experiment (RN50exclusive tweaks).",
    "v9": ("Extreme ArcFace (scale 35, margin 0.4) + Triplet OFF + center_loss_weight 0.0003 + Cosine LR  "
           "led to severe collapse (R1 28.9, mAP 47.2)."),
    "v10": "No ViTB/16 experiment (RN50exclusive tweaks)."
}

rn_remarks_full = {
    "v1": ("Baseline identical to ViTv1: full RN50 encoder, BNNeck + ArcFace + SupCon + Triplet + Center, "
           "Adam + Cosine; prompt frozen."),
    "v2": ("Inherited template diversity, AdamW split LR/WD, projection normalisation (as ViTv2)."),
    "v3": ("Scheduler switched to OneCycleLR (per batch) keeping other settings  modest gain."),
    "v4": ("Disabled Triplet & Center losses (ArcFace + SupCon + ID only)  performance dip."),
    "v5": ("ArcFace softened: scale 20, margin 0.3; all five losses ON  slight uplift."),
    "v6": ("Partial FT: unfreeze_blocks 2 with same loss set  results fell (R1 45.1, mAP 56.5)."),
    "v7": ("RN50specific tuning: lr_visual 0.0005, ArcFace scale 25 margin 0.35, still unfreeze_blocks 2."),
    "v8": ("Deeper partial FT: unfreeze_blocks 4; ArcFace scale 30 margin 0.35; Center back ON; Triplet kept; "
           "lr_visual 1e4; max_norm 0.5; CosineAnnealingLR kept  best mAP 64.3."),
    "v9": ("Full encoder again; ArcFace scale 35 margin 0.4; Triplet OFF; center_loss_weight 0.0003; "
           "lr_visual 7e5; Cosine LR  peak R1 59.8, mAP 69.8."),
    "v10": ("Finetune v9: keep center 0.0003, ArcFace scale 30 margin 0.4, Triplet ON, "
            "unfreeze_blocks 4, lr_visual 7e5  performance regressed to R1 51.5, mAP 62.3.")
}

# Convert to DataFrames
vit_df_full = pd.DataFrame({"model_version": list(vit_remarks_full.keys()),
                            "remarks": list(vit_remarks_full.values())})
rn_df_full = pd.DataFrame({"model_version": list(rn_remarks_full.keys()),
                           "remarks": list(rn_remarks_full.values())})

# Save CSVs
os.makedirs("result_logs", exist_ok=True)
vit_csv = "result_logs/remarks_vitb16_stage2.csv"
rn_csv = "result_logs/remarks_rn50_stage2.csv"
vit_df_full.to_csv(vit_csv, index=False)
rn_df_full.to_csv(rn_csv, index=False)

vit_csv, rn_csv
