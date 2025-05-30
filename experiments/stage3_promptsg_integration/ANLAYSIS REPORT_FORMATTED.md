
---

##  ViT-B/16   (Stage-3 PromptSG chronology)

| Ver.    | Configuration moves ( vs previous)                                                                                                                                                                                        | Loss / Head / Scheduler | Extra Optim./Reg. | Final scores                            |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ----------------- | --------------------------------------- |
| **v1**  |  **Baseline PromptSG** (joint tuning)<br> Linear classifier, Cross-Entropy + Triplet (SupCon OFF)<br> Pseudo-tokens, 1 cross-attn<br> AdamW (lr 1e-4 vis, 1e-5 rest), clip grad 5<br> cosine val (ID logits ignored) | CE + Tri                | norm+dropout      | **R1 62.38  mAP 71.57**                |
| **v2**  | ** L2 wd 1e-4, LR warm-up**, Cosine LR w/ warm-up<br> SupCon (temperature cfg) + auto loss-balancing<br> Early-stop patience 5**7**                                                                                    | CE + Tri + SupCon       | grad 5            | **R1 36.45  mAP 47.64** (destabilised) |
| **v3**  | ** Cross-Entropy OFF**  purely Triplet+SupCon                                                                                                                                                                            | Tri + SupCon            | same              | **R1 26.35  mAP 38.68**                |
| **v4**  | ** BNNeck + ArcFace head** (256-dim, reduction+BN)<br> Classifier switch requires ArcFace loss                                                                                                                           | ArcFace + Tri + SupCon  | same              | **R1 42.25  mAP 54.32**                |
| **v5**  | Paper-faithful **TextualInversionMLP (3-layer + BN)**<br> Prompt fusion 1  cross + 2  self attn<br> `use_composed_prompt true`                                                                                         | ArcFace + Tri + SupCon  | grad 5            | **R1 41.45  mAP 52.09**                |
| **v6**  | **Grad-clip 5  1**, BN neck retained<br> **lr\_clip\_visual & lr\_modules = 1e-6**                                                                                                                                       | ArcFace + Tri + SupCon  | grad 1            | **R1 80.83  mAP 86.63** (big recovery) |
| **v7**  | **Cross-Entropy removed again** (pure contrastive)                                                                                                                                                                         | Tri + SupCon            | same              |  v6 (no numbers given)                 |
| **v8**  | Keep v7 but **bnneck\_dim 1024, max\_norm 0.5**                                                                                                                                                                            | ArcFace + Tri + SupCon  | lr vis/mod 1e-4   | **R1 85.86  mAP 89.99**                |
| **v9**  | Same cfg **but epochs 60, patience 10**                                                                                                                                                                                    | same                    | same              | **R1 84.90  mAP 89.79**                |
| **v10** | **Prompt template changed** to surveillance-style string                                                                                                                                                                   | same                    | same              | **R1 84.87  mAP 89.35**                |
| **v11** | **Input resize 224128 portrait** (was 224224)                                                                                                                                                                            | same                    | same              | **R1 81.59  mAP 86.93**                |

---

###  ViT Track Insights

* **v6** shows how *stable gradients + tiny LR* rescued performance after v3 collapse.
* **v8** (bnneck 1024 & moderate clip) is peak (mAP  90). Prompt change (v10) didnt help; smaller portrait crops (v11) cost \~3 pp mAP.

---

##  RN50   (Stage-3 PromptSG chronology)

| Ver.    | Configuration moves ( vs previous)                                                               | Loss / Head / Scheduler         | Extra Optim./Reg. | Final scores                           |
| ------- | ------------------------------------------------------------------------------------------------- | ------------------------------- | ----------------- | -------------------------------------- |
| **v1**  | Same baseline as ViT-v1 (Linear, CE + Tri, SupCon OFF)                                            | CE + Tri                        | grad 5            | **R1 64.07  mAP 71.44**               |
| **v2**  | Warm-up, SupCon added, auto-balancing                                                             | CE + Tri + SupCon               | grad 5            | **R1 56.18  mAP 66.17**               |
| **v3**  | **CE removed** (Tri + SupCon only)                                                                | Tri + SupCon                    | same              | **R1 68.99  mAP 76.28** (best so far) |
| **v4**  | **BNNeck + ArcFace** head (dim 256)                                                               | ArcFace + Tri + SupCon          | same              | **R1 42.25  mAP 54.32** (drop)        |
| **v5**  | Paper-faithful Inversion + fusion (same as ViT-v5)                                                | ArcFace + Tri + SupCon          | grad 5            | **R1 45.70  mAP 55.54**               |
| **v6**  | Grad-clip 1, **lr 1e-6** (all mods)                                                               | ArcFace + Tri + SupCon          | grad 1            | **R1 50.08  mAP 60.29**               |
| **v7**  | **lr\_visual 5e-4** , ArcFace s 25 m 0.35, unfreeze 2 blk                                        | ArcFace + Tri + SupCon          | grad 1            | **R1 51.62  mAP 62.63**               |
| **v8**  | **unfreeze\_blocks 4**, ArcFace s 30 m 0.35, **lr 1e-4**, max\_norm 0.5, **Cosine sched removed** | ArcFace + Tri + SupCon + Center | grad 0.5          | **R1 61.23  mAP 69.75** (track peak)  |
| **v9**  | Same cfg but **epochs 60, patience 10**                                                           | same                            | same              | **R1 52.86  mAP 62.88**               |
| **v10** | **Prompt template changed** (surveillance)                                                        | same                            | same              | **R1 63.00  mAP 71.13** (near best)   |
| **v11** | **Resize 224128** portrait                                                                       | same                            | same              | **R1 58.25  mAP 67.29**               |

---

###  RN50 Track Insights

* RN50 likes **contrastive-only (v3)** and **deep BNNeck + ArcFace once LR / unfreeze are balanced (v8, v10)**.
* Removing Cosine scheduler and raising ArcFace scale to 30 helped (v8).
* Extended training (v9) over-fit; the new prompt in v10 gave a small lift.
* Portrait resize (v11) again costs \~4 pp mAP.

---

##  Cross-Stage-3 Take-aways

| Finding                                                     | Evidence                                                                        |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Gradient-clip & LR are the biggest stabilisers.**         | ViT v6 (clip 1, LR 1e-6) and RN50 v8 (clip 0.5, LR 1e-4) are peaks.             |
| **Contrastive-only works for RN50 but hurts ViT.**          | RN50 v3  mAP, ViT v3 collapsed.                                                |
| **BNNeck + ArcFace needs tuning or it degrades.**           | v4 drops on both tracks; only recovers when LR, scale, blocks adjusted (v6-v8). |
| **Prompt wording tweaks produce modest gains.**             | v10 boosts RN50, insignificant on ViT.                                          |
| **Portrait crops reduce retrieval quality in both models.** | v11 mAP down for ViT (-3) and RN50 (-4).                                        |




========================

# Create remark dictionaries
vit_remarks = {
    "v1": "Baseline PromptSG: Linear classifier, CE+Triplet (SupCon off); AdamW; grad clip 5; cosine validation only.",
    "v2": "Added weight_decay 1e-4, LR warmup, SupCon enabled with temperature cfg, loss autobalancing, cosine LR with warmup, patience 7.",
    "v3": "Removed CrossEntropy; pure Triplet+SupCon contrastive objective.",
    "v4": "Introduced BNNeck + ArcFace head (256d, reduction+BN); ArcFace loss added.",
    "v5": "Paperfaithful 3layer TextualInversionMLP with BN; 1cross+2self attention fusion; composed prompt used in train+val.",
    "v6": "Stability tweaks: grad clip 1, lr_clip_visual & lr_modules 1e6; large performance jump.",
    "v7": "Kept v6 config but continued CEfree (contrastive only); exploratory run.",
    "v8": "bnneck_dim 1024, max_norm 0.5; lr 1e4; Cosine scheduler removed; prompt/heads unchanged.",
    "v9": "Same as v8 but epochs 60 and patience 10 for longer training.",
    "v10": "Prompt template changed to surveillancestyle sentence.",
    "v11": "Input resize 224128 portrait (was 224224) to mimic handReID aspect."
}

rn_remarks = {
    "v1": "Baseline PromptSG: Linear classifier, CE+Triplet (SupCon off); AdamW; grad clip 5.",
    "v2": "Added weight_decay 1e-4, LR warmup, SupCon enabled, loss autobalancing.",
    "v3": "Removed CrossEntropy; contrastiveonly (Triplet+SupCon) giving first peak.",
    "v4": "BNNeck + ArcFace head (256d) introduced; big drop without tuning.",
    "v5": "Paperfaithful inversion MLP, refined fusion, composed prompt.",
    "v6": "Grad clip 1, ultralow LR (1e6) for visual modules.",
    "v7": "lr_visual 5e4 reduced, ArcFace scale 25 margin 0.35, unfreeze_blocks 2.",
    "v8": "unfreeze_blocks 4; ArcFace scale 30 margin 0.35; lr 1e4; max_norm 0.5; scheduler removed.",
    "v9": "Extended epochs 60, patience 10 (same cfg as v8).",
    "v10": "Surveillancestyle prompt template applied.",
    "v11": "Portrait resize 224128; rest unchanged."
}

# Convert to DataFrames
vit_df = pd.DataFrame({
    "model_version": list(vit_remarks.keys()),
    "remarks": list(vit_remarks.values())
})

rn_df = pd.DataFrame({
    "model_version": list(rn_remarks.keys()),
    "remarks": list(rn_remarks.values())
})

# Ensure output dir
os.makedirs("result_logs", exist_ok=True)
vit_path = "result_logs/stage3_vitb16_remarks.csv"
rn_path = "result_logs/stage3_rn50_remarks.csv"
vit_df.to_csv(vit_path, index=False)
rn_df.to_csv(rn_path, index=False)

vit_path, rn_path
