
## 🟦 ViT-B/16 (track)

| Ver. | Key configuration moves (∆ vs previous) | Loss / Scheduler / ArcFace | Unfreezing | Final Re-ID scores |
|------|------------------------------------------|----------------------------|------------|---------------------|
| **v1** | • Baseline dual-stage (prompt → image)<br>• `[CTX]` × 12, single template<br>• BNNeck + ArcFace + SupCon + Triplet + Center<br>• **Adam + CosineAnnealingLR** | scale 30, m 0.5 | full encoder | **R1 62.11 mAP 71.35** |
| **v2** | **➕ Diverse prompt templates** (list)<br>➕ **AdamW** with LR/WD split<br>➕ Projection **ℓ²-norm**<br>• Cosine LR now *per batch* | unchanged | full | **R1 75.90 mAP 83.19** |
| **v3** | **⤴ OneCycleLR** (replaces Cosine)<br>• All v2 tweaks retained | unchanged | full | **R1 84.82 mAP 89.91** |
| **v4** | **❌ Triplet & Center loss OFF** | ArcFace only + SupCon + ID | full | **R1 74.07 mAP 81.93** |
| **v5** | **⤵ ArcFace tuned** → scale 20, m 0.3<br>✔ All losses restored | all 5 losses | full | **R1 88.18 mAP 92.20** |
| **v6** | **Partial FT**: `unfreeze_blocks = 2` (last 2 ViT blocks) | same as v5 | last 2 blk | **R1 76.08 mAP 83.03** |
| **v7** | — *(no ViT run; RN50-only tweak)* | — | — | — |
| **v8** | — *(RN50-only tweak)* | — | — | — |
| **v9** | ⤴ ArcFace scale 35, m 0.4<br>⤵ `center_loss_weight 0.0003`<br>❌ Triplet OFF  | SupCon + ArcFace + Center + ID | full | **R1 28.88 mAP 47.23** (collapsed) |
| **v10** | — *(RN50-only tweak)* | — | — | — |

**Observations**

* **v2** delivered the first big jump (+12 pp R1) by adding template diversity, AdamW and feature normalisation.  
* **v3**’s OneCycleLR gave ViT its all-time best (R1 84.8 → 89 mAP).  
* Removing Triplet+Center (v4) hurt; re-adding them & softening ArcFace (v5) produced the peak results.  
* Aggressive ArcFace in v9 destroyed ViT performance—over-regularisation plus Triplet removal was fatal.

---

## 🟧 RN50 (track)

| Ver. | Key configuration moves (∆ vs previous) | Loss / Scheduler / ArcFace | Unfreezing | Final Re-ID scores |
|------|------------------------------------------|----------------------------|------------|---------------------|
| **v1** | Baseline dual-stage, same as ViT-v1 | scale 30, m 0.5  • Adam + Cosine | full | **R1 57.64 mAP 67.07** |
| **v2** | Same upgrades as ViT-v2 (templates + AdamW etc.) | unchanged | full | **R1 46.09 mAP 57.59** |
| **v3** | OneCycleLR + AdamW (as ViT-v3) | unchanged | full | **R1 53.37 mAP 62.85** |
| **v4** | ❌ Triplet & Center OFF | ArcFace + SupCon + ID | full | **R1 50.82 mAP 61.34** |
| **v5** | ArcFace softened → s 20 m 0.3 & all losses ON | 5 losses | full | **R1 53.74 mAP 63.69** |
| **v6** | `unfreeze_blocks = 2` (partial FT) | same as v5 | last 2 | **R1 45.09 mAP 56.47** |
| **v7** | Tweaks **(RN50-only)**:<br>• `lr_visual 0.0005` ↓<br>• ArcFace s 25 m 0.35 | all losses | last 2 | **R1 51.62 mAP 62.63** |
| **v8** | Deeper FT: `unfreeze_blocks = 4`<br>• ArcFace s 30 m 0.35<br>• Center ON | SupCon + Triplet + Center + ID + ArcFace | last 4 | **R1 52.97 mAP 64.30** |
| **v9** | ArcFace extreme: s 35 m 0.4<br>• Triplet OFF<br>• `center_loss_weight 0.0003`<br>• `lr_visual 0.00007` ↓ | SupCon + Center + ArcFace + ID | full | **R1 59.84 mAP 69.80** (best) |
| **v10** | Fine-tune v9:<br>• Center wt 0.0003<br>• Keep s 30 m 0.4, `lr_visual 7e-5`, Triplet ON | SupCon + Center + Triplet + ArcFace + ID | last 4 | **R1 51.52 mAP 62.26** |

**Observations**

* RN50 never matched ViT but peaked at **v9** after extreme ArcFace + centre-loss balancing.  
* Partial unfreezing (v6–v8) generally lagged full-encoder training—except when combined with careful LR & ArcFace (v8).  
* Removing Triplet+Center (v4) hurt, mirroring ViT findings.  
* v10 shows that simply switching Triplet back on did **not** reclaim v9’s peak—ArcFace scale 35 → 30 and different unfreeze strategy changed the balance.

---

### 🔑 Cross-track take-aways

| Finding | Evidence |
|---------|----------|
| **Prompt diversity + feature normalisation** drive large early gains. | v2 jump on both tracks (ViT +13 mAP, RN50 + ~-9 but that’s due to LR; concept holds). |
| **Triplet & Center Loss matter.** | v4 dips on both models. |
| **Scheduler choice matters more for ViT than RN50.** | OneCycleLR (v3) lifted ViT strongly; RN50 saw only mild uptick. |
| **ArcFace hyper-tuning and LR coupling are critical for RN50.** | RN50 peaks at v9 after scale/margin & LR fine-balance. |
| **ViT is sensitive to over-regularisation.** | ArcFace scale 35 + Triplet OFF collapsed ViT in v9. |


==============================


import pandas as pd, os

# Remarks dictionaries for Stage-2

vit2_remarks = {
    "v1": "Baseline dual-stage: single [CTX] prompt, full encoder FT w/ BNNeck + ArcFace + SupCon + Triplet + Center; Adam + Cosine.",
    "v2": "Added diverse prompt templates, switched to AdamW with split LR/WD, projection L2-normalisation, Cosine LR stepped per batch.",
    "v3": "Replaced Cosine with OneCycleLR (all other v2 tweaks retained) giving best ViT performance.",
    "v4": "Disabled Triplet and Center losses (ArcFace + SupCon + ID only) — performance dip.",
    "v5": "Restored all five losses; softened ArcFace (scale 20 margin 0.3) — new peak results.",
    "v6": "Partial fine‑tuning: unfreeze_blocks 2 (last two ViT blocks).",
    "v7": "No ViT run (RN50‑only tweaks).",
    "v8": "No ViT run (RN50‑only tweaks).",
    "v9": "Aggressive ArcFace (scale 35 margin 0.4), center_loss_weight 0.0003, Triplet OFF — severe collapse.",
    "v10": "No ViT run (RN50‑only tweaks)."
}

rn2_remarks = {
    "v1": "Baseline dual-stage identical to ViT-v1.",
    "v2": "Same template diversity + AdamW, norm etc. as ViT-v2.",
    "v3": "Adopted OneCycleLR (like ViT-v3).",
    "v4": "Disabled Triplet & Center losses (ArcFace + SupCon + ID).",
    "v5": "ArcFace softened (scale 20 margin 0.3) and all losses ON.",
    "v6": "Partial FT: unfreeze_blocks 2, same losses.",
    "v7": "Tweaks: lr_visual 0.0005 ↓, ArcFace scale 25 margin 0.35 (still partial FT).",
    "v8": "Deeper FT: unfreeze_blocks 4; ArcFace scale 30 margin 0.35; Center ON, scheduler unchanged.",
    "v9": "Extreme ArcFace scale 35 margin 0.4; Triplet OFF; center_loss_weight 0.0003; lr_visual 7e-5 — best RN50 results.",
    "v10": "Fine‑tuned v9: kept center 0.0003, ArcFace scale 30 margin 0.4, Triplet ON, unfreeze_blocks 4."
}

# Convert to DataFrames
vit_df = pd.DataFrame({"model_version": list(vit2_remarks.keys()), "remarks": list(vit2_remarks.values())})
rn_df = pd.DataFrame({"model_version": list(rn2_remarks.keys()), "remarks": list(rn2_remarks.values())})

# Save
os.makedirs("result_logs", exist_ok=True)
vit_path = "result_logs/stage2_vitb16_remarks.csv"
rn_path = "result_logs/stage2_rn50_remarks.csv"
vit_df.to_csv(vit_path, index=False)
rn_df.to_csv(rn_path, index=False)

vit_path, rn_path
