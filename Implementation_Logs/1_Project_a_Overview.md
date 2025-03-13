# Project Overview

## Project Title
**HandCLIP: Fine-Tuning CLIP for Hand-Based Identity Matching**

## Author
**Babu Pallam**  
P2849288@my365.dmu.ac.uk  
Supervisor: Dr. Nathanael L Baisa  
De Montfort University, Leicester, UK  

---

## 1. Project Background

Hand-based biometrics is gaining traction as an alternative to facial and fingerprint recognition. It offers:
- Uniqueness and accessibility
- Contactless, non-intrusive verification
- Use in smart devices and forensic applications

Traditional hand biometric recognition relies on CNN-based methods that extract handcrafted features like palmprints and hand geometry. However, these approaches often struggle with generalization under varied conditions such as lighting and hand posture.

CLIP (Contrastive Language-Image Pretraining), by OpenAI, has shown strong generalization abilities in vision-language tasks. The HandCLIP project aims to fine-tune CLIP for hand recognition by leveraging its cross-modal learning capabilities.

---

## 2. Problem Statement

- Existing hand recognition systems are heavily dependent on CNNs and handcrafted features.
- CLIP has not been fine-tuned or optimized for hand biometrics.
- There is an opportunity to enhance hand recognition using CLIP's vision-language capabilities.

---

## 3. Aim

Fine-tune CLIP for hand-based identity matching and evaluate its performance compared to existing CNN-based models (like MBA-Net).

---

## 4. Objectives

1. Adapt CLIP for hand biometrics by fine-tuning it on large-scale hand image datasets.
2. Compare HandCLIP with CNN-based methods in terms of accuracy and generalization.
3. Evaluate HandCLIP under variations in lighting, occlusions, and hand postures.
4. Leverage textual descriptions to enhance recognition via contrastive learning.

---

## 5. Proposed Solution (HandCLIP)

- Preprocess and augment hand image datasets.
- Fine-tune CLIP’s ViT-B/32 image encoder on hand datasets.
- Use text prompts describing hand features to support multimodal learning.
- Evaluate on metrics like mAP and Rank-1 accuracy.

---

## 6. Tools & Technologies

- **Programming Language**: Python  
- **Frameworks**: PyTorch, TensorFlow (if required)  
- **Pretrained Model**: CLIP (ViT-B/32)  
- **Datasets**: 11k Hands, PolyU Palmprint  
- **Hardware**: High-performance GPU (recommended)

---

## 7. Project Timeline (From Proposal)

| Phase                                    | Timeline                |
|------------------------------------------|-------------------------|
| Study & Data Collection                  | 29th Feb - 15th Mar 2025 |
| Implementation & Fine-Tuning CLIP        | 15th Mar - 10th Apr 2025 |
| Model Testing & Comparison               | 11th Apr - 28th Apr 2025 |
| Improving & Final Fine-Tuning            | 29th Apr - 10th May 2025 |
| Writing & Final Submission               | 10th May - 31st May 2025 |

---

## 8. Progress Summary

| Date       | Task                                      | Status    	|
|------------|-------------------------------------------|--------------|
| 01-03-2025 | Literature Review on CLIP and MBA-Net     | ✅ Completed	|
| 05-03-2025 | Dataset Collection (11k Hands)            | ✅ Completed	|
| 10-03-2025 | Dataset Preprocessing                     | ✅ Completed	|
| 15-03-2025 | CLIP Baseline Testing (without fine-tuning)| ⏳ Planned |

---

## 9. Risks & Mitigations

- **Data Availability**: Limited hand datasets → Use data augmentation.
- **Computational Costs**: Fine-tuning CLIP requires high GPU → Optimize batch size and use cloud resources if needed.
- **Model Adaptation**: CLIP pretrained on general images → Careful fine-tuning on hand-specific datasets.

---

## 10. Notes & Remarks

- Initial experiments with CLIP zero-shot classification show low accuracy on hand images.
- Text prompts are critical for enhancing performance—needs further research.
- Plan to test CLIP-SCGI and PromptSG ideas for potential improvements.

---

## 11. Next Steps

1. Complete dataset preprocessing and augmentation.
2. Implement CLIP fine-tuning for hand images.
3. Start integrating text prompts for contrastive learning.

