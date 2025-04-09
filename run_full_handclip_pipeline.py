import os
import subprocess

commands = [
    {
        "title": "🔧 Stage 1: Train Classifier (Frozen Text)",
        "cmd": "python experiments/stage1_train_classifier_frozen_text/train_stage1_frozen_text.py --config configs/train_stage1_frozen_text/train_vitb16_11k_dorsal_r.yml"
    },
    {
        "title": "🧪 Stage 1: Evaluation",
        "cmd": "python experiments/stage1_train_classifier_frozen_text/eval_stage1_frozen_text_single.py configs/eval_stage1_frozen_text/eval_vitb16_11k_dorsal_r.yml"
    },
    {
        "title": "🔧 Stage 2: Train CLIP ReID - Joint Training (Prompt + Image Encoder)",
        "cmd": "python experiments/stage2_clipreid_integration/train_stage2_joint.py --config configs/train_stage2_clip_reid/train_joint_vitb16_11k_dorsal_r.yml"
    },
    {
        "title": "🧪 Stage 2: Evaluation",
        "cmd": "python experiments/stage2_clipreid_integration/eval_stage2_joint.py configs/eval_stage2_clip_reid/eval_joint_vitb16_11k_dorsal_r.yml"
    },
    {
        "title": "🔧 Stage 3: Train PromptSG",
        "cmd": "python experiments/stage3_promptsg_integration/train_stage3_promptsg.py --config configs/train_stage3_promptsg/train_stage3_vitb16_11k_dorsal_r.yml"
    },
    {
        "title": "🧪 Stage 3: Evaluation",
        "cmd": "python experiments/stage3_promptsg_integration/eval_stage3_promptsg.py configs/eval_stage3_promptsg/eval_stage3_vitb16_11k_dorsal_r.yml"
    },
]

def run_pipeline():
    for step in commands:
        print(f"\n\033[94m{step['title']}\033[0m")
        print(f"➤ Running: {step['cmd']}\n")
        result = subprocess.run(step["cmd"], shell=True)
        if result.returncode != 0:
            print(f"\033[91m❌ Failed: {step['title']}\033[0m")
            break
        else:
            print(f"\033[92m✅ Completed: {step['title']}\033[0m")

if __name__ == "__main__":
    run_pipeline()
