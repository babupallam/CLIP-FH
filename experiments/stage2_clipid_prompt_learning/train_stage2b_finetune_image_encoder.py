import os
import sys
import yaml
import torch

# 🔧 Ensure local module imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if "datasets" not in os.listdir(PROJECT_ROOT):
    raise RuntimeError("❌ PROJECT_ROOT is misaligned. Check relative path in script.")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.clip_patch import load_clip_with_patch
from models.prompt_learner import PromptLearner
from datasets.build_dataloader import get_train_val_loaders
from engine.clipreid_trainer_stage2 import PromptLearnerTrainerStage2b


def main(config_path):
    # === Load YAML Config ===
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 🔹 Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    # === General Config Logging ===
    print("🚀 Starting Stage 2b - Image Encoder Fine-Tuning")
    print(f"Experiment : {cfg['experiment']}")
    print(f"Model      : {cfg['clip_model']} | Dataset: {cfg['dataset']} | Aspect: {cfg['aspect']}")
    print(f"Stage      : {cfg['stage']}")
    print(f"Batch Size : {cfg['batch_size']} | Epochs: {cfg['epochs']} | LR: {cfg['lr']}")
    print(f"Prompt Ctx : {cfg['n_ctx']} | Template: {cfg['prompt_template']}")
    print(f"Freeze Text: {cfg['freeze_text_encoder']} | Freeze Prompt: {cfg['freeze_prompt']}")

    # === Load CLIP model ===
    model_type = cfg.get("clip_model", "ViT-B/16")
    clip_model, _ = load_clip_with_patch(model_type=model_type, device=device, freeze_all=False)

    # === Freeze text encoder (if specified) ===
    if cfg.get("freeze_text_encoder", True):
        for param in clip_model.transformer.parameters():
            param.requires_grad = False
        for param in clip_model.token_embedding.parameters():
            param.requires_grad = False

    # === Get classnames from dataloader ===
    train_loader, _, num_classes = get_train_val_loaders(cfg)
    class_to_idx = train_loader.dataset.class_to_idx
    classnames = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    print(f"🧾 Number of classes found: {len(classnames)}")

    # === Initialize Prompt Learner ===
    prompt_learner = PromptLearner(
        classnames=classnames,
        clip_model=clip_model,
        n_ctx=cfg.get("n_ctx", 8),
        ctx_init=cfg.get("ctx_init", None),
        prompt_template=cfg.get("prompt_template", "a photo of a {}."),
        aspect=cfg["aspect"],
        device=device
    )

    # === Load trained prompt from Stage 2a ===
    prompt_ckpt = cfg.get("resume_prompt_from", "")
    assert os.path.exists(prompt_ckpt), f"❌ Prompt checkpoint not found: {prompt_ckpt}"
    print(f"🔁 Loading prompt from: {prompt_ckpt}")

    ckpt = torch.load(prompt_ckpt, map_location=device)

    # Safely remove tokenized_prompts if shape will mismatch
    if "tokenized_prompts" in ckpt:
        del ckpt["tokenized_prompts"]

    # Load the remaining weights
    load_result = prompt_learner.load_state_dict(ckpt, strict=False)
    print(f"⚠️ Loaded prompt with relaxed matching")
    print(f" - Missing keys   : {load_result.missing_keys}")
    print(f" - Unexpected keys: {load_result.unexpected_keys}")

    # === Freeze prompt learner if specified ===
    if cfg.get("freeze_prompt", True):
        for param in prompt_learner.parameters():
            param.requires_grad = False

    # === Trainer ===
    trainer = PromptLearnerTrainerStage2b(
        clip_model=clip_model,
        prompt_learner=prompt_learner,
        train_loader=train_loader,
        config=cfg,
        device=device
    )

    # === Start Training ===
    trainer.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
