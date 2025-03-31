import os
import sys

# 🔧 Ensure local module imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if "datasets" not in os.listdir(PROJECT_ROOT):
    raise RuntimeError("❌ PROJECT_ROOT is misaligned. Check relative path in script.")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os
import yaml
import torch
from models.clip_patch import load_clip_with_patch
from models.prompt_learner import PromptLearner
from datasets.build_dataloader import get_train_val_loaders


def validate_prompt_checkpoint(config_path):
    print("🔍 Validating Stage 2a Prompt Checkpoint...")

    # === Load YAML Config ===
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    # === Normalize model type for compatibility ===
    raw_model = cfg.get("model", "vitb16")
    model_type = raw_model.lower().replace("-", "").replace("/", "")

    # === Load frozen CLIP ===
    clip_model, _ = load_clip_with_patch(model_type, device, freeze_all=True)

    # === Load classnames from dataloader ===
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
        device=device
    )

    # === Load prompt checkpoint ===
    ckpt_path = cfg.get("resume_prompt_from", "")
    assert os.path.exists(ckpt_path), f"❌ Checkpoint not found: {ckpt_path}"
    print(f"📂 Loading prompt checkpoint from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # Remove incompatible buffer if present
    if "tokenized_prompts" in ckpt:
        print("⚠️ Removing tokenized_prompts from checkpoint (non-trainable buffer)...")
        del ckpt["tokenized_prompts"]

    # Load state dict with relaxed strict=False
    result = prompt_learner.load_state_dict(ckpt, strict=False)

    print("✅ Prompt checkpoint loaded successfully.")
    print(f" - Missing keys   : {result.missing_keys}")
    print(f" - Unexpected keys: {result.unexpected_keys}")

    # === Final sanity check ===
    with torch.no_grad():
        dummy_labels = torch.randint(0, len(classnames), (4,))
        dummy_prompts = prompt_learner.forward_batch(dummy_labels)
        print(f"🧪 forward_batch() successful: shape = {dummy_prompts.shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    validate_prompt_checkpoint(args.config)
