import sys                                # System-specific parameters and functions
import os                                 # OS module for path and environment management

# 🔧 Add the project root directory to PYTHONPATH to ensure internal imports work correctly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if "datasets" not in os.listdir(PROJECT_ROOT):
    raise RuntimeError("PROJECT_ROOT is misaligned. Check the relative path in the script.")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ===== External dependencies =====
import argparse
import yaml
import torch
from datasets.build_dataloader import get_train_val_loaders
from models.make_model import build_model
from engine.finetune_trainer_stage1 import FinetuneTrainerStage1


def generate_model_name(config):
    """
    Generate a unique model filename based on key hyperparameters.
    """
    stage = "stage1"
    strategy = config.get("variant", "na")
    model = config.get("model", "clip")
    dataset = config.get("dataset", "unk")
    aspect = config.get("aspect", "unk")
    epochs = f"e{config.get('epochs', 'x')}"
    lr = f"lr{str(config.get('lr', 'x')).replace('.', '').replace('-', '')}"
    batch = f"bs{config.get('batch_size', 'x')}"
    loss = f"loss{config.get('loss', 'ce')}"  # default to 'ce' if not specified

    name = f"{stage}_{strategy}_{model}_{dataset}_{aspect}_{epochs}_{lr}_{batch}_{loss}"
    return name


def get_incremental_checkpoint_path(save_dir, base_name):
    """
    Ensures a new checkpoint file is created every run with unique suffix.
    E.g., stage1_model_v1.pth, stage1_model_v2.pth, etc.
    """
    os.makedirs(save_dir, exist_ok=True)
    existing = [
        f for f in os.listdir(save_dir)
        if f.startswith(base_name) and f.endswith(".pth")
    ]

    if not existing:
        return os.path.join(save_dir, f"{base_name}_v1.pth")

    # Extract numerical suffixes
    suffixes = []
    for f in existing:
        parts = f.replace(".pth", "").split("_v")
        if len(parts) == 2 and parts[1].isdigit():
            suffixes.append(int(parts[1]))

    next_suffix = max(suffixes, default=1) + 1
    return os.path.join(save_dir, f"{base_name}_v{next_suffix}.pth")


def main(config_path):
    """
    Continues Stage 1 fine-tuning of a CLIP model from a saved checkpoint using a YAML config.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(config.get("resume_from", "")):
        raise FileNotFoundError(f"❌ Checkpoint not found at: {config.get('resume_from')}")

    # 🔹 Generate descriptive model name and checkpoint path
    exp_name = generate_model_name(config)
    config["experiment"] = exp_name
    config["save_path"] = get_incremental_checkpoint_path(config["save_dir"], exp_name)
    config["log_path"] = os.path.join(config["log_dir"], f"{exp_name}.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, num_classes = get_train_val_loaders(config)
    config["num_classes"] = num_classes

    clip_model, classifier = build_model(config, freeze_text=True)
    checkpoint = torch.load(config["resume_from"], map_location=device)

    # Load weights safely from structured or flat checkpoint
    if "model" in checkpoint and "classifier" in checkpoint:
        clip_model.load_state_dict(checkpoint["model"])
        classifier.load_state_dict(checkpoint["classifier"])
        print("✅ Loaded 'model' and 'classifier' from structured checkpoint.")
    else:
        clip_model.load_state_dict(checkpoint)
        print("✅ Loaded full model directly from flat checkpoint.")

    trainer = FinetuneTrainerStage1(clip_model, classifier, train_loader, config, device)
    trainer.train()

    print(f"💾 New checkpoint saved to: {config['save_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    main(args.config)
