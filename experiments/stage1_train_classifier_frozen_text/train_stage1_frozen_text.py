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


def main(config_path):
    """
    Main pipeline to train Stage 1 fine-tuning of CLIP model using a YAML configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 🔹 Construct filename from config
    exp_name = generate_model_name(config)
    config["experiment"] = exp_name
    config["save_path"] = os.path.join(config["save_dir"], f"{exp_name}.pth")
    config["log_path"] = os.path.join(config["log_dir"], f"{exp_name}.log")

    # 🔹 Automatically choose GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Load training and validation DataLoaders
    train_loader, _, num_classes = get_train_val_loaders(config)
    config["num_classes"] = num_classes

    # 🔹 Build the model components
    clip_model, classifier = build_model(config, freeze_text=True)

    # 🔹 Initialize trainer and begin training loop
    trainer = FinetuneTrainerStage1(clip_model, classifier, train_loader, config, device)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    main(args.config)
