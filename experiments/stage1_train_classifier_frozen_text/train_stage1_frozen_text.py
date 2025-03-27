import sys                                # System-specific parameters and functions
import os                                 # OS module for path and environment management

# 🔧 Add the project root directory to PYTHONPATH to ensure internal imports work correctly.
# This allows importing modules like `datasets`, `models`, `engine`, etc., without installing them as packages.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ===== External dependencies =====
import argparse                           # For parsing command-line arguments
import yaml                               # For reading YAML configuration files
import torch                              # PyTorch for device setup
from datasets.build_dataloader import get_train_val_loaders     # Loads training and validation data
from models.make_model import build_model                      # Constructs CLIP + classifier model
from engine.finetune_trainer_stage1 import FinetuneTrainerStage1  # Trainer class for fine-tuning Stage 1


def main(config_path):
    """
    Main pipeline to train Stage 1 fine-tuning of CLIP model using a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file containing training settings.
    """

    # 🔹 Load configuration from YAML file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 🔹 Automatically choose GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Load training and validation DataLoaders
    # We only use the training loader for this stage
    train_loader, _, num_classes = get_train_val_loaders(config)
    config["num_classes"] = num_classes  # Store class count in config for model building

    # 🔹 Build the model components
    # - Loads CLIP (ViT-B/16 or RN50)
    # - Adds a linear classifier on top of image encoder
    # - Freezes text encoder since it's not used here
    clip_model, classifier = build_model(config, freeze_text=True)

    # 🔹 Initialize trainer and begin training loop
    trainer = FinetuneTrainerStage1(clip_model, classifier, train_loader, config, device)
    trainer.train()


# Entry point of the script when run from command-line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    # 🔹 Execute main pipeline with provided config path
    main(args.config)
