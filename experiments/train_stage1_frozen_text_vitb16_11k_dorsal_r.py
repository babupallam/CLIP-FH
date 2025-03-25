import sys
import os

# 🔧 Force add project root (parent of /experiments) to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import argparse
import yaml
import torch
import os
from datasets.build_dataloader import get_train_val_loaders
from models.make_model import build_model
from engine.finetune_trainer_stage1 import FinetuneTrainerStage1



def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, _, num_classes = get_train_val_loaders(config)
    config["num_classes"] = num_classes

    # Build model
    clip_model, classifier = build_model(config, freeze_text=True)

    # Train
    trainer = FinetuneTrainerStage1(clip_model, classifier, train_loader, config, device)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)
