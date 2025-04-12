import argparse
import yaml
import torch
import os
import sys

# Add root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataloaders import get_train_val_loaders
from models.archived.make_model import build_model
from engine.archived.finetune_trainer_stage2_loss import FinetuneTrainerStage2

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, num_classes = get_train_val_loaders(config)
    config["num_classes"] = num_classes

    clip_model, classifier = build_model(config, freeze_text=True)
    trainer = FinetuneTrainerStage2(clip_model, classifier, train_loader, config, device)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
