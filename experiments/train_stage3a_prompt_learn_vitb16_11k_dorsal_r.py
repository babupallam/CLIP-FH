import os
import sys
import yaml
import torch
import argparse

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import clip
from datasets.build_dataloader import get_train_val_loaders
from models.prompt_learner import PromptLearner
from engine.clipreid_trainer_stage1 import ClipReIDPromptTrainer

# 👇 Patch CLIP with custom text encoder from embedding
from models.clip_patch import encode_text_from_embedding

def main(config_path):
    # 🔹 Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Load CLIP model
    model_name = "ViT-B/16" if config["model"] == "vitb16" else "RN50"
    clip_model, _ = clip.load(model_name, device=device)

    # 🔹 Monkey-patch CLIP to support prompt embeddings
    clip_model.encode_text_from_embedding = encode_text_from_embedding.__get__(clip_model)

    # 🔹 Load dataloader
    train_loader, _, num_classes = get_train_val_loaders(config)
    config["num_classes"] = num_classes

    # 🔹 Dummy class names (e.g., "ID_000", "ID_001", ...)
    class_names = [f"ID_{i:03d}" for i in range(num_classes)]

    # 🔹 Prompt learner: learns [X1]...[Xn] per class
    prompt_learner = PromptLearner(
        class_names=class_names,
        clip_model=clip_model,
        n_ctx=config.get("n_ctx", 4),
        prefix="A photo of a",
        suffix="person."
    ).to(device)

    # 🔹 Train prompt tokens with frozen CLIP
    trainer = ClipReIDPromptTrainer(clip_model, prompt_learner, train_loader, config, device)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
