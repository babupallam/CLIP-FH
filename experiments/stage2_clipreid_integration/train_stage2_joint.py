import os
import sys

# ===== Project Root =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch

# ===== Imports =====
from utils.clip_patch import load_clip_with_patch
from engine.prompt_learner import PromptLearner
from utils.dataloaders import get_train_val_loaders
from utils.loss.make_loss import build_loss

from utils.loss import CrossEntropyLoss
from utils.loss.triplet_loss import TripletLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import setup_logger
from utils.naming import build_filename
from engine.train_clipreid_stages import train_clipreid_prompt_stage, train_clipreid_image_stage



def train_joint(cfg_path):
    # === Load Config ===
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # === Build log file path ===
    os.makedirs(cfg["output_dir"], exist_ok=True)
    log_filename = build_filename(cfg, cfg.get("epochs_image"), stage="image", extension=".log", timestamped=False)
    log_path = os.path.join(cfg["output_dir"], log_filename)
    logger = setup_logger(log_path)

    def log(text):
        logger.info(text)

    # === Setup ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = cfg["model"]
    stage_mode = cfg.get("stage_mode", "prompt_then_image")
    epochs_prompt = cfg.get("epochs_prompt", 20)
    epochs_image = cfg.get("epochs_image", 20)
    lr = cfg["lr"]

    # === Load Model & Data ===
    clip_model, _ = load_clip_with_patch(model_type, device, freeze_all=True)
    train_loader, val_loader, num_classes = get_train_val_loaders(cfg)
    cfg["num_classes"] = num_classes

    class_to_idx = train_loader.dataset.class_to_idx
    classnames = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]

    # === Prompt Learner ===
    prompt_learner = PromptLearner(
        classnames=classnames,
        clip_model=clip_model,
        n_ctx=cfg["n_ctx"],
        ctx_init=cfg.get("ctx_init", None),
        prompt_template=cfg["prompt_template"],
        aspect=cfg["aspect"],
        device=device
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad,
               list(prompt_learner.parameters()) + list(clip_model.parameters())),
        lr=lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_prompt + epochs_image, eta_min=1e-6)
    loss_fn = build_loss(cfg["loss_list"], num_classes=num_classes,
                         feat_dim=clip_model.ln_final.weight.shape[0])

    ce_loss = CrossEntropyLoss()
    triplet_loss = TripletLoss(margin=0.3)

    if stage_mode in ["prompt_then_image", "prompt_only"]:
        cfg["loss_fn"] = loss_fn  # pass loss fn to stage
        train_clipreid_prompt_stage(
            clip_model=clip_model,
            prompt_learner=prompt_learner,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            log=log
        )

    if stage_mode in ["prompt_then_image", "image_only"]:
        train_clipreid_image_stage(
            clip_model=clip_model,
            prompt_learner=prompt_learner,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            log=log,
            loss_fn=loss_fn,
            ce_loss=ce_loss,
            triplet_loss=triplet_loss
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train_joint(args.config)