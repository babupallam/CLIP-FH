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

from utils.loss.cross_entropy_loss import CrossEntropyLoss
from utils.loss.triplet_loss import TripletLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR

from utils.naming import build_filename
from engine.train_clipreid_stages import train_clipreid_prompt_stage, train_clipreid_image_stage,evaluate_clipreid_after_training
from utils.logger import setup_logger
from utils.train_helpers import register_bnneck_and_arcface, unfreeze_clip_text_encoder, unfreeze_clip_image_encoder

from torch.optim import AdamW


def clipreid_integration(cfg_path):
    # === Load Config ===
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # === Build log file path ===
    os.makedirs(cfg["output_dir"], exist_ok=True)
    log_filename = build_filename(cfg, cfg.get("epochs_image"), stage="image", extension=".log", timestamped=False)
    log_path = os.path.join(cfg["output_dir"], log_filename)
    logger = setup_logger(log_path)

    # === Setup ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = cfg["model"]
    stage_mode = cfg.get("stage_mode", "prompt_then_image")
    epochs_prompt = cfg.get("epochs_prompt", 20)
    epochs_image = cfg.get("epochs_image", 20)
    lr = cfg["lr"]

    # === Load Model & Data ===
    clip_model, _ = load_clip_with_patch(model_type, device, freeze_all=False)
    clip_model.float()

    # Log CLIP text encoder embedding and output dims
    ctx_dim = clip_model.token_embedding.embedding_dim
    output_dim = clip_model.ln_final.weight.shape[0]
    logger.info(f"[DEBUG] CLIP Model = {model_type}  Token Embed Dim: {ctx_dim}, Final LN Dim: {output_dim}")

    best_model_state = {
        "clip_model": None,
        "prompt_learner": None
    }

    train_loader, val_loader, num_classes = get_train_val_loaders(cfg)
    cfg["num_classes"] = num_classes

    class_to_idx = train_loader.dataset.class_to_idx
    classnames = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]


    # === Prompt Learner ===
    prompt_learner = PromptLearner(
        classnames=classnames, #identities name
        cfg=cfg,
        clip_model=clip_model,
        n_ctx=cfg["n_ctx"],
        ctx_init=cfg.get("ctx_init", None),
        template=cfg["prompt_template"],
        aspect=cfg["aspect"],
        device=device
    )

    # Check PromptLearner structure
    proj = getattr(prompt_learner, 'proj', None)
    logger.info(f"[DEBUG] PromptLearner proj layer = {proj.__class__.__name__} | "
                f"in: {ctx_dim}, out: {output_dim}")

    # === Infer feature dimension using dummy forward pass
    with torch.no_grad():
        input_size = cfg.get("input_size", [224, 224])  # fallback to 224x224
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        dummy_feat = clip_model.encode_image(dummy_input)
        feat_dim = dummy_feat.shape[1]

        #Confirm image feature dim matches prompt proj output
        logger.info(f"[DEBUG] Dummy Image Feature Shape: {dummy_feat.shape}")
        logger.info(f"[DEBUG] Text Feature Output (via proj) should match: {output_dim}")
        if feat_dim != output_dim:
            logger.warning(f"[WARNING] Feature dim mismatch! Image: {feat_dim}, Text: {output_dim}")

    register_bnneck_and_arcface(
        model=clip_model,
        config=cfg,
        feat_dim=feat_dim,
        num_classes=num_classes,
        device=device,
        logger=logger.info
    )


    num_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    logger.info(f"Total trainable CLIP params: {num_params}")

    # for v2 from adam to adamw
    from torch.optim import AdamW

    from torch.optim import AdamW

    optimizer = AdamW([
        {
            "params": prompt_learner.parameters(),
            "lr": float(cfg.get("lr_prompt", 1e-4)),
            "weight_decay": float(cfg.get("weight_decay_prompt", 5e-4)),
        },
        {
            "params": clip_model.visual.parameters(),
            "lr": float(cfg.get("lr_visual", 1e-5)),
            "weight_decay": float(cfg.get("weight_decay_visual", 5e-4)),
        },
        {
            "params": clip_model.transformer.parameters(),
            "lr": float(cfg.get("lr_text", 5e-6)),
            "weight_decay": float(cfg.get("weight_decay_text", 5e-4)),
        },
    ])

    logger.info(f"[DEBUG] PromptLearner trainable params: {sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad)}")
    logger.info(f"[DEBUG] CLIP trainable params: {sum(p.numel() for p in clip_model.parameters() if p.requires_grad)}")

    # for v2
    # 1. Scheduler for Prompt stage
    #  The learning rate decreases in a smooth curve like a cosine wave.
    scheduler_prompt = CosineAnnealingLR(optimizer, T_max=epochs_prompt, eta_min=1e-6)

    # 2. Scheduler for Image stage
    # OneCycle policy: Learning rate first increases and then decreases.
    # More aggressive: Starts low → goes up → drops down again.
    scheduler_image = CosineAnnealingLR(optimizer, T_max=epochs_image, eta_min=1e-6)

    # for v3

    """
    # One cycle LR
    
    steps_per_epoch = len(train_loader)  # number of batches
    epochs_image = cfg["epochs_image"]
    total_steps = steps_per_epoch * epochs_image
    cfg["total_steps"] = total_steps

    # 1. Scheduler for Prompt stage
    scheduler_prompt = OneCycleLR(
        optimizer,
        max_lr=max([g["lr"] for g in optimizer.param_groups]),
        total_steps=epochs_prompt * len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        cycle_momentum=False,
    )

    # 2. Scheduler for Image stage
    scheduler_image = OneCycleLR(
        optimizer,
        max_lr=max([g["lr"] for g in optimizer.param_groups]),
        total_steps=epochs_image * len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        cycle_momentum=False,
    )

    """

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
            scheduler=scheduler_prompt,
            train_loader=train_loader,
            cfg=cfg,
            device=device,
            logger=logger,
            best_model_state = best_model_state
        )

    if best_model_state["prompt_learner"]:
        prompt_learner.load_state_dict(best_model_state["prompt_learner"])
        for p in prompt_learner.parameters():
            p.requires_grad_(False)

    # till v6
    # Unfreeze encoders BEFORE optimizer creation
    #unfreeze_clip_text_encoder(clip_model, logger.info)
    #unfreeze_clip_image_encoder(clip_model, logger.info)

    # for v6 -- unfreeze some layers for finetuning

    # ------------------------------------------------------------------------------
    # Unfreezing Strategy: unfreeze_blocks
    #
    # Controls how many blocks of the CLIP image encoder are unfrozen for fine-tuning.
    # This setting dynamically adapts based on the visual backbone (ViT or RN50).
    #
    # === For ViT (e.g., ViT-B/16): ===
    # - CLIP ViT has 12 transformer blocks (resblocks).
    # - Valid range: 0 to 12
    #   - 0 = freeze all layers
    #   - 2 = unfreeze last 2 blocks (recommended)
    #   - 12 = unfreeze all blocks (fully fine-tune ViT)
    #
    # === For RN50 (ResNet-50): ===
    # - CLIP RN50 has 4 main layers: layer1, layer2, layer3, layer4
    # - Valid range: 0 to 4
    #   - 0 = freeze all layers
    #   - 1 = unfreeze layer4 only
    #   - 2 = unfreeze layer3 and layer4 (recommended)
    #   - 4 = unfreeze all layers
    #
    # Note:
    # - Over-unfreezing may hurt generalization unless regularized well.
    # - This is helpful for staged fine-tuning (v6, v7 experiments, etc.)
    # ------------------------------------------------------------------------------
    unfreeze_blocks = cfg.get("unfreeze_blocks", 0)
    unfreeze_clip_image_encoder(clip_model, logger=logger.info, unfreeze_blocks=unfreeze_blocks)

    if stage_mode in ["prompt_then_image", "image_only"]:
        train_clipreid_image_stage(
            clip_model=clip_model,
            prompt_learner= prompt_learner,
            optimizer=optimizer,
            scheduler=scheduler_image,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            logger=logger,
            loss_fn=loss_fn,
            ce_loss=ce_loss,
            triplet_loss=triplet_loss,
            best_model_state = best_model_state
        )

    if best_model_state["clip_model"] and best_model_state["prompt_learner"]:
        clip_model.load_state_dict(best_model_state["clip_model"])
        prompt_learner.load_state_dict(best_model_state["prompt_learner"])
        logger.info("Restored BEST model for evaluation.")
    else:
        logger.warning("No best model stored  using final model for evaluation.")

    # === Final Evaluation: ReID metrics across 10 splits ===
    logger.info("[Eval] Starting evaluation across all 10 splits (ReID style)")
    evaluate_clipreid_after_training(
        clip_model=clip_model,
        prompt_learner=prompt_learner,
        cfg=cfg,
        logger=logger,
        device=device
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    clipreid_integration(args.config)