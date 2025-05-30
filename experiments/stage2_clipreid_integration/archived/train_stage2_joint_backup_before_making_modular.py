import os
import sys

import yaml
import torch
from datetime import datetime
from torch import nn
from tqdm import tqdm
import logging
from utils.save_load_models import save_checkpoint

# ===== Project Root =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ===== Imports =====
from utils.clip_patch import load_clip_with_patch
from engine.prompt_learner import PromptLearner
from utils.dataloaders import get_train_val_loaders
from utils.loss.make_loss import build_loss

from utils.loss.cross_entropy_loss import CrossEntropyLoss
from utils.loss.triplet_loss import TripletLoss
from torch.optim.lr_scheduler import CosineAnnealingLR


def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    return logger


def cache_image_features(clip_model, dataloader, device):
    clip_model.eval()
    image_features, labels = [], []

    with torch.no_grad():
        for images, label_batch in dataloader:
            images, label_batch = images.to(device), label_batch.to(device)
            feats = clip_model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            image_features.append(feats.cpu())
            labels.append(label_batch.cpu())

    return torch.cat(image_features), torch.cat(labels)


def unfreeze_image_encoder(clip_model, log):
    for param in clip_model.visual.parameters():
        param.requires_grad = True
    log("Unfroze CLIP image encoder.")


def freeze_prompt_learner(prompt_learner,log):
    for param in prompt_learner.parameters():
        param.requires_grad = False
    log("Prompt Learner frozen.")




def build_filename(config, epoches, stage, extension=".pth", timestamped=True):
    base = f"{config['experiment']}_{config['model']}_{config['dataset']}_{config['aspect']}"
    if stage == "prompt":
        base += f"_prompt_nctx{config['n_ctx']}_e{epoches}_lr{str(config['lr']).replace('.', '')}_bs{config['batch_size']}"
    elif stage == "image":
        base += f"_finetune_e{epoches}_lr{str(config['lr']).replace('.', '')}_bs{config['batch_size']}"
    if timestamped:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base += f"_{ts}"
    return base + extension


def validate(model, prompt_learner, val_loader, device, config, log):
    model.eval()
    prompt_learner.eval()
    all_img_feats, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            img_feats = model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            prompts = prompt_learner.forward_batch(labels)
            x = prompts + model.positional_embedding.unsqueeze(0)
            x = x.permute(1, 0, 2)
            x = model.transformer(x)
            x = x.permute(1, 0, 2)
            txt_feats = model.ln_final(x[:, 0, :])
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            all_img_feats.append(img_feats)
            all_labels.append(labels)

    img_feats = torch.cat(all_img_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    sim_matrix = img_feats @ txt_feats.T

    ranks = [1, 5, 10]
    metrics = {}

    sorted_indices = sim_matrix.argsort(dim=1, descending=True)
    aps = []
    for i in range(len(labels)):
        label = labels[i]
        ranking = labels[sorted_indices[i]]
        correct = (ranking == label).float()
        if correct.sum() == 0:
            continue
        precision_at_k = correct.cumsum(0) / (torch.arange(1, len(correct) + 1, device=device))
        ap = (precision_at_k * correct).sum() / correct.sum()
        aps.append(ap.item())
    metrics["mAP"] = sum(aps) / len(aps)

    for r in ranks:
        k = min(r, sim_matrix.size(1))  # Avoid topk crash
        correct = (sim_matrix.topk(k, dim=1).indices == torch.arange(sim_matrix.size(0), device=device).unsqueeze(1)).any(dim=1)
        metrics[f"rank{r}"] = correct.float().mean().item()

    log("\nValidation Results:")
    for k, v in metrics.items():
        log(f"{k.upper()}: {v*100:.2f}%")

    log(f"MAP: {metrics['mAP'] * 100:.2f}%")

    return metrics



def train_joint(cfg_path):
    # === Load Config ===
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # === Build log file path ===
    os.makedirs(cfg["output_dir"], exist_ok=True)
    log_filename = build_filename(cfg, cfg.get("epochs_image"), stage="image", extension=".log", timestamped=True)
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

    # === Stage 1: Prompt Tuning with Frozen CLIP ===
    if stage_mode in ["prompt_then_image", "prompt_only"]:
        image_feats, labels = cache_image_features(clip_model, train_loader, device)
        log(f"Cached {image_feats.shape[0]} features")
        best_rank1_prompt = 0.0  # Track best prompt-stage Rank-1
        for epoch in range(epochs_prompt):
            clip_model.eval()
            prompt_learner.train()
            indices = torch.randperm(len(labels))
            total_loss = 0

            pbar = tqdm(range(0, len(labels), cfg["batch_size"]), desc=f"Prompt Epoch {epoch + 1}/{epochs_prompt}")
            for i in pbar:
                idx = indices[i:i + cfg["batch_size"]]
                batch_feats = image_feats[idx].to(device)
                batch_labels = labels[idx].to(device)

                prompts = prompt_learner.forward_batch(batch_labels)
                x = prompts + clip_model.positional_embedding.unsqueeze(0)
                x = x.permute(1, 0, 2)
                x = clip_model.transformer(x)
                x = x.permute(1, 0, 2)
                text_feats = clip_model.ln_final(x[:, 0, :])
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                loss_i2t = loss_fn(features=batch_feats, text_features=text_feats,
                                   targets=batch_labels, mode="contrastive")
                loss_t2i = loss_fn(features=text_feats, text_features=batch_feats,
                                   targets=batch_labels, mode="contrastive")
                loss = loss_i2t + loss_t2i + 0.001 * (prompt_learner.ctx ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "prompt_norm": f"{prompt_learner.ctx.norm(dim=1).mean().item():.4f}"
                })

            # === Validate after each image fine-tune epoch
            metrics = validate(clip_model, prompt_learner, val_loader, device, cfg, log)
            if metrics["rank1"] > best_rank1_prompt:
                best_rank1_prompt = metrics["rank1"]
                best_path = os.path.join(cfg["save_dir"], build_filename(cfg, epoch, stage="prompt", extension="_BEST.pth",
                                                                         timestamped=False))
                save_checkpoint(
                    model=clip_model,
                    classifier=None,
                    optimizer=optimizer,
                    config=cfg,
                    epoch=epoch,
                    val_metrics=metrics,
                    path=best_path,
                    is_best=True,
                    scheduler=scheduler,
                    train_loss=total_loss
                )

                log(f" New BEST Prompt model saved at Rank-1 = {metrics['rank1'] * 100:.2f}%")

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "rank1": f"{metrics['rank1'] * 100:.2f}%",
                "mAP": f"{metrics['mAP'] * 100:.2f}%"
            })

            if epoch == epochs_prompt - 1:  # for Stage 2a
                final_path = os.path.join(cfg["save_dir"], build_filename(cfg, epochs_prompt, stage="prompt", extension="_FINAL.pth",
                                                                          timestamped=False))
                save_checkpoint(
                    model=clip_model,
                    classifier=None,
                    optimizer=optimizer,
                    config=cfg,
                    epoch=epochs_prompt,
                    val_metrics=metrics,
                    path=final_path,
                    is_best=False,
                    scheduler=scheduler,
                    train_loss=total_loss
                )

            log("=" * 60)
            log(f"[Epoch {epoch + 1}/{epochs_prompt}] Prompt Stage Summary")
            log(f"Prompt Loss         : {total_loss:.4f}")
            log(f"Rank-1 Accuracy     : {metrics['rank1'] * 100:.2f}%")
            log(f"Rank-5 Accuracy     : {metrics['rank5'] * 100:.2f}%")
            log(f"Rank-10 Accuracy    : {metrics['rank10'] * 100:.2f}%")
            log(f"Mean AP             : {metrics['mAP'] * 100:.2f}%")
            log("=" * 60)

            scheduler.step()

    # === Stage 2: Unfreeze and Fine-Tune Image Encoder ===
    if stage_mode in ["prompt_then_image", "image_only"]:
        unfreeze_image_encoder(clip_model, log)
        if cfg.get("freeze_prompt", True):
            freeze_prompt_learner(prompt_learner,log)

        best_rank1_image = 0.0  # Track best image-stage Rank-1
        for epoch in range(epochs_image):

            prompt_learner.train()
            clip_model.train()
            total_loss = 0

            pbar = tqdm(train_loader, desc=f"Image Epoch {epoch + 1}/{epochs_image}")
            for images, labels_batch in pbar:
                images, labels_batch = images.to(device), labels_batch.to(device)

                image_feats = clip_model.encode_image(images)
                image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

                prompts = prompt_learner.forward_batch(labels_batch)
                x = prompts + clip_model.positional_embedding.unsqueeze(0)
                x = x.permute(1, 0, 2)
                x = clip_model.transformer(x)
                x = x.permute(1, 0, 2)
                text_feats = clip_model.ln_final(x[:, 0, :])
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                loss_i2t = loss_fn(features=image_feats, text_features=text_feats,
                                   targets=labels_batch, mode="contrastive")
                loss_t2i = loss_fn(features=text_feats, text_features=image_feats,
                                   targets=labels_batch, mode="contrastive")

                # Assume you have a classifier layer for ID logits  add if missing
                if not hasattr(clip_model, "classifier"):
                    clip_model.classifier = nn.Linear(image_feats.size(1), num_classes).to(device)

                id_logits = clip_model.classifier(image_feats)
                id_loss = ce_loss(id_logits, labels_batch)
                tri_loss = triplet_loss(image_feats, labels_batch)

                loss = id_loss + tri_loss + loss_i2t + loss_t2i

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # === End of Epoch Validation (Stage 2b)
            metrics = validate(clip_model, prompt_learner, val_loader, device, cfg, log)
            if metrics["rank1"] > best_rank1_image:
                best_rank1_image = metrics["rank1"]
                best_path = os.path.join(cfg["save_dir"],
                                         build_filename(cfg, epoch, stage="image", extension="_BEST.pth", timestamped=False))
                save_checkpoint(
                    model=clip_model,
                    classifier=None,
                    optimizer=optimizer,
                    config=cfg,
                    epoch=epoch,
                    val_metrics=metrics,
                    path=best_path,
                    is_best=True,
                    scheduler=scheduler,
                    train_loss=total_loss
                )
                log(f"cNew BEST Image model saved at Rank-1 = {metrics['rank1'] * 100:.2f}%")

            log("=" * 60)
            log(f"[Epoch {epoch + 1}/{epochs_image}] Image Fine-Tune Stage Summary")
            log(f"Prompt Loss         : {total_loss:.4f}")
            log(f"Rank-1 Accuracy     : {metrics['rank1'] * 100:.2f}%")
            log(f"Rank-5 Accuracy     : {metrics['rank5'] * 100:.2f}%")
            log(f"Rank-10 Accuracy    : {metrics['rank10'] * 100:.2f}%")
            log(f"Mean AP             : {metrics['mAP'] * 100:.2f}%")
            log("=" * 60)

            if epoch == epochs_image - 1:
                final_path = os.path.join(cfg["save_dir"],
                                          build_filename(cfg, epochs_image, stage="image", extension="_FINAL.pth", timestamped=False))
                save_checkpoint(
                    model=clip_model,
                    classifier=None,
                    optimizer=optimizer,
                    config=cfg,
                    epoch=epochs_image,
                    val_metrics=metrics,
                    path=final_path,
                    is_best=False,
                    scheduler=scheduler,
                    train_loss=total_loss
                )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train_joint(args.config)