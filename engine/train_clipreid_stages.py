import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.save_load_models import save_checkpoint
from utils.eval_utils import validate
from utils.feature_cache import cache_image_features
from utils.naming import build_filename


def train_clipreid_prompt_stage(clip_model, prompt_learner, optimizer, scheduler,
                                train_loader, val_loader, cfg, device, log):
    image_feats, labels = cache_image_features(clip_model, train_loader, device)
    log(f"Cached {image_feats.shape[0]} features")

    best_rank1_prompt = 0.0
    for epoch in range(cfg["epochs_prompt"]):
        clip_model.eval()
        prompt_learner.train()
        indices = torch.randperm(len(labels))
        total_loss = 0

        pbar = tqdm(range(0, len(labels), cfg["batch_size"]), desc=f"Prompt Epoch {epoch + 1}/{cfg['epochs_prompt']}")
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

            loss_fn = cfg["loss_fn"]
            loss_i2t = loss_fn(features=batch_feats, text_features=text_feats, targets=batch_labels, mode="contrastive")
            loss_t2i = loss_fn(features=text_feats, text_features=batch_feats, targets=batch_labels, mode="contrastive")
            loss = loss_i2t + loss_t2i + 0.001 * (prompt_learner.ctx ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "prompt_norm": f"{prompt_learner.ctx.norm(dim=1).mean().item():.4f}"
            })

        metrics = validate(clip_model, prompt_learner, val_loader, device, cfg, log)
        if metrics["rank1"] > best_rank1_prompt:
            best_rank1_prompt = metrics["rank1"]
            best_path = os.path.join(cfg["save_dir"], build_filename(cfg, epoch, stage="prompt", extension="_BEST.pth", timestamped=False))
            save_checkpoint(clip_model, None, optimizer, cfg, epoch, metrics, best_path, True, scheduler, total_loss)
            log(f" New BEST Prompt model saved at Rank-1 = {metrics['rank1'] * 100:.2f}%")

        scheduler.step()


def train_clipreid_image_stage(clip_model, prompt_learner, optimizer, scheduler,
                               train_loader, val_loader, cfg, device, log,
                               loss_fn, ce_loss, triplet_loss):
    # Unfreeze image encoder
    for param in clip_model.visual.parameters():
        param.requires_grad = True
    log("Unfroze CLIP image encoder.")

    if cfg.get("freeze_prompt", True):
        for param in prompt_learner.parameters():
            param.requires_grad = False
        log("Prompt Learner frozen.")

    best_rank1_image = 0.0
    for epoch in range(cfg["epochs_image"]):
        prompt_learner.train()
        clip_model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Image Epoch {epoch + 1}/{cfg['epochs_image']}")
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

            # Optional: Classifier for ID loss
            if not hasattr(clip_model, "classifier"):
                clip_model.classifier = nn.Linear(image_feats.size(1), cfg["num_classes"]).to(device)

            id_logits = clip_model.classifier(image_feats)
            id_loss = ce_loss(id_logits, labels_batch)
            tri_loss = triplet_loss(image_feats, labels_batch)

            loss = id_loss + tri_loss + loss_i2t + loss_t2i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = validate(clip_model, prompt_learner, val_loader, device, cfg, log)
        if metrics["rank1"] > best_rank1_image:
            best_rank1_image = metrics["rank1"]
            best_path = os.path.join(cfg["save_dir"], build_filename(cfg, epoch, stage="image", extension="_BEST.pth", timestamped=False))
            save_checkpoint(clip_model, None, optimizer, cfg, epoch, metrics, best_path, True, scheduler, total_loss)
            log(f" New BEST Image model saved at Rank-1 = {metrics['rank1'] * 100:.2f}%")

        scheduler.step()
