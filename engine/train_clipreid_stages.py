import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.save_load_models import save_checkpoint
from utils.eval_utils import validate
from utils.feature_cache import cache_image_features
from utils.naming import build_filename
from utils.loss.center_loss import CenterLoss
from utils.loss.arcface import ArcFace
from utils.logger import setup_logger
from utils.train_helpers import freeze_entire_clip_model, unfreeze_clip_text_encoder,freeze_prompt_learner


def train_clipreid_prompt_stage(clip_model, prompt_learner, optimizer, scheduler,
                                train_loader, cfg, device, logger):
    # 0. Freeze all of CLIP for prompt learning
    freeze_entire_clip_model(clip_model, logger.info)

    # 1. Cache frozen image features
    image_feats, labels = cache_image_features(clip_model, train_loader, device)
    logger.info(f"Cached {image_feats.shape[0]} features for prompt learning.")

    # 2. Set model modes: CLIP frozen, prompt trainable
    clip_model.eval()  # for inference, keeps behavior consistent
    best_loss = float("inf")


    for epoch in range(cfg["epochs_prompt"]):
        prompt_learner.train()
        indices = torch.randperm(len(labels))  # shuffle indices
        total_loss = 0

        # 3. Mini-batch loop over frozen features
        pbar = tqdm(range(0, len(labels), cfg["batch_size"]), desc=f"Prompt Epoch {epoch + 1}/{cfg['epochs_prompt']}")
        for i in pbar:
            idx = indices[i:i + cfg["batch_size"]]
            batch_feats = image_feats[idx].to(device)
            batch_labels = labels[idx].to(device)

            # 4. Generate prompts → encode → get text features
            prompts = prompt_learner.forward_batch(batch_labels)
            x = prompts + clip_model.positional_embedding.unsqueeze(0)
            x = x.permute(1, 0, 2)
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            text_feats = clip_model.ln_final(x[:, 0, :])
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            # 5. Contrastive Loss (i2t and t2i)
            loss_fn = cfg["loss_fn"]
            loss_i2t = loss_fn(features=batch_feats, text_features=text_feats, targets=batch_labels, mode="contrastive")
            loss_t2i = loss_fn(features=text_feats, text_features=batch_feats, targets=batch_labels, mode="contrastive")

            # 6. Prompt regularization (L2 norm)
            loss = loss_i2t + loss_t2i + 0.001 * (prompt_learner.ctx ** 2).mean()

            # 7. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 8. Logging (live progress)
            # Metric computation
            prompt_norm = prompt_learner.ctx.norm(dim=1).mean().item()
            prompt_var = prompt_learner.ctx.var(dim=0).mean().item()
            prompt_grad = (prompt_learner.ctx.grad.norm().item()
                           if prompt_learner.ctx.grad is not None else 0.0)

            # Enhanced logging
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "p_norm": f"{prompt_norm:.4f}",
                "p_var": f"{prompt_var:.4f}",
                "p_grad": f"{prompt_grad:.4f}",
            })

        # 9. Epoch summary
        avg_loss = total_loss / len(indices)
        logger.info(f"[Epoch {epoch + 1}] Avg Prompt Loss: {avg_loss:.4f}")

        # 10. Save checkpoint (optional)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(cfg["save_dir"], build_filename(cfg, epoch, stage="prompt", extension="_BEST.pth", timestamped=False))
            save_checkpoint(
                clip_model, None, optimizer, cfg, epoch,
                {"loss": best_loss}, best_path,
                is_best=True, scheduler=scheduler,
                train_loss=best_loss
            )
            logger.info(f" New BEST Prompt model saved with loss = {best_loss:.4f}")

        # 11. Learning rate step
        scheduler.step()

    # 12. Save final model checkpoint
    final_path = os.path.join(cfg["save_dir"],
                              build_filename(cfg, epoch, stage="prompt", extension="_FINAL.pth", timestamped=False))
    save_checkpoint(
        clip_model, None, optimizer, cfg, epoch,
        {"loss": avg_loss}, final_path,
        is_best=False, scheduler=scheduler,
        train_loss=avg_loss
    )
    logger.info(f" Final Prompt model saved to: {final_path}")



def train_clipreid_image_stage(clip_model, prompt_learner, optimizer, scheduler,
                               train_loader, val_loader, cfg, device, logger,
                               loss_fn, ce_loss, triplet_loss):

    unfreeze_clip_text_encoder(clip_model, logger.info)

    if cfg.get("freeze_prompt", True):
        freeze_prompt_learner(prompt_learner, logger.info)
        logger.info("Prompt Learner frozen.")

    feat_dim = clip_model.bottleneck.num_features
    center_criterion = CenterLoss(num_classes=cfg["num_classes"], feat_dim=feat_dim, device=device)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.get("center_lr", 0.5))

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


            feats_bn = clip_model.bottleneck(image_feats)
            #id_logits = clip_model.classifier(feats_bn)
            arc_logits = clip_model.arcface(feats_bn, labels_batch)

            feat_norm = feats_bn.norm(dim=1).mean().item()
            arc_conf = arc_logits.softmax(dim=1).max(dim=1)[0].mean().item()

            center_loss_val = center_criterion(feats_bn, labels_batch)
            id_loss = ce_loss(arc_logits, labels_batch)
            tri_loss = triplet_loss(image_feats, labels_batch)

            #total loss
            loss = id_loss + tri_loss + loss_i2t + loss_t2i + cfg["center_loss_weight"] * center_loss_val

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            loss.backward()
            # Scale gradient manually
            for param in center_criterion.parameters():
                param.grad.data *= (1.0 / cfg["center_loss_weight"])

            optimizer.step()
            total_loss += loss.item()
            logger.info(f"[Epoch {epoch + 1}] Total Image Loss: {total_loss:.4f}")
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "feat_n": f"{feat_norm:.2f}",
                "arc_c": f"{arc_conf:.2f}",
                "id": f"{id_loss.item():.2f}",
                "tri": f"{tri_loss.item():.2f}",
                "cen": f"{center_loss_val.item():.2f}"
            })
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"[Epoch {epoch + 1}] Learning Rate: {current_lr:.6f}")
        logger.info(f"[Epoch {epoch + 1}] Running validation...")

        metrics = validate(clip_model, prompt_learner, val_loader, device, logger.info, cfg)
        if metrics["rank1"] > best_rank1_image:
            best_rank1_image = metrics["rank1"]
            best_path = os.path.join(cfg["save_dir"], build_filename(cfg, epoch+1, stage="image", extension="_BEST.pth", timestamped=False))
            save_checkpoint(clip_model, None, optimizer, cfg, epoch+1, metrics, best_path, True, scheduler, total_loss)
            logger.info(f" New BEST Image model saved at Rank-1 = {metrics['rank1'] * 100:.2f}%")
            logger.info(f"[Epoch {epoch + 1}] Rank-1: {metrics['rank1']:.2%}, mAP: {metrics['mAP']:.2%}")

        scheduler.step()

    # === Save FINAL checkpoint after last epoch
    final_path = os.path.join(cfg["save_dir"],
                              build_filename(cfg, epoch+1, stage="image", extension="_FINAL.pth", timestamped=False))
    save_checkpoint(
        clip_model, None, optimizer, cfg, epoch+1,
        metrics, final_path,
        is_best=False, scheduler=scheduler,
        train_loss=total_loss
    )
    logger.info(f"Final Image model saved to: {final_path}")

