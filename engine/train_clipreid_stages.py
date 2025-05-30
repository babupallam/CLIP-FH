import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.save_load_models import save_checkpoint
from utils.eval_utils import validate
from utils.naming import build_filename
from utils.loss.center_loss import CenterLoss
from utils.train_helpers import freeze_prompt_learner
from copy import deepcopy
from utils.feature_cache import cache_image_features
from torch.nn.functional import normalize


# ======================================================================
def train_clipreid_prompt_stage(clip_model, prompt_learner, optimizer, scheduler,
                                train_loader, cfg, device, logger, best_model_state):
    """
    Stage‑1 prompt learning using SupCon (CLIP‑ReID style).
    Uses your logger + tqdm exactly like before.
    """


    clip_model.eval()



    # 1) cache frozen image features once
    model_name = cfg["model"].lower()  # example: 'vitb16' or 'rn50'
    image_feats, labels = cache_image_features(clip_model, train_loader, device, model_name=model_name)
    logger.info(f"[Prompt] cached {len(labels)} frozen image feats")
    loss_fn = cfg["loss_fn"]                      # SupCon loss object built in main script

    best_loss = float("inf")

    for epoch in range(cfg["epochs_prompt"]):
        perm          = torch.randperm(len(labels))
        running_loss  = 0.0
        prompt_learner.train()

        from tqdm import tqdm
        pbar = tqdm(range(0, len(labels), cfg["batch_size"]),
                    desc=f"Prompt Epoch {epoch+1}/{cfg['epochs_prompt']}")

        for start in pbar:
            idx        = perm[start:start + cfg["batch_size"]]
            img_feats  = image_feats[idx].to(device)     # (B, dim)
            lab        = labels[idx].to(device)

            # ---- forward text branch -------------------------------------------------
            txt_embeds = prompt_learner(lab)  # (B, L, dim)

            # Add positional embeddings (detach so gradients flow only through prompts)
            pos_embed = clip_model.positional_embedding.unsqueeze(0).to(txt_embeds.device).detach()
            pos_embed = pos_embed.expand_as(txt_embeds)
            txt_embeds = txt_embeds + pos_embed

            # Transformer
            x = txt_embeds.permute(1, 0, 2)
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)

            # Normalize and pool
            txt_feats = clip_model.ln_final(x)
            txt_feats = txt_feats.mean(dim=1)
            txt_feats = torch.nn.functional.normalize(txt_feats, dim=-1)

            # Apply projection to BOTH
            txt_feats = prompt_learner.proj(txt_feats)

            if img_feats.shape != txt_feats.shape:
                logger.warning(
                    f"[WARNING] Feature dimension mismatch → img_feats: {img_feats.shape}, txt_feats: {txt_feats.shape}")

            #logger.info(f"[DEBUG] img_feats shape: {img_feats.shape}")
            #logger.info(f"[DEBUG] txt_feats shape: {txt_feats.shape}")

            # ---- SupCon losses -------------------------------------------------------
            # Suitable for prompt learning
            # Used in: CLIP-ReID, PromptSG, and other prompt-tuning papers.
            # why? Designed to bring same-class embeddings closer while pushing away different classes — perfect for ReID.
            # why? Works well with CLIP because it aligns learned text prompts with frozen (or fine-tuned) image embeddings.

            loss_i2t = loss_fn(
                features=img_feats,
                text_features=txt_feats,
                targets=lab,
                mode="contrastive"
            )

            loss_t2i = loss_fn(
                features=txt_feats,
                text_features=img_feats,
                targets=lab,
                mode="contrastive"
            )

            loss = loss_i2t + loss_t2i

            #print(f"loss_i2t: {type(loss_i2t)}, loss_t2i: {type(loss_t2i)}, loss: {type(loss)}")

            # ---- optimisation --------------------------------------------------------
            optimizer.zero_grad()
            assert isinstance(loss, torch.Tensor), f"Loss must be Tensor, got {type(loss)}"

            loss.backward()


            if prompt_learner.cls_ctx.grad is None:
                logger.warning("No gradient flowed into cls_ctx!")
            else:
                grad_norm = prompt_learner.cls_ctx.grad.norm().item()
                #logger.info(f"cls_ctx grad norm: {grad_norm:.6f}")

            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * lab.size(0)

            # ---- live progress bar metrics ------------------------------------------
            p_norm = prompt_learner.cls_ctx.norm(dim=2).mean().item()
            p_var  = prompt_learner.cls_ctx.var(dim=2).mean().item()
            p_grad = (prompt_learner.cls_ctx.grad.norm().item()
                      if prompt_learner.cls_ctx.grad is not None else 0.0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "p_norm": f"{p_norm:.4f}",
                "p_var": f"{p_var:.4f}",
                "p_grad": f"{p_grad:.4f}"
            })

        avg_loss = running_loss / len(labels)
        logger.info(f"[Epoch {epoch+1}] Avg Prompt Loss: {avg_loss:.4f}")

        # ---- remember best prompt weights in RAM ------------------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state["prompt_learner"] = prompt_learner.state_dict().copy()
            logger.info(f"new best prompt – {best_loss:.4f}")


def train_clipreid_image_stage(clip_model, prompt_learner, optimizer, scheduler,
                               train_loader, val_loader, cfg, device, logger,
                               loss_fn, ce_loss, triplet_loss, best_model_state):


    best_loss = float("inf")
    patience = cfg.get("early_stop_patience", 3)
    patience_counter = 0

    #for i, group in enumerate(optimizer.param_groups):
    #    for param in group["params"]:
    #        if hasattr(param, "name"):
    #            print(f"Param group {i}: {param.name}")

    # Count visual params manually
    image_encoder_params = list(clip_model.visual.parameters())
    trainable_visual = sum(p.numel() for p in image_encoder_params if p.requires_grad)
    print(f"[DEBUG] Trainable image encoder params: {trainable_visual}")

    #if cfg.get("freeze_prompt", True):
    #    freeze_prompt_learner(prompt_learner, logger.info)
    #    logger.info("Prompt Learner frozen.")

    feat_dim = clip_model.bottleneck.num_features
    center_criterion = CenterLoss(num_classes=cfg["num_classes"], feat_dim=feat_dim, device=device)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.get("center_lr", 0.5))

    best_rank1_image = 0.0
    for epoch in range(cfg["epochs_image"]):
        prompt_learner.train()
        #for name, param in clip_model.visual.named_parameters():
        #    if param.requires_grad:
        #        logger.info(f"[Trainable] {name}")


        clip_model.train()
        #if hasattr(clip_model, 'arcface'):
        #    arcface_weights = clip_model.arcface.weight
        #    logger.info(f"[ArcFace] Weight norm: {arcface_weights.norm():.4f}")

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
            image_feats = normalize(image_feats, dim=-1)

            # Match projection before computing SupCon
            text_feats = prompt_learner.proj(text_feats)

            # Before computing losses:
            text_feats = normalize(text_feats, dim=-1)

            if image_feats.shape[1] != text_feats.shape[1]:
                logger.warning(
                    f"[DIM MISMATCH] image_feats: {image_feats.shape}, text_feats: {text_feats.shape}")

            feats_bn = clip_model.bottleneck(image_feats)

            # Conditional Loss Computation
            if cfg.get("loss_use_id", True):
                id_logits = clip_model.classifier(feats_bn)
                id_loss = ce_loss(id_logits, labels_batch)
            else:
                id_loss = torch.tensor(0.0, device=device)

            if cfg.get("loss_use_arcface", True):
                arc_logits = clip_model.arcface(feats_bn, labels_batch)
                # Optionally compute accuracy for monitoring
                pred = id_logits.argmax(dim=1)
                acc = (pred == labels_batch).float().mean().item()

                feat_norm = feats_bn.norm(dim=1).mean().item()
                arc_conf = arc_logits.softmax(dim=1).max(dim=1)[0].mean().item()
                # Low confidence → uncertain prediction → class separation might be poor.
                # High confidence → sharp logits → better separation between identities.

                logger.info(f"[ArcFace] Confidence: {arc_conf:.4f}")
            else:
                arc_logits = torch.tensor(0.0, device=device)
                arc_conf = 0.0

            if cfg.get("loss_use_triplet", False):
                tri_loss = triplet_loss(image_feats, labels_batch)
            else:
                tri_loss = torch.tensor(0.0, device=device)

            if cfg.get("loss_use_center", False):
                center_loss_val = center_criterion(feats_bn, labels_batch)
            else:
                center_loss_val = torch.tensor(0.0, device=device)

            if cfg.get("loss_use_supcon", True):
                loss_i2t = loss_fn(image_feats, text_feats, labels_batch, mode="contrastive")
                loss_t2i = loss_fn(text_feats, image_feats, labels_batch, mode="contrastive")
            else:
                loss_i2t = loss_t2i = torch.tensor(0.0, device=device)

            # total loss computation

            loss = 0

            if cfg.get("loss_use_id", True):
                loss += id_loss

            if cfg.get("loss_use_triplet", False):
                loss += tri_loss

            if cfg.get("loss_use_supcon", True):
                loss += loss_i2t + loss_t2i

            if cfg.get("loss_use_center", False):
                loss += cfg["center_loss_weight"] * center_loss_val




            optimizer.zero_grad()
            optimizer_center.zero_grad()
            loss.backward()


            if cfg.get("loss_use_center", False):
                # Scale gradient manually
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / cfg["center_loss_weight"])

            #for name, param in clip_model.visual.named_parameters():
            #    if param.requires_grad and param.grad is not None:
            #        logger.info(f"[Grad OK] {name}: {param.grad.norm():.4f}")
            #        break


            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            #logger.info(f"[Epoch {epoch + 1}] Total Image Loss: {total_loss:.4f}")
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
        logger.info(f"[Epoch {epoch + 1}] Loss Breakdown: "
                    f"ID = {id_loss.item():.4f}, "
                    f"Triplet = {tri_loss.item():.4f}, "
                    f"Center = {center_loss_val.item():.4f}, "
                    f"i2t = {loss_i2t.item():.4f}, "
                    f"t2i = {loss_t2i.item():.4f}")

        metrics = validate(clip_model, prompt_learner, val_loader, device, logger.info, cfg)
        if metrics["rank1"] > best_rank1_image:
            best_rank1_image = metrics["rank1"]
            # Store best weights in memory
            best_model_state["clip_model"] = deepcopy(clip_model.state_dict())
            best_model_state["prompt_learner"] = deepcopy(prompt_learner.state_dict())
            patience_counter = 0
            best_path = os.path.join(cfg["save_dir"], build_filename(cfg, cfg["epochs_image"], stage="image", extension="_BEST.pth", timestamped=False))
            save_checkpoint(
                model=clip_model,
                classifier=None,
                optimizer=optimizer,
                config=cfg,
                epoch=cfg["epochs_image"],
                val_metrics=metrics,
                path=best_path,
                is_best=True,
                scheduler=scheduler,
                train_loss=total_loss,
                prompt_learner=prompt_learner
            )

            logger.info(f" New BEST Image model saved")
        else:
            patience_counter += 1
            logger.info(f"No improvement in loss. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered (prompt stage).")
                break




from engine.baseline_inference import extract_features, compute_similarity_matrix
from engine.evaluator import evaluate_rank
from utils.dataloaders import get_dataloader
import os
from utils.logger import setup_logger  # Ensure this exists

def evaluate_clipreid_after_training(clip_model, prompt_learner, cfg, logger, device):
    base_path = os.path.join("datasets", f"{'11khands' if cfg['dataset'] == '11k' else 'HD/Original Images'}", "train_val_test_split")
    if cfg["dataset"] == "11k":
        base_path = os.path.join(base_path + f"_{cfg['aspect']}")
    # === Create Evaluation Log File ===
    # Save evaluation log into a separate folder `eval_logs`
    eval_log_dir = os.path.join(cfg["log_dir"])
    os.makedirs(eval_log_dir, exist_ok=True)

    model_filename = build_filename(cfg, cfg["epochs_image"], stage="image", extension="_BEST.pth", timestamped=False)
    model_basename = os.path.splitext(os.path.basename(model_filename))[0]
    eval_log_name = f"eval_{model_basename}.log"
    eval_log_path = os.path.join(eval_log_dir, eval_log_name)

    eval_logger = setup_logger(eval_log_path, name="eval_logger")
    eval_logger.info(f"[Eval] Using base dataset path: {base_path}")

    all_rank1, all_rank5, all_rank10, all_map = [], [], [], []
    num_splits = cfg.get("num_splits", 10)

    for i in range(num_splits):
        query_path = os.path.join(base_path, f"query{i}")
        gallery_path = os.path.join(base_path, f"gallery{i}")

        if not os.path.exists(query_path) or not os.path.exists(gallery_path):
            eval_logger.warning(f"Skipping split {i}: missing {query_path} or {gallery_path}")
            continue

        eval_logger.info(f"[Eval] Split {i + 1}/{num_splits}")
        query_loader = get_dataloader(query_path, batch_size=cfg["batch_size"], shuffle=False, train=False)
        gallery_loader = get_dataloader(gallery_path, batch_size=cfg["batch_size"], shuffle=False, train=False)

        # Extract features (PromptLearner active)
        q_feats, q_labels = extract_features(clip_model, query_loader, device, prompt_learner=prompt_learner)
        g_feats, g_labels = extract_features(clip_model, gallery_loader, device, prompt_learner=prompt_learner)

        sim_matrix = compute_similarity_matrix(q_feats, g_feats)
        metrics = evaluate_rank(sim_matrix, q_labels, g_labels, topk=[1, 5, 10])
        metrics = {k: v * 100 for k, v in metrics.items()}

        eval_logger.info(f"  Rank-1 : {metrics['Rank-1']:.2f}%")
        eval_logger.info(f"  Rank-5 : {metrics['Rank-5']:.2f}%")
        eval_logger.info(f"  Rank-10: {metrics['Rank-10']:.2f}%")
        eval_logger.info(f"  mAP    : {metrics['mAP']:.2f}%")

        all_rank1.append(metrics["Rank-1"])
        all_rank5.append(metrics["Rank-5"])
        all_rank10.append(metrics["Rank-10"])
        all_map.append(metrics["mAP"])

    if all_rank1:
        eval_logger.info("===== Final ReID Metrics Across All Splits =====")
        eval_logger.info(f"Rank-1 Accuracy : {sum(all_rank1) / len(all_rank1):.2f}%")
        eval_logger.info(f"Rank-5 Accuracy : {sum(all_rank5) / len(all_rank5):.2f}%")
        eval_logger.info(f"Rank-10 Accuracy: {sum(all_rank10) / len(all_rank10):.2f}%")
        eval_logger.info(f"Mean AP         : {sum(all_map) / len(all_map):.2f}%")
    else:
        eval_logger.warning("No valid splits evaluated. Check dataset folders.")
