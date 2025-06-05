import os
import sys
import argparse
from os import utime

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


# Project structure
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataloaders import get_train_val_loaders
from utils.clip_patch import load_clip_with_patch
from utils.train_helpers import freeze_clip_text_encoder, build_promptsg_models
from utils.train_helpers import compose_prompt
from utils.save_load_models import save_checkpoint, save_promptsg_checkpoint
from utils.eval_utils import validate_promptsg
from utils.loss.supcon import SupConLoss
from utils.loss.triplet_loss import TripletLoss
from utils.naming import build_filename

from utils.logger import setup_logger


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === v11: patch for ViT positional embedding if using 224x128 input ===
def maybe_patch_clip_pos_embed(clip_model, input_hw):
    """
    Resizes CLIP positional embeddings for non-square inputs like 224x128.
    Supports both ViT (positional_embedding) and RN50 (attnpool.positional_embedding).
    Safe to call regardless of model type.
    """
    import torch.nn.functional as F

    h, w = input_hw
    # === ViT patching ===
    if hasattr(clip_model.visual, "positional_embedding"):
        pos_embed = clip_model.visual.positional_embedding  # [N, D]
        patch_size = (16, 16)

        grid_h = h // patch_size[0]
        grid_w = w // patch_size[1]
        new_len = grid_h * grid_w + 1

        if pos_embed.shape[0] != new_len:
            print(f"[v11-ViT] Interpolating ViT pos_embed: {pos_embed.shape[0]}  {new_len}")
            cls_token = pos_embed[:1, :]  # [1, D]
            spatial_tokens = pos_embed[1:, :]  # [N-1, D]

            # Dynamically infer original grid shape
            old_grid_len = spatial_tokens.shape[0]
            old_grid_h = old_grid_w = int(old_grid_len ** 0.5)
            assert old_grid_h * old_grid_w == old_grid_len, f"Cannot reshape {old_grid_len} into square grid"

            spatial = spatial_tokens.reshape(old_grid_h, old_grid_w, -1).permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
            resized = F.interpolate(spatial, size=(grid_h, grid_w), mode='bicubic', align_corners=False)
            new_spatial = resized.squeeze(0).permute(1, 2, 0).reshape(grid_h * grid_w, -1)

            new_pos_embed = torch.cat([cls_token, new_spatial], dim=0)
            clip_model.visual.positional_embedding = nn.Parameter(new_pos_embed)

    # === RN50 patching ===
    elif hasattr(clip_model.visual, "attnpool") and hasattr(clip_model.visual.attnpool, "positional_embedding"):
        pos_embed = clip_model.visual.attnpool.positional_embedding  # [50, D]
        feat_h, feat_w = h // 32, w // 32  # RN50 has stride-32
        new_len = feat_h * feat_w + 1

        if pos_embed.shape[0] != new_len:
            print(f"[v11-RN50] Interpolating RN50 attnpool pos_embed: {pos_embed.shape[0]}  {new_len}")
            cls_token = pos_embed[:1, :]  # [1, D]
            spatial_tokens = pos_embed[1:, :].T.unsqueeze(0)  # [1, D, N]
            resized = F.interpolate(spatial_tokens, size=new_len - 1, mode='linear', align_corners=False)
            resized = resized.squeeze(0).T  # [new_len-1, D]
            new_pos_embed = torch.cat([cls_token, resized], dim=0)  # [new_len, D]
            clip_model.visual.attnpool.positional_embedding = nn.Parameter(new_pos_embed)

def promptSG_integration(config):

    best_model_components = None  # Will store best model (clip, inversion, multi, classifier)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clip_model, _ = load_clip_with_patch(
        model_type=config["clip_model"],  # you already have "ViT-B/16"
        device=device,
        freeze_all=False  # Only image encoder remains trainable
    )

    clip_model.float()

    freeze_clip_text_encoder(clip_model)  # always freeze text encoder in PromptSG

    #  predicts the person/identity based on enhanced features.
    train_loader, val_loader, num_classes = get_train_val_loaders(config)
    # === v11: patch positional embeddings to support 224x128 (ViT & RN)
    sample_img = train_loader.dataset[0][0]  # (C, H, W)
    input_hw = (sample_img.shape[1], sample_img.shape[2])
    maybe_patch_clip_pos_embed(clip_model, input_hw)

    # === Build PromptSG modules
    #inversion_model, multimodal_module, classifier = build_promptsg_models(config, num_classes, device)
    # v4
    inversion_model, multimodal_module, reduction, bnneck, classifier = build_promptsg_models(config, num_classes, device)

    lr_clip = float(config['lr_clip_visual'])
    lr_modules = float(config['lr_modules'])
    weight_decay = float(config['weight_decay'])

    patience = config.get("early_stop_patience",10)
    patience_counter = 0

    optimizer = torch.optim.AdamW([
        {'params': inversion_model.parameters()},
        {'params': multimodal_module.parameters()},
        {'params': classifier.parameters()},
        {'params': clip_model.visual.parameters(), 'lr': lr_clip}
    ], lr=lr_modules, weight_decay=weight_decay)


    id_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    triplet_loss_fn = nn.TripletMarginLoss(margin=0.3)
    supcon_loss_fn = SupConLoss(temperature=config.get("supcon_temperature", 0.07))

    model_name = build_filename(config, config['epochs'], stage="stage3", extension=".pth", timestamped=False)
    #scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    log_path = os.path.join(config['output_dir'], model_name.replace('.pth', '.log'))
    logger = setup_logger(log_path)

    logger.info(f"Saving logs to: {log_path}")
    logger.info(f"Using {num_classes} classes | {len(train_loader)} training batches")


    #===== TRAINING ================

    logger.info("======= Stage 3: PromptSG Training =======")
    logger.info(f"Experiment Name : {config['experiment']}")
    logger.info(f"Model & Dataset : {config['model']}, {config['dataset']}, {config['aspect']}")
    logger.info(f"Freeze Text Enc.: {config['freeze_text_encoder']}")
    logger.info(f"Loss Weights    : ID={config['loss_id_weight']}, Tri={config['loss_tri_weight']}, SupCon={config['supcon_loss_weight']}")
    logger.info(f"LR={lr_modules} | Epochs={config['epochs']} | BatchSize={config['batch_size']}")

    best_acc1 = 0
    best_epoch = 0
    epoch_accs = []

    for epoch in range(1, config['epochs'] + 1):
        logger.info(f" Epoch {epoch}/{config['epochs']}")
        clip_model.visual.train()
        inversion_model.train()
        multimodal_module.train()
        classifier.train()

        total_loss = 0
        pbar = tqdm(train_loader, desc="Training")


        logger.info(f"[Epoch {epoch}] Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        for step, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            #1. Get image features from the CLIP image encoder.
            # Get image features from CLIP
            img_features = clip_model.encode_image(images).float()
            #2. Generate pseudo-token prompts using the inversion model.
            # Generate pseudo prompts using the inversion model:
            pseudo_tokens = inversion_model(img_features)

            # Compose full prompts: [prefix] + [pseudo] + [suffix]:
            template = config.get("prompt_template", "A detailed photo of a hand.")
            prefix, suffix = template.split("{aspect}")[0].strip(), template.split("{aspect}")[-1].strip()
            #3. Compose full prompts (prefix + pseudo + suffix).
            text_emb = compose_prompt(clip_model.encode_text, pseudo_tokens, templates=(prefix, suffix), device=device)

            #4. Fuse image and prompt embeddings using the multimodal module.
            # Fuse image & text using cross-attention:
            visual_emb = multimodal_module(text_emb, img_features.unsqueeze(1))  # [B, 3, 512]
            pooled = visual_emb.mean(dim=1)  # [B, 512]


            # >>> v2 begin
            pooled = F.normalize(pooled, dim=1)  # stabilise cosine losses
            pooled = F.dropout(pooled, p=0.1, training=clip_model.training)
            if step == 0 and epoch == 1:
                norm = pooled.norm(dim=1).mean().item()
                logger.info(f"[DEBUG] Pooled feature mean L2 norm: {norm:.4f}")

            # <<< v2 end
            #5. Normalize and pool features → pass to classifier.
            # Classify the identity:
            classifier_type = config.get("classifier", "linear").lower()
            #logger.info(f"[DEBUG] Classifier type: {classifier.__class__.__name__}")
            if classifier_type == "arcface":
                features = reduction(pooled)
                features_bn = bnneck(features)
                logits = classifier(features_bn, labels)
                conf = logits.softmax(dim=1).max(dim=1)[0].mean().item()
                logger.info(f"[DEBUG] ArcFace confidence: {conf:.4f}")
                '''
                    If ~0.9 = very confident

                    If < 0.5 = very confused  suggests:
                    
                    embeddings are untrained
                    
                    margin too strong
                    
                    loss too weak to shape the output
                '''

            else:
                logits = classifier(pooled)

            # 6. Compute all losses
            id_loss = id_loss_fn(logits, labels)
            triplet_loss_fn = TripletLoss(margin=0.3, mining='batch_hard')
            triplet_loss = triplet_loss_fn(pooled, labels)

            supcon_loss = supcon_loss_fn(pooled, text_emb.mean(dim=1), labels)

            supcon_weight = config['supcon_loss_weight'] * (epoch / config['epochs'])

            loss = (config['loss_tri_weight'] * triplet_loss +
                   supcon_weight * supcon_loss)

            loss = (config['loss_id_weight'] * id_loss +
                    config['loss_tri_weight'] * triplet_loss +
                    supcon_weight * supcon_loss)

            if torch.isnan(loss) or torch.isnan(id_loss) or torch.isnan(supcon_loss):
                logger.warning(f"NaN in loss detected at step {step} (epoch {epoch})")
                continue  # Skip this batch

            logger.info(
                f"[DEBUG] Loss Breakdown  ID: {id_loss.item():.4f}, SupCon: {supcon_loss.item():.4f}, Tri: {triplet_loss.item():.4f}")

            # 7. Backpropagate and update weights.
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(
                list(inversion_model.parameters()) +
                list(multimodal_module.parameters()) +
                list(classifier.parameters()) +
                list(clip_model.visual.parameters()),
                max_norm= 0.5  # reasonable threshold --- overfitting
            )

            total_loss += loss.item()
            pbar.set_postfix(ID=id_loss.item(), SupCon=supcon_loss.item(), Tri=triplet_loss.item(), Total=loss.item())

            if step == 0:
                logger.info(f"[DEBUG] Batch 0 - image shape: {images.shape}, labels: {labels.shape}")
                logger.info(f"[DEBUG] img_features: {img_features.shape}, prompt: {text_emb.shape}, pooled: {pooled.shape}")

        #scheduler.step()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"[Epoch {epoch}] Total Loss={avg_loss:.4f}")

        # === Validation: rank-K Acc ===
        logger.info(f"Validating epoch {epoch}...")


        model_components = (clip_model, inversion_model, multimodal_module, classifier)

        validation_metrics = validate_promptsg(
            model_components=(clip_model, inversion_model, multimodal_module, classifier),
            val_loader=val_loader,
            device=device,
            compose_prompt=compose_prompt,
            config=config  # optional, only needed for prompt_template
        )


        epoch_accs.append(validation_metrics["rank1_accuracy"])
        logger.info(f"[Epoch {epoch}] Avg Train Loss = {avg_loss:.4f} | ")

        logger.info("" + "=" * 20 + f" Epoch {epoch} Validation " + "=" * 20 + "")
        logger.info(f"Validation rank-1    : {validation_metrics['rank1_accuracy']:.2f}%")
        logger.info(f"Validation rank-5    : {validation_metrics['rank5_accuracy']:.2f}%")
        logger.info(f"Validation rank-10   : {validation_metrics['rank10_accuracy']:.2f}%")
        logger.info(f"Validation mAP      : {validation_metrics['mAP']:.2f}%")
        logger.info("=" * 60 + "")

        if validation_metrics["rank1_accuracy"] > best_acc1:
            # Save best model in memory (for final evaluation later)
            best_model_components = (
                clip_model, inversion_model, multimodal_module, classifier
            )
            best_acc1 = validation_metrics["rank1_accuracy"]
            best_epoch = epoch
            patience_counter = 0  # reset patience
            best_model_path = os.path.join(config['save_dir'], model_name.replace('.pth', '_BEST.pth'))
            logger.info(f"Saving best model at epoch {epoch} (Acc@1={best_acc1:.2f}%)  {best_model_path}")
            save_promptsg_checkpoint(
                model_components=(clip_model, inversion_model, multimodal_module, classifier),
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                val_metrics=validation_metrics,
                path=best_model_path,
            )
        else:
            patience_counter += 1
            logger.info(f"No improvement in Acc@1. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info(f"Early srankping at epoch {epoch} due to no improvement in rank-1 accuracy.")
                break

    """
    final_model_path= os.path.join(config['save_dir'], model_name.replace('.pth', '_FINAL.pth'))
    save_promptsg_checkpoint(
        model_components=(clip_model, inversion_model, multimodal_module, classifier),
        optimizer=optimizer,
        config=config,
        epoch=epoch,
        val_metrics=validation_metrics,
        path=final_model_path,  # same naming you already build
    )

    # === FINAL SUMMARY ===
    logger.info("====== FINAL SUMMARY ======")
    logger.info(f"Total epochs       : {config['epochs']}")
    logger.info(f"Avg Loss           : {sum(epoch_accs)/len(epoch_accs):.4f}")
    logger.info(f"Best Acc@1 Epoch   : {best_epoch}")
    logger.info(f"Best Acc@1         : {best_acc1:.2f}%")
    logger.info(f"Best mAP            : {validation_metrics['mAP']:.2f}%")
    logger.info("============================")

    """

    #======= EVALUATION ===============

    if best_model_components is not None:
        # === Create a separate log file for final evaluation ===
        import datetime
        os.makedirs("eval_logs", exist_ok=True)

        eval_log_name = os.path.basename(model_name).replace("_BEST", "").replace(".pth", "").replace(".pt", "")
        eval_log_name = "eval_" + eval_log_name + ".log"
        eval_log_path = os.path.join("eval_logs", eval_log_name)

        # Step 1: Open the evaluation log file in write mode with UTF-8 encoding
        with open(eval_log_path, "w", encoding="utf-8") as eval_log:
            # Step 2: Define a helper function to print and write evaluation logs
            def eval_log_info(text):
                print(text)
                eval_log.write(text + "\n")

            # Step 3: Start logging
            eval_log_info("Running final ReID-style evaluation using best model...\n")

            # Step 4: Set all model components to evaluation mode
            clip_model, inversion_model, multimodal_module, classifier = best_model_components
            clip_model.eval();
            inversion_model.eval();
            multimodal_module.eval();
            classifier.eval()

            from utils.dataloaders import get_dataloader
            from engine.baseline_inference import compute_similarity_matrix
            from engine.evaluator import evaluate_rank
            from engine.promptsg_inference import extract_features_promptsg

            base_path = os.path.join("datasets",
                                     "11khands" if config["dataset"] == "11k" else "HD/Original Images",
                                     f"train_val_test_split_{config['aspect']}")
            batch_size = config["batch_size"]
            num_splits = config.get("num_splits", 10)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            all_rank1, all_rank5, all_rank10, all_map = [], [], [], []

            for i in range(num_splits):
                query_path = os.path.join(base_path, f"query{i}")
                gallery_path = os.path.join(base_path, f"gallery{i}")

                if not os.path.exists(query_path) or not os.path.exists(gallery_path):
                    eval_log_info(f"Skipping missing split {i}: {query_path} / {gallery_path}")
                    continue

                eval_log_info(f" Evaluating Split {i + 1}/{num_splits}")

                query_loader = get_dataloader(query_path, batch_size=batch_size, shuffle=False, train=False)
                gallery_loader = get_dataloader(gallery_path, batch_size=batch_size, shuffle=False, train=False)

                q_feats, q_labels = extract_features_promptsg(
                    clip_model, inversion_model, multimodal_module, classifier,
                    query_loader, device, compose_prompt
                )
                g_feats, g_labels = extract_features_promptsg(
                    clip_model, inversion_model, multimodal_module, classifier,
                    gallery_loader, device, compose_prompt
                )

                sim_matrix = compute_similarity_matrix(q_feats, g_feats)
                metrics = evaluate_rank(sim_matrix, q_labels, g_labels, topk=[1, 5, 10])
                metrics = {k: v * 100 for k, v in metrics.items()}

                all_rank1.append(metrics.get("Rank-1", 0))
                all_rank5.append(metrics.get("Rank-5", 0))
                all_rank10.append(metrics.get("Rank-10", 0))
                all_map.append(metrics.get("mAP", 0))

                eval_log_info(f"[Split {i + 1}] Rank-1: {metrics['Rank-1']:.2f}%, Rank-5: {metrics['Rank-5']:.2f}%, "
                              f"Rank-10: {metrics['Rank-10']:.2f}%, mAP: {metrics['mAP']:.2f}%")

            if all_rank1:
                avg_rank1 = sum(all_rank1) / len(all_rank1)
                avg_rank5 = sum(all_rank5) / len(all_rank5)
                avg_rank10 = sum(all_rank10) / len(all_rank10)
                avg_map = sum(all_map) / len(all_map)

                eval_log_info("\n======= FINAL EVAL SUMMARY (Best Model) =======")
                eval_log_info(f"Avg Rank-1  : {avg_rank1:.2f}%")
                eval_log_info(f"Avg Rank-5  : {avg_rank5:.2f}%")
                eval_log_info(f"Avg Rank-10 : {avg_rank10:.2f}%")
                eval_log_info(f"Mean AP     : {avg_map:.2f}%")
                eval_log_info("===============================================")
            else:
                eval_log_info(" No valid splits evaluated.")
    else:
        logger.warning("Skipping final evaluation  best_model_components was not set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    promptSG_integration(config)