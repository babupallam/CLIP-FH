import os
import sys
import argparse
from os import utime

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm


# Project structure
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataloaders import get_train_val_loaders
from utils.clip_patch import load_clip_with_patch
from utils.train_helpers import freeze_clip_text_encoder, build_promptsg_models
from utils.train_helpers import compose_prompt
from utils.save_load_models import save_checkpoint
from utils.eval_utils import validate_promptsg
from utils.loss.supcon import SymmetricSupConLoss
from utils.naming import build_filename

from utils.logger import setup_logger


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clip_model, _ = load_clip_with_patch(
        model_type=config["clip_model"],  # you already have "ViT-B/16"
        device=device,
        freeze_all=False  # Only image encoder remains trainable
    )
    freeze_clip_text_encoder(clip_model)  # always freeze text encoder in PromptSG

    #  predicts the person/identity based on enhanced features.
    train_loader, val_loader, num_classes = get_train_val_loaders(config)
    # === Build PromptSG modules
    inversion_model, multimodal_module, classifier = build_promptsg_models(config, num_classes, device)


    lr_clip = float(config['lr_clip_visual'])
    lr_modules = float(config['lr_modules'])
    weight_decay = float(config['weight_decay'])

    optimizer = torch.optim.Adam([
        {'params': inversion_model.parameters()},
        {'params': multimodal_module.parameters()},
        {'params': clip_model.visual.parameters(), 'lr': lr_clip}
    ], lr=lr_modules, weight_decay=weight_decay)

    id_loss_fn = nn.CrossEntropyLoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.3)
    supcon_loss_fn = SymmetricSupConLoss()

    model_name = build_filename(config, config['epochs'], stage="stage3", extension=".pth", timestamped=False)

    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    log_path = os.path.join(config['output_dir'], model_name.replace('.pth', '.log'))
    logger = setup_logger(log_path)

    logger.info(f"Saving logs to: {log_path}")
    logger.info(f"Using {num_classes} classes | {len(train_loader)} training batches")

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

        for step, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            #with torch.no_grad(): not freexing image encoder
            img_features = clip_model.encode_image(images).float()

            pseudo_tokens = inversion_model(img_features)
            text_emb = compose_prompt(clip_model.encode_text, pseudo_tokens, device=device)
            visual_emb = multimodal_module(text_emb, img_features.unsqueeze(1))  # [B, 3, 512]
            pooled = visual_emb.mean(dim=1)  # [B, 512]

            logits = classifier(pooled)
            id_loss = id_loss_fn(logits, labels)
            triplet_loss = triplet_loss_fn(pooled, pooled, pooled)
            supcon_loss = supcon_loss_fn(pooled, text_emb.mean(dim=1), labels)

            loss = (config['loss_id_weight'] * id_loss +
                    config['loss_tri_weight'] * triplet_loss +
                    config['supcon_loss_weight'] * supcon_loss)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(ID=id_loss.item(), SupCon=supcon_loss.item(), Tri=triplet_loss.item(), Total=loss.item())

            if step == 0:
                logger.info(f"[DEBUG] Batch 0 - image shape: {images.shape}, labels: {labels.shape}")
                logger.info(f"[DEBUG] img_features: {img_features.shape}, prompt: {text_emb.shape}, pooled: {pooled.shape}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"[Epoch {epoch}] Total Loss={avg_loss:.4f}")

        # === Validation: Top-K Acc ===
        logger.info(f"Validating epoch {epoch}...")


        model_components = (clip_model, inversion_model, multimodal_module, classifier)

        validation_metrics = validate_promptsg(
            model_components=model_components,
            val_loader=val_loader,
            device=device,
            compose_prompt=compose_prompt
        )
        epoch_accs.append(validation_metrics["top1_accuracy"])
        logger.info(f"[Epoch {epoch}] Avg Train Loss = {avg_loss:.4f} | "
                    f"Val Acc@1 = {validation_metrics['top1_accuracy']:.2f}%, "
                    f"Acc@5 = {validation_metrics['top5_accuracy']:.2f}%, "
                    f"Acc@10 = {validation_metrics['top10_accuracy']:.2f}%")

        logger.info("" + "=" * 20 + f" Epoch {epoch} Validation " + "=" * 20 + "")
        logger.info(f"Avg Validation Loss : {validation_metrics['avg_val_loss']:.4f}")
        logger.info(f"Validation Top-1    : {validation_metrics['top1_accuracy']:.2f}%")
        logger.info(f"Validation Top-5    : {validation_metrics['top5_accuracy']:.2f}%")
        logger.info(f"Validation Top-10   : {validation_metrics['top10_accuracy']:.2f}%")
        logger.info(f"Validation mAP      : {validation_metrics['mAP']:.2f}%")
        logger.info("=" * 60 + "")

        if validation_metrics["top1_accuracy"] > best_acc1:
            best_acc1 = validation_metrics["top1_accuracy"]
            best_epoch = epoch

            best_model_path = os.path.join(config['save_dir'], model_name.replace('.pth', '_BEST.pth'))
            logger.info(f"Saving best model at epoch {epoch} (Acc@1={best_acc1:.2f}%)  {best_model_path}")
            save_checkpoint(
                model=clip_model,
                classifier=classifier,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                val_metrics=validation_metrics,
                path=best_model_path,
                is_best=True
            )

    save_checkpoint(
        model=clip_model,
        classifier=classifier,
        optimizer=optimizer,
        config=config,
        epoch=epoch,
        val_metrics=validation_metrics,
        path=os.path.join(config['save_dir'], model_name.replace('.pth', '_FINAL.pth')),
        is_best=False
    )

    # === FINAL SUMMARY ===
    logger.info("====== FINAL SUMMARY ======")
    logger.info(f"Total epochs       : {config['epochs']}")
    logger.info(f"Avg Loss           : {sum(epoch_accs)/len(epoch_accs):.4f}")
    logger.info(f"Best Acc@1 Epoch   : {best_epoch}")
    logger.info(f"Best Acc@1         : {best_acc1:.2f}%")
    logger.info(f"Best mAP            : {validation_metrics['mAP']:.2f}%")
    logger.info("============================")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train(config)
