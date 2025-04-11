import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from datetime import datetime
from tqdm import tqdm
from clip import clip

# Project structure
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from loss.contrastive_loss import SymmetricSupConLoss
from models.prompt_learner import TextualInversionMLP
from models.clip_patch import MultiModalInteraction
from datasets.build_dataloader import get_train_val_loaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compose_prompt(text_encoder, pseudo_token_embedding, templates=("A detailed photo of a", "hand.")):
    batch_size = pseudo_token_embedding.shape[0]
    prefix_tokens = clip.tokenize([templates[0]] * batch_size).to(device)
    suffix_tokens = clip.tokenize([templates[1]] * batch_size).to(device)
    with torch.no_grad():
        prefix_emb = text_encoder(prefix_tokens).float()
        suffix_emb = text_encoder(suffix_tokens).float()
    composed = torch.stack([prefix_emb, pseudo_token_embedding, suffix_emb], dim=1)  # [B, 3, D]
    return composed

def generate_model_name(cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (f"{cfg['stage']}_{cfg['variant']}_{cfg['model']}_{cfg['dataset']}_{cfg['aspect']}_"
            f"e{cfg['epochs']}_lr{cfg['lr_modules']}_bs{cfg['batch_size']}_"
            f"freezeText{cfg['freeze_text_encoder']}.pth")

def freeze_clip_text_encoder(clip_model):
    for param in clip_model.transformer.parameters():
        param.requires_grad = False
    for param in clip_model.token_embedding.parameters():
        param.requires_grad = False
    clip_model.positional_embedding.requires_grad = False
    clip_model.ln_final.requires_grad = False
    clip_model.text_projection.requires_grad = False


def train(config):

    clip_model, preprocess = clip.load(config['clip_model'], device=device)
    clip_model.eval()


    if config.get("freeze_text_encoder", True):
        freeze_clip_text_encoder(clip_model)

    # turns image features into pseudo-tokens.
    inversion_model = TextualInversionMLP(config['pseudo_token_dim'], config['pseudo_token_dim']).to(device)
    # connects text & vision using attention.
    multimodal_module = MultiModalInteraction(dim=config['pseudo_token_dim'],
                                              depth=config['transformer_layers']).to(device)
    #  predicts the person/identity based on enhanced features.
    classifier = nn.Linear(config['pseudo_token_dim'], get_train_val_loaders(config)[2]).to(device)

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

    train_loader, val_loader, num_classes = get_train_val_loaders(config)
    model_name = generate_model_name(config)

    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    log_path = os.path.join(config['output_dir'], model_name.replace('.pth', '.log'))

    print(f"\nSaving logs to: {log_path}")
    print(f"Using {num_classes} classes | {len(train_loader)} training batches")

    # === HEADER ===
    with open(log_path, 'w') as log_f:
        log_f.write("======= Stage 3: PromptSG Training =======\n")
        log_f.write(f"Experiment Name : {config['experiment']}\n")
        log_f.write(f"Model & Dataset : {config['model']}, {config['dataset']}, {config['aspect']}\n")
        log_f.write(f"Freeze Text Enc.: {config['freeze_text_encoder']}\n")
        log_f.write(f"Loss Weights    : ID={config['loss_id_weight']}, Tri={config['loss_tri_weight']}, SupCon={config['supcon_loss_weight']}\n")
        log_f.write(f"LR={lr_modules} | Epochs={config['epochs']} | BatchSize={config['batch_size']}\n\n")

        best_acc1 = 0
        best_epoch = 0
        epoch_accs = []

        for epoch in range(1, config['epochs'] + 1):
            print(f"\n Epoch {epoch}/{config['epochs']}")
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
                text_emb = compose_prompt(clip_model.encode_text, pseudo_tokens)
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
                    log_f.write(f"[DEBUG] Batch 0 - image shape: {images.shape}, labels: {labels.shape}\n")
                    log_f.write(f"[DEBUG] img_features: {img_features.shape}, prompt: {text_emb.shape}, pooled: {pooled.shape}\n")

            avg_loss = total_loss / len(train_loader)
            log_f.write(f"[Epoch {epoch}] Total Loss={avg_loss:.4f}\n")

            # === Validation: Top-K Acc ===
            print(f"Validating epoch {epoch}...")
            classifier.eval()
            correct_top1 = correct_top5 = correct_top10 = total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    img_features = clip_model.encode_image(images).float()
                    pseudo_tokens = inversion_model(img_features)
                    text_emb = compose_prompt(clip_model.encode_text, pseudo_tokens)
                    visual_emb = multimodal_module(text_emb, img_features.unsqueeze(1))
                    pooled = visual_emb.mean(dim=1)
                    logits = classifier(pooled)
                    _, pred_topk = logits.topk(10, dim=1)
                    total += labels.size(0)
                    correct_top1 += (pred_topk[:, :1] == labels.unsqueeze(1)).sum().item()
                    correct_top5 += (pred_topk[:, :5] == labels.unsqueeze(1)).sum().item()
                    correct_top10 += (pred_topk[:, :10] == labels.unsqueeze(1)).sum().item()

            acc1 = correct_top1 / total * 100
            acc5 = correct_top5 / total * 100
            acc10 = correct_top10 / total * 100
            epoch_accs.append(acc1)

            log_f.write(f"[Epoch {epoch}] Acc@1={acc1:.2f}% | Acc@5={acc5:.2f}% | Acc@10={acc10:.2f}%\n")

            if acc1 > best_acc1:
                best_acc1 = acc1
                best_epoch = epoch

                best_model_path = os.path.join(config['save_dir'], model_name.replace('.pth', '_BEST.pth'))
                print(f"Saving best model at epoch {epoch} (Acc@1={acc1:.2f}%) → {best_model_path}")
                torch.save({
                    'epoch': epoch,
                    'inversion_model_state_dict': inversion_model.state_dict(),
                    'multimodal_module_state_dict': multimodal_module.state_dict(),
                    'clip_visual_state_dict': clip_model.visual.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': config,
                    'top1_accuracy': acc1
                }, best_model_path)

            if epoch % config['save_frequency'] == 0 or epoch == config['epochs']:
                print(" Saving model...")
                torch.save({
                    'epoch': epoch,
                    'inversion_model_state_dict': inversion_model.state_dict(),
                    'multimodal_module_state_dict': multimodal_module.state_dict(),
                    'clip_visual_state_dict': clip_model.visual.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': config
                }, os.path.join(config['save_dir'], model_name.replace('.pth', '_FINAL.pth')))


            model_components = (clip_model, inversion_model, multimodal_module, classifier)

            validation_metrics = validate(model_components, val_loader, device)

            # Writing detailed validation logs
            log_f.write("\n" + "=" * 20 + f" Epoch {epoch} Validation " + "=" * 20 + "\n")
            log_f.write(f"Avg Validation Loss : {validation_metrics['avg_val_loss']:.4f}\n")
            log_f.write(f"Validation Top-1    : {validation_metrics['top1_accuracy']:.2f}%\n")
            log_f.write(f"Validation Top-5    : {validation_metrics['top5_accuracy']:.2f}%\n")
            log_f.write(f"Validation Top-10   : {validation_metrics['top10_accuracy']:.2f}%\n")
            log_f.write("=" * 60 + "\n\n")

        # === FINAL SUMMARY ===
        log_f.write("\n====== FINAL SUMMARY ======\n")
        log_f.write(f"Total epochs       : {config['epochs']}\n")
        log_f.write(f"Avg Loss           : {sum(epoch_accs)/len(epoch_accs):.4f}\n")
        log_f.write(f"Best Acc@1 Epoch   : {best_epoch}\n")
        log_f.write(f"Best Acc@1         : {best_acc1:.2f}%\n")
        log_f.write("============================\n")



def validate(model_components, val_loader, device):
    """
    Runs validation on the provided model components.
    Returns average validation loss and Top-1/5/10 accuracy.
    """
    clip_model, inversion_model, multimodal_module, classifier = model_components

    # Set models to eval mode
    clip_model.eval()
    inversion_model.eval()
    multimodal_module.eval()
    classifier.eval()

    total_samples = 0
    correct_top1 = correct_top5 = correct_top10 = 0
    total_val_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    print("\nStarting Validation Pass...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validation")):
            images, labels = images.to(device), labels.to(device)

            # Extract CLIP image features
            img_features = clip_model.encode_image(images).float()

            # Generate pseudo-tokens for identity-specific guidance
            pseudo_tokens = inversion_model(img_features)

            # Compose the full prompt from prefix, pseudo-token, and suffix
            text_embeddings = compose_prompt(clip_model.encode_text, pseudo_tokens)

            # Cross-attention: Text queries visual patches
            visual_embeddings = multimodal_module(text_embeddings, img_features.unsqueeze(1))

            # Average pooling of output tokens for classification
            pooled_emb = visual_embeddings.mean(dim=1)

            # Predict class logits
            logits = classifier(pooled_emb)

            # Compute classification loss
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            # Compute Top-K accuracy
            _, pred_topk = logits.topk(10, dim=1)
            match = pred_topk.eq(labels.unsqueeze(1).expand_as(pred_topk))
            total_samples += labels.size(0)
            correct_top1 += match[:, :1].sum().item()
            correct_top5 += match[:, :5].sum().item()
            correct_top10 += match[:, :10].sum().item()

            if batch_idx == 0:
                print(f"[DEBUG] First Val Batch → Image Shape: {images.shape} | Logits: {logits.shape}")

    # Normalize metrics
    avg_val_loss = total_val_loss / len(val_loader)
    acc1 = 100 * correct_top1 / total_samples
    acc5 = 100 * correct_top5 / total_samples
    acc10 = 100 * correct_top10 / total_samples

    print(
        f" Validation Results — Loss: {avg_val_loss:.4f} | Top1: {acc1:.2f}% | Top5: {acc5:.2f}% | Top10: {acc10:.2f}%")

    return {
        'avg_val_loss': avg_val_loss,
        'top1_accuracy': acc1,
        'top5_accuracy': acc5,
        'top10_accuracy': acc10
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train(config)
