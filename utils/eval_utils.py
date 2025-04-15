import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import torch.nn as nn

def validate(model, prompt_learner, val_loader, device, log, val_type="reid", batch_size=64, loss_fn=None):
    model.eval()
    if prompt_learner:
        prompt_learner.eval()

    all_feats, all_labels = [], []

    # === Step 1: Feature Extraction ===
    log("[Validation] Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Extracting"):
            images, labels = images.to(device), labels.to(device)
            feats = F.normalize(model.encode_image(images), dim=1)
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())

    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # === Step 2: Split query/gallery ===
    query_len = len(val_loader.dataset.datasets[0])
    gallery_len = len(val_loader.dataset.datasets[1])
    query_feats, gallery_feats = all_feats[:query_len], all_feats[query_len:]
    query_labels, gallery_labels = all_labels[:query_len], all_labels[query_len:]

    # === Step 3: Similarity Matrix ===
    sim_matrix = query_feats @ gallery_feats.T

    cmc = np.zeros(len(gallery_labels))
    aps = []

    # === Step 4: ReID Evaluation ===
    log("[Validation] Evaluating ranks and mAP...")
    for i in tqdm(range(len(query_labels)), desc="Computing Metrics"):
        query_label = query_labels[i].item()
        scores = sim_matrix[i]
        sorted_idx = torch.argsort(scores, descending=True)
        matches = (gallery_labels[sorted_idx] == query_label).numpy().astype(int)

        rank_pos = np.where(matches == 1)[0]
        if len(rank_pos) == 0:
            continue
        cmc[rank_pos[0]:] += 1

        try:
            ap = average_precision_score(matches, scores[sorted_idx].numpy())
            aps.append(ap)
        except:
            pass

    cmc /= len(query_labels)
    mAP = np.mean(aps)

    metrics = {
        'rank1': 100.0 * cmc[0],
        'rank5': 100.0 * cmc[4] if len(cmc) > 4 else 0.0,
        'rank10': 100.0 * cmc[9] if len(cmc) > 9 else 0.0,
        'mAP': 100.0 * mAP
    }

    log("\n[ReID Validation]")
    for k, v in metrics.items():
        log(f"{k.upper()}: {v:.2f}%")

    return metrics



import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score


@torch.no_grad()
def validate_stage1_prompts(model, prompt_learner, val_loader, device, log=print, batch_size=64):
    """
    Perform ReID-style evaluation of learned prompts after Stage 1 (CLIP-ReID),
    using cosine similarity between image features and prompt-generated text embeddings.

    Returns:
        dict: {'rank1': float, 'mAP': float}
    """
    model.eval()
    prompt_learner.eval()
    log("Starting Stage 1 ReID evaluation (prompt-based zero-shot)...")

    # === Step 1: Build text features for all class IDs ===
    log("Encoding prompt features for all class IDs...")
    num_classes = prompt_learner.num_classes
    all_text_features = []

    for start in range(0, num_classes, batch_size):
        end = min(start + batch_size, num_classes)
        batch_labels = torch.arange(start, end).to(device)
        prompts = prompt_learner.forward_batch(batch_labels)

        x = prompts + model.positional_embedding.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = model.transformer(x)
        x = x.permute(1, 0, 2)
        text_feats = model.ln_final(x[:, 0, :])
        text_feats = F.normalize(text_feats, dim=-1)
        all_text_features.append(text_feats)

    text_features = torch.cat(all_text_features, dim=0)  # shape [C, D]

    # === Step 2: Extract image features from val_loader ===
    image_features, labels = [], []

    for batch in tqdm(val_loader, desc="Extracting image features"):
        if len(batch) == 6:
            img, pid, *_ = batch
        else:
            img, pid = batch[:2]
        img = img.to(device)
        feats = F.normalize(model.encode_image(img), dim=-1)
        image_features.append(feats.cpu())
        labels.append(pid)

    image_features = torch.cat(image_features, dim=0)
    labels = torch.cat(labels, dim=0)

    # === Step 3: Compute similarity matrix ===
    sim_matrix = image_features @ text_features.T  # [N_img, N_text]
    preds = torch.argmax(sim_matrix, dim=1)

    # === Step 4: Compute CMC and mAP ===
    cmc = np.zeros(len(text_features))
    aps = []

    for i in tqdm(range(len(labels)), desc="Evaluating"):
        query_label = labels[i].item()
        scores = sim_matrix[i]
        sorted_idx = torch.argsort(scores, descending=True)
        matches = (torch.arange(len(text_features))[sorted_idx] == query_label).int().numpy()

        # Rank accuracy
        rank_pos = np.where(matches == 1)[0]
        if len(rank_pos) == 0:
            continue
        cmc[rank_pos[0]:] += 1

        # Average Precision
        try:
            ap = average_precision_score(matches, scores[sorted_idx].cpu().numpy())
            aps.append(ap)
        except:
            pass

    cmc /= len(labels)
    mAP = np.mean(aps) if aps else 0.0

    # === Step 5: Log results ===
    log("Stage 1 Evaluation Results (Prompt-Based)")
    log(f"mAP: {100.0 * mAP:.2f}%")
    log(f"Rank-1: {100.0 * cmc[0]:.2f}%")
    log(f"Rank-5: {100.0 * cmc[4]:.2f}%" if len(cmc) > 4 else "")
    log(f"Rank-10: {100.0 * cmc[9]:.2f}%" if len(cmc) > 9 else "")

    return {"rank1": cmc[0].item() * 100, "mAP": mAP * 100}

@torch.no_grad()
def validate_promptsg(model_components, val_loader, device, compose_prompt, loss_fn=None):
    """
    Validation for PromptSG: Uses composed prompts + cross-attention + classifier.
    Returns classification loss, top-k accuracy, and mAP using cosine similarity.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score
    from torch.nn.functional import normalize

    clip_model, inversion_model, multimodal_module, classifier = model_components
    clip_model.eval()
    inversion_model.eval()
    multimodal_module.eval()
    classifier.eval()

    total_samples = 0
    correct_top1 = correct_top5 = correct_top10 = 0
    total_val_loss = 0.0
    criterion = loss_fn or nn.CrossEntropyLoss()

    all_features = []
    all_labels = []

    print("\n[PromptSG] Starting Validation...")
    for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validation")):
        images, labels = images.to(device), labels.to(device)

        img_features = clip_model.encode_image(images).float()
        pseudo_tokens = inversion_model(img_features)
        text_embeddings = compose_prompt(clip_model.encode_text, pseudo_tokens, device=device)
        visual_embeddings = multimodal_module(text_embeddings, img_features.unsqueeze(1))
        pooled_emb = visual_embeddings.mean(dim=1)
        logits = classifier(pooled_emb)

        # Classification loss
        loss = criterion(logits, labels)
        total_val_loss += loss.item()

        # Accuracy
        _, pred_topk = logits.topk(10, dim=1)
        match = pred_topk.eq(labels.unsqueeze(1).expand_as(pred_topk))
        total_samples += labels.size(0)
        correct_top1 += match[:, :1].sum().item()
        correct_top5 += match[:, :5].sum().item()
        correct_top10 += match[:, :10].sum().item()

        # For mAP calculation
        all_features.append(pooled_emb.cpu())
        all_labels.append(labels.cpu())

        if batch_idx == 0:
            print(f"[DEBUG] Val Batch → Image Shape: {images.shape}, Logits: {logits.shape}")

    # Accuracy metrics
    avg_val_loss = total_val_loss / len(val_loader)
    acc1 = 100 * correct_top1 / total_samples
    acc5 = 100 * correct_top5 / total_samples
    acc10 = 100 * correct_top10 / total_samples

    # mAP computation
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_features = normalize(all_features, dim=1)

    sim_matrix = torch.matmul(all_features, all_features.T).cpu().numpy()
    all_labels_np = all_labels.cpu().numpy()
    mAP_total = 0.0
    num_queries = all_features.size(0)

    for i in range(num_queries):
        query_label = all_labels_np[i]
        sims = sim_matrix[i]
        mask = np.arange(num_queries) != i  # exclude self
        sims = sims[mask]
        targets = (all_labels_np[mask] == query_label).astype(int)

        if targets.sum() > 0:
            ap = average_precision_score(targets, sims)
            if not np.isnan(ap):
                mAP_total += ap

    mAP = mAP_total / num_queries

    print(f"[PromptSG] Val Results — Loss: {avg_val_loss:.4f} | Top1: {acc1:.2f}% | Top5: {acc5:.2f}% | Top10: {acc10:.2f}% | mAP: {mAP*100:.2f}%")

    return {
        'avg_val_loss': avg_val_loss,
        'top1_accuracy': acc1,
        'top5_accuracy': acc5,
        'top10_accuracy': acc10,
        'mAP': mAP * 100  # return as percentage
    }
