import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import torch.nn as nn
from torch.nn.functional import normalize

def validate(model, prompt_learner, val_loader, device, log, val_type="reid", batch_size=64, loss_fn=None):
    model.eval()  # Set model to evaluation mode
    if prompt_learner:
        prompt_learner.eval()  # Set prompt learner to eval mode if used

    all_feats, all_labels = [], []

    # === Step 1: Feature Extraction ===
    log("[Validation] Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Extracting"):
            images, labels = images.to(device), labels.to(device)
            # Extract image features and normalize them
            feats = F.normalize(model.encode_image(images), dim=1)
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all features and labels from batches
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # === Step 2: Split query/gallery ===
    # Get lengths from the combined dataset (assuming ReID-style splits)
    query_len = len(val_loader.dataset.datasets[0])
    gallery_len = len(val_loader.dataset.datasets[1])

    # Split features and labels into query and gallery sets
    query_feats, gallery_feats = all_feats[:query_len], all_feats[query_len:]
    query_labels, gallery_labels = all_labels[:query_len], all_labels[query_len:]

    # === Step 3: Similarity Matrix ===
    # Compute cosine similarity between query and gallery features
    sim_matrix = query_feats @ gallery_feats.T  # [num_query, num_gallery]

    cmc = np.zeros(len(gallery_labels))  # Cumulative Matching Curve
    aps = []  # Average Precision scores for mAP

    # === Step 4: ReID Evaluation ===
    log("[Validation] Evaluating ranks and mAP...")
    for i in tqdm(range(len(query_labels)), desc="Computing Metrics"):
        query_label = query_labels[i].item()
        scores = sim_matrix[i]  # Similarity scores for this query
        sorted_idx = torch.argsort(scores, descending=True)  # Sort gallery by score

        # Check if sorted gallery labels match the query label
        matches = (gallery_labels[sorted_idx] == query_label).numpy().astype(int)

        # Find the first correct match rank
        rank_pos = np.where(matches == 1)[0]
        if len(rank_pos) == 0:
            continue  # Skip if no match found
        cmc[rank_pos[0]:] += 1  # Update CMC curve from match position onward

        # Try to compute average precision for this query
        try:
            ap = average_precision_score(matches, scores[sorted_idx].numpy())
            aps.append(ap)
        except:
            pass  # Ignore errors in AP calculation

    # Normalize CMC and calculate mean average precision
    cmc /= len(query_labels)
    mAP = np.mean(aps)

    # Prepare metrics dictionary
    metrics = {
        'rank1': 100.0 * cmc[0],
        'rank5': 100.0 * cmc[4] if len(cmc) > 4 else 0.0,
        'rank10': 100.0 * cmc[9] if len(cmc) > 9 else 0.0,
        'mAP': 100.0 * mAP
    }

    # Print all metrics
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
def validate_promptsg(model_components, val_loader, device, compose_prompt, config=None):
    """
    PromptSG validation using cosine similarity for ReID-style metrics (Rank1, Rank5, Rank10, mAP).
    Classifier logits are ignored.
    """
    clip_model, inversion_model, multimodal_module, _ = model_components

    # Set all models to eval mode
    clip_model.eval()
    inversion_model.eval()
    multimodal_module.eval()

    all_features = []
    all_labels = []

    print("\n[PromptSG] Starting Validation...")

    # Loop through validation batches
    for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Extracting features")):
        images, labels = images.to(device), labels.to(device)

        # Get image features from CLIP
        img_features = clip_model.encode_image(images).float()

        # Generate pseudo text tokens from image features
        pseudo_tokens = inversion_model(img_features)

        # Use template from config if given, else default to plain format
        if config:
            template = config.get("prompt_template", "a photo of a {}")
        else:
            template = "a photo of a {}"

        # Split prefix and suffix from the template
        prefix, suffix = template.split("{aspect}")[0].strip(), template.split("{aspect}")[-1].strip()

        # Generate text embeddings using composed prompt with pseudo tokens
        text_emb = compose_prompt(clip_model.encode_text, pseudo_tokens, templates=(prefix, suffix), device=device)

        # Use the multi-modal module to combine text and visual embeddings
        visual_emb = multimodal_module(text_emb, img_features.unsqueeze(1))  # [B, 3, D]

        # Average the token-wise features to get final embedding
        pooled = visual_emb.mean(dim=1)

        # Normalize pooled embeddings and save
        pooled = F.normalize(pooled, dim=1)
        all_features.append(pooled.cpu())
        all_labels.append(labels.cpu())

        # Optional debug for first batch
        if batch_idx == 0:
            print(f"[DEBUG] Val Batch  Image Shape: {images.shape}, Logits: SKIPPED")

    # === Stack all features and labels
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Split query and gallery based on dataset structure
    query_len = len(val_loader.dataset.datasets[0])
    gallery_len = len(val_loader.dataset.datasets[1])

    query_feats = all_features[:query_len]
    gallery_feats = all_features[query_len:]
    query_labels = all_labels[:query_len]
    gallery_labels = all_labels[query_len:]

    # === Cosine similarity between query and gallery
    sim_matrix = query_feats @ gallery_feats.T  # [num_query, num_gallery]
    cmc = np.zeros(len(gallery_labels))  # Cumulative Matching Characteristic
    aps = []  # Average Precision scores

    # Compute ReID metrics for each query
    for i in range(len(query_labels)):
        query_label = query_labels[i].item()
        scores = sim_matrix[i]

        # Sort gallery indices by similarity (highest first)
        sorted_idx = torch.argsort(scores, descending=True)

        # Mark where correct matches occur
        matches = (gallery_labels[sorted_idx] == query_label).numpy().astype(int)

        # Find the first rank where correct match appears
        rank_pos = np.where(matches == 1)[0]
        if len(rank_pos) == 0:
            continue
        cmc[rank_pos[0]:] += 1

        # Compute average precision for the current query
        try:
            ap = average_precision_score(matches, scores[sorted_idx].numpy())
            aps.append(ap)
        except:
            pass  # Skip if precision can't be calculated

    # Final CMC and mAP
    cmc /= len(query_labels)
    mAP = np.mean(aps) if aps else 0.0

    # Print the evaluation results
    print(f"[PromptSG] ReID Validation Results")
    print(f"RANK1: {cmc[0] * 100:.2f}%")
    print(f"RANK5: {cmc[4] * 100:.2f}%" if len(cmc) > 4 else "")
    print(f"RANK10: {cmc[9] * 100:.2f}%" if len(cmc) > 9 else "")
    print(f"mAP: {mAP * 100:.2f}%")

    # Return all metrics
    return {
        'avg_val_loss': 0.0,  # Not using classifier/loss
        'rank1_accuracy': cmc[0] * 100,
        'rank5_accuracy': cmc[4] * 100 if len(cmc) > 4 else 0.0,
        'rank10_accuracy': cmc[9] * 100 if len(cmc) > 9 else 0.0,
        'mAP': mAP * 100
    }
