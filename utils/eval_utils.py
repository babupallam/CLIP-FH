import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

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
