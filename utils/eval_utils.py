import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score


def validate(model, prompt_learner, val_loader, device, log, val_type="classifier", batch_size=64, loss_fn=None):
    model.eval()
    if prompt_learner is not None:
        prompt_learner.eval()
    if hasattr(model, "classifier"):
        model.classifier.eval()

    all_feats, all_labels = [], []
    total_val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            feats = F.normalize(model.encode_image(images), dim=1)

            if loss_fn and hasattr(model, "classifier"):
                logits = model.classifier(feats)
                total_val_loss += loss_fn(logits, labels).item()

            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())

    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    results = {}

    if val_type in ("classifier", "both"):
        results.update(_evaluate_embedding_reid(all_feats, all_labels, total_val_loss, len(val_loader), log))

    if val_type in ("reid", "both") and prompt_learner is not None:
        results.update(_evaluate_prompt_reid(model, prompt_learner, all_feats, all_labels, device, log, batch_size))

    return results


def _evaluate_embedding_reid(all_feats, all_labels, total_val_loss, num_batches, log):
    sim_matrix = all_feats @ all_feats.T
    sim_matrix.fill_diagonal_(-1)

    N = len(all_labels)
    cmc = np.zeros(N)
    aps = []

    for i in range(N):
        query_label = all_labels[i].item()
        scores = sim_matrix[i]
        sorted_idx = torch.argsort(scores, descending=True)
        matches = (all_labels[sorted_idx] == query_label).numpy().astype(int)

        rank_pos = np.where(matches == 1)[0]
        if len(rank_pos) == 0:
            continue
        cmc[rank_pos[0]:] += 1

        try:
            ap = average_precision_score(matches, scores[sorted_idx].numpy())
            aps.append(ap)
        except ValueError:
            pass

    cmc /= N
    mAP = np.mean(aps)

    metrics = {
        'avg_val_loss': total_val_loss / num_batches,
        'rank1': 100.0 * cmc[0],
        'rank5': 100.0 * cmc[4] if len(cmc) > 4 else 0.0,
        'rank10': 100.0 * cmc[9] if len(cmc) > 9 else 0.0,
        'mAP': 100.0 * mAP
    }

    log("\n[Validation: Embedding ReID]")
    for k, v in metrics.items():
        log(f"{k.upper()}: {v:.2f}%")

    return metrics


def _evaluate_prompt_reid(model, prompt_learner, all_feats, all_labels, device, log, batch_size):
    txt_feats = []

    with torch.no_grad():
        for label_batch in torch.split(all_labels, batch_size):
            label_batch = label_batch.to(device)
            prompts = prompt_learner.forward_batch(label_batch)
            x = prompts + model.positional_embedding.unsqueeze(0)
            x = x.permute(1, 0, 2)
            x = model.transformer(x)
            x = x.permute(1, 0, 2)
            txt = model.ln_final(x[:, 0, :])
            txt_feats.append(F.normalize(txt, dim=1).cpu())

    txt_feats = torch.cat(txt_feats, dim=0)
    sim_matrix = all_feats @ txt_feats.T

    sorted_indices = sim_matrix.argsort(dim=1, descending=True)
    aps, metrics = [], {}

    for i in range(len(all_labels)):
        label = all_labels[i]
        ranking = all_labels[sorted_indices[i]]
        correct = (ranking == label).float()
        if correct.sum() == 0:
            continue
        precision_at_k = correct.cumsum(0) / torch.arange(1, len(correct) + 1)
        ap = (precision_at_k * correct).sum() / correct.sum()
        aps.append(ap.item())

    metrics["prompt_mAP"] = sum(aps) / len(aps)
    for r in [1, 5, 10]:
        k = min(r, sim_matrix.size(1))
        correct = (sim_matrix.topk(k, dim=1).indices == torch.arange(sim_matrix.size(0)).unsqueeze(1))
        metrics[f"prompt_rank{r}"] = correct.any(dim=1).float().mean().item() * 100.0

    log("\n[Validation: Prompt ReID]")
    for k, v in metrics.items():
        log(f"{k.upper()}: {v:.2f}%")

    return metrics
