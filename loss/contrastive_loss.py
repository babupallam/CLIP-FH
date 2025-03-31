import torch
import torch.nn.functional as F


def supcon_loss(image_features, text_features, labels, temperature=0.07):
    """
    Supervised contrastive loss: supports multiple positives per anchor (label-aware).
    Args:
        image_features: (B, D)
        text_features:  (B, D)
        labels:         (B,)
    Returns:
        SupCon-style symmetric loss.
    """
    assert image_features.shape == text_features.shape
    device = image_features.device
    B = image_features.shape[0]

    # Normalize
    image_features = F.normalize(image_features, dim=-1)
    text_features  = F.normalize(text_features, dim=-1)

    # Compute similarity matrix: (B, B)
    sim_i2t = image_features @ text_features.T / temperature
    sim_t2i = text_features @ image_features.T / temperature

    # Mask for positives (same labels)
    labels = labels.contiguous().view(-1, 1)  # shape: (B, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # shape: (B, B)

    def loss_fn(similarity, mask):
        # Exclude self-comparisons (diagonal)
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()  # stability
        exp_logits = torch.exp(logits) * (1 - torch.eye(B, device=device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()

    loss_i2t = loss_fn(sim_i2t, mask)
    loss_t2i = loss_fn(sim_t2i, mask)

    return (loss_i2t + loss_t2i) / 2

def clip_contrastive_loss(img_feats, txt_feats, temperature=0.07):
    """
    Minimal Li2t + Lt2i approach for a batch of B matched pairs.
    Assumes that row i in img_feats corresponds to row i in txt_feats (the correct match).
    B x D => B x B similarity matrix
    """
    # 1. Normalize
    img_feats = F.normalize(img_feats, dim=-1)
    txt_feats = F.normalize(txt_feats, dim=-1)

    # 2. Similarity matrix => shape (B, B)
    sim_matrix = img_feats @ txt_feats.t() / temperature

    # 3. Li2t: cross-entropy row-wise => correct pair is diag
    labels_for_image = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss_i2t = F.cross_entropy(sim_matrix, labels_for_image)

    # 4. Lt2i: cross-entropy col-wise => transpose sim
    loss_t2i = F.cross_entropy(sim_matrix.t(), labels_for_image)

    return 0.5 * (loss_i2t + loss_t2i)


