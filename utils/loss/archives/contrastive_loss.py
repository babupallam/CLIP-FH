import torch
import torch.nn.functional as F


def supcon_loss(text_features, image_features, t_labels, i_labels, temperature=1.0):
    """
    CLIP-ReID style Supervised Contrastive Loss (one-directional: text  image)

    Args:
        text_features:  (B, D) - prompt/text embeddings
        image_features: (B, D) - image embeddings
        t_labels:       (B,)   - text labels (identity)
        i_labels:       (B,)   - image labels (identity)
        temperature:    scalar for softmax scaling

    Returns:
        Scalar loss value
    """
    #print(f" supcon_loss called!")
    #print(f"   text_features shape: {text_features.shape}")
    #print(f"   image_features shape: {image_features.shape}")
    #print(f"   t_labels shape: {t_labels.shape}")
    #print(f"   i_labels shape: {i_labels.shape}")

    assert text_features.shape == image_features.shape, \
        f"Shape mismatch: {text_features.shape} vs {image_features.shape}"

    device = text_features.device
    batch_size = text_features.shape[0]

    # === Label-based positive mask ===
    labels = t_labels.contiguous().view(-1, 1)  # shape: (B, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)

    # === Similarity matrix (text  image) ===
    logits = torch.matmul(text_features, image_features.transpose(0, 1)) / temperature

    # === Numerical stability ===
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # === Compute log-softmax and mask out self-similarity ===
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    # === Contrastive loss: average over positive pairs ===
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()

    return loss



import torch.nn as nn

class SymmetricSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, img_emb, text_emb, labels):
        logits = self.similarity(img_emb.unsqueeze(1), text_emb.unsqueeze(0)) / self.temperature
        labels = labels.unsqueeze(1) == labels.unsqueeze(0)
        labels = labels.float()

        img2text_loss = -(labels * torch.log_softmax(logits, dim=1)).sum(1).mean()
        text2img_loss = -(labels * torch.log_softmax(logits, dim=0)).sum(0).mean()

        return (img2text_loss + text2img_loss) / 2
