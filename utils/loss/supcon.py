# utils/loss/supcontrast.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (for CLIP-ReID stage 1)
    """
    def __init__(self, device="cuda", temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features, contrast_features, labels, targets=None):
        """
        Args:
            features: Tensor of shape [B, D]
            contrast_features: Tensor of shape [B, D]
            labels: LongTensor of shape [B]
        Returns:
            loss: Scalar Tensor
        """
        features = F.normalize(features, dim=-1)
        contrast_features = F.normalize(contrast_features, dim=-1)

        batch_size = features.shape[0]
        logits = torch.div(torch.matmul(features, contrast_features.T), self.temperature)  # [B, B]

        # mask: positive pairs (same label, excluding self-pairing)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(self.device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(self.device)
        mask = mask * logits_mask  # zero out self-similarity

        # log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # mean log-prob of positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # final loss
        loss = -mean_log_prob_pos.mean()

        return loss  # MUST be a Tensor
