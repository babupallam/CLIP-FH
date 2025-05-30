import torch
import torch.nn as nn
import torch.nn.functional as F

# Supervised Contrastive Loss (used in CLIP-ReID stage 1)
class SupConLoss(nn.Module):
    def __init__(self, device="cuda", temperature=0.07):
        super(SupConLoss, self).__init__()

        # Temperature controls how sharp or smooth the softmax is
        self.temperature = temperature

        # Device to run the computation (e.g., "cuda" or "cpu")
        self.device = device

    def forward(self, features, contrast_features, labels, targets=None):
        """
        Compute supervised contrastive loss.

        Args:
            features: Tensor of shape [B, D] - main feature vectors
            contrast_features: Tensor of shape [B, D] - comparison feature vectors
            labels: Tensor of shape [B] - class labels for each sample
            targets: not used here, included for compatibility

        Returns:
            loss: A scalar Tensor with the final contrastive loss
        """

        # Normalize both sets of features to unit length
        features = F.normalize(features, dim=-1)
        contrast_features = F.normalize(contrast_features, dim=-1)

        batch_size = features.shape[0]

        # Compute cosine similarity and scale by temperature
        # Resulting shape: [B, B]
        logits = torch.div(torch.matmul(features, contrast_features.T), self.temperature)

        # Create a mask for positive pairs (same label), but remove self-comparisons
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(self.device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(self.device)
        mask = mask * logits_mask  # Remove self-pairs

        # Compute log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean log probability over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Final loss: average over the batch
        loss = -mean_log_prob_pos.mean()

        return loss  # Must return a tensor
