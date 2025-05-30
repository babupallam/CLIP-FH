import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, mining='batch_hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels):
        if self.mining == 'batch_hard':
            return self.batch_hard_triplet_loss(embeddings, labels)
        else:
            raise NotImplementedError("Only 'batch_hard' is supported.")

    def batch_hard_triplet_loss(self, embeddings, labels):
        """
        For each anchor, select hardest positive and hardest negative in the batch.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        n = embeddings.size(0)

        # Compute pairwise distance matrix [N, N]
        dist = torch.cdist(embeddings, embeddings, p=2)

        # Mask for positives and negatives
        labels = labels.unsqueeze(1)
        mask_pos = labels.eq(labels.T).float()
        mask_neg = 1.0 - mask_pos

        # For each anchor: hardest positive
        dist_ap = (dist * mask_pos).max(1)[0]
        dist_an = (dist + (1e5 * mask_pos)).min(1)[0]

        # Triplet loss
        y = torch.ones_like(dist_ap)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
