import torch
import torch.nn as nn
import torch.nn.functional as F

# Triplet loss with batch-hard mining
# It helps the model learn better embeddings by pulling similar items closer and pushing different ones apart
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, mining='batch_hard'):
        super().__init__()

        # Margin for the triplet loss (distance between positive and negative)
        self.margin = margin

        # Type of mining to use (currently supports only 'batch_hard')
        self.mining = mining

        # Use PyTorch's built-in MarginRankingLoss to calculate the triplet loss
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels):
        # Call the batch hard triplet loss function
        if self.mining == 'batch_hard':
            return self.batch_hard_triplet_loss(embeddings, labels)
        else:
            # If other mining types are given, raise error
            raise NotImplementedError("Only 'batch_hard' is supported.")

    def batch_hard_triplet_loss(self, embeddings, labels):
        """
        For each anchor embedding, find the hardest positive and hardest negative in the batch
        """

        # Normalize all embeddings to unit length
        embeddings = F.normalize(embeddings, dim=1)

        # Number of samples in the batch
        n = embeddings.size(0)

        # Compute pairwise distances between all embeddings
        # Resulting shape: [n, n]
        dist = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for positive and negative pairs
        labels = labels.unsqueeze(1)  # Shape: [n, 1]
        mask_pos = labels.eq(labels.T).float()  # Same labels = 1, others = 0
        mask_neg = 1.0 - mask_pos  # Opposite of positive mask

        # Hardest positive: max distance among positive pairs for each anchor
        dist_ap = (dist * mask_pos).max(1)[0]

        # Hardest negative: min distance among negative pairs for each anchor
        # Large number (1e5) added to positive pairs to exclude them from min calculation
        dist_an = (dist + (1e5 * mask_pos)).min(1)[0]

        # Ranking loss wants y = 1 (positive should be closer than negative)
        y = torch.ones_like(dist_ap)

        # Compute final triplet loss
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
