import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, features, labels):
        """
        features: Tensor of shape [batch_size, feat_dim]
        labels: Tensor of shape [batch_size]
        """
        anchors, positives, negatives = self.select_triplets(features, labels)
        if anchors is None:
            return torch.tensor(0.0, requires_grad=True).to(features.device)
        return self.ranking_loss(anchors, positives, negatives)

    def select_triplets(self, features, labels):
        """
        Selects valid (anchor, positive, negative) triplets from the batch
        """
        anchors, positives, negatives = [], [], []

        for i in range(len(features)):
            anchor_feat = features[i]
            anchor_label = labels[i]

            # Find all positives and negatives
            pos_mask = (labels == anchor_label).nonzero(as_tuple=True)[0]
            neg_mask = (labels != anchor_label).nonzero(as_tuple=True)[0]

            pos_mask = pos_mask[pos_mask != i]  # Exclude anchor itself

            if len(pos_mask) == 0 or len(neg_mask) == 0:
                continue

            # Pick one positive and one negative
            pos_feat = features[pos_mask[0]]
            neg_feat = features[neg_mask[0]]

            anchors.append(anchor_feat)
            positives.append(pos_feat)
            negatives.append(neg_feat)

        if len(anchors) == 0:
            return None, None, None

        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
