import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=30.0, margin=0.50):
        super(ArcFaceLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weights = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, features, labels):
        # Normalize features and weights
        features = F.normalize(features)
        weights = F.normalize(self.weights)

        # Cosine similarity
        cosine = F.linear(features, weights)

        # Add angular margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.margin)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = self.scale * (one_hot * target_logits + (1 - one_hot) * cosine)
        loss = F.cross_entropy(output, labels)
        return loss
