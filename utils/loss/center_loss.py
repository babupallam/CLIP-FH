# utils/loss/center_loss.py
import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device="cpu"):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum() / 2.0 / batch_size
