# utils/loss/center_loss.py

import torch
import torch.nn as nn

# This class defines the Center Loss
# It helps make features of the same class closer together
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device="cpu"):
        super(CenterLoss, self).__init__()

        # Save number of classes and feature dimension
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        # Create a learnable center for each class
        # Each center has the same dimension as the features
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        # Get the batch size
        batch_size = features.size(0)

        # Get the center for each label in the batch
        centers_batch = self.centers[labels]

        # Compute the squared distance between features and their class centers
        # Then average the total loss over the batch
        return ((features - centers_batch) ** 2).sum() / 2.0 / batch_size
