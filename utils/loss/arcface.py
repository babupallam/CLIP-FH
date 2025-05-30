import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()

        # Weight matrix [num_classes, feature_dim]
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))

        # Xavier initialization of weights
        nn.init.xavier_uniform_(self.weight)

        # Scaling factor and angular margin
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # Precompute cos(m) and sin(m) for efficiency
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        #self.margin = 0.5  # try lowering to 0.3
        #self.scale = 64  # try lowering to 30

        self.scale = 30  # more forgiving
        self.margin = 0.3  # easier class separation early on

        # Threshold for deciding margin application
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels):
        # Normalize both features and weights, then compute cosine similarity
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        # Compute sine from cosine using trig identity
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))

        # Apply angular margin
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply margin adjustment based on easy_margin setting
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels to apply phi only to correct classes
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Combine phi (for target class) and cosine (for other classes)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the output by predefined factor
        output *= self.s

        return output
