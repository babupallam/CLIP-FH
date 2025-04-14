# utils/loss/supcon.py

import torch
import torch.nn.functional as F
import torch.nn as nn

class SymmetricSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_emb, text_emb, labels):
        """
        img_emb: [B, D]
        text_emb: [B, D]
        labels: [B]
        """
        B = img_emb.size(0)

        # Normalize features
        img_emb = F.normalize(img_emb, dim=1)
        text_emb = F.normalize(text_emb, dim=1)

        # Similarity matrix: [B, B]
        sim_i2t = torch.matmul(img_emb, text_emb.T) / self.temperature
        sim_t2i = torch.matmul(text_emb, img_emb.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(labels.device)  # [B, B]

        # Contrastive loss (from SimCLR-style SupCon)
        def contrastive_loss(sim, mask):
            # subtracting max for stability
            sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
            exp_sim = torch.exp(sim)
            log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            loss = -(mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
            return loss.mean()

        loss_i2t = contrastive_loss(sim_i2t, mask)
        loss_t2i = contrastive_loss(sim_t2i, mask)

        return 0.5 * (loss_i2t + loss_t2i)
