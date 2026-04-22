import torch
import torch.nn as nn
import torch.nn.functional as F


class PrivacyLoss(nn.Module):
    def __init__(self, margin=0.7):
        super(PrivacyLoss, self).__init__()
        self.margin = margin

    def forward(self, feat_cover, feat_encoded):
        sim = F.cosine_similarity(feat_encoded, feat_cover, dim=1)
        target = 1 - self.margin
        loss = torch.mean(torch.log1p(torch.exp(sim - target)))
        return loss
