import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEscapeLoss(nn.Module):
    def __init__(self, clip_weight=1.0, dino_weight=0.5, escape_margin=0.5):
        super().__init__()
        self.clip_weight = clip_weight
        self.dino_weight = dino_weight
        self.escape_margin = escape_margin

    def set_epoch(self, epoch):
        pass

    def escape_loss(self, feat_cover, feat_encoded):
        sim = F.cosine_similarity(feat_encoded, feat_cover, dim=1)
        target = 1 - self.escape_margin
        loss = torch.mean(torch.log1p(torch.exp(sim - target)))
        return loss

    def forward(self, clip_cover, clip_encoded, dino_cover=None, dino_encoded=None):
        total_loss = 0.0
        loss_dict = {}

        clip_loss = self.escape_loss(clip_cover, clip_encoded)
        total_loss += self.clip_weight * clip_loss
        loss_dict['clip_escape'] = clip_loss.item()

        if dino_cover is not None and dino_encoded is not None:
            dino_loss = self.escape_loss(dino_cover, dino_encoded)
            total_loss += self.dino_weight * dino_loss
            loss_dict['dino_escape'] = dino_loss.item()

        loss_dict['total_privacy_loss'] = total_loss.item()
        return total_loss, loss_dict


class RetrievalEscapeLoss(nn.Module):
    def __init__(self, clip_weight=1.0, dino_weight=0.5, margin=0.5, **kwargs):
        super().__init__()
        self.escape_loss_fn = SimpleEscapeLoss(
            clip_weight=clip_weight,
            dino_weight=dino_weight,
            escape_margin=margin
        )

    def set_epoch(self, epoch):
        pass

    def forward(self, feat_cover, feat_encoded, dino_cover=None, dino_encoded=None, **kwargs):
        total_loss, loss_dict = self.escape_loss_fn(
            feat_cover, feat_encoded,
            dino_cover=dino_cover,
            dino_encoded=dino_encoded
        )
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
