import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureRedirectionLoss(nn.Module):
    def __init__(self, clip_weight=1.0, dino_weight=0.5, escape_margin=0.7):
        super().__init__()
        self.clip_weight = clip_weight
        self.dino_weight = dino_weight
        self.escape_margin = escape_margin

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
        loss_dict['clip_loss'] = clip_loss.item()

        if dino_cover is not None and dino_encoded is not None:
            dino_loss = self.escape_loss(dino_cover, dino_encoded)
            total_loss += self.dino_weight * dino_loss
            loss_dict['dino_loss'] = dino_loss.item()

        loss_dict['total_privacy_loss'] = total_loss.item()
        return total_loss, loss_dict


class RedirectionTripletLoss(nn.Module):
    def __init__(self, escape_margin=0.5, redirect_margin=0.6, max_target_sim=0.75):
        super().__init__()
        self.escape_margin = escape_margin
        self.redirect_margin = redirect_margin
        self.max_target_sim = max_target_sim

    def forward(self, feat_cover, feat_encoded, feat_target=None):
        sim_cover = F.cosine_similarity(feat_encoded, feat_cover, dim=1)
        target = 1 - self.escape_margin
        escape_loss = torch.mean(torch.log1p(torch.exp(sim_cover - target)))

        if feat_target is not None:
            sim_target = F.cosine_similarity(feat_encoded, feat_target, dim=1)
            redirect_loss = torch.mean(torch.relu(self.redirect_margin - sim_target))
            control_loss = torch.mean(torch.relu(sim_target - self.max_target_sim))
            total_loss = escape_loss + redirect_loss + 0.3 * control_loss
        else:
            total_loss = escape_loss

        return total_loss


class RetrievalEscapeLoss(nn.Module):
    def __init__(self, clip_weight=1.0, dino_weight=0.5,
                 triplet_weight=0.15, margin=0.7, triplet_margin=0.5, attacker_weight=0.5):
        super().__init__()
        self.clip_weight = clip_weight
        self.triplet_weight = triplet_weight
        self.margin = margin
        self.triplet_margin = triplet_margin
        self.attacker_weight = attacker_weight

        self.multi_feature_loss = FeatureRedirectionLoss(
            clip_weight=clip_weight,
            dino_weight=dino_weight,
            escape_margin=margin
        )
        self.triplet_loss = RedirectionTripletLoss(
            escape_margin=triplet_margin,
            redirect_margin=0.6,
            max_target_sim=0.75
        )

    def forward(self, feat_cover, feat_encoded, feat_negatives=None,
                dino_cover=None, dino_encoded=None,
                attacker_feat_cover=None, attacker_feat_encoded=None):
        multi_loss, loss_dict = self.multi_feature_loss(
            feat_cover, feat_encoded,
            dino_cover, dino_encoded
        )

        total_loss = multi_loss

        if attacker_feat_cover is not None and attacker_feat_encoded is not None:
            triplet_loss_val = self.triplet_loss(
                attacker_feat_cover, attacker_feat_encoded, feat_negatives
            )
            total_loss = total_loss + self.triplet_weight * triplet_loss_val
            loss_dict['triplet_loss'] = triplet_loss_val.item()

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
