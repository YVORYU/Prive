import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureRedirectionLoss(nn.Module):
    def __init__(self, clip_weight=1.0, dino_weight=0.5, escape_margin=0.7,
                 redirect_margin=0.6, max_target_sim=0.75,
                 clip_redirect_weight=0.5, dino_redirect_weight=0.3,
                 dino_redirect_margin=None, dino_max_target_sim=None,
                 text_escape_weight=0.5):
        super().__init__()
        self.clip_weight = clip_weight
        self.dino_weight = dino_weight
        self.escape_margin = escape_margin
        self.redirect_margin = redirect_margin
        self.max_target_sim = max_target_sim
        self.clip_redirect_weight = clip_redirect_weight
        self.dino_redirect_weight = dino_redirect_weight
        self.dino_redirect_margin = dino_redirect_margin if dino_redirect_margin is not None else redirect_margin
        self.dino_max_target_sim = dino_max_target_sim if dino_max_target_sim is not None else max_target_sim
        self.text_escape_weight = text_escape_weight

    def escape_loss(self, feat_cover, feat_encoded):
        sim = F.cosine_similarity(feat_encoded, feat_cover, dim=1)
        target = 1 - self.escape_margin
        loss = torch.mean(torch.log1p(torch.exp(sim - target)))
        return loss

    def directed_escape_loss(self, feat_cover, feat_encoded, feat_negatives=None,
                             redirect_margin=None, max_target_sim=None):
        escape = self.escape_loss(feat_cover, feat_encoded)

        if feat_negatives is None:
            return escape, torch.tensor(0.0, device=feat_cover.device), torch.tensor(0.0, device=feat_cover.device)

        r_margin = redirect_margin if redirect_margin is not None else self.redirect_margin
        m_sim = max_target_sim if max_target_sim is not None else self.max_target_sim
        sim_neg = F.cosine_similarity(feat_encoded, feat_negatives, dim=1)
        redirect = torch.mean(torch.log1p(torch.exp(r_margin - sim_neg)))
        control = torch.mean(torch.relu(sim_neg - m_sim))

        return escape, redirect, control

    def text_guided_escape_loss(self, feat_encoded, text_anchors):
        sim_text = F.cosine_similarity(feat_encoded, text_anchors, dim=1)
        loss = torch.mean(1 - sim_text)
        return loss

    def forward(self, clip_cover, clip_encoded, dino_cover=None, dino_encoded=None,
                clip_negatives=None, dino_negatives=None, text_anchors=None):
        total_loss = 0.0
        loss_dict = {}

        if clip_negatives is not None:
            clip_escape, clip_redirect, clip_control = self.directed_escape_loss(
                clip_cover, clip_encoded, clip_negatives)
            clip_loss = clip_escape + self.clip_redirect_weight * clip_redirect + 0.3 * clip_control
            loss_dict['clip_escape'] = clip_escape.item()
            loss_dict['clip_redirect'] = clip_redirect.item()
            loss_dict['clip_control'] = clip_control.item()
        else:
            clip_loss = self.escape_loss(clip_cover, clip_encoded)
            loss_dict['clip_escape'] = clip_loss.item()
            loss_dict['clip_redirect'] = 0.0
            loss_dict['clip_control'] = 0.0

        if text_anchors is not None:
            text_loss = self.text_guided_escape_loss(clip_encoded, text_anchors)
            clip_loss = clip_loss + self.text_escape_weight * text_loss
            loss_dict['text_escape'] = text_loss.item()
        else:
            loss_dict['text_escape'] = 0.0

        total_loss += self.clip_weight * clip_loss
        loss_dict['clip_loss'] = clip_loss.item()

        if dino_cover is not None and dino_encoded is not None:
            if dino_negatives is not None:
                dino_escape, dino_redirect, dino_control = self.directed_escape_loss(
                    dino_cover, dino_encoded, dino_negatives,
                    redirect_margin=self.dino_redirect_margin,
                    max_target_sim=self.dino_max_target_sim)
                dino_loss = dino_escape + self.dino_redirect_weight * dino_redirect + 0.3 * dino_control
                loss_dict['dino_escape'] = dino_escape.item()
                loss_dict['dino_redirect'] = dino_redirect.item()
                loss_dict['dino_control'] = dino_control.item()
            else:
                dino_loss = self.escape_loss(dino_cover, dino_encoded)
                loss_dict['dino_escape'] = dino_loss.item()
                loss_dict['dino_redirect'] = 0.0
                loss_dict['dino_control'] = 0.0

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
            redirect_loss = torch.mean(torch.log1p(torch.exp(self.redirect_margin - sim_target)))
            control_loss = torch.mean(torch.relu(sim_target - self.max_target_sim))
            total_loss = escape_loss + redirect_loss + 0.3 * control_loss
        else:
            total_loss = escape_loss

        return total_loss


class RetrievalEscapeLoss(nn.Module):
    def __init__(self, clip_weight=1.0, dino_weight=0.5,
                 triplet_weight=0.15, margin=0.7, triplet_margin=0.5, attacker_weight=0.5,
                 redirect_margin=0.6, max_target_sim=0.75,
                 clip_redirect_weight=0.5, dino_redirect_weight=0.3,
                 dino_redirect_margin=None, dino_max_target_sim=None,
                 text_escape_weight=0.5):
        super().__init__()
        self.clip_weight = clip_weight
        self.triplet_weight = triplet_weight
        self.margin = margin
        self.triplet_margin = triplet_margin
        self.attacker_weight = attacker_weight

        self.multi_feature_loss = FeatureRedirectionLoss(
            clip_weight=clip_weight,
            dino_weight=dino_weight,
            escape_margin=margin,
            redirect_margin=redirect_margin,
            max_target_sim=max_target_sim,
            clip_redirect_weight=clip_redirect_weight,
            dino_redirect_weight=dino_redirect_weight,
            dino_redirect_margin=dino_redirect_margin,
            dino_max_target_sim=dino_max_target_sim,
            text_escape_weight=text_escape_weight
        )
        self.triplet_loss = RedirectionTripletLoss(
            escape_margin=triplet_margin,
            redirect_margin=redirect_margin,
            max_target_sim=max_target_sim
        )

    def forward(self, feat_cover, feat_encoded, feat_negatives=None,
                dino_cover=None, dino_encoded=None,
                attacker_feat_cover=None, attacker_feat_encoded=None,
                clip_negatives=None, dino_negatives=None,
                text_anchors=None):
        multi_loss, loss_dict = self.multi_feature_loss(
            feat_cover, feat_encoded,
            dino_cover, dino_encoded,
            clip_negatives=clip_negatives,
            dino_negatives=dino_negatives,
            text_anchors=text_anchors
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
