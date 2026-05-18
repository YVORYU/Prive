import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureRedirectionLoss(nn.Module):
    def __init__(self, clip_weight=1.0, dino_weight=0.5, escape_margin=0.5,
                 clip_redirect_weight=0.3, dino_redirect_weight=0.2,
                 clip_redirect_margin=0.3, dino_redirect_margin=0.3,
                 clip_max_target_sim=0.55, dino_max_target_sim=0.55):
        super().__init__()
        self.clip_weight = clip_weight
        self.dino_weight = dino_weight
        self.escape_margin = escape_margin
        self.clip_redirect_weight = clip_redirect_weight
        self.dino_redirect_weight = dino_redirect_weight
        self.clip_redirect_margin = clip_redirect_margin
        self.dino_redirect_margin = dino_redirect_margin
        self.clip_max_target_sim = clip_max_target_sim
        self.dino_max_target_sim = dino_max_target_sim
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def escape_loss(self, feat_cover, feat_encoded, margin):
        sim = F.cosine_similarity(feat_encoded, feat_cover, dim=1)
        target = 1 - margin
        loss = torch.mean(torch.log1p(torch.exp(sim - target)))
        return loss, sim.mean().item()

    def redirect_loss(self, feat_encoded, hard_negatives, margin, max_target_sim):
        sim_to_neg = F.cosine_similarity(feat_encoded.unsqueeze(1), hard_negatives, dim=2)
        hardest_idx = sim_to_neg.argmax(dim=1)
        hardest_neg = hard_negatives[torch.arange(feat_encoded.size(0)), hardest_idx]
        sim_to_hardest = F.cosine_similarity(feat_encoded, hardest_neg, dim=1)
        redirect_target = 1 - margin
        redirect_loss = torch.mean(torch.log1p(torch.exp(redirect_target - sim_to_hardest)))
        control_loss = torch.mean(F.relu(sim_to_hardest - max_target_sim))
        return redirect_loss, control_loss

    def forward(self, clip_cover, clip_encoded, dino_cover=None, dino_encoded=None,
                clip_hard_negatives=None, dino_hard_negatives=None, escape_margin=None):
        if escape_margin is None:
            escape_margin = self.escape_margin

        total_loss = 0.0
        loss_dict = {}

        clip_escape, clip_sim = self.escape_loss(clip_cover, clip_encoded, escape_margin)
        total_loss += self.clip_weight * clip_escape
        loss_dict['clip_escape'] = clip_escape.item()
        loss_dict['clip_sim'] = clip_sim

        if clip_hard_negatives is not None and self.clip_redirect_weight > 0:
            clip_redir, clip_ctrl = self.redirect_loss(
                clip_encoded, clip_hard_negatives,
                self.clip_redirect_margin, self.clip_max_target_sim
            )
            total_loss += self.clip_redirect_weight * clip_redir
            loss_dict['clip_redirect'] = clip_redir.item()
            loss_dict['clip_control'] = clip_ctrl.item()

        if dino_cover is not None and dino_encoded is not None:
            dino_escape, dino_sim = self.escape_loss(dino_cover, dino_encoded, escape_margin)
            total_loss += self.dino_weight * dino_escape
            loss_dict['dino_escape'] = dino_escape.item()
            loss_dict['dino_sim'] = dino_sim

            if dino_hard_negatives is not None and self.dino_redirect_weight > 0:
                dino_redir, dino_ctrl = self.redirect_loss(
                    dino_encoded, dino_hard_negatives,
                    self.dino_redirect_margin, self.dino_max_target_sim
                )
                total_loss += self.dino_redirect_weight * dino_redir
                loss_dict['dino_redirect'] = dino_redir.item()
                loss_dict['dino_control'] = dino_ctrl.item()

        loss_dict['total_privacy_loss'] = total_loss.item()
        return total_loss, loss_dict
