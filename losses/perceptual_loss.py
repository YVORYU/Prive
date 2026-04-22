import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, use_l1=True, normalize_features=True):
        super(PerceptualLoss, self).__init__()
        self.use_l1 = use_l1
        self.normalize_features = normalize_features

        try:
            from torchvision.models import VGG19_Weights
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except ImportError:
            vgg = models.vgg19(pretrained=True)

        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:18])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.eval()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_to_imagenet(self, img):
        img = (img + 1) / 2
        return (img - self.imagenet_mean) / self.imagenet_std

    def normalize_features_fn(self, feat):
        mean = feat.mean(dim=[2, 3], keepdim=True)
        std = feat.std(dim=[2, 3], keepdim=True) + 1e-8
        normalized = (feat - mean) / std

        if normalized.size(1) >= 3:
            channel_parts = []
            for c in range(normalized.size(1)):
                channel_feat = normalized[:, c:c+1]
                if c < 3:
                    channel_mean = channel_feat.mean()
                    channel_std = channel_feat.std() + 1e-8
                    channel_parts.append((channel_feat - channel_mean) / channel_std)
                else:
                    channel_parts.append(channel_feat)
            normalized = torch.cat(channel_parts, dim=1)

        return normalized

    def forward(self, img1, img2):
        img1_norm = self.normalize_to_imagenet(img1)
        img2_norm = self.normalize_to_imagenet(img2)

        with torch.no_grad():
            feat1 = self.feature_extractor(img1_norm)

        feat2 = self.feature_extractor(img2_norm)

        if self.normalize_features:
            feat1_normalized = self.normalize_features_fn(feat1.detach())
            feat2_normalized = self.normalize_features_fn(feat2)
        else:
            feat1_normalized = feat1.detach()
            feat2_normalized = feat2

        if self.use_l1:
            loss = self.l1_loss(feat1_normalized, feat2_normalized)
        else:
            loss = self.mse_loss(feat1_normalized, feat2_normalized)
        return loss
