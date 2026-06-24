import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, message_length=30, n_channels=3, bilinear=False,
                 image_size=256, msg_budget=0.05, prv_budget=0.03,
                 phase1_prv_noise_smooth_kernel=7):
        super(Encoder, self).__init__()
        self.message_length = message_length
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.H = image_size
        self.W = image_size
        self.msg_budget = msg_budget
        self.prv_budget = prv_budget
        self.phase1_prv_noise_smooth_kernel = phase1_prv_noise_smooth_kernel

        channels = 128
        num_blocks = 16

        self.conv_layers = nn.Sequential(
            ConvBNRelu(3, channels)
        )
        for _ in range(num_blocks - 1):
            self.conv_layers.append(ConvBNRelu(channels, channels))

        self.after_concat_layer = ConvBNRelu(channels + 3 + message_length, channels)

        self.msg_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=1)
        )

        self.prv_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=1)
        )

    def forward(self, cover_image, message, phase=1, prv_noise_ratio=0.0):
        expanded_message = message.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.H, self.W)

        features = self.conv_layers(cover_image)
        concat = torch.cat([expanded_message, features, cover_image], dim=1)
        shared = self.after_concat_layer(concat)

        if phase == 2 and not self.msg_head[0].weight.requires_grad:
            with torch.no_grad():
                msg_residual = self.msg_head(shared)
                msg_residual = self.msg_budget * torch.tanh(msg_residual)
        else:
            msg_residual = self.msg_head(shared)
            msg_residual = self.msg_budget * torch.tanh(msg_residual)

        prv_residual = None
        if phase == 1:
            noise = torch.randn_like(cover_image)
            kernel = self.phase1_prv_noise_smooth_kernel
            if kernel > 1:
                if kernel % 2 == 0:
                    kernel += 1
                noise = F.avg_pool2d(noise, kernel_size=kernel, stride=1, padding=kernel // 2)
            noise = noise - noise.mean(dim=(2, 3), keepdim=True)
            noise = noise / (noise.std(dim=(2, 3), keepdim=True) + 1e-6)
            dummy_prv = self.prv_budget * prv_noise_ratio * torch.tanh(noise)
            encoded_image = cover_image + msg_residual + dummy_prv
            residual_penalty = torch.mean(msg_residual ** 2) + torch.mean(dummy_prv ** 2)
        else:
            prv_residual = self.prv_head(shared)
            prv_residual = self.prv_budget * torch.tanh(prv_residual)
            encoded_image = cover_image + msg_residual + prv_residual
            residual_penalty = torch.mean(msg_residual ** 2) + torch.mean(prv_residual ** 2)

        encoded_image = torch.clamp(encoded_image, -1, 1)

        return encoded_image, residual_penalty, msg_residual, prv_residual
