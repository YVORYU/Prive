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
    def __init__(self, message_length=30, n_channels=3, bilinear=False,image_size=256):
        super(Encoder, self).__init__()
        self.message_length = message_length
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.H = image_size
        self.W = image_size

        channels = 64
        num_blocks = 8

        self.conv_layers = nn.Sequential(
            ConvBNRelu(3, channels)
        )
        for _ in range(num_blocks - 1):
            self.conv_layers.append(ConvBNRelu(channels, channels))

        self.after_concat_layer = ConvBNRelu(channels + 3 + message_length, channels)

        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

    def forward(self, cover_image, message):
        expanded_message = message.unsqueeze(-1)
        expanded_message = expanded_message.unsqueeze(-1)
        expanded_message = expanded_message.expand(-1, -1, self.H, self.W)

        encoded_image = self.conv_layers(cover_image)

        concat = torch.cat([expanded_message, encoded_image, cover_image], dim=1)

        im_w = self.after_concat_layer(concat)

        encoded_image = self.final_layer(im_w)

        encoded_image = torch.tanh(encoded_image)

        residual_penalty = torch.mean((encoded_image - cover_image) ** 2)

        return encoded_image, residual_penalty