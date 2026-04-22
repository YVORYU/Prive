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


class Decoder(nn.Module):
    def __init__(self, message_length=30, n_channels=3):
        super(Decoder, self).__init__()
        self.message_length = message_length
        self.n_channels = n_channels

        decoder_channels = 64
        decoder_blocks = 8

        layers = [ConvBNRelu(3, decoder_channels)]
        for _ in range(decoder_blocks - 1):
            layers.append(ConvBNRelu(decoder_channels, decoder_channels))

        layers.append(ConvBNRelu(decoder_channels, message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(message_length, message_length)

    def forward(self, encoded_image):
        x = self.layers(encoded_image)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear(x)
        return x