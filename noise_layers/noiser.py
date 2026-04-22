import numpy as np
import torch.nn as nn
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization


"""
实现噪声层组合器，根据配置参数随机选择噪声层
"""
class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        self.noise_layers = nn.ModuleList()
        self._layer_names = []

        if len(noise_layers) == 0:
            self.noise_layers.append(Identity())
            self._layer_names.append('identity')
        else:
            for layer in noise_layers:
                if isinstance(layer, str):
                    if layer == 'JpegPlaceholder':
                        self.noise_layers.append(JpegCompression(device))
                        self._layer_names.append('jpeg')
                    elif layer == 'QuantizationPlaceholder':
                        self.noise_layers.append(Quantization(device))
                        self._layer_names.append('quant')
                    else:
                        raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
                                         f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
                else:
                    self.noise_layers.append(layer)
                    self._layer_names.append(layer.__class__.__name__)

    def forward(self, encoded_and_cover):
        idx = np.random.randint(0, len(self.noise_layers))
        return self.noise_layers[idx](encoded_and_cover)

