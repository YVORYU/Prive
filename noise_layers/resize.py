import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from noise_layers.crop import random_float
"""
实现"缩放"功能，随机缩放噪声图像的大小
"""
class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_range, interpolation_method='bilinear'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method


    def forward(self, noised_and_cover):
        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        noised_image = noised_and_cover[0]
        original_height = noised_image.shape[2]
        original_width = noised_image.shape[3]
        noised_image = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)
        noised_image = F.interpolate(
                                    noised_image,
                                    size=(original_height, original_width),
                                    mode=self.interpolation_method)
        return (noised_image, noised_and_cover[1])
