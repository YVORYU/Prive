import torch.nn as nn
import torch.nn.functional as F
from noise_layers.crop import random_float
# 本模块实现缩放（Resize）噪声层
# 主要功能是：将噪声图像按随机比例先缩放再缩放回原尺寸，模拟图像缩放处理造成的失真
# 这种"先放大/缩小再恢复"的操作会引入插值误差，对隐写信息造成扰动
# 缩放比例范围由 noise_argparser.py 中的 resize(min,max) 配置项控制
class Resize(nn.Module):
    # 缩放噪声层：将图像按随机比例缩放后再缩放回原始尺寸
    # 通过两次插值操作引入插值误差，模拟实际图像处理中的缩放失真
    # 目标尺寸 = 原始尺寸 * resize_ratio，然后再缩放回原始尺寸
    def __init__(self, resize_ratio_range, interpolation_method='bilinear'):
        # 初始化缩放噪声层
        # 参数:
        #   resize_ratio_range: 缩放比例范围 (min, max)，例如 (0.7, 0.9) 表示缩小到 70%~90%
        #   interpolation_method: 插值方法，默认 'bilinear'（双线性插值）
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]  # 缩放比例下界
        self.resize_ratio_max = resize_ratio_range[1]  # 缩放比例上界
        self.interpolation_method = interpolation_method  # 插值方法


    def forward(self, noised_and_cover):
        # 前向传播：先按随机比例缩放图像，再缩放回原始尺寸
        # 参数:
        #   noised_and_cover: 包含 [noised_image, cover_image] 的列表/元组
        # 返回:
        #   元组 (缩放后的噪声图像, cover_image)
        # 在 [resize_ratio_min, resize_ratio_max) 范围内随机采样一个缩放比例
        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        noised_image = noised_and_cover[0]  # 取出噪声图像
        original_height = noised_image.shape[2]  # 记录原始高度
        original_width = noised_image.shape[3]   # 记录原始宽度
        # 第一次插值：按随机比例缩放图像（scale_factor 同时作用于高和宽）
        noised_image = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)
        # 第二次插值：缩放回原始尺寸，引入插值误差
        noised_image = F.interpolate(
                                    noised_image,
                                    size=(original_height, original_width),
                                    mode=self.interpolation_method)
        return (noised_image, noised_and_cover[1])
