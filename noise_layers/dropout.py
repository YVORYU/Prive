import torch
import torch.nn as nn
import numpy as np
# 本模块实现 Dropout 噪声层
# 主要功能是：按照一定保留比例随机选择像素位置，被丢弃的像素用覆盖图像对应像素替换
# 这种逐像素的随机替换模拟了图像传输过程中部分像素丢失的场景
# 保留比例范围由 noise_argparser.py 中的 dropout(keep_min,keep_max) 配置项控制
class Dropout(nn.Module):
    # Dropout 噪声层：随机丢弃噪声图像中的部分像素，并用覆盖图像的像素填充
    # 与 Cropout 的区别：Dropout 是逐像素随机（散点式），Cropout 是按矩形区域随机
    def __init__(self, keep_ratio_range):
        # 初始化 Dropout 噪声层
        # 参数:
        #   keep_ratio_range: 像素保留比例范围 (keep_min, keep_max)，取值在 [0, 1]
        super(Dropout, self).__init__()
        self.keep_min = keep_ratio_range[0]  # 保留比例下界
        self.keep_max = keep_ratio_range[1]  # 保留比例上界


    def forward(self, noised_and_cover):
        # 前向传播：按随机保留比例生成二值掩码，融合噪声图像与覆盖图像
        # 参数:
        #   noised_and_cover: 包含 [noised_image, cover_image] 的列表/元组
        # 返回:
        #   元组 (融合后的图像, cover_image)

        noised_image = noised_and_cover[0]  # 取出噪声图像
        cover_image = noised_and_cover[1]   # 取出覆盖图像

        # 在 [keep_min, keep_max] 范围内随机采样一个保留比例
        mask_percent = np.random.uniform(self.keep_min, self.keep_max)

        # 按保留比例随机生成二值掩码（形状为 H x W），1 表示保留噪声像素，0 表示用覆盖像素替换
        mask = np.random.choice([0.0, 1.0], noised_image.shape[2:], p=[1 - mask_percent, mask_percent])
        # 将 numpy 掩码转换为 torch 张量，并放到与噪声图像相同的设备上
        mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)
        # mask_tensor.unsqueeze_(0)
        # mask_tensor.unsqueeze_(0)
        # 将掩码扩展到与噪声图像相同的形状（batch 和 channel 维度广播）
        mask_tensor = mask_tensor.expand_as(noised_image)
        # 融合：掩码为 1 处取噪声像素，掩码为 0 处取覆盖像素
        noised_image = noised_image * mask_tensor + cover_image * (1-mask_tensor)
        return (noised_image, cover_image)
