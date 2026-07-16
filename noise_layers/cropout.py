import torch
import torch.nn as nn
from noise_layers.crop import get_random_rectangle_inside
# 本模块实现 cropout（裁剪填充）噪声层
# 主要功能是：在噪声图像上随机选取一个矩形区域保留噪声像素，
# 矩形区域之外的像素用覆盖图像（原始图像）的对应像素替换
# 与 Crop 不同的是，Cropout 输出的图像尺寸与原图相同
# 该噪声层由 noise_argparser.py 中的 cropout((hmin,hmax),(wmin,wmax)) 配置项控制

class Cropout(nn.Module):
    # Cropout 噪声层：将噪声图像和覆盖图像融合为一张图
    # 在随机矩形窗口内取噪声图像像素，窗口外取覆盖图像像素
    # 最终输出图像尺寸与输入图像一致
    def __init__(self, height_ratio_range, width_ratio_range):
        # 初始化 Cropout 噪声层
        # 参数:
        #   height_ratio_range: 保留区域高度比例范围 (hmin, hmax)
        #   width_ratio_range:  保留区域宽度比例范围 (wmin, wmax)
        super(Cropout, self).__init__()
        self.height_ratio_range = height_ratio_range  # 保存高度比例范围
        self.width_ratio_range = width_ratio_range    # 保存宽度比例范围

    def forward(self, noised_and_cover):
        # 前向传播：根据随机矩形掩码融合噪声图像和覆盖图像
        # 参数:
        #   noised_and_cover: 包含 [noised_image, cover_image] 的列表/元组
        # 返回:
        #   noised_and_cover（其第 0 个元素已被替换为融合后的图像）
        noised_image = noised_and_cover[0]  # 取出噪声图像
        cover_image = noised_and_cover[1]   # 取出覆盖图像
        # 断言两图形状必须一致，否则无法逐像素融合
        assert noised_image.shape == cover_image.shape

        # 初始化全 0 掩码，与噪声图像同形状，后续在矩形窗口内填 1
        cropout_mask = torch.zeros_like(noised_image)
        # 在图像内部随机选取一个矩形窗口
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=noised_image,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)
        # 将掩码中矩形窗口区域置为 1，表示该区域保留噪声图像像素
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

        # 融合：窗口内取噪声图像像素，窗口外取覆盖图像像素
        noised_and_cover[0] = noised_image * cropout_mask + cover_image * (1-cropout_mask)
        return  noised_and_cover
