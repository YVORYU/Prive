import torch.nn as nn
import numpy as np
# 本模块实现随机裁剪噪声层
# 主要功能是按照指定的高度和宽度比例范围，对噪声图像和覆盖图像进行随机裁剪
# 用于模拟图像传输过程中的局部裁剪扰动，增强隐写鲁棒性
# 裁剪比例范围由 noise_argparser.py 中的 crop((hmin,hmax),(wmin,wmax)) 配置项控制

def random_float(min, max):
    # 在 [min, max) 区间内生成一个随机浮点数
    # 参数:
    #   min: 随机数下界
    #   max: 随机数上界
    # 返回:
    #   一个 [min, max) 范围内的随机浮点数
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    # 在图像内部随机选取一个矩形区域，作为裁剪窗口
    # 该函数类似于随机裁剪操作：高度和宽度比例分别在 height_ratio_range 和 width_ratio_range 范围内随机选取
    # 例如：若 height_ratio_range 为 (0.7, 0.9)，则会在该范围内随机取一个比例（例如 0.75），
    #   随后从 (0, 0.25) 中随机选取起始位置 rs，最终裁剪区间为 [rs, rs+0.75]
    # 这种方式保证裁剪起点在上下方向上等概率分布
    # 参数:
    #   image: 待裁剪的图像张量，形状为 (batch, channel, H, W)
    #   height_ratio_range: 保留高度比例的取值范围 (hmin, hmax)
    #   width_ratio_range:  保留宽度比例的取值范围 (wmin, wmax)
    # 返回:
    #   四元组 (height_start, height_end, width_start, width_end) 表示裁剪窗口的起止索引
    image_height = image.shape[2]  # 图像的高度（行数）
    image_width = image.shape[3]   # 图像的宽度（列数）

    # 根据随机比例计算裁剪后保留的高度和宽度（像素数）
    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[1]) * image_width))

    # 计算高度方向的裁剪起始位置：若保留高度等于原图高度，则起点为 0（无需裁剪）
    if remaining_height == image_height:
        height_start = 0
    else:
        # 否则在 [0, image_height - remaining_height] 范围内随机选取起点
        height_start = np.random.randint(0, image_height - remaining_height)

    # 计算宽度方向的裁剪起始位置：逻辑与高度方向一致
    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    # 返回裁剪窗口的 (h_start, h_end, w_start, w_end)
    return height_start, height_start+remaining_height, width_start, width_start+remaining_width


class Crop(nn.Module):
    # 随机裁剪噪声层：从图像的上下左右随机裁剪一部分
    # 裁剪量由 height_ratio_range 和 width_ratio_range 两个参数控制
    # 输入为 (noised_image, cover_image) 元组，对两张图应用相同的裁剪窗口
    def __init__(self, height_ratio_range, width_ratio_range):
        # 初始化裁剪噪声层
        # 参数:
        #   height_ratio_range: 高度方向保留比例范围 (hmin, hmax)
        #   width_ratio_range:  宽度方向保留比例范围 (wmin, wmax)
        super(Crop, self).__init__()
        self.height_ratio_range = height_ratio_range  # 保存高度比例范围
        self.width_ratio_range = width_ratio_range    # 保存宽度比例范围


    def forward(self, noised_and_cover):
        # 前向传播：对输入的 (噪声图像, 覆盖图像) 应用相同的随机裁剪
        # 参数:
        #   noised_and_cover: 包含两个元素的列表/元组 [noised_image, cover_image]
        # 返回:
        #   元组 (cropped_noised, cropped_cover)，即裁剪后的噪声图像和覆盖图像
        noised_image = noised_and_cover[0]  # 取出噪声图像（已嵌入消息的图像）
        cover_image = noised_and_cover[1]   # 取出原始覆盖图像
        # 计算随机裁剪窗口的起止索引
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, self.height_ratio_range, self.width_ratio_range)

        # 对噪声图像按窗口进行裁剪，使用 clone() 避免视图共享内存
        cropped_noised = noised_image[:, :, h_start:h_end, w_start:w_end].clone()
        # 对覆盖图像应用相同的裁剪窗口，保证两图空间对齐
        cropped_cover = cover_image[:, :, h_start:h_end, w_start:w_end].clone()

        return (cropped_noised, cropped_cover)
