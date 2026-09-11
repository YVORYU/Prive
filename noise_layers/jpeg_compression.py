import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 本模块实现可微的 JPEG 压缩噪声层
# 主要功能是：模拟 JPEG 压缩过程对图像造成的失真，包括 RGB->YUV 色彩空间转换、
# 8x8 分块 DCT 变换、按 Zigzag 顺序保留低频系数（高频截断）、IDCT 反变换、YUV->RGB 转换
# 通过卷积层实现 DCT/IDCT，使整个压缩过程可微，便于反向传播训练
# 该噪声层由 noise_argparser.py 中的 jpeg 配置项构造，以实例形式传入 Noiser

def gen_filters(size_x: int, size_y: int, dct_or_idct_fun: callable) -> np.ndarray:
    # 生成 DCT 或 IDCT 的卷积滤波器组
    # 每个 8x8 的滤波器对应一个 DCT/IDCT 基函数，共 size_x*size_y 个滤波器
    # 参数:
    #   size_x: 滤波器在 x 方向的尺寸（通常为 8）
    #   size_y: 滤波器在 y 方向的尺寸（通常为 8）
    #   dct_or_idct_fun: 计算 DCT 或 IDCT 系数的函数
    # 返回:
    #   形状为 (size_x*size_y, size_x, size_y) 的 numpy 数组，即滤波器权重
    tile_size_x = 8  # JPEG 标准的 8x8 分块尺寸
    # 初始化滤波器组：通道数 = size_x*size_y，每个滤波器为 size_x * size_y
    filters = np.zeros((size_x * size_y, size_x, size_y))
    # 遍历所有频率索引 (k_y, k_x)，每个索引对应一个 DCT/IDCT 基函数
    for k_y in range(size_y):
        for k_x in range(size_x):
            # 遍历所有空间位置 (n_y, n_x)
            for n_y in range(size_y):
                for n_x in range(size_x):
                    # 滤波器值为 y 方向和 x 方向 DCT/IDCT 基函数的乘积（可分离性）
                    filters[k_y * tile_size_x + k_x, n_y, n_x] = dct_or_idct_fun(n_y, k_y, size_y) * dct_or_idct_fun(n_x,
                                                                                                            k_x,
                                                                                                            size_x)
    return filters

def get_jpeg_yuv_filter_mask(image_shape: tuple, window_size: int, keep_count: int):
    # 生成 JPEG 量化截断掩码：按 Zigzag 顺序保留前 keep_count 个 DCT 系数，其余置 0
    # 参数:
    #   image_shape: 图像空间尺寸 (H, W)
    #   window_size: DCT 分块尺寸（通常为 8）
    #   keep_count: 每个分块中保留的低频系数个数
    # 返回:
    #   与图像同尺寸的二值掩码，1 表示保留该位置的 DCT 系数
    mask = np.zeros((window_size, window_size), dtype=np.uint8)  # 初始化单个 8x8 掩码为全 0

    # 按 Zigzag 扫描顺序对 8x8 内的所有位置排序
    # 排序键: (x+y, ...)，对角线之和小的在前（低频优先），同对角线上按奇偶性决定方向
    index_order = sorted(((x, y) for x in range(window_size) for y in range(window_size)),
                         key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))

    # 将 Zigzag 顺序前 keep_count 个位置置为 1（保留低频系数）
    for i, j in index_order[0:keep_count]:
        mask[i, j] = 1

    # 将 8x8 掩码平铺到整张图像尺寸（向上取整后裁剪到精确尺寸）
    return np.tile(mask, (int(np.ceil(image_shape[0] / window_size)),
                          int(np.ceil(image_shape[1] / window_size))))[0: image_shape[0], 0: image_shape[1]]


def dct_coeff(n, k, N):
    # 计算 DCT 变换的基函数系数（未归一化形式）
    # 参数:
    #   n: 空间位置索引
    #   k: 频率索引
    #   N: 变换尺寸（通常为 8）
    # 返回:
    #   DCT 基函数在 (n, k) 处的值
    return np.cos(np.pi / N * (n + 1. / 2.) * k)


def idct_coeff(n, k, N):
    # 计算 IDCT（逆 DCT）变换的基函数系数（含归一化）
    # 参数:
    #   n: 空间位置索引
    #   k: 频率索引
    #   N: 变换尺寸（通常为 8）
    # 返回:
    #   IDCT 基函数在 (n, k) 处的值
    return (int(0 == n) * (- 1 / 2) + np.cos(
        np.pi / N * (k + 1. / 2.) * n)) * np.sqrt(1 / (2. * N))


def rgb2yuv(image_rgb):
    # 将 RGB 图像转换为 YUV 色彩空间（使用 BT.601 标准系数）
    # JPEG 压缩在 YUV 空间进行，便于对亮度和色度分量分别处理
    # 参数:
    #   image_rgb: 输入 RGB 图像张量，形状 (batch, 3, H, W)
    # 返回:
    #   YUV 图像张量，形状与输入相同（新建张量，保留梯度连接）
    r = image_rgb[:, 0, :, :]  # R 通道
    g = image_rgb[:, 1, :, :]  # G 通道
    b = image_rgb[:, 2, :, :]  # B 通道
    # Y 分量（亮度）：按 BT.601 标准加权
    y = 0.299 * r + 0.587 * g + 0.114 * b
    # U 分量（色度蓝-黄）
    u = -0.168736 * r - 0.331264 * g + 0.5 * b
    # V 分量（色度红-青）
    v = 0.5 * r - 0.418688 * g - 0.081312 * b
    # 在通道维度上堆叠，返回新张量（避免原地赋值切断梯度流）
    return torch.stack([y, u, v], dim=1)


def yuv2rgb(image_yuv):
    # 将 YUV 图像转换回 RGB 色彩空间（使用 BT.601 标准系数）
    # 参数:
    #   image_yuv: 输入 YUV 图像张量，形状 (batch, 3, H, W)
    # 返回:
    #   RGB 图像张量（新建张量，保留梯度连接）
    y = image_yuv[:, 0, :, :]  # Y 分量
    u = image_yuv[:, 1, :, :]  # U 分量
    v = image_yuv[:, 2, :, :]  # V 分量
    # R 通道
    r = y + 1.402 * v
    # G 通道
    g = y - 0.344 * u - 0.714 * v
    # B 通道
    b = y + 1.772 * u
    # 在通道维度上堆叠，返回新张量
    return torch.stack([r, g, b], dim=1)


class JpegCompression(nn.Module):
    # 可微 JPEG 压缩噪声层
    # 通过卷积实现 DCT/IDCT，并对 DCT 系数按 Zigzag 顺序截断高频，模拟 JPEG 有损压缩
    # 整个流程可微，可在训练中反向传播
    def __init__(self, device, yuv_keep_weights = (25, 9, 9)):
        # 初始化 JPEG 压缩噪声层
        # 参数:
        #   device: 计算设备（cpu 或 cuda）
        #   yuv_keep_weights: Y/U/V 三个通道各自保留的 DCT 低频系数个数，默认 (25, 9, 9)
        #     Y 分量保留更多系数（保留更多亮度细节），U/V 分量保留较少（色度可压缩更多）
        super(JpegCompression, self).__init__()
        self.device = device  # 保存计算设备

        # 生成 DCT 卷积权重并转为张量，形状 (64, 1, 8, 8)，放到指定设备上
        self.dct_conv_weights = torch.tensor(gen_filters(8, 8, dct_coeff), dtype=torch.float32).to(self.device)
        # 在第 1 维增加通道维度，以便作为 conv2d 的卷积核
        self.dct_conv_weights.unsqueeze_(1)
        # 生成 IDCT 卷积权重并转为张量，形状 (64, 1, 8, 8)
        self.idct_conv_weights = torch.tensor(gen_filters(8, 8, idct_coeff), dtype=torch.float32).to(self.device)
        # 在第 1 维增加通道维度
        self.idct_conv_weights.unsqueeze_(1)

        self.yuv_keep_weights = yuv_keep_weights  # 保存各通道保留系数个数

        self.jpeg_mask = None  # 当前缓存的 JPEG 截断掩码

        # 预先生成一个较大的掩码 (1000x1000)，后续可通过切片复用于较小图像
        self.create_mask((1000, 1000))


    def create_mask(self, requested_shape):
        # 创建或重建 JPEG 截断掩码
        # 当 requested_shape 大于当前缓存掩码尺寸时，重新生成掩码
        # 参数:
        #   requested_shape: 所需掩码的空间尺寸 (H, W)
        if self.jpeg_mask is None or requested_shape[0] > self.jpeg_mask.shape[1] or requested_shape[1] > self.jpeg_mask.shape[2]:
            # 初始化掩码张量，形状 (3, H, W)，分别对应 Y/U/V 三个通道
            self.jpeg_mask = torch.empty((3,) + requested_shape, device=self.device)
            # 为每个通道生成对应的 Zigzag 截断掩码
            for channel, weights_to_keep in enumerate(self.yuv_keep_weights):
                mask = torch.from_numpy(get_jpeg_yuv_filter_mask(requested_shape, 8, weights_to_keep))
                self.jpeg_mask[channel] = mask

    def get_mask(self, image_shape):
        # 获取指定尺寸的 JPEG 截断掩码，若缓存不够大则重建
        # 参数:
        #   image_shape: 所需掩码的空间尺寸 (H, W)
        # 返回:
        #   形状为 (3, H, W) 的掩码张量
        if self.jpeg_mask.shape[1] < image_shape[0] or self.jpeg_mask.shape[2] < image_shape[1]:
            # 当前缓存掩码尺寸不足，重建
            self.create_mask(image_shape)
        # 从缓存中切片出所需尺寸的掩码并克隆返回
        return self.jpeg_mask[:, :image_shape[0], :image_shape[1]].clone()


    def apply_conv(self, image, filter_type: str):
        # 对图像应用 DCT 或 IDCT 卷积变换
        # 卷积步长为 8，等价于对 8x8 分块进行变换
        # 参数:
        #   image: 输入图像张量，形状 (batch, 3, H, W)
        #   filter_type: 'dct' 表示正向 DCT，'idct' 表示逆 DCT
        # 返回:
        #   变换后的图像张量，形状 (batch, 3, H, W)（其中 H/W 为分块数 * 8）

        # 根据滤波器类型选择对应的卷积核
        if filter_type == 'dct':
            filters = self.dct_conv_weights
        elif filter_type == 'idct':
            filters = self.idct_conv_weights
        else:
            raise ValueError('Unknown filter_type value.')

        image_conv_channels = []  # 用于存储每个通道卷积后的结果
        # 对每个通道独立进行卷积
        for channel in range(image.shape[1]):
            # 取出当前通道，并增加一个通道维度以适配 conv2d 输入
            # 使用非原地 unsqueeze 避免修改视图 strides 导致 autograd 版本冲突
            image_yuv_ch = image[:, channel, :, :].unsqueeze(1)
            # 以步长 8 进行卷积，输出 64 个通道（对应 8x8 个 DCT 系数）
            image_conv = F.conv2d(image_yuv_ch, filters, stride=8)
            # 调整维度顺序：将通道维度移到最后 -> (batch, H', W', 64)
            image_conv = image_conv.permute(0, 2, 3, 1)
            # 将最后一维 64 重塑为 (8, 8)，得到 (batch, H', W', 8, 8)
            image_conv = image_conv.view(image_conv.shape[0], image_conv.shape[1], image_conv.shape[2], 8, 8)
            # 调整维度：将 8x8 块的维度提前 -> (batch, H', 8, W', 8)
            image_conv = image_conv.permute(0, 1, 3, 2, 4)
            # 将块维度合并回空间维度 -> (batch, H'*8, W'*8)，即还原为图像空间尺寸
            image_conv = image_conv.contiguous().view(image_conv.shape[0],
                                                  image_conv.shape[1]*image_conv.shape[2],
                                                  image_conv.shape[3]*image_conv.shape[4])

            # 增加通道维度，以便后续按通道拼接
            # 使用非原地 unsqueeze 避免原地修改破坏计算图
            image_conv = image_conv.unsqueeze(1)

            image_conv_channels.append(image_conv)

        # 在通道维度上拼接所有通道的结果
        image_conv_stacked = torch.cat(image_conv_channels, dim=1)

        return image_conv_stacked


    def forward(self, noised_and_cover):
        # 前向传播：执行可微 JPEG 压缩流程
        # 流程：RGB -> 补零到 8 的倍数 -> YUV -> DCT -> 高频截断 -> IDCT -> RGB -> 去补零
        # 参数:
        #   noised_and_cover: 包含 [noised_image, cover_image] 的列表/元组
        # 返回:
        #   元组 (压缩后的噪声图像, cover_image)

        noised_image = noised_and_cover[0]  # 取出噪声图像
        # 计算高度方向需要补零的像素数，使高度为 8 的倍数（便于 8x8 分块 DCT）
        pad_height = (8 - noised_image.shape[2] % 8) % 8
        # 计算宽度方向需要补零的像素数
        pad_width = (8 - noised_image.shape[3] % 8) % 8

        # 在右下方向补零
        noised_image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(noised_image)

        # 将 RGB 图像转换为 YUV 色彩空间
        image_yuv = rgb2yuv(noised_image)

        # 断言补零后图像尺寸是 8 的倍数
        assert image_yuv.shape[2] % 8 == 0
        assert image_yuv.shape[3] % 8 == 0

        # 对 YUV 图像进行 DCT 变换
        image_dct = self.apply_conv(image_yuv, 'dct')
        # 获取 JPEG 压缩用的高频截断掩码
        # apply_conv 返回 (batch, 3, H, W)（经过 view 操作后）
        # 每个 8x8 块的 64 个系数在空间上连续存储
        mask = self.get_mask((image_dct.shape[2], image_dct.shape[3]))
        # mask 形状: (3, H, W)
        # image_dct 形状: (batch, 3, H, W)
        # 扩展 mask 以匹配 image_dct 进行广播
        mask = mask.unsqueeze(0).expand(image_dct.shape[0], -1, -1, -1)  # (batch, 3, H, W)
        # 将 DCT 系数与掩码相乘，截断高频系数
        image_dct_mask = torch.mul(image_dct, mask)

        # 对截断后的 DCT 系数进行逆 DCT 变换
        image_idct = self.apply_conv(image_dct_mask, 'idct')
        # 将 YUV 转换回 RGB
        image_ret_padded = yuv2rgb(image_idct)

        # 去除补零部分，恢复原始尺寸
        noised_image = image_ret_padded[:, :, :image_ret_padded.shape[2]-pad_height, :image_ret_padded.shape[3]-pad_width].clone()

        # 返回压缩后的噪声图像和原始覆盖图像
        return (noised_image, noised_and_cover[1])
