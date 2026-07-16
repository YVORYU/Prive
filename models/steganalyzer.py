# ==============================================================================
# 模块：隐写分析器（Steganalyzer，判别器）
# 职责：作为 GAN 中的判别器式二分类器，区分输入图像是「原始载体图像」还是「含密编码图像」。
#       使用谱归一化（spectral normalization）的卷积层稳定对抗训练，输出 logits
#       （未经过 sigmoid，配合 Hinge Loss 使用）。
# 架构：多层卷积下采样 + 全局平均池化 + 全连接层，最终输出单标量 logit。
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class Steganalyzer(nn.Module):
    # 隐写分析器主类（判别器）
    # 输入通道数为 3，使用谱归一化有助于训练稳定性
    def __init__(self, n_channels=3, use_spectral_norm=True):
        # 构造函数：初始化隐写分析器
        # 参数：
        #   n_channels (int): 输入图像通道数，默认 3（RGB）
        #   use_spectral_norm (bool): 是否对卷积层应用谱归一化，默认 True
        super(Steganalyzer, self).__init__()
        # 保存输入通道数
        self.n_channels = n_channels

        # 默认使用普通卷积层
        conv = nn.Conv2d
        # 如果使用谱归一化，定义一个 lambda 函数，用于在卷积层后应用谱归一化
        if use_spectral_norm:
            # 该 lambda 接收输入/输出通道数及其他关键字参数，返回谱归一化后的 Conv2d
            conv = lambda c_in, c_out, **kwargs: nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, **kwargs))

        # 卷积层架构
        # 每个卷积层后跟一个批量归一化层和一个 LeakyReLU 激活函数和一个池化层
        # 最后一个卷积层后跟一个全局平均池化层
        self.conv_layers = nn.Sequential(
            # 第 1 个卷积块：3 -> 64 通道
            conv(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 第 2 个卷积块：64 -> 64 通道，后接 2x 最大池化下采样
            conv(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 空间尺寸减半
            nn.MaxPool2d(2),

            # 第 3 个卷积块：64 -> 128 通道
            conv(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 第 4 个卷积块：128 -> 128 通道，后接 2x 最大池化下采样
            conv(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 空间尺寸减半
            nn.MaxPool2d(2),

            # 第 5 个卷积块：128 -> 256 通道
            conv(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 第 6 个卷积块：256 -> 256 通道，后接 2x 最大池化下采样
            conv(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 空间尺寸减半
            nn.MaxPool2d(2),

            # 第 7 个卷积块：256 -> 512 通道
            conv(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 第 8 个卷积块：512 -> 512 通道，后接自适应平均池化到 (1, 1)
            conv(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 全局平均池化，将空间维度压缩到 1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 全连接层架构
        # 第一个全连接层后跟一个 LeakyReLU 激活函数
        # 第二个全连接层输出单个 logit（无 sigmoid，配合 Hinge Loss 使用）
        self.fc_layers = nn.Sequential(
            # 512 -> 256 维全连接
            nn.Linear(512, 256),
            # LeakyReLU 激活，负斜率 0.2
            nn.LeakyReLU(0.2, inplace=True),
            # 256 -> 1 维全连接，输出二分类 logit
            nn.Linear(256, 1)
        )

    def forward(self, image):
        # 前向传播：对输入图像进行隐写分析，输出二分类 logits
        # 参数：
        #   image (Tensor): 输入图像，形状 (B, n_channels, H, W)
        # 返回：
        #   Tensor: 二分类 logits，形状 (B, 1)
        # 先通过卷积层，提取消息特征图
        x = self.conv_layers(image)
        # 将特征图展平为 (B, C) 形状，供全连接层使用
        x = x.view(x.size(0), -1)
        # 通过全连接层，输出判断结果 logits
        output = self.fc_layers(x)
        return output
