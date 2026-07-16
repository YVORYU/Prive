# ==============================================================================
# 模块：消息解码器（Decoder）
# 职责：从编码图像中预测二进制消息的 logits。
#       训练损失使用 BCEWithLogitsLoss，因此本模块输出的 logits 在计算训练损失前
#       不要额外加 sigmoid（sigmoid 已内置于 BCEWithLogitsLoss 中以提升数值稳定性）。
# 架构：多层 ConvBNRelu 堆叠 + 自适应平均池化 + 全连接层，将图像压缩为消息位 logits。
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    # 卷积 + 批归一化 + ReLU 的组合模块，作为解码器的基本构建单元
    # bias=False 因为后接 BatchNorm，偏置可被 BN 吸收

    def __init__(self, in_channels, out_channels):
        # 构造函数：初始化 ConvBNRelu 模块
        # 参数：
        #   in_channels (int): 输入通道数
        #   out_channels (int): 输出通道数
        super().__init__()
        # 将 3x3 卷积、BN、ReLU 串联为一个子模块
        self.conv = nn.Sequential(
            # 3x3 卷积，padding=1 保持空间尺寸，bias=False 因为后接 BN
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 批归一化，提升训练稳定性
            nn.BatchNorm2d(out_channels),
            # ReLU 激活函数，inplace=True 节省显存
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 前向传播：依次应用卷积、BN、ReLU
        # 参数：
        #   x (Tensor): 输入特征图
        # 返回：
        #   Tensor: 经 Conv-BN-ReLU 处理后的特征图
        return self.conv(x)


class Decoder(nn.Module):
    # 消息解码器主类
    # 职责：从编码图像中提取并预测二进制消息的 logits（未经 sigmoid）

    def __init__(self, message_length=30, n_channels=3):
        # 构造函数：初始化解码器
        # 参数：
        #   message_length (int): 二进制消息位数，默认 30
        #   n_channels (int): 输入图像通道数，默认 3（RGB）
        super(Decoder, self).__init__()
        # 保存消息位数，决定最终输出维度
        self.message_length = message_length
        # 保存输入图像通道数
        self.n_channels = n_channels

        # 解码器中间特征通道数
        decoder_channels = 64
        # 解码器卷积块数量
        decoder_blocks = 8

        # 构建卷积层列表：首个块将 3 通道图像映射到 decoder_channels 维特征
        layers = [ConvBNRelu(3, decoder_channels)]
        # 追加 (decoder_blocks - 1) 个 ConvBNRelu 块，提取深层特征
        for _ in range(decoder_blocks - 1):
            layers.append(ConvBNRelu(decoder_channels, decoder_channels))

        # 最后一个卷积块将通道数降到 message_length，每个通道对应一位消息
        layers.append(ConvBNRelu(decoder_channels, message_length))

        # 自适应平均池化将空间维度压缩到 (1, 1)，得到每个通道的全局表征
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        # 将所有层封装为 nn.Sequential
        self.layers = nn.Sequential(*layers)

        # 全连接层：对池化后的特征做线性变换，输出最终的消息 logits
        self.linear = nn.Linear(message_length, message_length)

    def forward(self, encoded_image):
        # 前向传播：从编码图像预测消息 logits
        # 参数：
        #   encoded_image (Tensor): 编码图像，形状 (B, 3, H, W)，取值范围 [-1, 1]
        # 返回：
        #   Tensor: 消息 logits，形状 (B, message_length)
        #        （注意：未经过 sigmoid，配合 BCEWithLogitsLoss 使用）

        # 通过卷积层和池化层提取特征，输出形状 (B, message_length, 1, 1)
        x = self.layers(encoded_image)
        # 移除空间维度，形状变为 (B, message_length)
        x = x.squeeze(-1).squeeze(-1)
        # 通过全连接层输出最终的消息 logits
        x = self.linear(x)
        return x
