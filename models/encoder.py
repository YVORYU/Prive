# ==============================================================================
# 模块：双残差编码器（Dual-Residual Encoder）
# 职责：在载体图像（cover_image）中同时嵌入两类残差：
#   1. msg_residual：消息残差，用于携带二进制消息，由 msg_head 输出并受 msg_budget 约束；
#   2. prv_residual：隐私残差，用于偏移 CLIP/DINOv2 等内容特征以降低 CBIR 检索关联风险，
#      由 prv_head 输出并受 prv_budget 约束，且会乘以梯度驱动的纹理掩码以集中扰动。
#
# 训练流程：
#   - Phase 1：msg_head 学习基础隐写能力；prv_head 冻结，注入经平滑和零均值单位方差归一化的
#     随机噪声作为预适应（dummy_prv），privacy loss 权重为 0。
#   - Phase 2：冻结 msg_head，prv_head 学习隐私扰动；通过纹理掩码将扰动集中在纹理复杂区域。
#   - 编码图像最终会被 clamp 到 [-1, 1] 区间，保证像素合法。
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    # 卷积 + 批归一化 + ReLU 的组合模块，作为编码器的基本构建单元
    # 用于提取图像特征和共享表征，bias=False 是因为后接 BatchNorm，偏置可被吸收

    def __init__(self, in_channels, out_channels):
        # 构造函数：初始化 ConvBNRelu 模块
        # 参数：
        #   in_channels (int): 输入通道数
        #   out_channels (int): 输出通道数
        super().__init__()
        # 使用 nn.Sequential 将 3x3 卷积、BN、ReLU 串联成一个不可拆分的子模块
        self.conv = nn.Sequential(
            # 3x3 卷积，padding=1 保持空间尺寸，bias=False 因为后接 BN
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 批归一化，加速收敛并提升训练稳定性
            nn.BatchNorm2d(out_channels),
            # ReLU 激活，inplace=True 节省显存
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 前向传播：对输入张量依次应用卷积、BN、ReLU
        # 参数：
        #   x (Tensor): 输入特征图
        # 返回：
        #   Tensor: 经 Conv-BN-ReLU 处理后的特征图
        return self.conv(x)


class Encoder(nn.Module):
    # 双残差编码器主类
    # 职责：将消息和隐私信息同时嵌入载体图像，输出编码图像以及对应的残差和正则项

    def __init__(self, message_length=30, n_channels=3, bilinear=False,
                 image_size=256, msg_budget=0.05, prv_budget=0.03,
                 phase1_prv_noise_smooth_kernel=7,
                 prv_residual_smooth_kernel=1, prv_residual_smooth_sigma=1.0,
                 texture_mask_strength=5.0, texture_mask_smooth_kernel=5):
        # 构造函数：初始化编码器
        # 参数：
        #   message_length (int): 二进制消息位数，默认 30
        #   n_channels (int): 图像通道数，默认 3（RGB）
        #   bilinear (bool): 是否使用双线性上采样（保留接口，当前未使用）
        #   image_size (int): 输入图像的空间尺寸（正方形），默认 256
        #   msg_budget (float): 消息残差的预算上界，控制嵌入强度
        #   prv_budget (float): 隐私残差的预算上界，控制扰动强度
        #   phase1_prv_noise_smooth_kernel (int): Phase 1 随机隐私噪声的平滑核大小
        #   prv_residual_smooth_kernel (int): 隐私残差平滑核大小（保留接口）
        #   prv_residual_smooth_sigma (float): 隐私残差平滑高斯 sigma（保留接口）
        #   texture_mask_strength (float): 纹理掩码 sigmoid 的缩放强度
        #   texture_mask_smooth_kernel (int): 纹理掩码梯度图的平滑核大小
        super(Encoder, self).__init__()
        # 保存消息位数，用于扩展消息到空间维度
        self.message_length = message_length
        # 保存图像通道数
        self.n_channels = n_channels
        # 保存是否使用双线性上采样标志（保留接口）
        self.bilinear = bilinear
        # 图像高度，等于 image_size
        self.H = image_size
        # 图像宽度，等于 image_size
        self.W = image_size
        # 消息残差预算，用于 tanh 缩放
        self.msg_budget = msg_budget
        # 隐私残差预算，用于 tanh 缩放
        self.prv_budget = prv_budget
        # Phase 1 随机噪声平滑核大小
        self.phase1_prv_noise_smooth_kernel = phase1_prv_noise_smooth_kernel
        # 纹理掩码 sigmoid 强度系数
        self.texture_mask_strength = texture_mask_strength
        # 纹理掩码梯度图平滑核大小
        self.texture_mask_smooth_kernel = texture_mask_smooth_kernel

        # 预留的隐私残差平滑核张量缓存（当前未使用，初始化为 None）
        self._prv_smooth_kernel = None

        # 共享特征通道数
        channels = 128
        # 共享卷积块数量
        num_blocks = 16

        # 共享卷积层：首个块将 3 通道图像映射到 channels 维特征
        self.conv_layers = nn.Sequential(
            ConvBNRelu(3, channels)
        )
        # 追加 (num_blocks - 1) 个 ConvBNRelu 块，逐步提取深层特征
        for _ in range(num_blocks - 1):
            self.conv_layers.append(ConvBNRelu(channels, channels))

        # 拼接后的融合层：输入维度 = 消息 + 共享特征 + 原图，输出 channels 维共享表征
        self.after_concat_layer = ConvBNRelu(channels + 3 + message_length, channels)

        # 消息头：从共享特征预测消息残差，最后一层用 1x1 卷积输出 3 通道
        self.msg_head = nn.Sequential(
            # 3x3 卷积，保持通道数
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            # 批归一化
            nn.BatchNorm2d(channels),
            # ReLU 激活
            nn.ReLU(inplace=True),
            # 第二个 3x3 卷积，进一步非线性变换
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            # 批归一化
            nn.BatchNorm2d(channels),
            # ReLU 激活
            nn.ReLU(inplace=True),
            # 1x1 卷积，将通道数降为 3，输出与图像同维度的残差
            nn.Conv2d(channels, 3, kernel_size=1)
        )

        # 隐私头：结构与 msg_head 相同，用于预测隐私残差
        # Phase 1 中此模块被冻结，Phase 2 中解冻参与训练
        self.prv_head = nn.Sequential(
            # 3x3 卷积
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            # 批归一化
            nn.BatchNorm2d(channels),
            # ReLU 激活
            nn.ReLU(inplace=True),
            # 第二个 3x3 卷积
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            # 批归一化
            nn.BatchNorm2d(channels),
            # ReLU 激活
            nn.ReLU(inplace=True),
            # 1x1 卷积，输出 3 通道隐私残差
            nn.Conv2d(channels, 3, kernel_size=1)
        )

    def _compute_texture_mask(self, cover_image):
        # 计算梯度驱动的纹理掩码，用于将隐私扰动集中在纹理复杂区域
        # 参数：
        #   cover_image (Tensor): 载体图像，形状 (B, 3, H, W)
        # 返回：
        #   Tensor: 纹理掩码，形状 (B, 1, H, W)，取值范围 [0.15, 1.0]
        # 在无梯度上下文中计算，掩码本身不参与反向传播
        with torch.no_grad():
            # 将 RGB 转为单通道灰度图，用于计算图像梯度
            gray = cover_image.mean(dim=1, keepdim=True)
            # 垂直方向梯度（高度方向差分），形状 (B,1,H-1,W)
            grad_h = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
            # 水平方向梯度（宽度方向差分），形状 (B,1,H,W-1)
            grad_w = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
            # 对垂直梯度在高度方向补 0，恢复到 (B,1,H,W)
            grad_h = F.pad(grad_h, (0, 0, 0, 1))
            # 对水平梯度在宽度方向补 0，恢复到 (B,1,H,W)
            grad_w = F.pad(grad_w, (0, 1, 0, 0))
            # 总梯度 = 垂直 + 水平，反映该像素的纹理复杂度
            grad = grad_h + grad_w
            # 取纹理掩码平滑核大小
            k = self.texture_mask_smooth_kernel
            # 当核大小 > 1 时对梯度图进行平滑，避免掩码过于尖锐
            if k > 1:
                # avg_pool2d 要求核大小为奇数，若为偶数则 +1
                if k % 2 == 0:
                    k += 1
                # 使用平均池化平滑梯度图，stride=1 保持尺寸
                grad = F.avg_pool2d(grad, kernel_size=k, stride=1, padding=k // 2)
            # 通过 sigmoid 将梯度映射到 [0,1]，strength 控制掩码锐度
            mask = torch.sigmoid(grad * self.texture_mask_strength)
            # 将掩码下界 clamp 到 0.15，保证平滑区域仍保留一定扰动
            mask = torch.clamp(mask, min=0.15, max=1.0)
        return mask

    def forward(self, cover_image, message, phase=1, prv_noise_ratio=0.0):
        # 前向传播：根据当前阶段生成编码图像
        # 参数：
        #   cover_image (Tensor): 载体图像，形状 (B, 3, H, W)，取值范围 [-1, 1]
        #   message (Tensor): 二进制消息，形状 (B, message_length)
        #   phase (int): 训练阶段，1 或 2
        #   prv_noise_ratio (float): Phase 1 中随机隐私噪声的注入比例，渐进增大
        # 返回：
        #   encoded_image (Tensor): 编码后图像，形状同 cover_image
        #   residual_penalty (Tensor): 残差 L2 正则项，用于约束嵌入强度
        #   msg_residual (Tensor): 消息残差
        #   prv_residual (Tensor or None): 隐私残差；Phase 1 返回 None

        # 将消息 (B, L) 扩展到空间维度 (B, L, H, W)，与图像特征对齐
        expanded_message = message.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.H, self.W)

        # 通过共享卷积层提取载体图像特征
        features = self.conv_layers(cover_image)
        # 在通道维拼接 [消息, 共享特征, 原图]，作为融合层输入
        concat = torch.cat([expanded_message, features, cover_image], dim=1)
        # 融合层输出共享表征 shared，供 msg_head 和 prv_head 使用
        shared = self.after_concat_layer(concat)

        # ===== 消息残差计算 =====
        # Phase 2 且 msg_head 被冻结时，在 no_grad 下计算以节省显存
        if phase == 2 and not self.msg_head[0].weight.requires_grad:
            with torch.no_grad():
                # 通过 msg_head 预测原始消息残差
                msg_residual = self.msg_head(shared)
                # 使用 tanh 将残差约束在 [-msg_budget, msg_budget] 范围内
                msg_residual = self.msg_budget * torch.tanh(msg_residual)
        else:
            # Phase 1 或 msg_head 未冻结时正常计算，保留梯度用于反传
            msg_residual = self.msg_head(shared)
            # 同样用 tanh 缩放到预算范围内
            msg_residual = self.msg_budget * torch.tanh(msg_residual)

        # 隐私残差初始化为 None，Phase 1 不返回真实隐私残差
        prv_residual = None
        if phase == 1:
            # ===== Phase 1：注入随机隐私噪声作为预适应 =====
            # 生成与载体图像同形状的标准正态随机噪声
            noise = torch.randn_like(cover_image)
            # 取 Phase 1 噪声平滑核大小
            kernel = self.phase1_prv_noise_smooth_kernel
            # 当核 > 1 时对噪声进行空间平滑，使噪声更具结构性而非纯像素级
            if kernel > 1:
                # 强制核大小为奇数
                if kernel % 2 == 0:
                    kernel += 1
                # 使用平均池化平滑噪声
                noise = F.avg_pool2d(noise, kernel_size=kernel, stride=1, padding=kernel // 2)
            # 对每个样本的空间维度做零均值化
            noise = noise - noise.mean(dim=(2, 3), keepdim=True)
            # 对每个样本的空间维度做单位方差归一化，加 1e-6 防止除零
            noise = noise / (noise.std(dim=(2, 3), keepdim=True) + 1e-6)
            # 缩放到隐私预算范围内，并乘以渐进比例 prv_noise_ratio
            dummy_prv = self.prv_budget * prv_noise_ratio * torch.tanh(noise)
            # 编码图像 = 原图 + 消息残差 + 随机隐私噪声
            encoded_image = cover_image + msg_residual + dummy_prv
            # 计算 Phase 1 残差正则项（消息残差 + 噪声残差的 L2 均值）
            residual_penalty = torch.mean(msg_residual ** 2) + torch.mean(dummy_prv ** 2)
        else:
            # ===== Phase 2：学习真实隐私残差 =====
            # 通过 prv_head 预测隐私残差
            prv_residual = self.prv_head(shared)
            # 使用 tanh 约束到 [-prv_budget, prv_budget]
            prv_residual = self.prv_budget * torch.tanh(prv_residual)
            # 计算纹理掩码，将扰动集中在纹理复杂区域
            texture_mask = self._compute_texture_mask(cover_image)
            # 将隐私残差逐像素乘以纹理掩码
            prv_residual = prv_residual * texture_mask
            # 编码图像 = 原图 + 消息残差 + 隐私残差
            encoded_image = cover_image + msg_residual + prv_residual
            # 计算 Phase 2 残差正则项（消息残差 + 隐私残差的 L2 均值）
            residual_penalty = torch.mean(msg_residual ** 2) + torch.mean(prv_residual ** 2)

        # 将编码图像 clamp 到 [-1, 1]，保证像素值合法
        encoded_image = torch.clamp(encoded_image, -1, 1)

        # 返回编码图像、残差正则项、消息残差、隐私残差（Phase 1 为 None）
        return encoded_image, residual_penalty, msg_residual, prv_residual
