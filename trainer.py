# 模块职责：
# trainer.py 是 PrivSteg 项目的训练核心模块，负责以下任务：
#   1. 初始化编码器、解码器、判别器（隐写分析器）、内容识别器（CLIP/DINOv2）等模型
#   2. 构建优化器、损失函数、数据加载器、噪声层
#   3. 执行两阶段训练：Phase 1 训练基础隐写与对抗能力；Phase 2 冻结 msg_head 训练隐私扰动
#   4. 维护 CLIP/DINO 特征的 memory bank（用于 hard negative mining）
#   5. 提供 checkpoint 保存、TensorBoard 日志、可视化样本保存等辅助功能
#   6. 在 Phase 2 验证 bit accuracy 低于阈值时触发 decoder recovery 步骤

import os  # 操作系统相关接口，用于路径拼接与文件存在性检查
import torch  # PyTorch 主库，张量计算与自动求导
import torch.nn as nn  # 神经网络模块，提供损失函数等
import torch.nn.functional as F  # 函数式接口，提供 relu、normalize、interpolate 等
import torch.optim as optim  # 优化器模块，Adam 等
import timm  # PyTorch Image Models，用于加载 DINOv2 ViT-S/14 模型
from torch.utils.data import DataLoader  # 数据加载器，支持批量、打乱、多进程
import numpy as np  # 数值计算（虽然本文件未直接使用，但保留以备扩展）
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 日志写入器
from torchvision.utils import save_image, make_grid  # 图像保存与网格拼接工具

# 导入项目内部的模型、损失、数据集与噪声模块
from models import Encoder, Decoder, Steganalyzer, ContentRecognizer  # 编码器、解码器、判别器、CLIP 内容识别器
from losses import SSIMLoss, FeatureRedirectionLoss  # SSIM 损失与 CLIP/DINO 特征偏移损失
from data import SteganographyDataset  # 隐写数据集，递归加载图像并生成随机消息
from noise_layers.noiser import Noiser  # 噪声合成器，按概率施加信道噪声
import config  # 全局配置模块，定义 Config 类


# 特征记忆库：用于存储历史载体图像的 CLIP/DINO 特征向量，
# 在 Phase 2 训练中作为 hard negative 池，促使编码图像的特征远离
# 这些"接近原始载体"的特征，从而降低 CBIR 图像检索关联风险。
class FeatureMemoryBank:
    # 构造函数：初始化固定大小的环形特征缓冲区
    # 参数:
    #   feature_dim: 单条特征向量维度（CLIP=512, DINOv2 ViT-S/14=384）
    #   memory_size: 缓冲区容量（条数），默认 1000
    #   device: 张量所在设备
    def __init__(self, feature_dim, memory_size=1000, device='cuda'):
        self.feature_dim = feature_dim  # 记录特征维度
        self.memory_size = memory_size  # 记录容量上限
        self.device = device  # 记录设备
        # 预分配容量 x 维度的零张量，作为环形缓冲区底层存储
        self.memory_bank = torch.zeros(memory_size, feature_dim, device=device)
        self.memory_ptr = 0  # 当前写入位置的环形指针
        self.memory_count = 0  # 当前已存入的有效特征数量（不超过 memory_size）

    # 更新记忆库：将一批新特征写入环形缓冲区
    # 参数:
    #   features: 形如 [B, D] 的特征张量，会被 L2 归一化后写入
    # 不返回任何值；更新内部指针与计数
    @torch.no_grad()
    def update(self, features):
        # 先做 L2 归一化，确保后续相似度比较在单位球面上进行
        features = F.normalize(features.float(), p=2, dim=1)
        batch_size = features.shape[0]  # 当前批次的样本数
        ptr = self.memory_ptr  # 起始写入位置

        # 若写入会越过末尾，则分两段写入以实现环形回绕
        if ptr + batch_size > self.memory_size:
            remaining = self.memory_size - ptr  # 末尾剩余可写入条数
            self.memory_bank[ptr:] = features[:remaining]  # 写入末尾
            self.memory_bank[:batch_size - remaining] = features[remaining:]  # 剩余从头写入
            self.memory_ptr = batch_size - remaining  # 更新指针到回绕后位置
        else:
            # 未越界则直接连续写入
            self.memory_bank[ptr:ptr + batch_size] = features
            self.memory_ptr = (ptr + batch_size) % self.memory_size  # 指针前进并取模

        # 有效计数单调递增但不超过容量上限
        self.memory_count = min(self.memory_count + batch_size, self.memory_size)

    # 检索 hard negatives：对每个查询特征，返回记忆库中与之最相似的 k 个特征
    # 参数:
    #   features: 查询特征张量 [B, D]（应为归一化后的 CLIP/DINO 特征）
    #   k: 每个查询返回的 hard negative 数量，默认 5
    # 返回:
    #   [B, k, D] 的 hard negative 特征张量；若记忆库为空则返回 None
    def get_hard_negatives(self, features, k=5):
        # 记忆库尚未填充时直接返回 None，调用方需判空
        if self.memory_count == 0:
            return None

        features = features.float()  # 转为 float（查询特征也应是归一化的）
        # 仅在有效范围内取记忆特征，避免使用尚未填充的零向量
        valid_memory = self.memory_bank[:self.memory_count]
        # 通过矩阵乘法计算查询与每个记忆特征的余弦相似度（因为已归一化）
        similarities = torch.mm(features, valid_memory.t())
        # k 不能超过当前记忆库实际容量
        actual_k = min(k, self.memory_count)
        # 取相似度最大的 k 个索引（这些是最"危险"的 hard negative）
        _, indices = similarities.topk(actual_k, dim=1)
        # 根据索引从记忆库中取出对应特征
        hard_negatives = valid_memory[indices]

        return hard_negatives

    # 导出记忆库状态字典，用于 checkpoint 持久化
    # 返回: 包含 memory_bank/cpu 张量、指针、计数的字典
    def state_dict(self):
        return {
            'memory_bank': self.memory_bank.cpu(),  # 移到 CPU 以避免设备相关保存问题
            'memory_ptr': self.memory_ptr,
            'memory_count': self.memory_count
        }

    # 从状态字典恢复记忆库
    # 参数: state_dict - 之前由 state_dict() 导出的字典
    def load_state_dict(self, state_dict):
        # 把保存的张量移回当前设备
        self.memory_bank = state_dict['memory_bank'].to(self.device)
        self.memory_ptr = state_dict['memory_ptr']
        self.memory_count = state_dict['memory_count']


# 训练器：PrivSteg 训练流程的核心类，封装两阶段训练所需的所有组件与流程。
# 职责包括：
#   - 实例化 Encoder/Decoder/Steganalyzer/ContentRecognizer 与可选的 DINOv2
#   - 配置 Adam 优化器、SSIM/MSE/BCE/特征偏移等损失
#   - 解析噪声配置并构建 Noiser
#   - 提供 train_step / recovery_step / validate / save_checkpoint 等方法
class Trainer:
    # 构造函数：根据传入的 Config 实例完成所有模型、优化器、损失与噪声层的初始化
    # 参数:
    #   cfg: config.Config 实例，包含所有训练超参与路径配置
    def __init__(self, cfg):
        self.cfg = cfg  # 保存配置对象，后续 step 中会读取阶段权重等
        self.device = cfg.device  # 训练设备（cuda 或 cpu）
        # 创建 TensorBoard 写入器，日志输出到 cfg.log_dir
        self.writer = SummaryWriter(cfg.log_dir)

        # 实例化编码器：双残差结构，输出 msg_residual（消息扰动）与 prv_residual（隐私扰动）
        self.encoder = Encoder(
            message_length=cfg.message_length,  # 二进制消息位数（默认 30）
            n_channels=3,  # RGB 三通道
            image_size=cfg.image_size,  # 图像分辨率（默认 256）
            msg_budget=cfg.msg_budget,  # 消息残差预算（L∞ 限制，默认 0.04）
            prv_budget=cfg.prv_budget,  # 隐私残差预算
            phase1_prv_noise_smooth_kernel=cfg.phase1_prv_noise_smooth_kernel,  # Phase1 隐私噪声平滑核大小
            texture_mask_strength=cfg.texture_mask_strength,  # 纹理掩码强度
            texture_mask_smooth_kernel=cfg.texture_mask_smooth_kernel  # 纹理掩码平滑核大小
        ).to(self.device)
        # 解码器：从（含噪）编码图像预测 message logits
        self.decoder = Decoder(message_length=cfg.message_length, n_channels=3).to(self.device)
        # 判别器（隐写分析器）：spectral norm 卷积分类器，输出真实/隐写 logits
        self.steganalyzer = Steganalyzer(n_channels=3, use_spectral_norm=True).to(self.device)
        # 内容识别器：冻结的 CLIP ViT-B/32，用于提取图像/文本特征并做归一化
        self.content_recognizer = ContentRecognizer(device=self.device).to(self.device)

        # 编码器与解码器共用一个 Adam 优化器（不同学习率）
        self.optimizer_encoder_decoder = optim.Adam([
            {"params": self.encoder.parameters(), "lr": cfg.lr_encoder},  # 编码器学习率
            {"params": self.decoder.parameters(), "lr": cfg.lr_decoder},  # 解码器学习率
        ])
        # 判别器单独的 Adam 优化器
        self.optimizer_steganalyzer = optim.Adam(self.steganalyzer.parameters(), lr=cfg.lr_steganalyzer)

        # 各类损失函数实例化并放到训练设备上
        self.ssim_loss = SSIMLoss().to(self.device)  # 结构相似度损失，衡量视觉质量
        self.mse_loss = nn.MSELoss()  # 像素级均方误差
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()  # 消息位损失（结合 sigmoid）

        # 特征偏移损失：组合 CLIP/DINO escape loss 与可选 hard negative redirection
        self.retrieval_escape_loss = FeatureRedirectionLoss(
            clip_weight=cfg.clip_weight,  # CLIP escape 权重
            dino_weight=cfg.dino_weight,  # DINO escape 权重（0 表示不使用）
            escape_margin=cfg.privacy_margin_start,  # escape hinge 的初始 margin
            clip_redirect_weight=cfg.clip_redirect_weight,  # CLIP hard negative redirect 权重
            dino_redirect_weight=cfg.dino_redirect_weight,  # DINO hard negative redirect 权重
            clip_redirect_margin=cfg.clip_redirect_margin,  # CLIP redirect margin
            dino_redirect_margin=cfg.dino_redirect_margin,  # DINO redirect margin
            clip_max_target_sim=cfg.clip_max_target_sim,  # CLIP 目标相似度上限
            dino_max_target_sim=cfg.dino_max_target_sim  # DINO 目标相似度上限
        ).to(self.device)

        self.use_clip_redirect = cfg.use_clip_redirect  # 是否启用 CLIP/DINO hard negative redirect
        self.clip_memory_bank = None  # CLIP 特征记忆库，按需创建
        self.dino_memory_bank = None  # DINO 特征记忆库，按需创建

        # 若启用 CLIP redirect，则创建 CLIP 特征记忆库（CLIP ViT-B/32 图像特征维度为 512）
        if self.use_clip_redirect:
            self.clip_memory_bank = FeatureMemoryBank(
                feature_dim=512,
                memory_size=cfg.clip_memory_size,  # 记忆库容量
                device=self.device
            )
            self.clip_hard_negative_k = cfg.clip_hard_negative_k  # 每个 query 取多少 hard negatives

        self.dino_weight = cfg.dino_weight  # DINO 权重，>0 表示启用 DINOv2
        # 当 DINO 权重大于 0 时加载 DINOv2 ViT-S/14 作为额外的内容特征提取器
        if self.dino_weight > 0:
            # 允许从 cfg 读取模型权重路径，缺失则回退到 None
            dino_model_path = getattr(cfg, 'dino_model_path', None)
            # 通过 timm 创建 ViT-S/14（DINOv2），不加载预训练权重，去除分类头，输入 224x224
            self.dino_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=224)
            # 仅在权重文件存在时才尝试加载
            if dino_model_path and os.path.exists(dino_model_path):
                # 加载 checkpoint 到指定设备
                state_dict = torch.load(dino_model_path, map_location=self.device)
                # 兼容多种 checkpoint 封装格式（含 'state_dict' 或 'model' 键）
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']

                # 以下处理位置编码插值：当 checkpoint 的 patch 网格与当前模型不一致时做双三次插值
                model_pos_embed = self.dino_model.pos_embed.data  # 当前模型的位置编码
                ckpt_pos_embed = state_dict.get('pos_embed', None)  # checkpoint 中的位置编码
                if ckpt_pos_embed is not None and ckpt_pos_embed.shape != model_pos_embed.shape:
                    # 拆出 cls token 与 patch token 分别处理
                    cls_token = ckpt_pos_embed[:, :1, :]  # cls token 不参与插值
                    patch_embed = ckpt_pos_embed[:, 1:, :]  # patch token 部分
                    embed_dim = patch_embed.shape[-1]  # 嵌入维度
                    # 由 patch 数反推原 patch 网格边长
                    orig_grid_size = int(patch_embed.shape[1] ** 0.5)
                    # 由当前模型 pos_embed 反推新 patch 网格边长
                    new_grid_size = int((model_pos_embed.shape[1] - 1) ** 0.5)
                    # 整理成 [B, C, H, W] 以使用 F.interpolate
                    patch_embed = patch_embed.reshape(1, orig_grid_size, orig_grid_size, embed_dim).permute(0, 3, 1, 2)
                    # 双三次插值到目标网格大小
                    patch_embed = F.interpolate(patch_embed, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
                    # 整理回 [1, num_patches, embed_dim]
                    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, embed_dim)
                    # 拼回 cls token，覆盖 state_dict 中的 pos_embed
                    state_dict['pos_embed'] = torch.cat([cls_token, patch_embed], dim=1)
                    print(f'DINOv2 pos_embed interpolated from {orig_grid_size}x{orig_grid_size} to {new_grid_size}x{new_grid_size}')

                # strict=False 容许部分权重缺失
                self.dino_model.load_state_dict(state_dict, strict=False)
                print(f'DINOv2 model loaded from {dino_model_path}')
            else:
                # 权重文件不存在时仅给出警告，使用随机初始化的 DINOv2
                print(f'Warning: DINOv2 model not found at {dino_model_path}, using random initialization')
            self.dino_model = self.dino_model.to(self.device)
            self.dino_model.eval()  # DINOv2 仅作为特征提取器，固定为 eval 模式
            # 冻结所有参数，不参与训练
            for param in self.dino_model.parameters():
                param.requires_grad = False
            # ImageNet 标准归一化均值/方差，用于 DINOv2 输入预处理
            self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

            # 若同时启用 CLIP redirect，则为 DINO 也建一个记忆库（ViT-S/14 输出 384 维）
            if self.use_clip_redirect:
                self.dino_memory_bank = FeatureMemoryBank(
                    feature_dim=384,
                    memory_size=cfg.clip_memory_size,
                    device=self.device
                )
        else:
            # 未启用 DINO 时设为 None，extract_dino_features 会直接返回 None
            self.dino_model = None

        # 解析噪声配置字符串，构建噪声层列表
        noise_layers = []
        if cfg.noise_config != 'identity':
            # 延迟导入噪声解析器，避免无噪声场景下的不必要依赖
            from noise_argparser import parse_crop, parse_cropout, parse_dropout, parse_resize
            # 配置串以 '+' 分隔多个噪声命令
            split_commands = cfg.noise_config.split('+')
            for command in split_commands:
                command = command.replace(' ', '')  # 去除空格便于前缀匹配
                if command.startswith('cropout'):
                    noise_layers.append(parse_cropout(command))  # cropout：外部裁剪后填充
                elif command.startswith('crop'):
                    noise_layers.append(parse_crop(command))  # crop：内部裁剪
                elif command.startswith('dropout'):
                    noise_layers.append(parse_dropout(command))  # dropout：随机像素丢弃
                elif command.startswith('resize'):
                    noise_layers.append(parse_resize(command))  # resize：随机缩放
                elif command.startswith('jpeg'):
                    from noise_layers.jpeg_compression import JpegCompression
                    noise_layers.append(JpegCompression(self.device))  # JPEG 压缩
                elif command.startswith('quant'):
                    from noise_layers.quantization import Quantization
                    noise_layers.append(Quantization(self.device))  # 量化噪声

        # 用 Noiser 封装噪声层；若列表为空则 Noiser 内部默认使用 identity
        self.noiser = Noiser(noise_layers, self.device).to(self.device)
        # 标记是否实际存在非 identity 噪声（影响对抗输入与视觉损失目标的选择）
        self.has_noise = not (len(self.noiser._layer_names) == 1 and self.noiser._layer_names[0] == 'identity')

        self.best_loss = float('inf')  # 记录历史最佳验证损失，用于 best checkpoint 判断
        self.phase2_msg_head_frozen = False  # 标记 Phase 2 中 msg_head 是否已被冻结
        self.disc_train_counter = 0  # 判别器训练计数（用于调度，预留字段）

    # 构建训练与验证的 DataLoader
    # 返回: (train_loader, val_loader) 两个 DataLoader
    def get_dataloaders(self):
        # 训练集：从 data_dir/train_dir 递归加载图像
        train_dataset = SteganographyDataset(
            root_dir=os.path.join(self.cfg.data_dir, self.cfg.train_dir),
            image_size=self.cfg.image_size,  # resize 到配置尺寸
            message_length=self.cfg.message_length  # 每张图随机生成该长度的二进制消息
        )

        # 验证集：从 data_dir/val_dir 加载
        val_dataset = SteganographyDataset(
            root_dir=os.path.join(self.cfg.data_dir, self.cfg.val_dir),
            image_size=self.cfg.image_size,
            message_length=self.cfg.message_length
        )

        # 训练 loader：打乱顺序，开启 pin_memory 加速到 GPU 的拷贝
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,  # 多进程数据加载
            pin_memory=True
        )
        # 验证 loader：不打乱，确保可复现
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    # 提取 DINOv2 特征
    # 参数:
    #   image: [-1, 1] 区间的图像张量 [B, 3, H, W]
    #   allow_grad: 是否允许梯度回传到 DINOv2 输入（编码图像需要 True）
    # 返回:
    #   归一化后的 DINO 特征张量 [B, 384]；若 DINO 未启用则返回 None
    def extract_dino_features(self, image, allow_grad=False):
        # DINO 模型未启用时直接返回 None
        if not hasattr(self, 'dino_model') or self.dino_model is None:
            return None
        # 在指定梯度上下文中执行，便于编码图像分支反向传播
        with torch.set_grad_enabled(allow_grad):
            # DINOv2 期望 224x224 输入，先做双线性下采样
            image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            image_normalized = (image_resized + 1) / 2  # 从 [-1, 1] 还原到 [0, 1]
            # 应用 ImageNet 标准化
            image_normalized = (image_normalized - self.dino_mean) / self.dino_std
            features = self.dino_model(image_normalized)  # 前向得到 [B, 384] 特征
            # L2 归一化（加小常数避免除零），便于后续 cosine 比较
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-7)
        return features

    # 冻结编码器的 msg_head：Phase 2 中使用，停止消息残差头的学习
    def freeze_msg_head(self):
        for param in self.encoder.msg_head.parameters():
            param.requires_grad = False
        self.phase2_msg_head_frozen = True  # 同步状态标记

    # 解冻 msg_head：恢复 msg_head 参数的可训练性
    def unfreeze_msg_head(self):
        for param in self.encoder.msg_head.parameters():
            param.requires_grad = True
        self.phase2_msg_head_frozen = False

    # 冻结编码器的 prv_head：Phase 1 中使用，防止隐私残差头过早学习
    def freeze_prv_head(self):
        for param in self.encoder.prv_head.parameters():
            param.requires_grad = False

    # 解冻 prv_head：Phase 2 中开启隐私残差头的训练
    def unfreeze_prv_head(self):
        for param in self.encoder.prv_head.parameters():
            param.requires_grad = True

    # 冻结共享层（conv_layers 与 after_concat_layer）
    # Phase 2 后期使用，避免共享特征被隐私目标带偏
    def freeze_shared_layers(self):
        for param in self.encoder.conv_layers.parameters():
            param.requires_grad = False
        for param in self.encoder.after_concat_layer.parameters():
            param.requires_grad = False

    # 解冻共享层：Phase 2 前期允许共享层参与训练以适应隐私目标
    def unfreeze_shared_layers(self):
        for param in self.encoder.conv_layers.parameters():
            param.requires_grad = True
        for param in self.encoder.after_concat_layer.parameters():
            param.requires_grad = True

    # 通用工具：批量设置某个模块所有参数的 requires_grad
    # 参数:
    #   module: 任意 nn.Module
    #   requires_grad: True/False
    def _set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad

    # 计算 FFT 高频能量损失（仅 Phase 2 启用）
    # 用途：惩罚隐私残差中的高频成分，避免在频域暴露隐写痕迹
    # 参数: residual - 隐私残差张量 [B, 3, H, W]
    # 返回: 标量损失（高频幅值均值的平方根）
    @staticmethod
    def _fft_high_freq_loss(residual):
        # 二维 FFT（正交归一化），得到复数频谱
        fft = torch.fft.fft2(residual, norm='ortho')
        amplitude = torch.abs(fft)  # 取幅值
        # 将零频移到中心，便于构造同心圆掩码
        amplitude = torch.fft.fftshift(amplitude, dim=(-2, -1))
        H, W = residual.shape[-2], residual.shape[-1]  # 空间尺寸
        cy, cx = H // 2, W // 2  # 中心（零频）位置
        # 构造每行/每列的坐标，便于计算到中心的距离
        y = torch.arange(H, device=residual.device).float().unsqueeze(1)
        x = torch.arange(W, device=residual.device).float().unsqueeze(0)
        # 到中心的欧氏距离矩阵
        dist = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        radius = min(H, W) // 4  # 高频判定半径
        mask = (dist > radius).float()  # 半径之外为高频区域
        # 返回高频幅值均值的平方根（加小常数防数值问题）
        return torch.sqrt((amplitude * mask).mean() + 1e-8)

    # 判别器 Hinge 损失：margin 默认 1.0
    # 参数:
    #   real_logits: 真实载体图像的判别 logits
    #   fake_logits: 隐写图像的判别 logits
    # 返回: 标量损失 L_D
    def _disc_hinge_loss(self, real_logits, fake_logits):
        # 从 cfg 读取 margin，未配置时回退到 1.0
        margin = getattr(self.cfg, 'disc_hinge_margin', 1.0)
        # 真实图像希望 logit > margin
        real_loss = F.relu(margin - real_logits).mean()
        # 隐写图像希望 logit < -margin
        fake_loss = F.relu(margin + fake_logits).mean()
        return real_loss + fake_loss

    # 生成器（编码器）非饱和对抗损失：使用 softplus 形式
    # 参数: fake_logits - 隐写图像在判别器上的 logits
    # 返回: 标量损失 L_adv
    def _gen_adv_loss(self, fake_logits):
        # softplus(-logit) 等价于 -log(sigmoid(logit))，非饱和形式更稳定
        return F.softplus(-fake_logits).mean()

    # 计算判别器的诊断指标
    # 参数: real_logits / fake_logits 同上
    # 返回: 字典，包含真实/隐写的 logit、概率均值与判别准确率
    def _disc_metrics(self, real_logits, fake_logits):
        real_prob = torch.sigmoid(real_logits)  # 真实图像判为真的概率
        fake_prob = torch.sigmoid(fake_logits)  # 隐写图像判为真的概率
        # 真实图像 logit > 0 视为判对
        real_acc = (real_logits > 0).float().mean()
        # 隐写图像 logit < 0 视为判对
        fake_acc = (fake_logits < 0).float().mean()
        # 综合准确率取均值，用于门控判别器更新
        disc_acc = 0.5 * (real_acc + fake_acc)
        return {
            'D_real_logit': real_logits.mean().item(),  # 真实图像平均 logit
            'D_fake_logit': fake_logits.mean().item(),  # 隐写图像平均 logit
            'D_real_prob': real_prob.mean().item(),  # 真实图像平均概率
            'D_fake_prob': fake_prob.mean().item(),  # 隐写图像平均概率
            'D_acc': disc_acc.item()  # 判别器准确率
        }

    # 选择对抗训练用的判别器输入
    # 当存在信道噪声时，使用含噪图像作为判别器输入，使其对噪声鲁棒；
    # 否则直接使用原始载体/编码图像。
    # 参数: 原始与含噪的 cover_image / encoded_image
    # 返回: (real_input, fake_input) 两个张量
    def _adversarial_inputs(self, cover_image, encoded_image, noised_cover_image, noised_encoded_image):
        if self.has_noise:
            return noised_cover_image, noised_encoded_image
        return cover_image, encoded_image

    # 计算对抗损失与判别器指标的组合
    # 参数: real_logits / fake_logits
    # 返回: (L_D, L_adv, discr_cover_bce, discr_encod_bce, metrics)
    #   - L_D: 判别器 Hinge 损失
    #   - L_adv: 生成器对抗损失
    #   - discr_cover_bce / discr_encod_bce: BCE 形式的判别器损失（仅用于日志记录）
    #   - metrics: 判别器诊断指标字典
    def _adversarial_losses_and_metrics(self, real_logits, fake_logits):
        L_D = self._disc_hinge_loss(real_logits, fake_logits)  # 判别器 Hinge 损失
        L_adv = self._gen_adv_loss(fake_logits)  # 生成器对抗损失
        # BCE 形式损失（标签：真实为 1，隐写为 0），用于日志对比
        discr_cover_bce = self.bce_with_logits_loss(real_logits, torch.ones_like(real_logits))
        discr_encod_bce = self.bce_with_logits_loss(fake_logits, torch.zeros_like(fake_logits))
        metrics = self._disc_metrics(real_logits, fake_logits)
        return L_D, L_adv, discr_cover_bce, discr_encod_bce, metrics

    # 根据阶段与 epoch 设置编码器各子模块的冻结状态
    # 参数:
    #   phase: 1 或 2
    #   epoch: 当前 epoch（Phase 2 用于决定是否冻结共享层）
    def _set_phase_freeze(self, phase, epoch):
        if phase == 1:
            # Phase 1：解冻 msg_head、冻结 prv_head、解冻共享层
            # 编码器仅学习消息残差，prv_head 注入预适应噪声不参与训练
            self.unfreeze_msg_head()
            self.freeze_prv_head()
            self.unfreeze_shared_layers()
        else:
            # Phase 2：冻结 msg_head（保护已学的消息嵌入），解冻 prv_head 学习隐私扰动
            self.freeze_msg_head()
            self.unfreeze_prv_head()
            # 前 1/3 epoch 允许共享层参与训练以适应隐私目标，之后冻结以稳定消息能力
            if epoch <= self.cfg.num_epochs_phase2 // 3:
                self.unfreeze_shared_layers()
            else:
                self.freeze_shared_layers()

    # 单个训练 step：完成一次前向、损失计算、判别器更新、生成器更新
    # 参数:
    #   batch: 数据集返回的字典，包含 cover_image 与 message
    #   phase: 训练阶段（1 或 2）
    #   epoch: 当前 epoch，用于调度权重与冻结策略
    # 返回:
    #   losses 字典，包含本 step 各项损失与诊断指标（用于日志/TensorBoard）
    def train_step(self, batch, phase, epoch=1):
        # 根据阶段设置编码器各子模块的 requires_grad
        self._set_phase_freeze(phase, epoch)

        # 取出载体图像与对应二进制消息，并搬到训练设备
        cover_image = batch['cover_image'].to(self.device)
        message = batch['message'].to(self.device)

        # 取本阶段/epoch 对应的损失权重字典
        weights = self.cfg.get_phase_weights(phase, epoch)
        # Phase 1 按 epoch 渐进引入隐私噪声比例；Phase 2 不再使用噪声预适应
        prv_noise_ratio = self.cfg.get_prv_noise_ratio(epoch) if phase == 1 else 0.0

        # 编码器前向：输出编码图像、残差惩罚项、消息残差、隐私残差
        encoded_image, residual_penalty, msg_residual, prv_residual = self.encoder(
            cover_image, message, phase=phase, prv_noise_ratio=prv_noise_ratio
        )

        # 对编码图像与原始载体施加随机噪声，模拟信道失真
        noised_encoded_image, noised_cover_image = self.noiser((encoded_image, cover_image))

        # 解码器从含噪编码图像预测 message logits（注意：未加 sigmoid，BCE 内部处理）
        message_pred = self.decoder(noised_encoded_image)

        # ===== 视觉损失（原始图像 vs 编码图像） =====
        L_pixel = self.mse_loss(encoded_image, cover_image)  # 像素 MSE
        L_ssim = self.ssim_loss(encoded_image, cover_image)  # SSIM 损失
        L_visual = self.cfg.lambda_pixel * L_pixel + self.cfg.lambda_ssim * L_ssim  # 加权视觉损失

        # ===== 视觉损失（含噪版本，强化对噪声鲁棒性） =====
        # 有噪声时目标用含噪载体，否则用原始载体
        visual_target_noised = noised_cover_image if self.has_noise else cover_image
        L_pixel_noised = self.mse_loss(noised_encoded_image, visual_target_noised)
        L_ssim_noised = self.ssim_loss(noised_encoded_image, visual_target_noised)
        L_visual_noised = self.cfg.lambda_pixel * L_pixel_noised + self.cfg.lambda_ssim * L_ssim_noised

        # ===== 消息损失：BCEWithLogits，目标是原 message =====
        L_message = self.bce_with_logits_loss(message_pred, message)

        # ===== 判别器分支：计算判别损失并按门控更新判别器 =====
        # 选择判别器输入（含噪或原始）
        disc_real_input, disc_stego_input = self._adversarial_inputs(
            cover_image, encoded_image, noised_cover_image, noised_encoded_image
        )
        # detach 防止梯度流回生成器
        fake_logits_for_disc = self.steganalyzer(disc_stego_input.detach())
        real_logits = self.steganalyzer(disc_real_input.detach())
        # 计算判别损失、生成器对抗损失、BCE 日志项与指标
        L_D, _, discr_cover_bce, discr_encod_bce, disc_metrics = self._adversarial_losses_and_metrics(
            real_logits, fake_logits_for_disc
        )

        # 判别器门控更新：仅在对抗权重 > 0 且判别准确率未过高（<0.70）时更新
        # 避免判别器过强导致生成器梯度消失
        D_updated = 0.0
        if weights.get('adv', 0.0) > 0 and disc_metrics['D_acc'] < getattr(self.cfg, 'disc_acc_high', 0.70):
            self.optimizer_steganalyzer.zero_grad()
            L_D.backward()  # 仅判别器参数有梯度
            # 梯度裁剪，防止训练不稳定
            torch.nn.utils.clip_grad_norm_(self.steganalyzer.parameters(), max_norm=1.0)
            self.optimizer_steganalyzer.step()
            D_updated = 1.0  # 标记本 step 已更新判别器

        # ===== 生成器对抗损失：再前向一次以拿到可回传梯度的 stego_pred =====
        self._set_requires_grad(self.steganalyzer, False)  # 临时关闭判别器梯度
        steganalyzer_was_training = self.steganalyzer.training  # 记录原训练状态
        self.steganalyzer.eval()  # 切到 eval 影响 BN/Dropout
        stego_pred = self.steganalyzer(disc_stego_input)  # 此时梯度可回传到编码器
        L_adv = self._gen_adv_loss(stego_pred)  # 生成器对抗损失
        if steganalyzer_was_training:
            self.steganalyzer.train()  # 恢复训练状态
        self._set_requires_grad(self.steganalyzer, True)  # 恢复判别器可训练

        # ===== CLIP 内容特征：用于 escape / redirect 损失 =====
        # 载体特征 detach，仅作为锚点
        feat_cover = self.content_recognizer(cover_image).detach()
        # 编码图像特征保留梯度，使隐私损失能反传到编码器
        feat_encoded = self.content_recognizer(encoded_image)

        # DINO 特征：仅当启用 DINO 时返回张量，否则 None
        dino_cover = self.extract_dino_features(cover_image)
        dino_encoded = self.extract_dino_features(encoded_image, allow_grad=True)

        clip_hard_negatives = None
        dino_hard_negatives = None

        # 更新 CLIP 记忆库并取出当前批次的 hard negatives
        if self.use_clip_redirect and self.clip_memory_bank is not None:
            self.clip_memory_bank.update(feat_cover.detach())
            clip_hard_negatives = self.clip_memory_bank.get_hard_negatives(
                feat_encoded.detach(), k=self.clip_hard_negative_k
            )

        # 更新 DINO 记忆库并取出 hard negatives（仅当 DINO 启用且记忆库存在）
        if self.use_clip_redirect and self.dino_memory_bank is not None and dino_cover is not None:
            self.dino_memory_bank.update(dino_cover.detach())
            dino_hard_negatives = self.dino_memory_bank.get_hard_negatives(
                dino_encoded.detach(), k=self.clip_hard_negative_k
            )

        # 当前阶段/epoch 的 escape margin（Hinge 边界，渐进扩大）
        escape_margin = self.cfg.get_escape_margin(phase, epoch)

        # 计算隐私 escape + redirect 损失，返回总损失与子项字典
        L_privacy, privacy_loss_dict = self.retrieval_escape_loss(
            feat_cover, feat_encoded,
            dino_cover=dino_cover,
            dino_encoded=dino_encoded,
            clip_hard_negatives=clip_hard_negatives,
            dino_hard_negatives=dino_hard_negatives,
            escape_margin=escape_margin
        )

        # ===== 残差惩罚与 FFT 高频惩罚 =====
        residual_penalty_weight = weights.get('residual_penalty', 0.05)  # 残差预算惩罚权重
        fft_weight = weights.get('fft', 0.0)  # FFT 高频惩罚权重（仅 Phase 2 > 0）
        # 仅在启用 FFT 且存在隐私残差时计算
        if fft_weight > 0 and prv_residual is not None:
            L_fft = self._fft_high_freq_loss(prv_residual)
        else:
            L_fft = torch.tensor(0.0, device=self.device)
        # 视觉损失加权（含噪版本按 noised 权重叠加）
        L_visual_weighted = weights['visual'] * (L_visual + weights['noised'] * L_visual_noised)
        # 非视觉损失：消息 + 对抗 + 残差惩罚
        L_non_visual = (weights['message'] * L_message +
                       weights['adv'] * L_adv +
                       residual_penalty_weight * residual_penalty)
        L_privacy_weighted = weights['privacy'] * L_privacy  # 隐私损失加权
        L_fft_weighted = fft_weight * L_fft  # FFT 损失加权
        # 总损失：视觉 + 非视觉 + 隐私 + FFT
        L_total = L_visual_weighted + L_non_visual + L_privacy_weighted + L_fft_weighted

        # ===== 反向传播与生成器/编码器/解码器更新 =====
        self.optimizer_encoder_decoder.zero_grad()
        L_total.backward()

        # Phase 2 中 msg_head 被冻结：清除可能由残差惩罚等引入的梯度，确保不更新
        if phase == 2 and self.phase2_msg_head_frozen:
            for param in self.encoder.msg_head.parameters():
                param.grad = None

        # 对编码器与解码器参数做联合梯度裁剪
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), max_norm=1.0)

        self.optimizer_encoder_decoder.step()  # 执行一步参数更新

        # ===== 计算 bit accuracy 用于监控 =====
        with torch.no_grad():
            # 将 logits 转为二进制预测（>0.5 视为 1）
            message_pred_binary = (torch.sigmoid(message_pred) > 0.5).float()
            bit_accuracy = (message_pred_binary == message).float().mean().item()

        # 汇总本 step 的所有损失与指标，供日志记录
        losses = {
            'L_pixel': L_pixel.item(),  # 像素 MSE
            'L_ssim': L_ssim.item(),  # SSIM 损失
            'L_visual': L_visual.item(),  # 视觉损失
            'L_pixel_noised': L_pixel_noised.item(),  # 含噪像素 MSE
            'L_ssim_noised': L_ssim_noised.item(),  # 含噪 SSIM
            'L_visual_noised': L_visual_noised.item(),  # 含噪视觉损失
            'L_message': L_message.item(),  # 消息损失
            'L_adv': L_adv.item(),  # 生成器对抗损失
            'L_D': L_D.item(),  # 判别器损失
            'discr_cover_bce': discr_cover_bce.item(),  # 真实图像 BCE
            'discr_encod_bce': discr_encod_bce.item(),  # 隐写图像 BCE
            'L_privacy': L_privacy.item(),  # 隐私 escape+redirect 损失
            'L_residual_penalty': residual_penalty.item(),  # 残差预算惩罚
            'L_fft': L_fft.item(),  # FFT 高频损失
            'bit_accuracy': bit_accuracy,  # 比特准确率
            'escape_margin': escape_margin,  # 当前 escape margin
            'prv_noise_ratio': prv_noise_ratio,  # 当前隐私噪声比例
            'D_updated': D_updated,  # 本 step 是否更新了判别器
            'adv_weight': weights.get('adv', 0.0)  # 对抗权重
        }
        losses.update(disc_metrics)  # 合并判别器诊断指标

        # 合并隐私损失的子项（前缀 privacy_）
        for key, value in privacy_loss_dict.items():
            losses[f'privacy_{key}'] = value if isinstance(value, float) else value

        return losses

    # 解码器恢复步骤：当 Phase 2 中 bit accuracy 跌破阈值时调用
    # 思路：冻结编码器与 prv_head，仅用"干净"编码图像（只含 msg_residual）
    #       重新训练解码器若干 batch，恢复其消息解码能力
    # 参数:
    #   train_loader: 训练数据加载器
    #   num_batches: 恢复训练的 batch 数量，默认 5
    def recovery_step(self, train_loader, num_batches=5):
        self.encoder.eval()  # 编码器切到 eval，BN/Dropout 固定
        self.decoder.train()  # 仅解码器参与训练

        self.freeze_prv_head()  # 冻结隐私残差头，避免被误更新

        # 使用一个临时的 Adam 优化器，学习率为正常值的一半，避免剧烈震荡
        temp_optimizer = optim.Adam(self.decoder.parameters(), lr=self.cfg.lr_decoder * 0.5)

        # 遍历训练 loader，仅处理前 num_batches 个 batch
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break

            cover_image = batch['cover_image'].to(self.device)
            message = batch['message'].to(self.device)

            # 编码器前向不需要梯度；以 Phase 1 模式生成纯消息残差
            with torch.no_grad():
                _, _, msg_residual, _ = self.encoder(
                    cover_image, message, phase=1, prv_noise_ratio=0.0
                )
                # 仅叠加消息残差，不含隐私扰动，得到"干净"编码图像
                clean_encoded = cover_image + msg_residual
                # 限制到 [-1, 1] 与训练时一致
                clean_encoded = torch.clamp(clean_encoded, -1, 1)

            # 解码器在干净编码图像上预测 message logits
            message_pred = self.decoder(clean_encoded)
            # 仅用消息损失训练解码器
            L_msg = self.bce_with_logits_loss(message_pred, message)

            temp_optimizer.zero_grad()
            L_msg.backward()
            # 梯度裁剪保持稳定
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
            temp_optimizer.step()

        self.unfreeze_prv_head()  # 恢复 prv_head 可训练
        self.encoder.train()  # 编码器回到 train 模式

    # 验证函数：在验证集上评估当前模型的所有损失与指标
    # 参数:
    #   val_loader: 验证集 DataLoader
    #   phase: 当前训练阶段
    #   epoch: 当前 epoch
    #   save_vis: 是否保存可视化样本（cover 与 encoded 对比图）
    # 返回:
    #   total_losses 字典，包含所有指标在验证集上的 batch 平均值
    def validate(self, val_loader, phase, epoch, save_vis=False):
        # 将所有模型切到 eval，固定 BN/Dropout
        self.encoder.eval()
        self.decoder.eval()
        self.steganalyzer.eval()

        total_losses = {}  # 累加各项损失
        num_batches = 0  # 已处理的 batch 数

        sample_images = []  # 用于可视化的样本
        max_samples = 8  # 最多采集 8 个样本
        all_batches = list(val_loader)  # 一次性拉取所有 batch（验证集通常较小）
        # 统计验证集总样本数
        num_total_samples = sum(batch['cover_image'].size(0) for batch in all_batches)
        if num_total_samples < max_samples:
            # 总样本不足时全部选中
            selected_indices = list(range(num_total_samples))
        else:
            # 否则随机抽取 max_samples 个样本索引
            import random
            selected_indices = random.sample(range(num_total_samples), max_samples)
        selected_indices_set = set(selected_indices)  # 转集合便于 O(1) 查找

        # 验证时使用与训练一致的噪声比例与 escape margin
        prv_noise_ratio = self.cfg.get_prv_noise_ratio(epoch) if phase == 1 else 0.0
        escape_margin = self.cfg.get_escape_margin(phase, epoch)

        with torch.no_grad():  # 验证全程不需要梯度
            current_idx = 0  # 全局样本索引，用于匹配可视化选中索引
            for batch in all_batches:
                cover_image = batch['cover_image'].to(self.device)
                message = batch['message'].to(self.device)

                weights = self.cfg.get_phase_weights(phase, epoch)  # 阶段权重

                # 编码器前向（与 train_step 一致）
                encoded_image, residual_penalty, msg_residual, prv_residual = self.encoder(
                    cover_image, message, phase=phase, prv_noise_ratio=prv_noise_ratio
                )
                # 施加噪声
                noised_encoded_image, noised_cover_image = self.noiser((encoded_image, cover_image))
                # 解码器预测
                message_pred = self.decoder(noised_encoded_image)

                # 视觉损失（原始）
                L_pixel = self.mse_loss(encoded_image, cover_image)
                L_ssim = self.ssim_loss(encoded_image, cover_image)
                L_visual = self.cfg.lambda_pixel * L_pixel + self.cfg.lambda_ssim * L_ssim

                # 视觉损失（含噪）
                visual_target_noised = noised_cover_image if self.has_noise else cover_image
                L_pixel_noised = self.mse_loss(noised_encoded_image, visual_target_noised)
                L_ssim_noised = self.ssim_loss(noised_encoded_image, visual_target_noised)
                L_visual_noised = self.cfg.lambda_pixel * L_pixel_noised + self.cfg.lambda_ssim * L_ssim_noised

                # 消息损失
                L_message = self.bce_with_logits_loss(message_pred, message)

                # 对抗损失（验证时只评估不更新）
                disc_real_input, disc_stego_input = self._adversarial_inputs(
                    cover_image, encoded_image, noised_cover_image, noised_encoded_image
                )
                stego_pred = self.steganalyzer(disc_stego_input)
                cover_pred = self.steganalyzer(disc_real_input)
                L_D, L_adv, discr_cover_bce, discr_encod_bce, disc_metrics = self._adversarial_losses_and_metrics(
                    cover_pred, stego_pred
                )

                # CLIP/DINO 特征与隐私损失
                feat_cover = self.content_recognizer(cover_image)
                feat_encoded = self.content_recognizer(encoded_image)

                dino_cover = self.extract_dino_features(cover_image)
                dino_encoded = self.extract_dino_features(encoded_image)

                L_privacy, _ = self.retrieval_escape_loss(
                    feat_cover, feat_encoded,
                    dino_cover=dino_cover,
                    dino_encoded=dino_encoded,
                    escape_margin=escape_margin
                )

                # 残差惩罚与 FFT 损失
                residual_penalty_weight = weights.get('residual_penalty', 0.05)
                fft_weight = weights.get('fft', 0.0)
                if fft_weight > 0 and prv_residual is not None:
                    L_fft = self._fft_high_freq_loss(prv_residual)
                else:
                    L_fft = torch.tensor(0.0, device=self.device)
                # 验证用总损失（与 train_step 一致结构）
                L_total = (weights['visual'] * (L_visual + weights['noised'] * L_visual_noised) +
                          weights['message'] * L_message +
                          weights['adv'] * L_adv +
                          weights['privacy'] * L_privacy +
                          residual_penalty_weight * residual_penalty +
                          fft_weight * L_fft)

                # 计算本 batch 的 bit accuracy
                message_pred_binary = (torch.sigmoid(message_pred) > 0.5).float()
                bit_accuracy = (message_pred_binary == message).float().mean()

                # 累加各项标量损失（.item() 触发同步）
                for key, value in [
                    ('L_pixel', L_pixel), ('L_ssim', L_ssim),
                    ('L_visual', L_visual),
                    ('L_pixel_noised', L_pixel_noised), ('L_ssim_noised', L_ssim_noised),
                    ('L_visual_noised', L_visual_noised),
                    ('L_message', L_message), ('L_adv', L_adv), ('L_D', L_D),
                    ('discr_cover_bce', discr_cover_bce), ('discr_encod_bce', discr_encod_bce),
                    ('L_privacy', L_privacy), ('L_residual_penalty', residual_penalty),
                    ('L_fft', L_fft),
                    ('bit_accuracy', bit_accuracy)
                ]:
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value.item()

                # 累加判别器诊断指标
                for key, value in disc_metrics.items():
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value

                # 累加辅助字段（验证时 D_updated 恒为 0）
                for key, value in [
                    ('D_updated', 0.0), ('adv_weight', weights.get('adv', 0.0)),
                    ('prv_noise_ratio', prv_noise_ratio)
                ]:
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value

                num_batches += 1

                # 收集可视化样本：遍历 batch 中每个样本，匹配选中索引
                for i in range(cover_image.size(0)):
                    if current_idx in selected_indices_set:
                        sample_images.append({
                            'cover': cover_image[i].cpu(),  # 移到 CPU 节省显存
                            'encoded': encoded_image[i].cpu()
                        })
                    current_idx += 1

        # 对所有累加值取 batch 平均
        for key in total_losses:
            total_losses[key] /= num_batches

        # 可视化：将 cover 与 encoded 拼成上下两行的对比图并保存
        if save_vis and sample_images:
            save_samples = sample_images[:5]  # 只取前 5 个样本

            cover_images = [img['cover'] for img in save_samples]
            encoded_images = [img['encoded'] for img in save_samples]

            # 各自拼成 1x5 网格
            cover_grid = make_grid(cover_images, nrow=5, padding=2, normalize=False)
            encoded_grid = make_grid(encoded_images, nrow=5, padding=2, normalize=False)
            # 上下拼接（dim=1 表示沿高度方向拼接）
            combined = torch.cat([cover_grid, encoded_grid], dim=1)

            # 保存到 checkpoint_dir/visualizations 目录
            save_dir = os.path.join(self.cfg.checkpoint_dir, 'visualizations')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'phase{phase}_epoch{epoch}.png')

            # 从 [-1, 1] 还原到 [0, 1] 再裁剪保存
            combined_normalized = (combined + 1) / 2
            combined_normalized = torch.clamp(combined_normalized, 0, 1)
            save_image(combined_normalized, save_path)
            print(f' [OK] Visualization saved: {save_path}')

        # 恢复所有模型到 train 模式
        self.encoder.train()
        self.decoder.train()
        self.steganalyzer.train()

        return total_losses

    # 预热记忆库：在正式训练前用若干 batch 的载体特征填充 CLIP/DINO 记忆库
    # 用途：避免训练初期记忆库为空导致 hard negative mining 失效
    # 参数:
    #   train_loader: 训练数据加载器
    #   num_batches: 预热使用的 batch 数量，默认 50
    def warmup_memory_bank(self, train_loader, num_batches=50):
        # 若两个记忆库都未启用，则无需预热
        if self.clip_memory_bank is None and self.dino_memory_bank is None:
            return

        print(f'Warming up Memory Banks with {num_batches} batches...')

        with torch.no_grad():  # 预热阶段不需要梯度
            for i, batch in enumerate(train_loader):
                if i >= num_batches:
                    break
                cover_image = batch['cover_image'].to(self.device)

                # 用 CLIP 提取载体特征并写入对应记忆库
                if self.clip_memory_bank is not None:
                    clip_feat = self.content_recognizer(cover_image)
                    self.clip_memory_bank.update(clip_feat)

                # 用 DINOv2 提取载体特征并写入对应记忆库（若启用）
                if self.dino_memory_bank is not None:
                    dino_feat = self.extract_dino_features(cover_image)
                    if dino_feat is not None:
                        self.dino_memory_bank.update(dino_feat)

        # 汇总打印两个记忆库的最终有效容量
        info_parts = []
        if self.clip_memory_bank is not None:
            info_parts.append(f'CLIP: {self.clip_memory_bank.memory_count}')
        if self.dino_memory_bank is not None:
            info_parts.append(f'DINO: {self.dino_memory_bank.memory_count}')
        print(f'Memory Banks warmed up. [{", ".join(info_parts)}]')

    # 保存 checkpoint：将当前训练状态序列化到磁盘
    # 同时保存两份：带 phase/epoch 后缀的历史版本 + 始终最新的 checkpoint.pth
    # 参数:
    #   epoch: 当前 epoch
    #   phase: 当前训练阶段
    def save_checkpoint(self, epoch, phase):
        # 组装 checkpoint 字典，包含模型、优化器与训练状态
        checkpoint = {
            'epoch': epoch,  # 当前 epoch
            'phase': phase,  # 当前阶段
            'encoder_state_dict': self.encoder.state_dict(),  # 编码器权重
            'decoder_state_dict': self.decoder.state_dict(),  # 解码器权重
            'steganalyzer_state_dict': self.steganalyzer.state_dict(),  # 判别器权重
            'optimizer_encoder_decoder': self.optimizer_encoder_decoder.state_dict(),  # 编/解优化器状态
            'optimizer_steganalyzer': self.optimizer_steganalyzer.state_dict(),  # 判别器优化器状态
            'best_loss': self.best_loss,  # 历史最佳损失
            'phase2_msg_head_frozen': self.phase2_msg_head_frozen  # msg_head 冻结标记
        }

        # 若启用 CLIP 记忆库，则一并保存其状态
        if self.clip_memory_bank is not None:
            checkpoint['clip_memory_bank'] = self.clip_memory_bank.state_dict()

        # 若启用 DINO 记忆库，则一并保存其状态
        if self.dino_memory_bank is not None:
            checkpoint['dino_memory_bank'] = self.dino_memory_bank.state_dict()

        # 保存带阶段与 epoch 后缀的历史版本，便于回溯
        path = os.path.join(self.cfg.checkpoint_dir, f'checkpoint_phase{phase}_epoch{epoch}.pth')
        torch.save(checkpoint, path)

        # 同时覆盖最新的 checkpoint.pth，供 continue 命令直接读取
        latest_path = os.path.join(self.cfg.checkpoint_dir, 'checkpoint.pth')
        torch.save(checkpoint, latest_path)
