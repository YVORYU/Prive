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
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 日志写入器
from torchvision.utils import save_image, make_grid  # 图像保存与网格拼接工具

# 导入项目内部的模型、损失、数据集与噪声模块
from models import Encoder, Decoder, Steganalyzer, ContentRecognizer  # 编码器、解码器、判别器、CLIP 内容识别器
from losses import SSIMLoss, FeatureRedirectionLoss  # SSIM 损失与 CLIP/DINO 特征偏移损失
from data import SteganographyDataset  # 隐写数据集，递归加载图像并生成随机消息
from noise_layers.noiser import Noiser  # 噪声合成器，按概率施加信道噪声


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
    #   exclude_sim: 候选排除阈值；与 query 相似度超过该值的候选（自身副本、
    #     近重复图、同身份照片）被排除，避免重定向目标退化为"自己"
    # 返回:
    #   [B, k, D] 的 hard negative 特征张量；若记忆库为空则返回 None
    def get_hard_negatives(self, features, k=5, exclude_sim=None):
        # 记忆库尚未填充时直接返回 None，调用方需判空
        if self.memory_count == 0:
            return None

        features = features.float()  # 转为 float（查询特征也应是归一化的）
        # 仅在有效范围内取记忆特征，避免使用尚未填充的零向量
        valid_memory = self.memory_bank[:self.memory_count]
        # 通过矩阵乘法计算查询与每个记忆特征的余弦相似度（因为已归一化）
        similarities = torch.mm(features, valid_memory.t())
        # 排除过于相似的候选：它们会使重定向与 escape 目标对冲（拉向自己/同人）
        if exclude_sim is not None:
            similarities = similarities.masked_fill(similarities > exclude_sim, -1.0)
        # k 不能超过当前记忆库实际容量
        actual_k = min(k, self.memory_count)
        # 取相似度最大的 k 个索引（这些是排除后最"危险"的 hard negative）
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
            texture_mask_smooth_kernel=cfg.texture_mask_smooth_kernel,  # 纹理掩码平滑核大小
            prv_lowres_size=getattr(cfg, 'prv_lowres_size', 96)  # 隐私残差低分辨率约束尺寸
        ).to(self.device)
        # 解码器：从（含噪）编码图像预测 message logits
        self.decoder = Decoder(message_length=cfg.message_length, n_channels=3).to(self.device)
        # 判别器（隐写分析器）：spectral norm 卷积分类器，输出真实/隐写 logits
        self.steganalyzer = Steganalyzer(n_channels=3, use_spectral_norm=True).to(self.device)
        # 内容识别器：冻结的 CLIP ViT-B/32，用于提取图像/文本特征并做归一化
        self.content_recognizer = ContentRecognizer(device=self.device).to(self.device)

        # 编码器与解码器共用一个 Adam 优化器（不同学习率）；
        # prv_head 与 prv_trunk（隐私独立支路）共用单独参数组、学习率 ×prv_head_lr_mult：
        # CLIP 逃逸方向逐图不同，Adam 在多目标拉扯下信噪比失效（权重放大 loss 不改变步长），
        # 只有提高 lr 能放大其实际更新步长
        prv_prefixes = ('prv_head', 'prv_trunk')
        encoder_non_prv = [p for n, p in self.encoder.named_parameters() if not n.startswith(prv_prefixes)]
        encoder_prv = [p for n, p in self.encoder.named_parameters() if n.startswith(prv_prefixes)]
        self.optimizer_encoder_decoder = optim.Adam([
            {"params": encoder_non_prv, "lr": cfg.lr_encoder},  # 编码器主体学习率
            {"params": encoder_prv, "lr": cfg.lr_encoder * getattr(cfg, 'prv_head_lr_mult', 5.0)},  # prv_head 加速
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
            clip_redirect_margin=cfg.clip_redirect_margin,  # CLIP redirect margin（遗留）
            dino_redirect_margin=cfg.dino_redirect_margin,  # DINO redirect margin（遗留）
            clip_max_target_sim=cfg.clip_max_target_sim,  # CLIP 目标相似度上限
            dino_max_target_sim=cfg.dino_max_target_sim,  # DINO 目标相似度上限
            clip_redirect_target_sim=getattr(cfg, 'clip_redirect_target_sim', 0.80),  # CLIP 重定向目标相似度
            dino_redirect_target_sim=getattr(cfg, 'dino_redirect_target_sim', 0.55)  # DINO 重定向目标相似度
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
        # best_loss 所属阶段：Phase 1 准则是 L_message，Phase 2 准则是 clip_sim+dino_sim，
        # 两者数值不可比。train.py 检测到阶段切换时重置 best_loss，否则从 Phase 1
        # 进入 Phase 2 后 best 分数永远高于 Phase 1 收敛的 L_message，best 冻结不再更新
        self.best_loss_phase = None
        self.phase2_msg_head_frozen = False  # 标记 Phase 2 中 msg_head 是否已被冻结

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

    # 冻结隐私支路（prv_head 与 prv_trunk）：Phase 1 中使用，防止隐私残差头过早学习
    def freeze_prv_head(self):
        for param in self.encoder.prv_head.parameters():
            param.requires_grad = False
        for param in self.encoder.prv_trunk.parameters():
            param.requires_grad = False

    # 解冻隐私支路：Phase 2 中开启 prv_head 与 prv_trunk 的训练
    def unfreeze_prv_head(self):
        for param in self.encoder.prv_head.parameters():
            param.requires_grad = True
        for param in self.encoder.prv_trunk.parameters():
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
    # 用途：惩罚隐私残差中的高频成分，避免在频域暴露隐写痕迹/可见条纹
    # 参数:
    #   residual - 隐私残差张量 [B, 3, H, W]
    #   radius_ratio - 高频判定半径占奈奎斯特半径的比例（0.25 覆盖周期<8px 的条纹频段）
    # 返回: 标量损失（高频幅值均值，线性形式梯度稳定）
    # 实现说明（修复点）：
    #   1) 原半径 min(H,W)//4 只覆盖 r>64 的极高频，周期 4~8px 的可见条纹（r=32~64）完全豁免；
    #      现半径由 radius_ratio 控制，默认奈奎斯特的 0.25 倍（256 图约 r=32）。
    #   2) 原 sqrt(mean(A)) 形态梯度随违规增大而递减（病态）；现改为 mean(A) 线性形式，
    #      梯度恒定，与违规程度成正比。
    @staticmethod
    def _fft_high_freq_loss(residual, radius_ratio=0.25):
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
        # 奈奎斯特半径 = min(H,W)//2，按配置比例确定高频判定半径
        radius = max(1, int((min(H, W) // 2) * radius_ratio))
        mask = (dist > radius).float()  # 半径之外为高频区域
        # 返回高频幅值均值（线性惩罚，梯度稳定）
        return (amplitude * mask).mean()

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

    # 内环逐图优化（方案 C 的教师生成器）：
    # 冻结全部网络参数，把当前 batch 的低分辨率隐私残差场（与编码器内部低通约束
    # 同一变量空间）作为唯一优化变量，跑 distill_inner_steps 步 Adam 直接优化。
    # 目标函数与逐图精修试点（val/refine_encode.py，已实测 clip_sim 0.94→0.55~0.65
    # 且 bit_acc/PSNR/SSIM 不劣化）完全一致：
    #   escape(CLIP/DINO，margin 取 end 值) + 消息双分支 BCE + 视觉 MSE/SSIM
    # 约束链与训练同构（低通→tanh×预算→纹理掩码→clamp）。
    # 返回教师隐私残差（detached）；步数<=0 或无隐私残差时返回 None。
    # 说明：内环反向只经过 delta（叶子），不触碰编码器参数；Balance 段 decoder
    # 可训练时其梯度会被内环污染，但随后 train_step 的 zero_grad() 会清除，无副作用。
    def _refine_teacher_residual(self, cover_image, message, msg_residual, prv_residual,
                                 feat_cover, dino_cover):
        cfg = self.cfg
        steps = getattr(cfg, 'distill_inner_steps', 0)
        if steps <= 0 or prv_residual is None:
            return None
        H, W = cover_image.shape[-2], cover_image.shape[-1]
        lowres = getattr(cfg, 'prv_lowres_size', 96)
        budget = self.encoder.prv_budget

        # 从网络当前输出热启动（与逐图精修一致）：反解 tanh 前的低分辨率场
        with torch.no_grad():
            texture_mask = self.encoder._compute_texture_mask(cover_image)
            ratio = (prv_residual.detach() / (budget * texture_mask + 1e-8)).clamp(-0.95, 0.95)
            delta = F.adaptive_avg_pool2d(torch.atanh(ratio), (lowres, lowres))
        delta = delta.detach().clone()
        delta.requires_grad_(True)
        opt = optim.Adam([delta], lr=getattr(cfg, 'distill_lr', 0.1))

        w_privacy = getattr(cfg, 'distill_w_privacy', 12.0)
        w_message = getattr(cfg, 'distill_w_message', 10.0)
        w_visual = getattr(cfg, 'distill_w_visual', 15.0)
        msg_residual = msg_residual.detach()

        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            # 与编码器 Phase 2 相同的约束链：低通上采样 -> tanh 预算 -> 纹理掩码
            prv_up = F.interpolate(delta, size=(H, W), mode='bicubic', align_corners=False)
            prv = budget * torch.tanh(prv_up) * texture_mask
            encoded = torch.clamp(cover_image + msg_residual + prv, -1, 1)

            # 逃逸损失：escape-only（无 redirect），margin 固定 end 值（与验证配方一致）
            feat_enc = self.content_recognizer(encoded)
            dino_enc = self.extract_dino_features(encoded, allow_grad=True) if dino_cover is not None else None
            L_priv, _ = self.retrieval_escape_loss(
                feat_cover, feat_enc,
                dino_cover=dino_cover, dino_encoded=dino_enc,
                escape_margin=cfg.privacy_margin_end,
                dino_escape_margin=cfg.dino_escape_margin_end
            )

            # 消息双分支：干净 + 含噪各 0.5（Phase 2 干净分支比例即 0.5）
            L_msg = 0.5 * self.bce_with_logits_loss(self.decoder(encoded), message)
            noised_enc, _ = self.noiser((encoded, cover_image))
            L_msg = L_msg + 0.5 * self.bce_with_logits_loss(self.decoder(noised_enc), message)

            L_vis = self.mse_loss(encoded, cover_image) + self.ssim_loss(cover_image, encoded)

            L = w_privacy * L_priv + w_message * L_msg + w_visual * L_vis
            L.backward()
            opt.step()

        with torch.no_grad():
            prv_up = F.interpolate(delta, size=(H, W), mode='bicubic', align_corners=False)
            return budget * torch.tanh(prv_up) * texture_mask

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
            # 全程冻结共享层：隐私目标不再改动 shared 表征，
            # 避免冻结的 msg_head 因输入分布漂移而导致解码率下跌
            self.freeze_shared_layers()
            # Sprint 段额外冻结 decoder：纯逃逸冲刺期间消息目标关闭（权重 0），
            # decoder 无需更新也避免其适应暂时崩坏的残差分布；
            # Refine 终段同样冻结 decoder（val/refine_encode.py 语义：残差自适应
            # 既有解码器而非解码器迁就残差，学生目标中的消息锚直接穿过冻结
            # 的 decoder 反向约束残差）；
            # Balance 段解冻恢复训练。该冻结只影响生成器优化器侧的梯度回传，
            # recovery_step 使用独立的临时优化器，不受此标记影响
            sprint_epochs = getattr(self.cfg, 'phase2_sprint_epochs', 0)
            in_refine = self.cfg.is_phase2_refine_segment(epoch)
            self._set_requires_grad(self.decoder, epoch > sprint_epochs and not in_refine)

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

        # 解码器双分支前向：干净编码图 + 含噪编码图分别预测 message logits
        # （注意：未加 sigmoid，BCE 内部处理）
        message_pred = self.decoder(noised_encoded_image)
        message_pred_clean = self.decoder(encoded_image)

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

        # ===== 消息损失：干净 + 含噪双分支 BCEWithLogits =====
        # 干净分支保底提供稳定梯度通路（冷启动不再依赖含噪信号在噪声后的信噪比）；
        # 含噪分支在解码器启动之后自然学会抗噪，相当于内置课程学习。
        # 实测仅靠含噪单分支时，JPEG/resize 会摧毁随机初始化的微弱 msg 信号，
        # 解码器永远等不到启动时刻，L_message 锁死在 ln2。
        # 干净分支权重按 get_msg_clean_ratio 调度：Phase 1 前期 0.8（加速冷启动，
        # 避免含噪分支的噪声梯度稀释有效梯度），后期退火到 0.5 均衡抗噪
        clean_ratio = self.cfg.get_msg_clean_ratio(phase, epoch)
        L_message = ((1.0 - clean_ratio) * self.bce_with_logits_loss(message_pred, message) +
                     clean_ratio * self.bce_with_logits_loss(message_pred_clean, message))

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

        # ===== 生成器对抗损失：仅当 adv 权重 > 0 时才前向一次拿到可回传梯度的 stego_pred =====
        if weights.get('adv', 0.0) > 0:
            self._set_requires_grad(self.steganalyzer, False)  # 临时关闭判别器梯度
            steganalyzer_was_training = self.steganalyzer.training  # 记录原训练状态
            self.steganalyzer.eval()  # 切到 eval 影响 BN/Dropout
            stego_pred = self.steganalyzer(disc_stego_input)  # 此时梯度可回传到编码器
            L_adv = self._gen_adv_loss(stego_pred)  # 生成器对抗损失
            if steganalyzer_was_training:
                self.steganalyzer.train()  # 恢复训练状态
            self._set_requires_grad(self.steganalyzer, True)  # 恢复判别器可训练
        else:
            # adv=0：判别器仅作监控指标，不向编码器回传对抗梯度（避免引入伪影方向）
            L_adv = torch.tensor(0.0, device=self.device)

        # ===== CLIP/DINO 内容特征：用于 escape / redirect 损失 =====
        # 载体特征 detach，仅作为锚点
        feat_cover = self.content_recognizer(cover_image).detach()
        dino_cover = self.extract_dino_features(cover_image)

        privacy_weight = weights.get('privacy', 0.0)
        if privacy_weight > 0:
            # Phase 2 双分支特征提取：
            #   干净分支——梯度不经噪声层雅可比，质量高，作为逃逸训练的主力；
            #     （quant 的傅里叶近似梯度在 1~21 倍间震荡、jpeg 近似梯度劣质，
            #       若只在含噪图上提特征，2/3 的时间隐私梯度是坏信号，实测完全推不动）
            #   含噪分支——保证逃逸对 jpeg/resize 等信道失真鲁棒（传播场景必经压缩）
            feat_encoded_clean = self.content_recognizer(encoded_image)
            feat_encoded_noised = self.content_recognizer(noised_encoded_image)
            dino_encoded_clean = self.extract_dino_features(encoded_image, allow_grad=True)
            dino_encoded_noised = self.extract_dino_features(noised_encoded_image, allow_grad=True)
        else:
            # Phase 1：privacy 不参与优化，仅提取干净分支特征作监控（无梯度，省算力）
            with torch.no_grad():
                feat_encoded_clean = self.content_recognizer(encoded_image)
                dino_encoded_clean = self.extract_dino_features(encoded_image)
            feat_encoded_noised = None
            dino_encoded_noised = None

        clip_hard_negatives = None
        dino_hard_negatives = None

        # Refine 终段：redirect 关闭（val/refine_encode.py 精修配方为 escape-only）。
        # redirect 把编码特征拉向 hard negative 人群（目标 0.85），而人群聚集在
        # 原图邻域，与 escape 的"远离原图"存在方向对冲——精修试点证明
        # escape-only 可达 0.57，终段以逃逸深度优先
        in_refine = phase == 2 and self.cfg.is_phase2_refine_segment(epoch)

        # 先从记忆库检索 hard negatives（先查询、后入库，避免当前 batch 命中自身）；
        # exclude_sim 按特征空间分别标定，排除与 query 过近的候选（近重复/同身份照片），
        # 防止重定向目标退化为"自己"而与 escape loss 对冲。
        # 查询使用干净分支特征（与入库的 cover 特征同分布，redirect 目标也定义在干净空间）
        if self.use_clip_redirect and self.clip_memory_bank is not None and not in_refine:
            clip_hard_negatives = self.clip_memory_bank.get_hard_negatives(
                feat_encoded_clean.detach(), k=self.clip_hard_negative_k,
                exclude_sim=getattr(self.cfg, 'clip_redirect_exclude_sim', 0.92)
            )

        # 检索 DINO 记忆库的 hard negatives（仅当 DINO 启用且记忆库存在）
        if self.use_clip_redirect and self.dino_memory_bank is not None and dino_cover is not None and not in_refine:
            dino_hard_negatives = self.dino_memory_bank.get_hard_negatives(
                dino_encoded_clean.detach(), k=self.clip_hard_negative_k,
                exclude_sim=getattr(self.cfg, 'dino_redirect_exclude_sim', 0.87)
            )

        # 当前阶段/epoch 的 escape margin（Hinge 边界，渐进扩大；CLIP/DINO 独立调度）
        escape_margin = self.cfg.get_escape_margin(phase, epoch)
        dino_escape_margin = self.cfg.get_dino_escape_margin(phase, epoch)

        # 干净分支隐私损失：escape + redirect（redirect 目标定义在干净特征空间，仅在此分支启用）
        L_privacy_clean, privacy_loss_dict = self.retrieval_escape_loss(
            feat_cover, feat_encoded_clean,
            dino_cover=dino_cover,
            dino_encoded=dino_encoded_clean,
            clip_hard_negatives=clip_hard_negatives,
            dino_hard_negatives=dino_hard_negatives,
            escape_margin=escape_margin,
            dino_escape_margin=dino_escape_margin
        )

        if privacy_weight > 0 and feat_encoded_noised is not None:
            # 含噪分支隐私损失：仅 escape（无 redirect），保证逃逸对信道失真鲁棒
            L_privacy_noised, noised_privacy_dict = self.retrieval_escape_loss(
                feat_cover, feat_encoded_noised,
                dino_cover=dino_cover,
                dino_encoded=dino_encoded_noised,
                escape_margin=escape_margin,
                dino_escape_margin=dino_escape_margin
            )
            # 双分支加权融合：干净分支主导逃逸方向（梯度质量高），
            # 含噪分支仅锚定传播场景鲁棒性（其噪声层雅可比会稀释有效梯度）
            # Sprint 段含噪权重经 get_privacy_noised_ratio 调度为 0（纯逃逸冲刺）
            noised_ratio = self.cfg.get_privacy_noised_ratio(phase, epoch)
            L_privacy = (1.0 - noised_ratio) * L_privacy_clean + noised_ratio * L_privacy_noised
            # 传播场景的含噪 sim 单独记录（best checkpoint 判定面向传播效果）
            privacy_loss_dict['clip_sim_noised'] = noised_privacy_dict.get('clip_sim', 0.0)
            privacy_loss_dict['dino_sim_noised'] = noised_privacy_dict.get('dino_sim', 0.0)
        else:
            L_privacy = L_privacy_clean

        # 检索完成后再将当前 batch 的 cover 特征入库，供后续 batch 作为候选
        if self.use_clip_redirect and self.clip_memory_bank is not None:
            self.clip_memory_bank.update(feat_cover.detach())
        if self.use_clip_redirect and self.dino_memory_bank is not None and dino_cover is not None:
            self.dino_memory_bank.update(dino_cover.detach())

        # ===== 方案 C：内环逐图优化蒸馏（Phase 2）=====
        # 教师：冻结网络，对当前 batch 的低分辨率隐私残差场做少量 Adam 步直接优化
        # （与 val/refine_encode.py 逐图精修同一目标函数），得到探针级逃逸方向；
        # 学生：把 prv_head 的输出拉向教师残差，把该方向蒸进单次前向。
        # 教师目标含消息 BCE，蒸馏在 Sprint 段（message 权重 0）也起到保解码锚的作用
        # 关键：损失必须按教师能量归一化成相对误差。原始残差空间 MSE 量级 ~1e-5
        # （残差 RMS 仅 ~0.015，师生差异 RMS ~0.003），加权 10 后贡献 ~1e-4，
        # 对比总损失 ~10+（privacy 15）完全不可见——实测 A100 训练全程
        # L_distill 恒定 9e-5 不收敛，方案 C 数值上从未生效，Phase 2 只剩
        # 跨图方向互相抵消的 escape 梯度（0.01/10 轮的慢速下降根源）。
        # 归一化后 L_distill = 师生相对平方误差（当前约 0.4），加权 10 后
        # 与 privacy 同量级，棘轮机制真正启动
        L_distill = torch.tensor(0.0, device=self.device)
        if phase == 2 and getattr(self.cfg, 'distill_inner_steps', 0) > 0 and prv_residual is not None:
            teacher_prv = self._refine_teacher_residual(
                cover_image, message, msg_residual, prv_residual, feat_cover, dino_cover
            )
            if teacher_prv is not None:
                teacher_ref = teacher_prv.detach()
                L_distill = self.mse_loss(prv_residual, teacher_ref) / (
                    teacher_ref.pow(2).mean() + 1e-7
                )

        # ===== 残差惩罚与 FFT 高频惩罚 =====
        residual_penalty_weight = weights.get('residual_penalty', 0.05)  # 残差预算惩罚权重
        fft_weight = weights.get('fft', 0.0)  # FFT 高频惩罚权重（仅 Phase 2 > 0）
        # 仅在启用 FFT 且存在隐私残差时计算
        if fft_weight > 0 and prv_residual is not None:
            L_fft = self._fft_high_freq_loss(
                prv_residual, getattr(self.cfg, 'fft_high_freq_radius_ratio', 0.25)
            )
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
        # 蒸馏损失加权（Phase 2 且启用内环蒸馏时 > 0）
        L_distill_weighted = getattr(self.cfg, 'distill_weight', 0.0) * L_distill
        # 总损失：视觉 + 非视觉 + 隐私 + FFT + 蒸馏
        L_total = L_visual_weighted + L_non_visual + L_privacy_weighted + L_fft_weighted + L_distill_weighted

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

        # ===== 计算 bit accuracy 用于监控（含噪分支为主指标，干净分支辅助观察冷启动） =====
        with torch.no_grad():
            # 将 logits 转为二进制预测（>0.5 视为 1）
            message_pred_binary = (torch.sigmoid(message_pred) > 0.5).float()
            bit_accuracy = (message_pred_binary == message).float().mean().item()
            message_pred_clean_binary = (torch.sigmoid(message_pred_clean) > 0.5).float()
            bit_accuracy_clean = (message_pred_clean_binary == message).float().mean().item()

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
            'L_distill': L_distill.item(),  # 蒸馏损失（学生残差 vs 教师残差）
            'bit_accuracy': bit_accuracy,  # 比特准确率（含噪分支，确权主指标）
            'bit_accuracy_clean': bit_accuracy_clean,  # 干净分支比特准确率（冷启动观察用）
            'escape_margin': escape_margin,  # 当前 CLIP escape margin
            'dino_escape_margin': dino_escape_margin,  # 当前 DINO escape margin
            'prv_noise_ratio': prv_noise_ratio,  # 当前隐私噪声比例
            'D_updated': D_updated,  # 本 step 是否更新了判别器
            'adv_weight': weights.get('adv', 0.0)  # 对抗权重
        }
        losses.update(disc_metrics)  # 合并判别器诊断指标

        # 合并隐私损失的子项（前缀 privacy_）
        for key, value in privacy_loss_dict.items():
            losses[f'privacy_{key}'] = value

        return losses

    # 解码器恢复步骤：当 Phase 2 中 bit accuracy 跌破阈值时调用
    # 思路：冻结编码器，用"完整编码图像（含隐私残差 + 信道噪声）"
    #       重新训练解码器若干 batch，使恢复分布与实际解码分布一致
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

            # 编码器前向不需要梯度；以 Phase 2 模式生成含隐私残差的完整编码图像
            with torch.no_grad():
                encoded_image, _, _, _ = self.encoder(
                    cover_image, message, phase=2, prv_noise_ratio=0.0
                )
                # 施加与训练一致的信道噪声，保证恢复分布 = 实际解码分布
                noised_encoded, _ = self.noiser((encoded_image, cover_image))

            # 解码器在含噪完整编码图像上预测 message logits
            message_pred = self.decoder(noised_encoded)
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

        # 验证时使用与训练一致的噪声比例与 escape margin（CLIP/DINO 独立调度）
        prv_noise_ratio = self.cfg.get_prv_noise_ratio(epoch) if phase == 1 else 0.0
        escape_margin = self.cfg.get_escape_margin(phase, epoch)
        dino_escape_margin = self.cfg.get_dino_escape_margin(phase, epoch)

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
                # 解码器双分支预测（与 train_step 一致）
                message_pred = self.decoder(noised_encoded_image)
                message_pred_clean = self.decoder(encoded_image)

                # 视觉损失（原始）
                L_pixel = self.mse_loss(encoded_image, cover_image)
                L_ssim = self.ssim_loss(encoded_image, cover_image)
                L_visual = self.cfg.lambda_pixel * L_pixel + self.cfg.lambda_ssim * L_ssim

                # 视觉损失（含噪）
                visual_target_noised = noised_cover_image if self.has_noise else cover_image
                L_pixel_noised = self.mse_loss(noised_encoded_image, visual_target_noised)
                L_ssim_noised = self.ssim_loss(noised_encoded_image, visual_target_noised)
                L_visual_noised = self.cfg.lambda_pixel * L_pixel_noised + self.cfg.lambda_ssim * L_ssim_noised

                # 消息损失（干净 + 含噪双分支，与训练一致）
                L_message = 0.5 * (
                    self.bce_with_logits_loss(message_pred, message) +
                    self.bce_with_logits_loss(message_pred_clean, message)
                )

                # 对抗损失（验证时只评估不更新）
                disc_real_input, disc_stego_input = self._adversarial_inputs(
                    cover_image, encoded_image, noised_cover_image, noised_encoded_image
                )
                stego_pred = self.steganalyzer(disc_stego_input)
                cover_pred = self.steganalyzer(disc_real_input)
                L_D, L_adv, discr_cover_bce, discr_encod_bce, disc_metrics = self._adversarial_losses_and_metrics(
                    cover_pred, stego_pred
                )

                # CLIP/DINO 特征与隐私损失（与训练一致的双分支结构）
                feat_cover = self.content_recognizer(cover_image)
                feat_encoded_clean = self.content_recognizer(encoded_image)

                dino_cover = self.extract_dino_features(cover_image)
                dino_encoded_clean = self.extract_dino_features(encoded_image)

                # 干净分支：escape（redirect 在验证时不查询 memory bank，与训练 Phase 1 行为一致）
                L_privacy_clean, privacy_loss_dict = self.retrieval_escape_loss(
                    feat_cover, feat_encoded_clean,
                    dino_cover=dino_cover,
                    dino_encoded=dino_encoded_clean,
                    escape_margin=escape_margin,
                    dino_escape_margin=dino_escape_margin
                )

                if weights.get('privacy', 0.0) > 0:
                    # 含噪分支：仅 escape，锚定传播场景（与训练 Phase 2 一致）
                    feat_encoded_noised = self.content_recognizer(noised_encoded_image)
                    dino_encoded_noised = self.extract_dino_features(noised_encoded_image)
                    L_privacy_noised, noised_privacy_dict = self.retrieval_escape_loss(
                        feat_cover, feat_encoded_noised,
                        dino_cover=dino_cover,
                        dino_encoded=dino_encoded_noised,
                        escape_margin=escape_margin,
                        dino_escape_margin=dino_escape_margin
                    )
                    # 与训练一致的含噪分支权重（Sprint 段为 0）
                    noised_ratio = self.cfg.get_privacy_noised_ratio(phase, epoch)
                    L_privacy = (1.0 - noised_ratio) * L_privacy_clean + noised_ratio * L_privacy_noised
                    privacy_loss_dict['clip_sim_noised'] = noised_privacy_dict.get('clip_sim', 0.0)
                    privacy_loss_dict['dino_sim_noised'] = noised_privacy_dict.get('dino_sim', 0.0)
                else:
                    L_privacy = L_privacy_clean

                # 残差惩罚与 FFT 损失
                fft_weight = weights.get('fft', 0.0)
                if fft_weight > 0 and prv_residual is not None:
                    L_fft = self._fft_high_freq_loss(
                        prv_residual, getattr(self.cfg, 'fft_high_freq_radius_ratio', 0.25)
                    )
                else:
                    L_fft = torch.tensor(0.0, device=self.device)

                # 计算本 batch 的 bit accuracy（含噪主指标 + 干净分支）
                message_pred_binary = (torch.sigmoid(message_pred) > 0.5).float()
                bit_accuracy = (message_pred_binary == message).float().mean()
                message_pred_clean_binary = (torch.sigmoid(message_pred_clean) > 0.5).float()
                bit_accuracy_clean = (message_pred_clean_binary == message).float().mean()

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
                    ('bit_accuracy', bit_accuracy),
                    ('bit_accuracy_clean', bit_accuracy_clean)
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

                # 累加隐私相似度指标（用于多目标 best checkpoint 判定与日志监控）；
                # _noised 后缀为传播场景（含噪）指标，Phase 2 才存在
                for key in ['clip_sim', 'dino_sim', 'clip_sim_noised', 'dino_sim_noised']:
                    if key in privacy_loss_dict:
                        if key not in total_losses:
                            total_losses[key] = 0
                        total_losses[key] += privacy_loss_dict[key]

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
    #   extra_path: 可选的额外保存路径（如 best checkpoint）
    def save_checkpoint(self, epoch, phase, extra_path=None):
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
            'best_loss_phase': self.best_loss_phase,  # 该最佳值按哪个阶段的准则记录
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

        # 可选：额外保存一份到指定路径（如 checkpoint_best.pth）
        if extra_path is not None:
            torch.save(checkpoint, extra_path)
