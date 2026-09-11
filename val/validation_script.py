# 模块名: validation_script.py
# 模块职责: PrivSteg 项目的单张图像验证脚本。
# 该脚本加载训练好的 checkpoint，并支持以下验证模式:
#   - encode (编码): 将二进制消息嵌入到载体图像中
#   - decode (解码): 从编码图像中预测消息比特
#   - discriminator (判别器): 检测图像是否被隐写分析器判定为含密
#   - similarity (相似度): 对比编码图与载体图的 CLIP/DINO/SSIM/PSNR 相似度
#   - full (完整): 编码 -> 逐图精修 -> 解码-验证-判别-相似度检查
# full 模式默认在编码后执行逐图精修（与 refine_encode.py 同配方）：冻结全部网络，
# 仅对 96x96 低分辨率隐私残差场跑 Adam 优化，只在比特 100% 正确的状态里挑
# clip_sim 最低者，并以网络原始残差为初始候选——保证精修结果不劣于直接编码。
# (--refine-steps 0 可关闭精修；--refine-budget/--refine-margin 可覆盖约束与目标)
# 命令示例: python val/validation_script.py -c checkpoint.pth -i test.jpg -m "101000..." --mode full -o output
# 图像归一化到 [-1, 1] 区间，便于模型推理

import os  # 操作系统接口，用于文件路径检查与目录创建
import argparse  # 命令行参数解析
import time  # 计时模块，用于精修耗时统计
import torch  # PyTorch 主库，张量计算与模型推理
import torch.nn.functional as F  # PyTorch 函数式接口，提供 interpolate/cosine_similarity 等
import numpy as np  # 数值计算库，用于数组运算
from PIL import Image  # 图像读取与保存
from torchvision import transforms  # 图像预处理流水线 (Resize/ToTensor/Normalize)
import timm  # PyTorch Image Models，用于加载 DINOv2 ViT 模型

# 本脚本位于 val/ 子目录：定位项目根并加入 sys.path（import models/losses/config 等根模块），
# 同时把工作目录切到项目根，保证 ./models/dinov2_vits14_pretrain.pth 等相对路径
# 与从项目根直接运行时的行为完全一致
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from models import Encoder, Decoder, Steganalyzer, ContentRecognizer  # 项目核心模型组件
from losses import SSIMLoss  # 结构相似性损失，用于评估编码图像质量
from losses.advanced_privacy_loss import FeatureRedirectionLoss  # CLIP/DINO escape 损失，精修目标
from noise_layers.noiser import Noiser  # 信道噪声采样器，精修消息分支的含噪项
import config  # 项目配置模块，提供 Config 类


# _project_out_chroma_lowfreq 函数功能: 剔除隐私残差的 8x8 网格低频色度分量，返回修正后的残差。
# 背景：深挖配方实测残差能量 77% 落在色度通道、低频占其 44%，且总残差低频色度的 97% 来自
# prv_residual——构成"发灰/蒙层"感知劣化主体（PSNR/SSIM 对其失明）。梯度惩罚实测无效
# （逃逸梯度与色度方向高度平行，w=40~5000 均被淹没），故用结构化事后投影：精修收敛后把
# prv 的低频色度整支剔除。只动 8x8 网格（32px 块）尺度——更粗尺度是逃逸承重结构
# （实测剔除 16/32 网格后 clip 升至 0.57/0.80）。调用方需门控：投影后 bit_acc 必须
# 100% 且 clip_sim ≤ (1-margin)，否则回退未投影结果。
def _project_out_chroma_lowfreq(prv, grid=8):
    # prv: (B,3,H,W) RGB 隐私残差
    r, g, b = prv[:, 0:1], prv[:, 1:2], prv[:, 2:3]
    # BT.601 RGB->YCbCr 色度行
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    # 低频分量: grid x grid 自适应均值池化后双线性上采样回全分辨率
    cb_low = F.interpolate(F.adaptive_avg_pool2d(cb, grid), size=prv.shape[-2:],
                           mode='bilinear', align_corners=False)
    cr_low = F.interpolate(F.adaptive_avg_pool2d(cr, grid), size=prv.shape[-2:],
                           mode='bilinear', align_corners=False)
    # 逆变换剔除低频色度 (Y 不变): dR=1.402 dCr, dG=-0.344 dCb-0.714 dCr, dB=1.772 dCb
    out = prv.clone()
    out[:, 0:1] = out[:, 0:1] - 1.402 * cr_low
    out[:, 1:2] = out[:, 1:2] + 0.344136 * cb_low + 0.714136 * cr_low
    out[:, 2:3] = out[:, 2:3] - 1.772 * cb_low
    return out


# ValidationScript 类职责:
#   封装 PrivSteg 项目单图验证流程所需的所有组件与功能。
#   在初始化阶段构建编码器、解码器、隐写分析器、内容识别器 (CLIP)、
#   SSIM 损失模块，并可选地加载 DINOv2 模型用于特征级相似度评估。
#   提供编码、解码、判别器预测、相似度检查与完整验证等多种调用方式。
class ValidationScript:
    # __init__ 方法功能: 初始化验证脚本所需的所有模型、损失模块和预处理流水线。
    # 参数:
    #   checkpoint_path (str|None): checkpoint 文件路径，若提供则会加载权重到对应模型。
    #   device (torch.device|None): 指定运行设备，若为 None 则自动选择 CUDA/CPU。
    def __init__(self, checkpoint_path=None, device=None):
        # 确定运行设备: 优先使用外部传入，否则按 CUDA 可用性自动选择
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 实例化配置对象，从中读取所有模型超参与默认值
        self.cfg = config.Config()

        # 构造编码器，使用 Config 中的消息长度、通道数、图像尺寸与残差预算；
        # prv_lowres_size 必须与训练一致（它不改变权重形状，只改变 forward 的低通约束行为）
        self.encoder = Encoder(
            message_length=self.cfg.message_length,
            n_channels=3,
            image_size=self.cfg.image_size,
            msg_budget=self.cfg.msg_budget,
            prv_budget=self.cfg.prv_budget,
            prv_lowres_size=self.cfg.prv_lowres_size
        ).to(self.device)

        # 构造解码器，从编码图像预测消息比特 logits
        self.decoder = Decoder(
            message_length=self.cfg.message_length,
            n_channels=3
        ).to(self.device)

        # 构造隐写分析器 (判别器)，启用 spectral norm 提高训练稳定性
        self.steganalyzer = Steganalyzer(n_channels=3, use_spectral_norm=True).to(self.device)

        # 构造内容识别器 (CLIP)，用于评估编码图像与载体图像的特征相似度
        self.content_recognizer = ContentRecognizer(device=self.device).to(self.device)

        # 构造 SSIM 损失模块，用于衡量编码图像与载体图的结构相似性
        self.ssim_loss = SSIMLoss().to(self.device)

        # 信道噪声采样器（精修含噪消息分支使用），懒加载并缓存
        self._noiser = None

        # 加载 DINOv2 模型
        # 仅当 Config.dino_weight > 0 时启用 DINOv2，否则跳过以节省显存
        self.dino_model = None
        self.dino_mean = None
        self.dino_std = None
        # 读取 DINOv2 权重路径，未配置时返回 None
        dino_model_path = getattr(self.cfg, 'dino_model_path', None)
        if getattr(self.cfg, 'dino_weight', 0) > 0:
            # 通过 timm 创建 DINOv2 ViT-S/14 模型，关闭 pretrained 自动下载，
            # num_classes=0 表示只保留特征提取部分，img_size=224 是 DINOv2 的标准输入尺寸
            self.dino_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=224)
            # 若权重文件存在则加载，否则保持随机初始化并打印警告
            if dino_model_path and os.path.exists(dino_model_path):
                # 加载权重文件到 CPU/GPU (按 device)，weights_only=False 允许加载非纯张量结构
                state_dict = torch.load(dino_model_path, map_location=self.device, weights_only=False)
                # 兼容不同保存格式: 优先取 state_dict 字段，再退到 model 字段
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']

                # 获取当前模型的位置编码 pos_embed 张量，用于后续可能的形状对齐
                model_pos_embed = self.dino_model.pos_embed.data
                # 取 checkpoint 中的 pos_embed，若不存在则为 None
                ckpt_pos_embed = state_dict.get('pos_embed', None)
                # 当 checkpoint 的 pos_embed 与当前模型形状不一致时执行 2D 插值
                # 这通常发生在 patch_size 或输入分辨率不同的情况下
                if ckpt_pos_embed is not None and ckpt_pos_embed.shape != model_pos_embed.shape:
                    # 切出 CLS token (第 0 个位置)，单独保留不做插值
                    cls_token = ckpt_pos_embed[:, :1, :]
                    # 切出 patch tokens，需要在 2D 网格上插值
                    patch_embed = ckpt_pos_embed[:, 1:, :]
                    embed_dim = patch_embed.shape[-1]
                    # 原始 patch 网格边长 (假设为方形)
                    orig_grid_size = int(patch_embed.shape[1] ** 0.5)
                    # 当前模型期望的 patch 网格边长 (排除 CLS token 后开方)
                    new_grid_size = int((model_pos_embed.shape[1] - 1) ** 0.5)
                    # 重排为 (1, H, W, C) -> (1, C, H, W)，便于双三次插值
                    patch_embed = patch_embed.reshape(1, orig_grid_size, orig_grid_size, embed_dim).permute(0, 3, 1, 2)
                    # 双三次插值到目标网格尺寸
                    patch_embed = F.interpolate(patch_embed, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
                    # 恢复到 (1, N, C) 排布
                    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, embed_dim)
                    # 将 CLS token 与插值后的 patch tokens 在序列维度上拼接，覆盖原值
                    state_dict['pos_embed'] = torch.cat([cls_token, patch_embed], dim=1)

                # 以非严格模式加载，允许有部分键不匹配 (例如分类头)
                self.dino_model.load_state_dict(state_dict, strict=False)
                print(f"DINOv2 模型已加载: {dino_model_path}")
            else:
                # 权重缺失时仅给出警告，使用随机初始化的权重继续运行
                print(f"警告: DINOv2 模型未找到: {dino_model_path}，使用随机初始化")
            # 将 DINOv2 模型移至目标设备并切换为评估模式
            self.dino_model = self.dino_model.to(self.device)
            self.dino_model.eval()
            # 冻结所有参数，验证阶段不需要梯度更新
            for param in self.dino_model.parameters():
                param.requires_grad = False
            # DINOv2 推理时使用的标准 ImageNet 归一化均值 (形状: 1,3,1,1)
            self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            # DINOv2 推理时使用的标准 ImageNet 归一化标准差 (形状: 1,3,1,1)
            self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        else:
            # dino_weight=0 时直接跳过 DINOv2 加载，相似度评估将仅基于 CLIP
            print("DINOv2 未启用 (dino_weight=0)")

        # 输入图像预处理流水线: 缩放到目标尺寸 -> 转 tensor -> 归一化到 [-1, 1]
        # mean=std=0.5 等价于将 [0,1] 映射到 [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 反向预处理流水线: 将 [-1, 1] 还原到 [0, 1] 区间并裁剪
        # 数学上等价于 (x - mean) / std 的逆运算，其中 mean=-1, std=2
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
        ])

        # 若提供了 checkpoint 路径，则在初始化末尾加载权重
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    # load_checkpoint 方法功能: 从指定路径加载 checkpoint，并将各模型权重恢复到对应模块。
    # 参数:
    #   checkpoint_path (str): checkpoint 文件路径 (.pth)。
    # 加载过程会打印缺失键与多余键，并对 steganalyzer 的 fc_layers 命名做兼容性映射。
    def load_checkpoint(self, checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        # 记录 checkpoint 路径，供可视化报告的元信息展示
        self.checkpoint_path = checkpoint_path
        # 加载 checkpoint 字典到目标设备，weights_only=False 允许加载非张量结构
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 内部辅助函数: 报告 load_state_dict 返回的缺失键与多余键
        def _report(module_name, result):
            missing, unexpected = result
            if missing:
                print(f"[{module_name}] 缺失键: {list(missing)}")
            if unexpected:
                print(f"[{module_name}] 多余键: {list(unexpected)}")

        # 若 checkpoint 包含编码器权重则加载
        if 'encoder_state_dict' in checkpoint:
            # 形状过滤：prv_head 输入已扩为 256 通道（新增 prv_trunk 独立支路），
            # 旧 checkpoint 的隐私头卷积权重形状不匹配，跳过后隐私残差部分随机初始化
            # （消息通路权重完整加载，解码能力不受影响）；新结构 checkpoint 全量加载
            encoder_state = checkpoint['encoder_state_dict']
            model_state = self.encoder.state_dict()
            compatible = {k: v for k, v in encoder_state.items()
                          if k in model_state and model_state[k].shape == v.shape}
            skipped = [k for k in encoder_state if k not in compatible]
            if skipped:
                print(f"[encoder] 跳过 {len(skipped)} 个形状不兼容的权重（旧隐私头结构，隐私残差不可信）: {skipped[:6]}")
            result = self.encoder.load_state_dict(compatible, strict=False)
            _report('encoder', result)
        # 若 checkpoint 包含解码器权重则加载
        if 'decoder_state_dict' in checkpoint:
            result = self.decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)
            _report('decoder', result)
        # 若 checkpoint 包含隐写分析器权重则加载，并执行 fc_layers 命名兼容映射
        if 'steganalyzer_state_dict' in checkpoint:
            # 拷贝一份 state_dict，避免修改原始 checkpoint
            steganalyzer_state_dict = checkpoint['steganalyzer_state_dict'].copy()

            # 获取当前模型与 checkpoint 中所有键的集合，用于比对差异
            model_keys = set(self.steganalyzer.state_dict().keys())
            ckpt_keys = set(steganalyzer_state_dict.keys())
            # 遍历 checkpoint 中存在但模型中缺失的键，尝试做 fc_layers 命名重映射
            for ckpt_key in list(ckpt_keys - model_keys):
                # 仅处理 fc_layers 下的参数 (例如不同版本间的全连接层命名变化)
                if not ckpt_key.startswith('fc_layers.'):
                    continue
                # 截取第 2 个点之后的部分作为键后缀 (例如 "0.weight" 中的 "weight")
                suffix = ckpt_key.split('.', 2)[2]
                # 在模型独有键中查找以 fc_layers. 开头、且后缀相同的候选键，按字典序排序
                candidates = sorted(
                    k for k in (model_keys - ckpt_keys)
                    if k.startswith('fc_layers.') and k.endswith('.' + suffix)
                )
                # 若找到候选键，则将 checkpoint 中旧键的值迁移到新键名下
                if candidates:
                    target = candidates[0]
                    steganalyzer_state_dict[target] = steganalyzer_state_dict.pop(ckpt_key)
                    print(f"映射键: {ckpt_key} -> {target}")

            # 以非严格模式加载映射后的 steganalyzer 权重
            result = self.steganalyzer.load_state_dict(steganalyzer_state_dict, strict=False)
            _report('steganalyzer', result)

        # 将所有模型切换到评估模式 (eval)，关闭 dropout/BN 更新等训练行为
        self.encoder.eval()
        self.decoder.eval()
        self.steganalyzer.eval()
        self.content_recognizer.eval()
        print("模型加载完成!")

    # load_image 方法功能: 读取并预处理一张图像，返回模型可用的张量 (NCHW)。
    # 参数:
    #   image_path (str): 输入图像文件路径。
    # 返回:
    #   torch.Tensor: 形状为 (1, 3, H, W) 的张量，已归一化到 [-1, 1] 并放到目标设备。
    def load_image(self, image_path):
        print(f"加载图像: {image_path}")
        # 用 PIL 打开图像并强制转换为 RGB 三通道，避免 RGBA/灰度图导致通道不一致
        image = Image.open(image_path).convert('RGB')
        # 应用预处理流水线 (Resize/ToTensor/Normalize)，并增加 batch 维度
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        print(f"图像形状: {image_tensor.shape}")
        return image_tensor

    # process_message 方法功能: 处理输入的消息字符串或消息文件，输出模型可用的消息张量。
    # 参数:
    #   message_input (str): 二进制消息字符串或包含消息的文件路径。
    # 返回:
    #   message_tensor (torch.Tensor): 形状 (1, message_length) 的 float 张量。
    #   message_str (str): 经补齐/截断后的最终消息字符串。
    #   original_length (int): 输入消息的原始长度 (用于后续仅校验原始位)。
    def process_message(self, message_input):
        # 若输入是文件路径，则从文件中读取消息内容并 strip
        if os.path.isfile(message_input):
            with open(message_input, 'r') as f:
                message_str = f.read().strip()
        else:
            # 否则直接当作二进制字符串处理
            message_str = message_input

        # 仅保留 '0' 和 '1' 字符，过滤掉空格/换行/其他字符
        message_str = ''.join(c for c in message_str if c in '01')

        # 记录原始长度，后续仅对原始长度部分校验准确率，避免补 0 干扰结果
        original_length = len(message_str)
        # 目标长度来自配置，必须与模型训练时一致
        target_length = self.cfg.message_length

        # 长度不足时在末尾补 '0'，并打印提示
        if original_length < target_length:
            message_str = message_str + '0' * (target_length - original_length)
            print(f"消息长度 {original_length} < 目标长度 {target_length}，已自动补0")
        # 长度超出时截断到目标长度，并打印提示
        elif original_length > target_length:
            message_str = message_str[:target_length]
            print(f"消息长度 {original_length} > 目标长度 {target_length}，已自动截断")

        print(f"最终消息长度: {len(message_str)}")
        print(f"消息内容: {message_str}")

        # 将字符串逐字符转为 float 张量，并添加 batch 维度后放到目标设备
        message_tensor = torch.tensor([float(c) for c in message_str], dtype=torch.float32).unsqueeze(0).to(self.device)
        return message_tensor, message_str, original_length

    # encode_image 方法功能: 调用编码器将二进制消息嵌入到载体图像中，返回编码图像与残差惩罚。
    # 参数:
    #   cover_image (torch.Tensor): 载体图像张量，形状 (1, 3, H, W)，范围 [-1, 1]。
    #   message (torch.Tensor): 待嵌入消息张量，形状 (1, message_length)。
    # 返回:
    #   encoded_image (torch.Tensor): 编码后的图像张量，与 cover_image 同形状。
    #   residual_penalty (torch.Tensor): 编码器输出的残差惩罚标量，用于监控残差强度。
    # 注意: 验证阶段使用 phase=2 且 prv_noise_ratio=0.0，禁用随机隐私噪声。
    def encode_image(self, cover_image, message):
        print("\n" + "="*60)
        print("开始编码...")

        # 推理阶段关闭梯度计算，节省显存并加速
        with torch.no_grad():
            # 调用编码器，phase=2 表示激活 prv_head，prv_noise_ratio=0.0 表示不注入随机噪声
            encoded_image, residual_penalty, msg_residual, prv_residual = self.encoder(
                cover_image, message, phase=2, prv_noise_ratio=0.0
            )

        # 打印残差惩罚值，用于监控编码强度
        print(f"残差惩罚值: {residual_penalty.item():.6f}")
        return encoded_image, residual_penalty

    # _extract_dino_features_grad 方法功能: 提取 DINOv2 特征但保留输入梯度。
    # 与 extract_dino_features 的预处理完全一致，仅去掉 no_grad 包装——
    # 逐图精修需要对编码图像求梯度，不能在提取特征时截断计算图。
    # 参数:
    #   image (torch.Tensor): 输入图像张量，范围 [-1, 1]。
    # 返回:
    #   torch.Tensor|None: L2 归一化的 DINOv2 特征；模型未启用时返回 None。
    def _extract_dino_features_grad(self, image):
        if self.dino_model is None:
            return None
        image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        x = (image_resized + 1) / 2
        x = (x - self.dino_mean) / self.dino_std
        feats = self.dino_model(x)
        return feats / (feats.norm(dim=-1, keepdim=True) + 1e-7)

    # _build_noiser 方法功能: 按训练噪声配置懒加载并缓存信道采样器。
    # 解析逻辑与 trainer.py 一致；精修的含噪消息分支用它模拟传播信道。
    # 返回:
    #   Noiser: 信道噪声采样器（首次调用时构建，之后复用缓存）。
    def _build_noiser(self):
        if self._noiser is None:
            from noise_argparser import parse_crop, parse_cropout, parse_dropout, parse_resize
            from noise_layers.jpeg_compression import JpegCompression
            from noise_layers.quantization import Quantization
            layers = []
            if self.cfg.noise_config != 'identity':
                for command in self.cfg.noise_config.split('+'):
                    command = command.replace(' ', '')
                    if command.startswith('cropout'):
                        layers.append(parse_cropout(command))
                    elif command.startswith('crop'):
                        layers.append(parse_crop(command))
                    elif command.startswith('dropout'):
                        layers.append(parse_dropout(command))
                    elif command.startswith('resize'):
                        layers.append(parse_resize(command))
                    elif command.startswith('jpeg'):
                        layers.append(JpegCompression(self.device))
                    elif command.startswith('quant'):
                        layers.append(Quantization(self.device))
            self._noiser = Noiser(layers, self.device).to(self.device)
        return self._noiser

    # _refine_metrics 方法功能: 以验证同口径计算精修过程的监控指标。
    # 参数:
    #   encoded (torch.Tensor): 编码图像张量。
    #   message (torch.Tensor): 目标消息张量。
    #   cover (torch.Tensor): 载体图像张量（计算 PSNR/SSIM 用）。
    #   feat_cover_clip/feat_cover_dino (torch.Tensor|None): 预提取的载体特征（避免重复前向）。
    #   prv (torch.Tensor|None): 隐私残差（计算 RMS 用）。
    # 返回:
    #   dict: clip_sim/dino_sim/bit_acc/psnr/ssim/prv_rms。
    def _refine_metrics(self, encoded, message, cover, feat_cover_clip, feat_cover_dino, prv=None):
        with torch.no_grad():
            feat_enc = self.content_recognizer(encoded)
            clip_sim = F.cosine_similarity(feat_cover_clip, feat_enc, dim=1).item()
            dino_sim = None
            if feat_cover_dino is not None:
                dino_enc = self._extract_dino_features_grad(encoded)
                dino_sim = F.cosine_similarity(feat_cover_dino, dino_enc, dim=1).item()
            pred = (torch.sigmoid(self.decoder(encoded)) > 0.5).float()
            bit_acc = (pred == message).float().mean().item()
            mse = torch.mean((encoded - cover) ** 2).item()
            psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')
            ssim = 1 - self.ssim_loss(cover, encoded).item()
            result = {'clip_sim': clip_sim, 'dino_sim': dino_sim, 'bit_acc': bit_acc,
                      'psnr': psnr, 'ssim': ssim}
            if prv is not None:
                result['prv_rms'] = prv.pow(2).mean().sqrt().item()
                # 色度 RMS：蒙层监测指标（色度残差 RMS，感知劣化主体，PSNR/SSIM 测不到）
                r, g, b = prv[:, 0:1], prv[:, 1:2], prv[:, 2:3]
                cb = -0.168736 * r - 0.331264 * g + 0.5 * b
                cr = 0.5 * r - 0.418688 * g - 0.081312 * b
                result['chroma_rms'] = torch.sqrt((cb ** 2 + cr ** 2).mean()).item()
        return result

    # refine_encoded_image 方法功能: 编码期逐图精修（与 refine_encode.py 完全同配方）。
    # 冻结全部网络参数，仅把 96x96 低分辨率隐私残差场作为优化变量跑 steps 步 Adam:
    #   L = w_privacy*(CLIP/DINO escape) + w_message*(干净+含噪 BCE) + w_visual*(MSE+SSIM)
    #   收敛后可选色度蒙层投影（chroma_projection, 见 _project_out_chroma_lowfreq）
    # 约束链与训练同构: 96 低通 -> 双三次上采样 -> tanh*prv_budget -> 纹理掩码 -> clamp。
    # 消息残差 msg_residual 完全取自冻结网络、不参与优化；只在比特 100% 正确的状态里
    # 挑选 clip_sim 最低者，且以网络原始残差作为初始候选——保证精修结果不劣于直接编码。
    # 参数:
    #   cover_image (torch.Tensor): 载体图像张量 (1,3,H,W)，范围 [-1,1]。
    #   message (torch.Tensor): 消息张量 (1,message_length)。
    #   steps (int): Adam 优化步数。
    #   lr (float): Adam 学习率。
    #   w_privacy/w_message/w_visual (float): 精修目标三项权重（验证配方 12/10/15）。
    #   margin_end (float|None): CLIP escape margin 覆盖，None 取 Config 默认。
    #   prv_budget (float|None): 隐私残差预算覆盖，None 取 Config 默认。
    #   log_interval (int): 指标打印间隔步数。
    #   verbose (bool): 是否打印精修过程日志（批量评估时可关闭以减少输出）。
    #   chroma_projection (bool|None): 收敛后剔除 prv 低频色度蒙层（None 取 Config 默认）。
    # 返回:
    #   (encoded_refined, report): 精修后的编码图像；report 含基线/精修指标与逐步日志。
    def refine_encoded_image(self, cover_image, message, steps=60, lr=0.1,
                             w_privacy=12.0, w_message=10.0, w_visual=15.0,
                             margin_end=None, prv_budget=None, log_interval=10,
                             verbose=True, chroma_projection=None):
        cfg = self.cfg
        H, W = cover_image.shape[-2], cover_image.shape[-1]
        budget = prv_budget if prv_budget is not None else cfg.prv_budget
        margin = margin_end if margin_end is not None else cfg.privacy_margin_end
        # 色度蒙层事后投影：None 取 Config 默认（False = 关闭，回退原精修配方）
        if chroma_projection is None:
            chroma_projection = getattr(cfg, 'refine_chroma_projection', False)

        if verbose:
            print("\n" + "="*60)
            print(f"开始逐图精修 (steps={steps}, lr={lr}, budget={budget}, margin={margin}, "
                  f"chroma_projection={chroma_projection})...")

        # 冻结编码器/解码器参数（记录原状态，结束后恢复）
        saved_states = []
        for module in (self.encoder, self.decoder):
            for p in module.parameters():
                saved_states.append((p, p.requires_grad))
                p.requires_grad = False

        try:
            with torch.no_grad():
                # 基线编码：消息残差与初始隐私残差均取自冻结网络
                encoded_base, _, msg_residual, prv_residual = self.encoder(
                    cover_image, message, phase=2, prv_noise_ratio=0.0
                )
                feat_cover_clip = self.content_recognizer(cover_image)
                feat_cover_dino = self._extract_dino_features_grad(cover_image)
                texture_mask = self.encoder._compute_texture_mask(cover_image)
                base_metrics = self._refine_metrics(
                    encoded_base, message, cover_image,
                    feat_cover_clip, feat_cover_dino, prv=prv_residual
                )

            # 精修变量初始化: 逆约束链（纹理掩码/tanh 预算）把网络残差映射回
            # 低分辨率 pre-tanh 场，从网络已学方向热启动
            with torch.no_grad():
                ratio = (prv_residual / (budget * texture_mask + 1e-8)).clamp(-0.95, 0.95)
                delta0 = F.adaptive_avg_pool2d(
                    torch.atanh(ratio), (cfg.prv_lowres_size, cfg.prv_lowres_size)
                )

            def build_encoded(d):
                # 与训练同构的约束链: 低分辨率场 -> 双三次上采样 -> tanh*预算 -> 纹理掩码 -> clamp
                prv_up = F.interpolate(d, size=(H, W), mode='bicubic', align_corners=False)
                prv = budget * torch.tanh(prv_up) * texture_mask
                return torch.clamp(cover_image + msg_residual + prv, -1, 1), prv

            def eval_delta(d):
                with torch.no_grad():
                    enc, prv = build_encoded(d)
                    return self._refine_metrics(
                        enc, message, cover_image, feat_cover_clip, feat_cover_dino, prv=prv
                    )

            # 网络原始残差作为初始最优候选（精修不劣化保证）
            m_init = eval_delta(delta0)
            best = (m_init['clip_sim'], delta0.clone(), m_init) if m_init['bit_acc'] >= 1.0 else None

            delta = torch.nn.Parameter(delta0.clone())
            opt = torch.optim.Adam([delta], lr=lr)
            bce = torch.nn.BCEWithLogitsLoss()
            mse = torch.nn.MSELoss()
            escape = FeatureRedirectionLoss(
                clip_weight=1.0, dino_weight=cfg.dino_weight,
                escape_margin=margin,
                clip_redirect_weight=0.0, dino_redirect_weight=0.0
            )
            noiser = self._build_noiser()
            clean_ratio = cfg.get_msg_clean_ratio(2, 1)

            t0 = time.time()
            log = []
            for step in range(1, steps + 1):
                opt.zero_grad(set_to_none=True)
                encoded, _ = build_encoded(delta)

                # 隐私逃逸: CLIP/DINO 二次 Hinge escape
                feat_enc = self.content_recognizer(encoded)
                dino_enc = self._extract_dino_features_grad(encoded) if feat_cover_dino is not None else None
                L_priv, _ = escape(
                    feat_cover_clip, feat_enc,
                    dino_cover=feat_cover_dino, dino_encoded=dino_enc,
                    escape_margin=margin,
                    dino_escape_margin=cfg.dino_escape_margin_end
                )

                # 消息锚: 干净 + 含噪双分支 BCE（残差不得破坏解码）
                L_msg = clean_ratio * bce(self.decoder(encoded), message)
                noised_enc, _ = noiser((encoded, cover_image))
                L_msg = L_msg + (1.0 - clean_ratio) * bce(self.decoder(noised_enc), message)

                # 视觉锚: MSE + SSIM
                L_vis = mse(encoded, cover_image) + self.ssim_loss(cover_image, encoded)

                L = w_privacy * L_priv + w_message * L_msg + w_visual * L_vis
                L.backward()
                opt.step()

                if step == 1 or step % log_interval == 0 or step == steps:
                    m = eval_delta(delta)
                    log.append({'step': step, **m})
                    if verbose:
                        dino_str = f"{m['dino_sim']:.4f}" if m['dino_sim'] is not None else '  --  '
                        chroma_str = (f"{m['chroma_rms']:.4f}"
                                      if m.get('chroma_rms') is not None else '  --  ')
                        print(f"    step {step:3d} | clip {m['clip_sim']:.4f} | dino {dino_str} | "
                              f"bit {m['bit_acc']*100:5.1f}% | psnr {m['psnr']:5.2f} | "
                              f"ssim {m['ssim']:.4f} | prv_rms {m['prv_rms']:.4f} "
                              f"| chroma_rms {chroma_str}")
                    # 最优状态必须比特全对，且 clip_sim 最低
                    if m['bit_acc'] >= 1.0 and (best is None or m['clip_sim'] < best[0]):
                        best = (m['clip_sim'], delta.detach().clone(), m)

            # 没有任何比特全对的状态时退回最终步（诚实暴露消息失败）
            final_delta = best[1] if best is not None else delta.detach().clone()
            with torch.no_grad():
                encoded_refined, prv_refined = build_encoded(final_delta)
                # 色度蒙层事后投影: 剔除 prv 的 8x8 网格低频色度（"发灰/蒙层"主体，
                # 梯度惩罚实测无效——逃逸方向与色度方向高度平行，只能结构化剔除）。
                # 门控: 投影后 bit_acc 必须 100% 且 clip_sim 仍 ≤ (1-margin) 逃逸目标，
                # 否则回退未投影结果（投影是增益项，绝不允许劣化两个硬约束）
                projected = False
                if chroma_projection:
                    prv_proj = _project_out_chroma_lowfreq(prv_refined)
                    enc_proj = torch.clamp(cover_image + msg_residual + prv_proj, -1, 1)
                    m_proj = self._refine_metrics(
                        enc_proj, message, cover_image,
                        feat_cover_clip, feat_cover_dino, prv=prv_proj
                    )
                    if m_proj['bit_acc'] >= 1.0 and m_proj['clip_sim'] <= 1.0 - margin:
                        encoded_refined, prv_refined = enc_proj, prv_proj
                        projected = True
                refined_metrics = self._refine_metrics(
                    encoded_refined, message, cover_image,
                    feat_cover_clip, feat_cover_dino, prv=prv_refined
                )
            refine_time = time.time() - t0

            if verbose:
                b, r = base_metrics, refined_metrics
                chroma_note = (f" | chroma {b.get('chroma_rms', float('nan')):.4f}"
                               f"->{r.get('chroma_rms', float('nan')):.4f}"
                               ) if r.get('chroma_rms') is not None else ""
                proj_note = " | 色度投影已应用" if projected else ""
                print(f"精修完成: clip {b['clip_sim']:.4f} -> {r['clip_sim']:.4f} | "
                      f"bit {r['bit_acc']*100:.2f}% | psnr {r['psnr']:.2f} | "
                      f"ssim {r['ssim']:.4f}{chroma_note}{proj_note} | 用时 {refine_time:.0f}s")

            report = {
                'base_metrics': base_metrics,
                'refined_metrics': refined_metrics,
                'log': log,
                'time_sec': refine_time,
                'steps': steps, 'lr': lr, 'budget': budget, 'margin': margin,
                'chroma_projection_applied': projected
            }
            return encoded_refined, report
        finally:
            # 恢复网络参数的原 requires_grad 状态
            for p, state in saved_states:
                p.requires_grad = state

    # decode_message 方法功能: 调用解码器从编码图像中预测消息比特。
    # 参数:
    #   encoded_image (torch.Tensor): 编码后的图像张量。
    # 返回:
    #   message_pred_binary (torch.Tensor): 二值化后的预测消息 (0/1 float)。
    #   message_pred_sigmoid (torch.Tensor): sigmoid 后的预测概率。
    # 注意: 解码器输出 logits，需要在外部手动加 sigmoid 才能得到概率。
    def decode_message(self, encoded_image):
        print("\n" + "="*60)
        print("开始解码...")

        # 推理阶段关闭梯度
        with torch.no_grad():
            # 解码器输出原始 logits，未经过 sigmoid
            message_pred = self.decoder(encoded_image)

        # 对 logits 应用 sigmoid 得到 (0, 1) 区间的概率
        message_pred_sigmoid = torch.sigmoid(message_pred)
        # 以 0.5 为阈值二值化，得到最终的 0/1 比特预测
        message_pred_binary = (message_pred_sigmoid > 0.5).float()

        return message_pred_binary, message_pred_sigmoid

    # verify_message_extraction 方法功能: 对比原始消息与解码消息，统计比特准确率。
    # 参数:
    #   original_message (torch.Tensor): 原始消息张量。
    #   decoded_message (torch.Tensor): 解码得到的二值消息张量。
    #   original_length (int|None): 输入消息的原始长度，若小于 message_length 则只校验该长度。
    # 返回:
    #   bit_accuracy (float): 比特准确率 (0~1)。
    #   correct_bits (int): 正确比特数。
    #   total_bits (int): 参与比较的比特总数。
    def verify_message_extraction(self, original_message, decoded_message, original_length=None):
        print("\n" + "="*60)
        print("消息提取验证:")

        # 去掉 batch 维度并转回 CPU 上的 numpy 数组，便于逐比特比较
        original_binary = original_message.squeeze().cpu().numpy()
        decoded_binary = decoded_message.squeeze().cpu().numpy()

        # 若原始长度小于消息长度，则只比较原始部分，避免补 0 干扰结果
        if original_length is not None and original_length < len(original_binary):
            original_binary = original_binary[:original_length]
            decoded_binary = decoded_binary[:original_length]
            print(f"(仅计算原始输入的 {original_length} 位，不包括补位0)")

        # 统计两序列对应位相同的数量
        correct_bits = np.sum(original_binary == decoded_binary)
        total_bits = len(original_binary)
        # 计算比特准确率
        bit_accuracy = correct_bits / total_bits

        # 打印原始与解码消息，便于人工核对
        print(f"原始消息: {''.join(str(int(b)) for b in original_binary)}")
        print(f"解码消息: {''.join(str(int(b)) for b in decoded_binary)}")
        print(f"比特准确率: {bit_accuracy*100:.2f}%")
        print(f"正确比特数: {correct_bits}/{total_bits}")
        print(f"错误比特数: {total_bits - correct_bits}")

        return bit_accuracy, correct_bits, total_bits

    # discriminator_predict 方法功能: 调用隐写分析器对输入图像做隐写检测预测。
    # 参数:
    #   image (torch.Tensor): 待检测的图像张量。
    #   image_name (str): 用于日志展示的图像名称 (例如 "原始图像"/"隐写图像")。
    # 返回:
    #   stego_prob (float): 隐写概率 (1 - sigmoid(logit))，越大越像含密图像。
    #   prediction.item() (float): 判别器输出的原始 logit 值。
    # 说明: 隐写分析器输出"原始图像"概率 (sigmoid 后)，隐写概率 = 1 - 该值。
    def discriminator_predict(self, image, image_name="图像"):
        print("\n" + "="*60)
        print(f"判别器预测 ({image_name}):")

        # 推理阶段关闭梯度
        with torch.no_grad():
            # 判别器输出 logits (标量)
            prediction = self.steganalyzer(image)
            # sigmoid 得到判为"原始图像"的概率
            probability = torch.sigmoid(prediction)

        # 取出"原始图像"概率 (标量)
        cover_prob = probability.item()
        # 隐写概率 = 1 - 原始概率
        stego_prob = 1.0 - cover_prob
        print(f"判别器输出 (logits): {prediction.item():.6f}")
        print(f"原始图像概率 (sigmoid): {cover_prob*100:.2f}%")
        print(f"隐写概率 (1-sigmoid): {stego_prob*100:.2f}%")
        # 以 0.5 为阈值给出最终判定
        print(f"判断结果: {'原始图像' if cover_prob > 0.5 else '隐写图像'}")

        return stego_prob, prediction.item()

    # extract_dino_features 方法功能: 用 DINOv2 模型提取图像的特征向量 (已归一化)。
    # 参数:
    #   image (torch.Tensor): 输入图像张量，范围 [-1, 1]。
    # 返回:
    #   torch.Tensor|None: DINOv2 输出的 L2 归一化特征；若 DINOv2 未启用则返回 None。
    # 处理流程: 缩放到 224x224 -> 反归一化回 [0,1] -> ImageNet 归一化 -> 模型推理 -> L2 归一化。
    def extract_dino_features(self, image):
        # 若 DINOv2 未启用则直接返回 None
        if self.dino_model is None:
            return None
        with torch.no_grad():
            # 将图像缩放到 DINOv2 标准输入尺寸 224x224
            image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            # 从 [-1, 1] 反归一化回 [0, 1] 区间
            image_normalized = (image_resized + 1) / 2
            # 应用 ImageNet 标准均值/标准差归一化
            image_normalized = (image_normalized - self.dino_mean) / self.dino_std
            # 通过 DINOv2 提取特征 (num_classes=0，输出 backbone 特征)
            features = self.dino_model(image_normalized)
            # L2 归一化特征向量，便于后续 cosine 相似度计算 (加 1e-7 避免除零)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-7)
        return features

    # privacy_similarity 方法功能: 对比载体图像与编码图像的多维度相似度，评估隐私保护效果。
    # 参数:
    #   cover_image (torch.Tensor): 载体图像张量。
    #   encoded_image (torch.Tensor): 编码后的图像张量。
    # 返回:
    #   dict: 包含 CLIP/DINO 余弦相似度、欧氏距离、SSIM、PSNR 等指标的字典。
    # 评估维度:
    #   - CLIP 特征余弦相似度/欧氏距离 (反映 CBIR 检索关联风险)
    #   - DINOv2 特征余弦相似度/欧氏距离 (更贴近视觉感知的特征)
    #   - SSIM 结构相似性 (反映视觉质量)
    #   - PSNR 峰值信噪比 (反映像素级保真度)
    def privacy_similarity(self, cover_image, encoded_image):
        print("\n" + "="*60)
        print("隐私相似度判断:")

        # 提取 CLIP 特征 (推理阶段不计算梯度)
        with torch.no_grad():
            feat_cover = self.content_recognizer(cover_image)
            feat_encoded = self.content_recognizer(encoded_image)

        # CLIP 特征的余弦相似度，dim=1 表示在特征维度上计算
        cosine_sim = F.cosine_similarity(feat_cover, feat_encoded, dim=1).item()

        # CLIP 特征的 L2 欧氏距离
        euclidean_dist = torch.norm(feat_encoded - feat_cover, p=2, dim=1).item()

        # CLIP/DINO 余弦相似度理论范围 [-1, 1]，clamp 到 [0, 1] 后用作隐私指标
        # similarity_prob 越大表示越容易被 CBIR 关联到原图，隐私保护越弱
        similarity_prob = max(0.0, min(1.0, cosine_sim))

        # DINO 相似度
        # 仅在 DINOv2 启用时计算
        dino_sim = None
        dino_euclidean_dist = None
        if self.dino_model is not None:
            # 分别提取载体图与编码图的 DINOv2 特征
            feat_cover_dino = self.extract_dino_features(cover_image)
            feat_encoded_dino = self.extract_dino_features(encoded_image)
            # DINOv2 特征的余弦相似度
            dino_sim = F.cosine_similarity(feat_cover_dino, feat_encoded_dino, dim=1).item()
            # DINOv2 特征的 L2 欧氏距离
            dino_euclidean_dist = torch.norm(feat_encoded_dino - feat_cover_dino, p=2, dim=1).item()
            # 同样将 DINO 余弦相似度 clamp 到 [0, 1] 用作隐私指标
            dino_sim_prob = max(0.0, min(1.0, dino_sim))
        else:
            # DINOv2 未启用时设为 None，下游判断会跳过该指标
            dino_sim_prob = None

        # 计算 SSIM: SSIMLoss 返回的是 1 - SSIM，所以用 1 减回去还原
        ssim_value = 1 - self.ssim_loss(cover_image, encoded_image).item()

        # 计算像素级 MSE
        mse = torch.mean((encoded_image - cover_image) ** 2).item()
        # PSNR 公式: 10 * log10(MAX^2 / MSE)；数据范围 [-1, 1]，故 MAX^2 = 4
        if mse > 0:
            psnr = 10 * np.log10(4.0 / mse)
        else:
            # 完全一致时 PSNR 视为无穷大
            psnr = float('inf')

        # 打印各项相似度指标
        print(f"CLIP 余弦相似度: {cosine_sim:.6f}")
        print(f"CLIP 欧氏距离: {euclidean_dist:.6f}")
        if dino_sim is not None:
            print(f"DINO 余弦相似度: {dino_sim:.6f}")
            print(f"DINO 欧氏距离: {dino_euclidean_dist:.6f}")
        print(f"SSIM: {ssim_value:.6f}")
        print(f"PSNR: {psnr:.2f} dB")
        # 打印隐私判断结论: 相似度越高，保护程度越低
        print("\n隐私判断结论:")
        print(f"  - CLIP 特征相似度: {similarity_prob*100:.2f}%")
        print(f"  - CLIP 隐私保护程度: {(1-similarity_prob)*100:.2f}%")
        if dino_sim is not None:
            print(f"  - DINO 特征相似度: {dino_sim_prob*100:.2f}%")
            print(f"  - DINO 隐私保护程度: {(1-dino_sim_prob)*100:.2f}%")

        # 返回所有指标的字典，便于后续汇总与外部使用
        return {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'similarity_probability': similarity_prob,
            'dino_similarity': dino_sim,
            'dino_similarity_probability': dino_sim_prob,
            'dino_euclidean_distance': dino_euclidean_dist,
            'ssim': ssim_value,
            'psnr': psnr
        }

    # save_image 方法功能: 将模型输出的图像张量反归一化后保存为 PNG 文件。
    # 参数:
    #   image_tensor (torch.Tensor): 图像张量，形状 (1, 3, H, W)，范围 [-1, 1]。
    #   save_path (str): 输出 PNG 文件路径。
    def save_image(self, image_tensor, save_path):
        # 去掉 batch 维度，应用反向归一化 ([-1,1] -> [0,1])，转回 CPU numpy
        image_np = self.inverse_transform(image_tensor.squeeze(0)).cpu().numpy()
        # 将 CHW 排布转换为 HWC，便于 PIL 处理
        image_np = np.transpose(image_np, (1, 2, 0))
        # 将 [0,1] float 转为 [0,255] uint8
        image_np = (image_np * 255).astype(np.uint8)
        # 从 numpy 数组构造 PIL 图像并保存
        image = Image.fromarray(image_np)
        image.save(save_path)
        print(f"图像已保存: {save_path}")

    # run_full_validation 方法功能: 执行完整的验证流程，包括编码、逐图精修、解码、消息验证、判别器预测、相似度检查。
    # 参数:
    #   image_path (str): 载体图像路径。
    #   message (str): 待嵌入的二进制消息或消息文件路径。
    #   output_dir (str|None): 输出目录路径，若为 None 则在原图旁创建 <name>_stego_output 目录。
    #   refine_steps (int): 逐图精修步数（默认 60；0 关闭精修）。
    #   refine_lr (float): 精修 Adam 学习率。
    #   refine_budget (float|None): 精修隐私残差预算覆盖，None 取 Config 默认。
    #   refine_margin (float|None): 精修 CLIP escape margin 覆盖，None 取 Config 默认。
    #   refine_chroma_projection (int|None): 收敛后剔除 prv 低频色度蒙层（None 取 Config 默认）。
    # 返回:
    #   dict: 包含比特准确率、判别器概率、隐私相似度结果、精修报告等汇总指标。
    def run_full_validation(self, image_path, message, output_dir=None,
                            refine_steps=30, refine_lr=0.1,
                            refine_budget=None, refine_margin=None,
                            refine_chroma_projection=None):
        print("\n" + "#"*60)
        print("# 完整验证流程")
        print("#"*60)

        # 1. 加载载体图像并预处理为模型输入张量
        cover_image = self.load_image(image_path)
        # 2. 处理消息字符串/文件，得到消息张量与原始长度
        message_tensor, message_str, original_length = self.process_message(message)

        # 3. 调用编码器将消息嵌入载体图像（基线编码）
        encoded_image, residual_penalty = self.encode_image(cover_image, message_tensor)

        # 3.5 逐图精修（默认开启）: 对隐私残差再优化 refine_steps 步深挖逃逸，
        # 消息与视觉质量由精修目标函数硬保证；后续所有验证步骤（解码/判别器/
        # 相似度/保存）都基于精修后的图像，确保输出图像达标
        refine_report = None
        if refine_steps and refine_steps > 0:
            encoded_image, refine_report = self.refine_encoded_image(
                cover_image, message_tensor,
                steps=refine_steps, lr=refine_lr,
                margin_end=refine_margin, prv_budget=refine_budget,
                chroma_projection=(None if refine_chroma_projection is None
                                   else bool(refine_chroma_projection))
            )

        # 4. 调用解码器从编码图像中提取消息
        decoded_message, decoded_prob = self.decode_message(encoded_image)

        # 5. 对比原始消息与解码消息，计算比特准确率
        bit_accuracy, correct_bits, total_bits = self.verify_message_extraction(
            message_tensor, decoded_message, original_length
        )

        # 6. 判别器对原始图像做预测，得到原图被判为含密的概率
        cover_prob, _ = self.discriminator_predict(cover_image, "原始图像")
        # 7. 判别器对编码图像做预测，得到隐写图被判为含密的概率
        stego_prob, _ = self.discriminator_predict(encoded_image, "隐写图像")

        # 8. 计算 CLIP/DINO/SSIM/PSNR 等多维度隐私相似度指标
        privacy_results = self.privacy_similarity(cover_image, encoded_image)

        # 若未指定输出目录，则在原图同目录下创建 <image_name>_stego_output 子目录
        if output_dir is None:
            image_dir = os.path.dirname(os.path.abspath(image_path))
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.join(image_dir, f"{image_name}_stego_output")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        # 构造原图与编码图的保存路径
        cover_save_path = os.path.join(output_dir, 'cover_image.png')
        encoded_save_path = os.path.join(output_dir, 'encoded_image.png')
        # 保存原图与编码图
        self.save_image(cover_image, cover_save_path)
        self.save_image(encoded_image, encoded_save_path)
        print(f"\n输出目录: {output_dir}")

        # 9. 生成可视化报告 report.html（自包含 HTML，浏览器直接打开；
        #    matplotlib 缺失或生成失败时跳过，不影响验证流程本身）
        try:
            from visual_report import generate_html_report
            report_path = generate_html_report(
                output_dir,
                cover_image=cover_image,
                encoded_image=encoded_image,
                message_bits=message_tensor.squeeze(0).cpu().numpy(),
                decoded_bits=decoded_message.squeeze(0).cpu().numpy(),
                decoded_probs=decoded_prob.squeeze(0).cpu().numpy(),
                bit_accuracy=bit_accuracy,
                correct_bits=correct_bits,
                total_bits=total_bits,
                cover_prob=cover_prob,
                stego_prob=stego_prob,
                privacy_results=privacy_results,
                refine_report=refine_report,
                meta={
                    'checkpoint': getattr(self, 'checkpoint_path', None) or '(not provided)',
                    'input image': os.path.basename(image_path),
                    'device': str(self.device),
                    'refine steps': refine_steps if refine_steps else 'off',
                    'refine budget': refine_budget if refine_budget is not None else self.cfg.prv_budget,
                    'refine margin': refine_margin if refine_margin is not None else self.cfg.privacy_margin_end,
                    'message length': total_bits,
                    'dino_margin': self.cfg.dino_escape_margin_end,
                }
            )
            if report_path:
                print(f"可视化报告: {report_path}")
        except Exception as e:
            print(f"可视化报告生成失败: {e}")

        # 打印验证结果汇总
        print("\n" + "#"*60)
        print("# 验证结果汇总")
        print("#"*60)
        print(f"消息提取准确率: {bit_accuracy*100:.2f}%")
        print(f"判别器预测 - 原图隐写概率: {cover_prob*100:.2f}%")
        print(f"判别器预测 - 隐写图隐写概率: {stego_prob*100:.2f}%")
        print(f"CLIP 隐私相似度: {privacy_results['similarity_probability']*100:.2f}%")
        print(f"CLIP 隐私保护程度: {(1-privacy_results['similarity_probability'])*100:.2f}%")
        # 若 DINOv2 启用，则额外打印 DINO 相关指标
        if privacy_results.get('dino_similarity') is not None:
            print(f"DINO 隐私相似度: {privacy_results['dino_similarity_probability']*100:.2f}%")
            print(f"DINO 隐私保护程度: {(1-privacy_results['dino_similarity_probability'])*100:.2f}%")
        # 逐图精修前后对比（精修开启时）
        if refine_report is not None:
            b, r = refine_report['base_metrics'], refine_report['refined_metrics']
            print(f"逐图精修: CLIP 相似度 {b['clip_sim']:.4f} -> {r['clip_sim']:.4f} "
                  f"(隐私保护 {(1-b['clip_sim'])*100:.2f}% -> {(1-r['clip_sim'])*100:.2f}%)")
            print(f"          比特准确率 {b['bit_acc']*100:.2f}% -> {r['bit_acc']*100:.2f}% | "
                  f"PSNR {b['psnr']:.2f} -> {r['psnr']:.2f} dB | "
                  f"SSIM {b['ssim']:.4f} -> {r['ssim']:.4f}")
        print("#"*60)

        # 返回汇总结果字典
        return {
            'bit_accuracy': bit_accuracy,
            'cover_discriminator_prob': cover_prob,
            'stego_discriminator_prob': stego_prob,
            'privacy_results': privacy_results,
            'refine_report': refine_report
        }

    # run_discriminator_only 方法功能: 仅运行判别器模式，对输入图像做隐写检测。
    # 参数:
    #   image_path (str): 待检测图像路径。
    # 返回:
    #   prob (float): 隐写概率 (越大越像含密图像)。
    #   logit (float): 判别器原始 logit 输出。
    def run_discriminator_only(self, image_path):
        print("\n" + "#"*60)
        print("# 判别器预测模式")
        print("#"*60)

        # 加载并预处理输入图像
        image = self.load_image(image_path)
        # 调用判别器预测
        prob, logit = self.discriminator_predict(image, "输入图像")

        # 打印判别器预测结果
        print("\n" + "#"*60)
        print("# 判别器预测结果")
        print("#"*60)
        print(f"隐写概率: {prob*100:.2f}%")
        # 以 0.5 为阈值给出最终判定
        print(f"判断结果: {'隐写图像' if prob > 0.5 else '原始图像'}")
        print("#"*60)

        return prob, logit

    # run_decode_only 方法功能: 仅运行解码模式，从输入图像中提取嵌入的消息比特。
    # 参数:
    #   image_path (str): 待解码的图像路径 (通常是已编码的隐写图像)。
    # 返回:
    #   message_str (str): 解码得到的二进制字符串。
    #   decoded_prob (torch.Tensor): sigmoid 后的解码概率张量。
    def run_decode_only(self, image_path):
        print("\n" + "#"*60)
        print("# 解码模式")
        print("#"*60)

        # 加载并预处理输入图像
        image = self.load_image(image_path)
        # 调用解码器预测消息
        decoded_message, decoded_prob = self.decode_message(image)

        # 去掉 batch 维度并转回 CPU numpy，便于转换为字符串
        decoded_binary = decoded_message.squeeze().cpu().numpy()
        # 将比特数组转为字符串形式
        message_str = ''.join(str(int(b)) for b in decoded_binary)

        # 打印解码结果
        print("\n" + "#"*60)
        print("# 解码结果")
        print("#"*60)
        print(f"解码消息: {message_str}")
        print(f"消息长度: {len(message_str)} bits")
        print("#"*60)

        return message_str, decoded_prob

    # run_similarity_only 方法功能: 仅运行相似度判断模式，比较用户提供的原图与隐写图。
    # 参数:
    #   cover_image_path (str): 原始载体图像路径。
    #   stego_image_path (str): 隐写图像路径。
    # 返回:
    #   dict: 包含 CLIP/DINO/SSIM/PSNR 等相似度指标的字典。
    def run_similarity_only(self, cover_image_path, stego_image_path):
        print("\n" + "#"*60)
        print("# 相似度判断模式")
        print("#"*60)

        # 分别加载原图与隐写图
        cover_image = self.load_image(cover_image_path)
        stego_image = self.load_image(stego_image_path)

        # 计算多维度隐私相似度指标
        privacy_results = self.privacy_similarity(cover_image, stego_image)

        # 打印相似度判断结果
        print("\n" + "#"*60)
        print("# 相似度判断结果")
        print("#"*60)
        print(f"CLIP 余弦相似度: {privacy_results['cosine_similarity']:.6f}")
        print(f"CLIP 欧氏距离: {privacy_results['euclidean_distance']:.6f}")
        # 若 DINOv2 启用，则额外打印 DINO 相似度与距离
        if privacy_results.get('dino_similarity') is not None:
            print(f"DINO 余弦相似度: {privacy_results['dino_similarity']:.6f}")
            print(f"DINO 欧氏距离: {privacy_results['dino_euclidean_distance']:.6f}")
        print(f"SSIM: {privacy_results['ssim']:.6f}")
        print(f"PSNR: {privacy_results['psnr']:.2f} dB")
        print("\n结论:")
        print(f"  - CLIP 相似概率: {privacy_results['similarity_probability']*100:.2f}%")
        print(f"  - CLIP 隐私保护程度: {(1-privacy_results['similarity_probability'])*100:.2f}%")
        # 若 DINOv2 启用，则额外打印 DINO 隐私指标
        if privacy_results.get('dino_similarity') is not None:
            print(f"  - DINO 相似概率: {privacy_results['dino_similarity_probability']*100:.2f}%")
            print(f"  - DINO 隐私保护程度: {(1-privacy_results['dino_similarity_probability'])*100:.2f}%")
        print("#"*60)

        return privacy_results


# main 函数功能: 脚本 CLI 入口，解析命令行参数并按指定模式调度 ValidationScript。
# 流程:
#   1. 解析命令行参数 (checkpoint/image/cover/message/output-dir/device/mode)
#   2. 根据模式做参数校验，缺失关键参数时打印警告或错误
#   3. 实例化 ValidationScript (加载 checkpoint)
#   4. 按 mode 分发到对应的运行方法 (full/discriminator/decode/similarity)
def main():
    # 创建参数解析器，描述本脚本用途
    parser = argparse.ArgumentParser(description='PrivSteg 验证脚本')

    # --checkpoint/-c: 模型 checkpoint 路径，判别器/解码/完整模式推荐提供
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='检查点文件路径 (判别器模式推荐提供)')
    # --image/-i: 必填，输入图像路径 (待验证的隐写图像或载体图)
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='输入图像路径 (隐写图像)')
    # --cover: 原图路径，相似度判断模式必需
    parser.add_argument('--cover', type=str, default=None,
                        help='原图路径 (相似度判断模式必需)')
    # --message/-m: 待嵌入的二进制消息，可以是字符串或文件路径；默认 30 个 '0'
    parser.add_argument('--message', '-m', type=str, default='0' * 30,
                        help='消息 (二进制字符串或消息文件路径)')
    # --output-dir/-o: 输出目录，用于保存编码图等产物
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='输出目录')
    # --device: 指定运行设备，可选 cuda 或 cpu
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')
    # --mode: 验证模式枚举，默认 full
    parser.add_argument('--mode', type=str, choices=['full', 'discriminator', 'decode', 'similarity'], default='full',
                        help='验证模式: full=完整验证, discriminator=仅判别器预测, decode=仅解码消息, similarity=相似度判断')
    # --refine-steps: full 模式编码后的逐图精修步数（与 refine_encode.py 同配方）
    parser.add_argument('--refine-steps', type=int, default=30,
                        help='逐图精修步数 (0=关闭精修, 默认 30; 精修后图像用于全部后续验证与保存)')
    # --refine-lr: 精修 Adam 学习率
    parser.add_argument('--refine-lr', type=float, default=0.1,
                        help='逐图精修 Adam 学习率 (默认 0.1)')
    # --refine-budget: 精修隐私残差预算覆盖（默认 0.08 深挖配方）
    parser.add_argument('--refine-budget', type=float, default=0.08,
                        help='精修隐私残差预算 (默认 0.08 深挖配方)')
    # --refine-margin: 精修 CLIP escape margin 覆盖（默认 0.50 深挖配方）
    parser.add_argument('--refine-margin', type=float, default=0.50,
                        help='精修 CLIP escape margin (默认 0.50 深挖配方)')
    # --refine-chroma-projection: 收敛后剔除隐私残差 8x8 网格低频色度（消"蒙层/发灰"；1 开 0 关）
    parser.add_argument('--refine-chroma-projection', type=int, default=None,
                        help='精修收敛后剔除 prv 低频色度蒙层 (默认取 Config.refine_chroma_projection; '
                             '1 开启 / 0 关闭)。门控: 投影后 bit_acc 须 100%% 且 clip_sim 仍达标, '
                             '否则自动回退未投影结果')

    # 解析命令行参数
    args = parser.parse_args()

    # 若指定了 device 字符串则构造 torch.device，否则为 None (交给 ValidationScript 自动选择)
    device = torch.device(args.device) if args.device else None

    # 判别器模式未提供 checkpoint 时给出警告，权重未训练预测无意义
    if args.mode == 'discriminator' and args.checkpoint is None:
        print("警告: 判别器模式未提供检查点，将使用随机初始化的权重，预测结果无意义!")
        print("建议使用: -c <checkpoint_path> 指定检查点路径")

    # 解码模式未提供 checkpoint 时给出警告
    if args.mode == 'decode' and args.checkpoint is None:
        print("警告: 解码模式未提供检查点，将使用随机初始化的权重，解码结果无意义!")
        print("建议使用: -c <checkpoint_path> 指定检查点路径")

    # 相似度模式必须提供 --cover 原图路径，否则直接退出
    if args.mode == 'similarity' and args.cover is None:
        print("错误: 相似度判断模式需要指定原图路径!")
        print("建议使用: --cover <cover_image_path> 指定原图路径")
        return

    # 实例化 ValidationScript，构造所有模型并加载 checkpoint (若提供)
    validator = ValidationScript(checkpoint_path=args.checkpoint, device=device)

    # 按用户指定的模式分发执行
    if args.mode == 'full':
        # 完整验证模式: 未提供 checkpoint 时给出警告，仍可继续 (使用未训练模型)
        if args.checkpoint is None:
            print("警告: 完整验证模式需要检查点，使用未训练的模型进行验证...")
        validator.run_full_validation(
            image_path=args.image,
            message=args.message,
            output_dir=args.output_dir,
            refine_steps=args.refine_steps,
            refine_lr=args.refine_lr,
            refine_budget=args.refine_budget,
            refine_margin=args.refine_margin,
            refine_chroma_projection=args.refine_chroma_projection
        )
    elif args.mode == 'discriminator':
        # 仅判别器预测模式
        prob, _ = validator.run_discriminator_only(image_path=args.image)
    elif args.mode == 'decode':
        # 仅解码模式
        message_str, decoded_prob = validator.run_decode_only(image_path=args.image)
    else:
        # 相似度判断模式 (此时 args.cover 已确保存在)
        validator.run_similarity_only(cover_image_path=args.cover, stego_image_path=args.image)


# 脚本直接执行入口: 调用 main 函数
if __name__ == '__main__':
    main()
