# config.py
# 模块职责：定义 PrivSteg 项目的全局配置类 Config。
# Config 使用类属性集中管理训练、验证、模型构造和损失调度所需的所有配置项，
# 包括设备选择、图像尺寸、消息长度、批大小、各模型学习率、两阶段训练轮数、
# 对抗训练参数、损失权重、消息/隐私残差预算、纹理掩码参数、Phase 1 渐进噪声调度、
# CLIP/DINOv2 隐私重定向参数、恢复触发阈值、数据/检查点路径以及噪声层配置等。
# 此外提供按阶段和轮次动态计算损失权重、隐私 escape margin、Phase 1 哑隐私噪声比例的工具方法。
# 当 README 与当前实现不一致时，以本文件代码为准。
import torch


class Config:
    # 配置类：集中存放训练流程所需的所有超参数与路径配置。
    # 所有属性均为类属性，可直接通过 Config 或实例访问；train.py 中会基于命令行参数覆盖部分属性。

    # 设备选择：优先使用 CUDA GPU，不可用时回退到 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 图像尺寸（正方形边长），输入图像会被 resize 到该尺寸
    image_size = 256
    # 嵌入的二进制消息长度（比特数）
    message_length = 30
    # 训练批大小
    batch_size = 8

    # 编码器学习率
    lr_encoder = 1e-3
    # 解码器学习率
    lr_decoder = 1e-3
    # 隐写分析器（判别器）学习率，通常比编/解码器小一个数量级以稳定对抗训练
    lr_steganalyzer = 1e-4

    # Phase 1（基础+对抗训练）的总轮数
    num_epochs_phase1 = 150
    # Phase 2（隐私训练）的总轮数
    num_epochs_phase2 = 100

    # 判别器损失阈值（遗留兼容项，当前主要由 disc_acc_high 门控更新）
    discriminator_loss_threshold = 0.6
    # 判别器准确率门控上限：达到该值时暂停判别器更新，避免判别器过强
    disc_acc_high = 0.70
    # 判别器 Hinge Loss 的 margin
    disc_hinge_margin = 1.0

    # 像素重建损失权重
    lambda_pixel = 1.0
    # SSIM 结构相似度损失权重
    lambda_ssim = 1.0
    # FFT 高频能量惩罚权重（Phase 2 用于约束隐私残差的高频成分）
    lambda_fft = 10.0

    # 消息残差预算（约束 msg_residual 的幅值）
    msg_budget = 0.04
    # 隐私残差预算（约束 prv_residual 的幅值）
    prv_budget = 0.04

    # 纹理掩码强度：控制梯度驱动掩码对隐私扰动的放大倍数
    texture_mask_strength = 3.0
    # 纹理掩码平滑核大小（平均池化核，用于平滑梯度掩码）
    texture_mask_smooth_kernel = 5

    # Phase 1 对抗损失权重三分段调度：起始（前 1/3）/中间（中 1/3）/末段（后 1/3）
    phase1_adv_start = 0.0
    phase1_adv_mid = 0.25
    phase1_adv_end = 0.75
    # Phase 2 对抗损失权重（恒定）
    phase2_adv = 0.5

    # Phase 1 哑隐私噪声调度：训练到该比例轮次之前不注入噪声
    phase1_prv_noise_start = 0.50
    # Phase 1 哑隐私噪声到达上限的训练比例轮次
    phase1_prv_noise_end = 0.90
    # Phase 1 哑隐私噪声相对于 prv_budget 的最大比例
    phase1_prv_noise_max_ratio = 0.50
    # 哑隐私噪声 ramp 的幂次（>1 表示前期增长慢、后期增长快）
    phase1_prv_noise_power = 2.0
    # 哑隐私噪声平滑核大小（平均池化核，使噪声更自然）
    phase1_prv_noise_smooth_kernel = 7

    # Phase 2 隐私 escape margin 渐进调度的起始值与终止值
    privacy_margin_start = 0.15
    privacy_margin_end = 0.30

    # 是否启用 CLIP 特征重定向（CBIR 检索逃逸）
    use_clip_redirect = False
    # CLIP memory bank 容量（用于存储历史特征作为 hard negative）
    clip_memory_size = 5000
    # 每次采样 hard negative 的数量
    clip_hard_negative_k = 5
    # CLIP escape loss 的 margin
    clip_redirect_margin = 0.35
    # CLIP 重定向目标相似度上限（超过则不再推动）
    clip_max_target_sim = 0.55
    # CLIP 重定向损失权重
    clip_redirect_weight = 0.6
    # DINOv2 重定向损失权重
    dino_redirect_weight = 0.3
    # DINOv2 escape loss 的 margin
    dino_redirect_margin = 0.3
    # DINOv2 重定向目标相似度上限
    dino_max_target_sim = 0.55

    # CLIP 总权重（content recognizer 中的 CLIP 分支权重）
    clip_weight = 1.0
    # DINOv2 总权重（>0 时加载 DINOv2 权重；缺失则警告并随机初始化）
    dino_weight = 0.5
    # DINOv2 预训练权重文件路径
    dino_model_path = './models/dinov2_vits14_pretrain.pth'


    # memory bank 预热批次数（Phase 1 开始时收集多少 batch 填充 memory bank）
    memory_warmup_batches = 50

    # Phase 2 触发 decoder recovery 的 bit accuracy 阈值（低于此值则启动恢复）
    bit_accuracy_threshold = 0.90
    # decoder recovery 的步数
    recovery_steps = 10

    # 数据集根目录（由命令行设置）
    data_dir = None
    # 训练子目录名
    train_dir = 'train'
    # 验证子目录名
    val_dir = 'val'

    # 检查点目录名（会被 train.py 改写为实验目录下的绝对路径）
    checkpoint_dir = 'checkpoints'
    # 日志/TensorBoard 目录名（会被 train.py 改写为实验目录下的绝对路径）
    log_dir = 'logs'

    # DataLoader 工作进程数
    num_workers = 4
    # 每隔多少轮保存一次 checkpoint
    save_interval = 10

    # 噪声层配置字符串（'identity' 表示无噪声；可由命令行 --noise 覆盖）
    noise_config = 'identity'

    def get_phase_weights(self, phase, epoch=1):
        # 根据训练阶段和轮次返回各损失项的权重字典。
        # 参数：
        #   phase: 训练阶段（1 或 2）
        #   epoch: 当前轮次（从 1 开始计数）
        # 返回：包含 visual/message/adv/privacy/noised/residual_penalty/fft 权重的字典
        if phase == 1:
            # Phase 1：将总轮数三等分，对抗权重按段切换
            total = self.num_epochs_phase1
            third = total // 3
            if epoch <= third:
                # 前 1/3：对抗权重为起始值（默认 0，先稳定隐写能力）
                adv = self.phase1_adv_start
            elif epoch <= 2 * third:
                # 中 1/3：对抗权重提升到中间值
                adv = self.phase1_adv_mid
            else:
                # 后 1/3：对抗权重提升到末段值
                adv = self.phase1_adv_end
            # Phase 1 不计算隐私损失（privacy=0），不施加 FFT 惩罚（fft=0）
            return {
                'visual': 10.0,
                'message': 10.0,
                'adv': adv,
                'privacy': 0.0,
                'noised': 1.0,
                'residual_penalty': 0.5,
                'fft': 0.0
            }
        elif phase == 2:
            # Phase 2：将总轮数三等分，逐步提升隐私权重、降低消息权重
            total = self.num_epochs_phase2
            third = total // 3
            if epoch <= third:
                # 前 1/3：隐私权重较低，消息权重较高（保证可解码性）
                privacy = 2.0
                message = 20.0
            elif epoch <= 2 * third:
                # 中 1/3：隐私权重提升，消息权重略降
                privacy = 4.0
                message = 15.0
            else:
                # 后 1/3：隐私权重最高，消息权重最低
                privacy = 5.0
                message = 10.0
            # Phase 2 启用 FFT 高频惩罚，约束隐私残差频谱
            return {
                'visual': 10.0,
                'message': message,
                'adv': self.phase2_adv,
                'privacy': privacy,
                'noised': 0.3,
                'residual_penalty': 0.5,
                'fft': self.lambda_fft
            }
        else:
            # 非法的阶段值，抛出错误防止误用
            raise ValueError(f'Invalid phase: {phase}')

    def get_escape_margin(self, phase, epoch):
        # 计算 Phase 2 隐私 escape margin（特征重定向损失的 Hinge margin）。
        # margin 随 epoch 从 privacy_margin_start 线性增长到 privacy_margin_end。
        # 参数：
        #   phase: 训练阶段（1 或 2）
        #   epoch: 当前轮次（从 1 开始计数）
        # 返回：当前轮次的 escape margin 标量
        if phase == 1:
            # Phase 1 不计算隐私损失，escape margin 恒为 0
            return 0.0
        # Phase 2 总轮数
        total = self.num_epochs_phase2
        # 训练进度比例 [0, 1]：(epoch-1)/(total-1)，epoch 从 1 开始
        progress = min(1.0, (epoch - 1) / max(1, total - 1))
        # 线性插值得到当前 margin
        return self.privacy_margin_start + (self.privacy_margin_end - self.privacy_margin_start) * progress

    def get_prv_noise_ratio(self, epoch):
        # 计算 Phase 1 哑隐私噪声相对于 prv_budget 的注入比例。
        # 噪声在 warmup_end 之前为 0，在 warmup_end~ramp_end 之间按幂函数 ramp，
        # 在 ramp_end 之后保持最大比例。
        # 参数：
        #   epoch: Phase 1 当前轮次（从 1 开始计数）
        # 返回：哑隐私噪声比例（标量，0~phase1_prv_noise_max_ratio）
        # Phase 1 总轮数
        total = self.num_epochs_phase1
        # 噪声开始注入的训练比例对应的轮次
        warmup_end = total * self.phase1_prv_noise_start
        # 噪声达到上限的训练比例对应的轮次
        ramp_end = total * self.phase1_prv_noise_end
        if epoch <= warmup_end:
            # warmup 阶段：不注入哑隐私噪声
            return 0.0
        elif epoch <= ramp_end:
            # ramp 阶段：按幂函数从 0 增长到最大比例
            progress = (epoch - warmup_end) / max(1.0, ramp_end - warmup_end)
            # 将 progress 裁剪到 [0, 1] 防止数值越界
            progress = min(1.0, max(0.0, progress))
            # 按幂次（默认 2.0）计算当前比例
            return self.phase1_prv_noise_max_ratio * (progress ** self.phase1_prv_noise_power)
        else:
            # ramp 结束后：保持最大比例
            return self.phase1_prv_noise_max_ratio