# train.py
# 模块职责：PrivSteg 项目的 CLI 入口。
# 解析 new 和 continue 两个子命令：
#   - new：在 experiments/<name>/ 下创建新实验目录，写入 config.json 与 train.log，
#          初始化 Trainer，按需从 --continue-from-folder 加载 checkpoint 初始化模型，
#          然后运行两阶段训练循环。
#   - continue：从已有实验目录加载 config.json 与 checkpoint，恢复训练状态后继续训练。
# 另外提供随机种子设置、配置序列化、日志初始化、batch/epoch 损失记录以及
# 阶段训练循环（含 Phase 2 msg_head 冻结、bit accuracy 下降时的 decoder recovery）等辅助函数。
import argparse  # 命令行参数解析
import torch  # PyTorch 主库，用于张量计算与 checkpoint 加载
import random  # Python 内置随机数生成器，用于设置种子
import numpy as np  # NumPy，用于设置随机种子
import os  # 操作系统接口，用于路径拼接与目录创建
import json  # JSON 序列化，用于保存/读取 config.json
from datetime import datetime  # 时间戳，用于日志头部
from trainer import Trainer  # 训练器，封装模型、优化器、训练/验证 step
from noise_argparser import NoiseArgParser  # 自定义 argparse Action，解析 --noise 参数
import config  # 全局配置模块，定义 Config 类


def set_seed(seed=42):
    # 设置全局随机种子，保证训练可复现。
    # 参数：seed - 随机种子（默认 42）
    # 设置 Python random、NumPy、PyTorch 的种子，并在 GPU 可用时启用确定性 cuDNN
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # GPU 可用时同步设置 CUDA 种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 启用确定性算法、关闭 cuDNN 自动调优以保证可复现（可能略降性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_config(config_path, cfg):
    # 将 Config 实例的关键配置项序列化为 JSON 文件，便于实验复现与 continue 命令读取。
    # 参数：
    #   config_path - 输出的 config.json 路径
    #   cfg - Config 实例
    config_dict = {
        # 数据集与目录配置
        'data_dir': cfg.data_dir,
        'train_dir': cfg.train_dir,
        'val_dir': cfg.val_dir,
        # 图像与消息配置
        'image_size': cfg.image_size,
        'message_length': cfg.message_length,
        'batch_size': cfg.batch_size,
        # 训练轮数
        'num_epochs_phase1': cfg.num_epochs_phase1,
        'num_epochs_phase2': cfg.num_epochs_phase2,
        # 判别器/对抗配置
        'discriminator_loss_threshold': cfg.discriminator_loss_threshold,
        'disc_acc_high': cfg.disc_acc_high,
        'disc_hinge_margin': cfg.disc_hinge_margin,
        # 学习率
        'lr_encoder': cfg.lr_encoder,
        'lr_decoder': cfg.lr_decoder,
        'lr_steganalyzer': cfg.lr_steganalyzer,
        # 损失权重与残差预算
        'lambda_pixel': cfg.lambda_pixel,
        'lambda_ssim': cfg.lambda_ssim,
        'msg_budget': cfg.msg_budget,
        'prv_budget': cfg.prv_budget,
        'lambda_fft': cfg.lambda_fft,
        # 纹理掩码参数
        'texture_mask_strength': cfg.texture_mask_strength,
        'texture_mask_smooth_kernel': cfg.texture_mask_smooth_kernel,
        # Phase 1 对抗权重三分段
        'phase1_adv_start': cfg.phase1_adv_start,
        'phase1_adv_mid': cfg.phase1_adv_mid,
        'phase1_adv_end': cfg.phase1_adv_end,
        'phase2_adv': cfg.phase2_adv,
        # Phase 1 哑隐私噪声调度
        'phase1_prv_noise_start': cfg.phase1_prv_noise_start,
        'phase1_prv_noise_end': cfg.phase1_prv_noise_end,
        'phase1_prv_noise_max_ratio': cfg.phase1_prv_noise_max_ratio,
        'phase1_prv_noise_power': cfg.phase1_prv_noise_power,
        'phase1_prv_noise_smooth_kernel': cfg.phase1_prv_noise_smooth_kernel,
        # Phase 2 隐私 escape margin 调度
        'privacy_margin_start': cfg.privacy_margin_start,
        'privacy_margin_end': cfg.privacy_margin_end,
        # CLIP/DINOv2 重定向配置
        'use_clip_redirect': cfg.use_clip_redirect,
        'clip_memory_size': cfg.clip_memory_size,
        'clip_hard_negative_k': cfg.clip_hard_negative_k,
        'clip_redirect_margin': cfg.clip_redirect_margin,
        'clip_max_target_sim': cfg.clip_max_target_sim,
        'clip_redirect_weight': cfg.clip_redirect_weight,
        'dino_redirect_weight': cfg.dino_redirect_weight,
        'dino_redirect_margin': cfg.dino_redirect_margin,
        'dino_max_target_sim': cfg.dino_max_target_sim,
        'clip_redirect_target_sim': cfg.clip_redirect_target_sim,
        'dino_redirect_target_sim': cfg.dino_redirect_target_sim,
        'clip_redirect_exclude_sim': cfg.clip_redirect_exclude_sim,
        'dino_redirect_exclude_sim': cfg.dino_redirect_exclude_sim,
        'dino_escape_margin_start': cfg.dino_escape_margin_start,
        'dino_escape_margin_end': cfg.dino_escape_margin_end,
        'msg_clean_ratio_start': cfg.msg_clean_ratio_start,
        'msg_clean_ratio_end': cfg.msg_clean_ratio_end,
        'privacy_noised_ratio': cfg.privacy_noised_ratio,
        'privacy_noised_ratio_sprint': getattr(cfg, 'privacy_noised_ratio_sprint', 0.0),
        'phase2_sprint_epochs': cfg.phase2_sprint_epochs,
        'phase2_sprint_visual': cfg.phase2_sprint_visual,
        'phase2_sprint_privacy': cfg.phase2_sprint_privacy,
        # 必须持久化：continue 恢复配置依赖 config.json，缺失会使精修段静默失效
        'phase2_refine_epochs': getattr(cfg, 'phase2_refine_epochs', 0),
        'prv_head_lr_mult': cfg.prv_head_lr_mult,
        'distill_inner_steps': getattr(cfg, 'distill_inner_steps', 0),
        'distill_lr': getattr(cfg, 'distill_lr', 0.1),
        'distill_weight': getattr(cfg, 'distill_weight', 0.0),
        'distill_w_privacy': getattr(cfg, 'distill_w_privacy', 12.0),
        'distill_w_message': getattr(cfg, 'distill_w_message', 10.0),
        'distill_w_visual': getattr(cfg, 'distill_w_visual', 15.0),
        'clip_weight': cfg.clip_weight,
        'dino_weight': cfg.dino_weight,
        'dino_model_path': cfg.dino_model_path,
        # 结构/频域约束配置
        'prv_lowres_size': cfg.prv_lowres_size,
        'fft_high_freq_radius_ratio': cfg.fft_high_freq_radius_ratio,
        # 恢复触发阈值
        'bit_accuracy_threshold': cfg.bit_accuracy_threshold,
        'recovery_steps': cfg.recovery_steps,
        # 噪声层与 IO 配置
        'noise_config': cfg.noise_config,
        'num_workers': cfg.num_workers,
        'save_interval': cfg.save_interval,
        'memory_warmup_batches': cfg.memory_warmup_batches
    }
    # 以 UTF-8 编码、4 空格缩进写入 JSON
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4)


def load_encoder_state_compatible(encoder, state_dict):
    # 加载编码器权重，自动跳过形状不匹配的项。
    # 背景：prv_head 输入从 128 通道扩为 256 通道（新增 prv_trunk 独立支路），
    # 旧 checkpoint 的 prv_head 卷积权重形状不匹配；这些层与新增的 prv_trunk
    # 保持新初始化、由 Phase 2 重新训练，消息通路（conv_layers/after_concat_layer/
    # msg_head）与 decoder 的权重完整复用，消息解码能力不受影响。
    # 参数：
    #   encoder - Encoder 模型实例
    #   state_dict - checkpoint 中的 encoder_state_dict
    model_state = encoder.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in model_state and model_state[k].shape == v.shape}
    skipped = [k for k in state_dict if k not in compatible]
    if skipped:
        print(f"[Checkpoint] 跳过 {len(skipped)} 个与新结构不兼容的编码器权重（隐私支路相关，Phase 2 重训）: {skipped[:6]}")
    encoder.load_state_dict(compatible, strict=False)


def setup_logger(log_file, cfg, append=False):
    # 初始化训练日志文件，写入日志头部、配置摘要与网络架构说明。
    # 参数：
    #   log_file - 日志文件路径
    #   cfg - Config 实例（用于输出关键配置项）
    #   append - 是否以追加模式打开（continue 命令时为 True）
    # 追加模式（继续训练）或覆盖模式（新训练）
    mode = 'a' if append else 'w'
    with open(log_file, mode, encoding='utf-8') as f:
        if append:
            # 追加模式：写入分隔线与"Resuming Training"标题
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Resuming Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        else:
            # 覆盖模式：写入分隔线与"Training Log"标题
            f.write("=" * 80 + "\n")
            f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        # 写入配置摘要部分
        f.write("Configuration:\n")
        f.write("-" * 40 + "\n")
        # 仅记录与训练行为强相关的关键配置（与 save_config 的全集略有不同）
        config_dict = {
            'data_dir': cfg.data_dir,
            'msg_budget': cfg.msg_budget,
            'prv_budget': cfg.prv_budget,
            'lambda_fft': cfg.lambda_fft,
            'texture_mask_strength': cfg.texture_mask_strength,
            'texture_mask_smooth_kernel': cfg.texture_mask_smooth_kernel,
            'disc_acc_high': cfg.disc_acc_high,
            'phase1_adv_start': cfg.phase1_adv_start,
            'phase1_adv_mid': cfg.phase1_adv_mid,
            'phase1_adv_end': cfg.phase1_adv_end,
            'phase2_adv': cfg.phase2_adv,
            'phase1_prv_noise_start': cfg.phase1_prv_noise_start,
            'phase1_prv_noise_end': cfg.phase1_prv_noise_end,
            'phase1_prv_noise_max_ratio': cfg.phase1_prv_noise_max_ratio,
            'phase1_prv_noise_power': cfg.phase1_prv_noise_power,
            'phase1_prv_noise_smooth_kernel': cfg.phase1_prv_noise_smooth_kernel,
            'privacy_margin_start': cfg.privacy_margin_start,
            'privacy_margin_end': cfg.privacy_margin_end,
            'dino_escape_margin_start': cfg.dino_escape_margin_start,
            'dino_escape_margin_end': cfg.dino_escape_margin_end,
            'use_clip_redirect': cfg.use_clip_redirect,
            'clip_redirect_weight': cfg.clip_redirect_weight,
            'dino_redirect_weight': cfg.dino_redirect_weight,
            'clip_redirect_target_sim': cfg.clip_redirect_target_sim,
            'dino_redirect_target_sim': cfg.dino_redirect_target_sim,
            'clip_redirect_exclude_sim': cfg.clip_redirect_exclude_sim,
            'dino_redirect_exclude_sim': cfg.dino_redirect_exclude_sim,
            'msg_clean_ratio_start': cfg.msg_clean_ratio_start,
            'msg_clean_ratio_end': cfg.msg_clean_ratio_end,
            'privacy_noised_ratio': cfg.privacy_noised_ratio,
            'privacy_noised_ratio_sprint': getattr(cfg, 'privacy_noised_ratio_sprint', 0.0),
            'phase2_sprint_epochs': cfg.phase2_sprint_epochs,
            'phase2_sprint_visual': cfg.phase2_sprint_visual,
            'phase2_sprint_privacy': cfg.phase2_sprint_privacy,
            'phase2_refine_epochs': getattr(cfg, 'phase2_refine_epochs', 0),
            'prv_head_lr_mult': cfg.prv_head_lr_mult,
            'distill_inner_steps': getattr(cfg, 'distill_inner_steps', 0),
            'distill_lr': getattr(cfg, 'distill_lr', 0.1),
            'distill_weight': getattr(cfg, 'distill_weight', 0.0),
            'prv_lowres_size': cfg.prv_lowres_size,
            'fft_high_freq_radius_ratio': cfg.fft_high_freq_radius_ratio,
            'noise_config': cfg.noise_config,
            'bit_accuracy_threshold': cfg.bit_accuracy_threshold,
            'num_epochs_phase1': cfg.num_epochs_phase1,
            'num_epochs_phase2': cfg.num_epochs_phase2,
        }
        # 逐项写入关键配置
        for key, value in config_dict.items():
            f.write(f"  {key}: {value}\n")

        # 写入网络架构说明部分
        f.write("\nNetwork Architecture:\n")
        f.write("-" * 40 + "\n")
        # 编码器：双残差结构（msg_head + prv_head），通道数 128，残差块数 16
        f.write("  Encoder: Dual-residual (msg_head + prv_head), channels=128, blocks=16\n")
        # 解码器：CNN，输出经 sigmoid
        f.write("  Decoder: CNN with sigmoid output\n")
        # 隐写分析器：CNN，使用 SpectralNorm 稳定对抗训练
        f.write("  Steganalyzer: CNN with SpectralNorm\n")
        # 内容识别器：冻结的 CLIP
        f.write("  ContentRecognizer: CLIP (frozen)\n")
        f.write("\n")


def log_batch_losses(log_file, batch_idx, losses, phase):
    # 将单个 batch 的各项损失与判别器统计信息追加写入训练日志。
    # 参数：
    #   log_file - 日志文件路径
    #   batch_idx - 当前 batch 索引
    #   losses - trainer.train_step 返回的损失字典
    #   phase - 当前训练阶段（1 或 2）
    with open(log_file, 'a', encoding='utf-8') as f:
        # 写入阶段与 batch 索引前缀
        f.write(f"Phase {phase} - Batch {batch_idx}: ")
        # 视觉重建损失
        f.write(f"L_visual={losses['L_visual']:.6f}, ")
        # 消息解码损失（BCEWithLogits）
        f.write(f"L_message={losses['L_message']:.6f}, ")
        # 生成器对抗损失
        f.write(f"L_adv={losses['L_adv']:.6f}, ")
        # 判别器损失（默认 0，键不存在时回退）
        f.write(f"L_D={losses.get('L_D', 0):.6f}, ")
        # 判别器准确率
        f.write(f"D_acc={losses.get('D_acc', 0):.4f}, ")
        # 判别器是否更新（0/1）
        f.write(f"D_updated={losses.get('D_updated', 0):.0f}, ")
        # 判别器对真实载体图的预测概率
        f.write(f"D_real_prob={losses.get('D_real_prob', 0):.4f}, ")
        # 判别器对编码图的预测概率
        f.write(f"D_fake_prob={losses.get('D_fake_prob', 0):.4f}, ")
        # 判别器对载体图的 BCE 损失
        f.write(f"discr_cover_bce={losses['discr_cover_bce']:.6f}, ")
        # 判别器对编码图的 BCE 损失
        f.write(f"discr_encod_bce={losses['discr_encod_bce']:.6f}, ")
        # 隐私重定向损失
        f.write(f"L_privacy={losses['L_privacy']:.6f}, ")
        # 比特准确率
        f.write(f"bit_accuracy={losses['bit_accuracy']:.4f}, ")
        # Phase 1 哑隐私噪声注入比例
        f.write(f"prv_noise_ratio={losses.get('prv_noise_ratio', 0):.4f}, ")
        # Phase 2 隐私 escape margin
        f.write(f"escape_margin={losses.get('escape_margin', 0):.4f}\n")


def log_epoch_losses(log_file, epoch, train_losses, val_losses, phase):
    # 将一个 epoch 的训练与验证损失摘要写入训练日志。
    # 参数：
    #   log_file - 日志文件路径
    #   epoch - 当前轮次
    #   train_losses - 训练损失均值字典
    #   val_losses - 验证损失字典
    #   phase - 当前训练阶段（1 或 2）
    with open(log_file, 'a', encoding='utf-8') as f:
        # 写入分隔线与 epoch 摘要标题
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Phase {phase} - Epoch {epoch} Summary\n")
        f.write("=" * 80 + "\n")

        # 写入训练损失部分
        f.write("Training Losses:\n")
        for key, value in train_losses.items():
            f.write(f"  {key}: {value:.6f}\n")

        # 写入验证损失部分
        f.write("\nValidation Losses:\n")
        for key, value in val_losses.items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\n")


def run_training_loop(trainer, train_loader, val_loader, cfg, start_phase, start_epoch, log_file):
    # 两阶段训练主循环：依次执行 Phase 1 与 Phase 2，每个阶段内按 epoch 训练与验证。
    # 参数：
    #   trainer - Trainer 实例
    #   train_loader - 训练集 DataLoader
    #   val_loader - 验证集 DataLoader
    #   cfg - Config 实例
    #   start_phase - 起始阶段（1 或 2，用于断点续训）
    #   start_epoch - 起始阶段内的起始轮次（用于断点续训）
    #   log_file - 日志文件路径
    # 遍历阶段 1、2（range 上界 3 不含 3）
    for phase in range(start_phase, 3):
        # 根据阶段名动态获取该阶段的总轮数
        num_epochs = getattr(cfg, f'num_epochs_phase{phase}')
        if num_epochs <= 0:
            # 总轮数非正则跳过该阶段
            print(f'Skipping Phase {phase} (num_epochs={num_epochs})')
            continue
        # 起始阶段从断点轮次开始，其它阶段从第 1 轮开始
        phase_start_epoch = start_epoch if phase == start_phase else 1

        print(f'Starting Phase {phase}')

        if phase == 1 and phase_start_epoch == 1:
            # Phase 1 首轮：预热 memory bank（收集若干 batch 填充 CLIP/DINO 特征库）
            trainer.warmup_memory_bank(train_loader, num_batches=getattr(cfg, 'memory_warmup_batches', 50))

        if phase == 2 and phase_start_epoch == 1:
            # 直入 Phase 2（--epochs-phase1 0 热启动）时 memory bank 未经预热，
            # redirect 损失将无 hard negative 可检索；若 checkpoint 未携带 bank 数据则补预热
            clip_count = trainer.clip_memory_bank.memory_count if trainer.clip_memory_bank is not None else 0
            dino_count = trainer.dino_memory_bank.memory_count if trainer.dino_memory_bank is not None else 0
            if clip_count == 0 and dino_count == 0:
                trainer.warmup_memory_bank(train_loader, num_batches=getattr(cfg, 'memory_warmup_batches', 50))

        if phase == 2 and not trainer.phase2_msg_head_frozen:
            # Phase 2 进入前若 msg_head 尚未冻结，则冻结它，仅训练 prv_head + decoder
            trainer.freeze_msg_head()
            sprint_epochs = getattr(cfg, 'phase2_sprint_epochs', 0)
            print('Phase 2: msg_head frozen, training prv_head + decoder only')
            if sprint_epochs > 0:
                print(f'Phase 2 两段式：Sprint（纯逃逸冲刺，decoder 冻结）{sprint_epochs} 轮 -> Balance（恢复正常组合）{cfg.num_epochs_phase2 - sprint_epochs} 轮')

        # 从起始轮次训练到总轮数
        for epoch in range(phase_start_epoch, num_epochs + 1):
            # 将编码器、解码器、隐写分析器切换到训练模式
            trainer.encoder.train()
            trainer.decoder.train()
            trainer.steganalyzer.train()

            # 调度用相对轮次（本次运行的阶段内进度），而非绝对轮号。
            # 否则从旧 checkpoint（如 phase2 epoch 100）热启动重训 Phase 2 时，
            # epoch 恒大于 sprint_epochs，Sprint/Balance/权重三分段永远不会触发
            sched_epoch = epoch - phase_start_epoch + 1

            # 累积当前 epoch 各项损失
            epoch_losses = {}
            num_batches = 0

            # 遍历训练集每个 batch
            for batch_idx, batch in enumerate(train_loader):
                # 执行单步训练，返回损失字典
                losses = trainer.train_step(batch, phase, sched_epoch)
                # 累加各项损失到 epoch 统计
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value

                num_batches += 1
                # 写入 batch 级日志
                log_batch_losses(log_file, batch_idx, losses, phase)

            # 计算 epoch 平均损失
            for key in epoch_losses:
                epoch_losses[key] /= num_batches

            # 是否在该 epoch 保存可视化样本
            save_vis = epoch % cfg.save_interval == 0
            # 在验证集上评估
            val_losses = trainer.validate(val_loader, phase, sched_epoch, save_vis)
            # 写入 epoch 级日志
            log_epoch_losses(log_file, epoch, epoch_losses, val_losses, phase)

            # 打印训练/验证比特准确率
            print(f'Phase {phase} - Epoch {epoch} - Train Bit Acc: {epoch_losses["bit_accuracy"]:.4f} - Val Bit Acc: {val_losses["bit_accuracy"]:.4f}')

            # 将训练损失写入 TensorBoard
            for key, value in epoch_losses.items():
                trainer.writer.add_scalar(f'Phase{phase}/Train/{key}', value, epoch)
            # 将验证损失写入 TensorBoard
            for key, value in val_losses.items():
                trainer.writer.add_scalar(f'Phase{phase}/Val/{key}', value, epoch)

            # 判断是否为当前最佳（多目标准则）：
            #   Phase 1 以验证消息损失为准（先学好隐写/解码能力）；
            #   Phase 2 在 bit_accuracy 达标（≥ 阈值）的约束下最小化 clip_sim + dino_sim，
            #   未达标时加大惩罚项，保证选出的 checkpoint 确权与隐私同时最优
            if phase == 1:
                best_score = val_losses['L_message']
            else:
                # 优先使用传播场景（含噪）相似度作为逃逸判定标准；不存在时回退干净指标
                clip_sim = val_losses.get('clip_sim_noised', val_losses.get('clip_sim', 2.0))
                dino_sim = val_losses.get('dino_sim_noised', val_losses.get('dino_sim', 2.0))
                best_score = clip_sim + dino_sim
                if val_losses.get('bit_accuracy', 0.0) < getattr(cfg, 'bit_accuracy_threshold', 0.90):
                    best_score += 10.0  # 解码未达标的 checkpoint 一律排在后面
            # best 准则按阶段不同（Phase 1: L_message；Phase 2: clip+dino）且数值不可比——
            # L_message 收敛后约 0.1~0.3，而 sim 之和约 1.4~1.9，不重置则 Phase 2 的
            # best 分数永远高于 Phase 1 遗留值，best 会冻结在 Phase 1 不再更新。
            # 检测到准则所属阶段变化时重置历史最优（continue 续训同阶段则保留）
            if trainer.best_loss_phase != phase:
                trainer.best_loss = float('inf')
                trainer.best_loss_phase = phase
            is_best = best_score < trainer.best_loss
            if is_best:
                # 更新最佳记录并额外保存一份 best checkpoint
                trainer.best_loss = best_score
                trainer.save_checkpoint(epoch, phase,
                                        extra_path=os.path.join(cfg.checkpoint_dir, 'checkpoint_best.pth'))
            if epoch % cfg.save_interval == 0:
                # 按保存间隔写 checkpoint
                trainer.save_checkpoint(epoch, phase)

            if phase == 2:
                # Sprint 段（纯逃逸冲刺）：decoder 冻结、消息目标关闭，
                # bit_acc 暂时下降是预期行为，跳过 recovery 且明确标注日志
                sprint_epochs = getattr(cfg, 'phase2_sprint_epochs', 0)
                in_refine = cfg.is_phase2_refine_segment(sched_epoch)
                if sched_epoch <= sprint_epochs:
                    print(f'  [SPRINT] epoch {sched_epoch}/{sprint_epochs}: privacy-only sprint (bit_acc drop expected, recovery skipped)')
                elif in_refine:
                    # Refine 终段：decoder 冻结是精修配方的设计语义（残差自适应
                    # 既有解码器），recovery 重训 decoder 会使消息锚的目标漂移；
                    # 学生目标含 message 权重 10，bit_acc 应由锚自身维持
                    print(f'  [REFINE] epoch {sched_epoch}: escape-only refine (message anchor active, recovery skipped)')
                else:
                    # Balance 段：检查验证比特准确率是否低于阈值，触发 decoder recovery
                    val_bit_acc = val_losses.get('bit_accuracy', 1.0)
                    threshold = getattr(cfg, 'bit_accuracy_threshold', 0.90)
                    if val_bit_acc < threshold:
                        print(f'  [RECOVERY] bit_accuracy={val_bit_acc:.4f} < {threshold}, running recovery steps...')
                        # 获取恢复步数
                        recovery_steps = getattr(cfg, 'recovery_steps', 10)
                        # 在干净消息上重新训练 decoder
                        trainer.recovery_step(train_loader, num_batches=recovery_steps)
                        print('  [RECOVERY] Done, decoder retrained on clean messages')

    print('Training completed!')
    # 关闭 TensorBoard writer
    trainer.writer.close()


def main():
    # CLI 主入口：解析命令行参数并分发到 new 或 continue 子流程。
    # 父解析器，描述程序用途
    parent_parser = argparse.ArgumentParser(description='Training of GAN Steganography with Privacy Protection')
    # 创建子命令解析器（new / continue）
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')

    # ===== new 子命令：启动新实验 =====
    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    # 数据集根目录（默认 dataset1，可用 -d 覆盖）
    new_run_parser.add_argument('--data-dir', '-d', type=str, default='./dataset1',
                                help='The directory where data is stored (default: ./dataset1).')
    # 训练子目录名
    new_run_parser.add_argument('--train-dir', type=str, default='train',
                                help='Subdirectory name for training data (default: train).')
    # 验证子目录名
    new_run_parser.add_argument('--val-dir', type=str, default='val',
                                help='Subdirectory name for validation data (default: val).')
    # 批大小
    new_run_parser.add_argument('--batch-size', '-b', type=int, default=config.Config.batch_size,
                                help='The batch size.')
    # 实验名（决定 experiments/<name>/ 目录）
    new_run_parser.add_argument('--name', type=str, default='default',
                                help='The name of experiment.')
    # 图像尺寸（正方形边长）
    new_run_parser.add_argument('--size', '-s', type=int, default=256,
                                help='The size of images (images are square so this is height and width).')
    # 消息比特长度
    new_run_parser.add_argument('--message', '-m', type=int, default=30, help='The length in bits of watermark.')
    # 从另一个实验目录的 checkpoint 初始化（用于迁移学习）
    new_run_parser.add_argument('--continue-from-folder', '-c', default='', type=str,
                                help='The folder from where to continue a previous run. Leave blank if you are starting a new experiment.')
    # 是否开启 TensorBoard 日志
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    # 是否启用混合精度训练
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')
    # 噪声层配置字符串，使用自定义 NoiseArgParser 解析
    new_run_parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
                                help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")
    # 随机种子
    new_run_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Phase 1 / Phase 2 总轮数
    new_run_parser.add_argument('--epochs-phase1', type=int, default=config.Config.num_epochs_phase1,
                                help='Number of epochs for phase 1 (basic+adversarial training).')
    new_run_parser.add_argument('--epochs-phase2', type=int, default=config.Config.num_epochs_phase2,
                                help='Number of epochs for phase 2 (privacy training).')
    # 判别器损失阈值（遗留兼容）
    new_run_parser.add_argument('--discriminator-loss-threshold', type=float, default=config.Config.discriminator_loss_threshold,
                                help='Legacy discriminator loss threshold kept for compatibility.')
    # 判别器准确率门控上限
    new_run_parser.add_argument('--disc-acc-high', type=float, default=config.Config.disc_acc_high,
                                help='Pause discriminator updates when discriminator accuracy reaches this value.')
    # 判别器 Hinge Loss margin
    new_run_parser.add_argument('--disc-hinge-margin', type=float, default=config.Config.disc_hinge_margin,
                                help='Margin used by discriminator hinge loss.')

    # Phase 1 对抗权重三分段
    new_run_parser.add_argument('--phase1-adv-start', type=float, default=config.Config.phase1_adv_start,
                                help='Phase 1 adversarial weight during the first third.')
    new_run_parser.add_argument('--phase1-adv-mid', type=float, default=config.Config.phase1_adv_mid,
                                help='Phase 1 adversarial weight during the second third.')
    new_run_parser.add_argument('--phase1-adv-end', type=float, default=config.Config.phase1_adv_end,
                                help='Phase 1 adversarial weight during the final third.')
    # Phase 2 对抗权重
    new_run_parser.add_argument('--phase2-adv', type=float, default=config.Config.phase2_adv,
                                help='Phase 2 adversarial weight.')

    # Phase 1 哑隐私噪声调度参数
    new_run_parser.add_argument('--phase1-prv-noise-start', type=float, default=config.Config.phase1_prv_noise_start,
                                help='Phase 1 fraction before dummy privacy noise starts ramping.')
    new_run_parser.add_argument('--phase1-prv-noise-end', type=float, default=config.Config.phase1_prv_noise_end,
                                help='Phase 1 fraction where dummy privacy noise reaches its cap.')
    new_run_parser.add_argument('--phase1-prv-noise-max-ratio', type=float, default=config.Config.phase1_prv_noise_max_ratio,
                                help='Maximum Phase 1 dummy privacy noise ratio.')
    new_run_parser.add_argument('--phase1-prv-noise-power', type=float, default=config.Config.phase1_prv_noise_power,
                                help='Power used for the Phase 1 dummy privacy noise ramp.')
    new_run_parser.add_argument('--phase1-prv-noise-smooth-kernel', type=int, default=config.Config.phase1_prv_noise_smooth_kernel,
                                help='Average-pooling kernel for smoothing Phase 1 dummy privacy noise.')

    # 各模型学习率
    new_run_parser.add_argument('--lr-encoder', type=float, default=config.Config.lr_encoder,
                                help='Learning rate for encoder')
    new_run_parser.add_argument('--lr-decoder', type=float, default=config.Config.lr_decoder,
                                help='Learning rate for decoder')
    new_run_parser.add_argument('--lr-steganalyzer', type=float, default=config.Config.lr_steganalyzer,
                                help='Learning rate for steganalyzer')

    # 损失权重与 IO 配置
    new_run_parser.add_argument('--lambda-pixel', type=float, default=config.Config.lambda_pixel,
                                help='Weight for pixel loss.')
    new_run_parser.add_argument('--lambda-ssim', type=float, default=config.Config.lambda_ssim,
                                help='Weight for SSIM loss.')
    new_run_parser.add_argument('--num-workers', type=int, default=config.Config.num_workers,
                                help='Number of data loading workers.')
    new_run_parser.add_argument('--save-interval', type=int, default=config.Config.save_interval,
                                help='Save checkpoint every N epochs.')
    # Phase 2 Sprint 段轮数（纯逃逸冲刺）
    new_run_parser.add_argument('--sprint-epochs', type=int, default=config.Config.phase2_sprint_epochs,
                                help='Number of Phase 2 sprint (privacy-only) epochs.')
    # Phase 2 终段精修轮数（val/refine_encode.py 同配方收尾）
    new_run_parser.add_argument('--refine-epochs', type=int, default=config.Config.phase2_refine_epochs,
                                help='Number of final Phase 2 refine epochs (refine_encode-style objective; '
                                     'warm-start a pure refine run with --refine-epochs == --epochs-phase2).')
    # 方案 C 内环蒸馏参数（--distill-steps 0 可关闭蒸馏）
    new_run_parser.add_argument('--distill-steps', type=int, default=config.Config.distill_inner_steps,
                                help='Inner-loop refinement steps per batch for teacher distillation (0 disables).')
    new_run_parser.add_argument('--distill-weight', type=float, default=config.Config.distill_weight,
                                help='Weight of the distillation loss (student vs teacher residual).')
    # 隐私残差预算：逃逸深度近似随预算平方增长（cos 相似度下降 ∝ 特征位移²），
    # 当前默认 0.04 下 PSNR 42.8dB/SSIM 0.971 仍有充足视觉余量，可按需放大
    new_run_parser.add_argument('--prv-budget', type=float, default=config.Config.prv_budget,
                                help='Privacy residual budget (amplitude upper bound of prv_residual).')
    # CLIP escape margin：二次 Hinge 在 sim < (1-margin) 后梯度归零，margin 决定逃逸地板。
    # CPU 逐图精修实测（60 步，bit_acc 全程 100%）：budget 0.08 下 margin 0.30/0.40/0.50
    # 分别对应 clip_sim 0.575/0.547/0.440（PSNR 39.9/39.4/39.0）——margin 是第一杠杆
    new_run_parser.add_argument('--privacy-margin-end', type=float, default=config.Config.privacy_margin_end,
                                help='Final CLIP escape margin (escape target sim = 1 - margin).')
    new_run_parser.add_argument('--dino-margin-end', type=float, default=config.Config.dino_escape_margin_end,
                                help='Final DINO escape margin (DINO target sim = 1 - margin).')

    # ===== continue 子命令：从已有实验目录恢复训练 =====
    continue_parser = subparsers.add_parser('continue', help='continue a previous run')
    # 已有实验目录（必填）
    continue_parser.add_argument('--folder', '-f', required=True, type=str,
                                  help='The folder from where to continue a previous run.')

    # 解析命令行参数
    args = parent_parser.parse_args()

    # 未指定子命令时打印帮助并退出
    if args.command is None:
        parent_parser.print_help()
        return

    if args.command == 'new':
        # ===== new 子命令处理：创建新实验并启动训练 =====
        # 设置随机种子保证可复现
        set_seed(args.seed)

        # 创建 Config 实例并用命令行参数覆盖默认值
        cfg = config.Config()
        # 数据集与图像配置
        cfg.data_dir = args.data_dir
        cfg.train_dir = args.train_dir
        cfg.val_dir = args.val_dir
        cfg.batch_size = args.batch_size
        cfg.message_length = args.message
        cfg.image_size = args.size

        # 训练轮数与对抗配置
        cfg.num_epochs_phase1 = args.epochs_phase1
        cfg.num_epochs_phase2 = args.epochs_phase2
        cfg.discriminator_loss_threshold = args.discriminator_loss_threshold
        cfg.disc_acc_high = args.disc_acc_high
        cfg.disc_hinge_margin = args.disc_hinge_margin
        cfg.phase1_adv_start = args.phase1_adv_start
        cfg.phase1_adv_mid = args.phase1_adv_mid
        cfg.phase1_adv_end = args.phase1_adv_end
        cfg.phase2_adv = args.phase2_adv
        # Phase 1 哑隐私噪声调度参数
        cfg.phase1_prv_noise_start = args.phase1_prv_noise_start
        cfg.phase1_prv_noise_end = args.phase1_prv_noise_end
        cfg.phase1_prv_noise_max_ratio = args.phase1_prv_noise_max_ratio
        cfg.phase1_prv_noise_power = args.phase1_prv_noise_power
        cfg.phase1_prv_noise_smooth_kernel = args.phase1_prv_noise_smooth_kernel

        # 学习率
        cfg.lr_encoder = args.lr_encoder
        cfg.lr_decoder = args.lr_decoder
        cfg.lr_steganalyzer = args.lr_steganalyzer

        # 损失权重
        cfg.lambda_pixel = args.lambda_pixel
        cfg.lambda_ssim = args.lambda_ssim

        # Phase 2 调度与蒸馏
        cfg.phase2_sprint_epochs = args.sprint_epochs
        cfg.phase2_refine_epochs = args.refine_epochs
        cfg.distill_inner_steps = args.distill_steps
        cfg.distill_weight = args.distill_weight
        # 隐私残差预算（CLI 可放大逃逸深度，Encoder/教师内环均从该值构建约束链）
        cfg.prv_budget = args.prv_budget
        # 逃逸 margin 终值：Sprint/Refine 段直接取该值；教师内环目标同源联动
        cfg.privacy_margin_end = args.privacy_margin_end
        cfg.dino_escape_margin_end = args.dino_margin_end

        # IO 配置
        cfg.num_workers = args.num_workers
        cfg.save_interval = args.save_interval

        # 若指定了噪声配置则覆盖默认 identity
        if args.noise:
            cfg.noise_config = args.noise

        # 创建实验目录结构 experiments/<name>/
        experiment_name = args.name
        experiment_dir = os.path.join('experiments', experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        # 检查点与 TensorBoard 子目录
        cfg.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        cfg.log_dir = os.path.join(experiment_dir, 'tensorboard')
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        # 初始化日志与配置文件
        log_file = os.path.join(experiment_dir, 'train.log')
        setup_logger(log_file, cfg)
        config_path = os.path.join(experiment_dir, 'config.json')
        save_config(config_path, cfg)

        # 实例化 Trainer
        trainer = Trainer(cfg)

        # 默认从 Phase 1 第 1 轮开始
        start_epoch = 1
        start_phase = 1

        if args.continue_from_folder:
            # 若指定 --continue-from-folder，则从该目录加载 checkpoint 初始化模型
            checkpoint_path = os.path.join(args.continue_from_folder, 'checkpoints', 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                # 加载 checkpoint 到对应设备
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                # 恢复编码器（跳过与新结构不匹配的隐私支路权重）、解码器、隐写分析器权重
                load_encoder_state_compatible(trainer.encoder, checkpoint['encoder_state_dict'])
                trainer.decoder.load_state_dict(checkpoint['decoder_state_dict'])
                trainer.steganalyzer.load_state_dict(checkpoint['steganalyzer_state_dict'])
                # 恢复优化器状态；新增 prv_trunk 参数会使旧 checkpoint 的参数组
                # 尺寸不匹配（ValueError），此时使用全新 Adam 状态（Adam 预热很快，可接受）
                try:
                    trainer.optimizer_encoder_decoder.load_state_dict(checkpoint['optimizer_encoder_decoder'])
                    trainer.optimizer_steganalyzer.load_state_dict(checkpoint['optimizer_steganalyzer'])
                except ValueError:
                    print('[Checkpoint] 优化器状态与新结构不匹配（新增 prv_trunk 参数），改用全新 Adam 状态')
                # 恢复 CLIP memory bank（若存在）
                if trainer.clip_memory_bank is not None and 'clip_memory_bank' in checkpoint:
                    trainer.clip_memory_bank.load_state_dict(checkpoint['clip_memory_bank'])
                # 恢复 DINO memory bank（若存在）
                if trainer.dino_memory_bank is not None and 'dino_memory_bank' in checkpoint:
                    trainer.dino_memory_bank.load_state_dict(checkpoint['dino_memory_bank'])
                # 权重初始化语义：new 命令的训练计划（阶段/轮数）完全由本次 CLI 参数决定，
                # 从 Phase 1 第 1 轮起算。checkpoint 的绝对轮号不能带过来——例如旧
                # Phase 2 epoch 100 会使 start_epoch=101，Phase 2 循环 range(101, N) 为空，
                # 一轮都不会训练。断点续训请用 continue 子命令。
                # best_loss 也不继承：旧值是 Phase 1 L_message 准则，与 Phase 2 的
                # clip_sim+dino_sim best 判定不可比，继承会让 best 永远不触发。
                trainer.best_loss = float('inf')
                trainer.phase2_msg_head_frozen = False
                start_epoch = 1
                start_phase = 1
                print(f"[Init] Weights loaded from {args.continue_from_folder} "
                      f"(source: phase {checkpoint['phase']} epoch {checkpoint['epoch']}); "
                      f"fresh plan: phase1={cfg.num_epochs_phase1}, phase2={cfg.num_epochs_phase2}")
            else:
                # checkpoint 不存在则从零开始
                print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")

        # 构建 DataLoader 并启动训练循环
        train_loader, val_loader = trainer.get_dataloaders()
        run_training_loop(trainer, train_loader, val_loader, cfg, start_phase, start_epoch, log_file)

    elif args.command == 'continue':
        # ===== continue 子命令处理：从已有实验目录恢复训练 =====
        # 实验目录（即 --folder 指定的路径）
        checkpoint_dir = args.folder
        # 读取 config.json 恢复配置
        config_path = os.path.join(checkpoint_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            cfg = config.Config()
            # 将 config.json 中的键值逐项写回 cfg（仅覆盖已存在的属性）
            for key, value in config_dict.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        else:
            # 配置文件缺失则使用默认配置
            cfg = config.Config()
        # 设置检查点与 TensorBoard 子目录
        cfg.checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
        cfg.log_dir = os.path.join(checkpoint_dir, 'tensorboard')
        log_file = os.path.join(checkpoint_dir, 'train.log')
        # 以追加模式初始化日志
        setup_logger(log_file, cfg, append=True)

        # 实例化 Trainer
        trainer = Trainer(cfg)
        checkpoint_path = os.path.join(cfg.checkpoint_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            # 加载 checkpoint 到对应设备
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            # 恢复编码器（跳过与新结构不匹配的隐私支路权重）、解码器、隐写分析器权重
            load_encoder_state_compatible(trainer.encoder, checkpoint['encoder_state_dict'])
            trainer.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            trainer.steganalyzer.load_state_dict(checkpoint['steganalyzer_state_dict'])
            # 恢复优化器状态；结构变更导致参数组尺寸不匹配时改用全新 Adam 状态
            try:
                trainer.optimizer_encoder_decoder.load_state_dict(checkpoint['optimizer_encoder_decoder'])
                trainer.optimizer_steganalyzer.load_state_dict(checkpoint['optimizer_steganalyzer'])
            except ValueError:
                print('[Checkpoint] 优化器状态与新结构不匹配（新增 prv_trunk 参数），改用全新 Adam 状态')
            # 恢复 CLIP memory bank（若存在）
            if trainer.clip_memory_bank is not None and 'clip_memory_bank' in checkpoint:
                trainer.clip_memory_bank.load_state_dict(checkpoint['clip_memory_bank'])
            # 恢复 DINO memory bank（若存在）
            if trainer.dino_memory_bank is not None and 'dino_memory_bank' in checkpoint:
                trainer.dino_memory_bank.load_state_dict(checkpoint['dino_memory_bank'])
            # 恢复最佳损失记录（旧格式 checkpoint 无 best_loss_phase 字段时置 None，
            # 首轮按当前阶段准则重置重判，避免跨准则比较）
            trainer.best_loss = checkpoint.get('best_loss', float('inf'))
            trainer.best_loss_phase = checkpoint.get('best_loss_phase', None)
            # 恢复 msg_head 冻结状态
            trainer.phase2_msg_head_frozen = checkpoint.get('phase2_msg_head_frozen', False)
            if trainer.phase2_msg_head_frozen:
                trainer.freeze_msg_head()
            # 从断点的下一轮继续
            start_epoch = checkpoint['epoch'] + 1
            start_phase = checkpoint['phase']
            print(f"Resuming from Phase {start_phase}, Epoch {start_epoch}")
        else:
            # checkpoint 不存在则报错并退出
            print(f"Checkpoint not found at {checkpoint_path}")
            return

        # 构建 DataLoader 并启动训练循环
        train_loader, val_loader = trainer.get_dataloaders()
        run_training_loop(trainer, train_loader, val_loader, cfg, start_phase, start_epoch, log_file)


# 脚本入口：直接运行 train.py 时调用 main()
if __name__ == '__main__':
    main()
