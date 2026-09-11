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
    # 训练批大小（A100 标准配置；本地 CPU 冒烟请在命令行显式传 --batch-size 1）
    batch_size = 35

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
    lambda_fft = 50.0
    # FFT 高频判定半径占奈奎斯特半径（min(H,W)//2）的比例。
    # 0.25 对应半径 min(H,W)//8（256 图约 r=32），可覆盖周期 4~8px 的可见条纹频段
    fft_high_freq_radius_ratio = 0.25

    # 消息残差预算（约束 msg_residual 的幅值）
    msg_budget = 0.04
    # 隐私残差预算（约束 prv_residual 的幅值）。
    # 默认 0.08（深挖配方）：逃逸深度近似随预算平方增长（cos 下降 ∝ 特征位移²），
    # 0.04→0.08 实测精修 clip_sim 0.634→0.575，PSNR 42.8→39.9 仍在视觉高质量区间
    prv_budget = 0.08
    # 隐私残差低分辨率约束：prv_head 输出先池化到该尺寸再上采样回全分辨率，
    # 从结构上消除逐像素高频自由度，抑制条纹伪影。
    # 96 的取值依据：CLIP ViT-B/32 的 patch 为 16×16 像素，对 patch 尺度中频结构
    # （32~96 周期/图）最敏感；48×48 低通（<24 周期）恰好把残差约束在 CLIP 最不敏感
    # 的频段，实测 privacy 损失占 88% 但 sim 几乎不降。96×96 网格≈2.7px/格，
    # 上采样后仍结构性消除 >96 周期的高频条纹，FFT 惩罚兜底不变
    prv_lowres_size = 96

    # 纹理掩码强度：控制梯度驱动掩码对隐私扰动的放大倍数
    texture_mask_strength = 3.0
    # 纹理掩码平滑核大小（平均池化核，用于平滑梯度掩码）
    texture_mask_smooth_kernel = 5

    # Phase 1 对抗损失权重三分段调度：起始（前 1/3）/中间（中 1/3）/末段（后 1/3）
    # 默认全 0：本场景（检索逃逸 + 溯源水印）不依赖隐写分析对抗；
    # 判别器失效时 L_adv 只会向残差注入额外高频伪影，需要时可重新调高
    phase1_adv_start = 0.0
    phase1_adv_mid = 0.0
    phase1_adv_end = 0.0
    # Phase 2 对抗损失权重（恒定）
    phase2_adv = 0.0

    # Phase 1 哑隐私噪声调度：训练到该比例轮次之前不注入噪声。
    # 注意：随机噪声会淹没 msg 信号、阻碍消息通路的冷启动，
    # 必须推迟到消息学习基本成型之后（0.8 对应 Phase 1 后 20% 轮次）
    phase1_prv_noise_start = 0.80
    # Phase 1 哑隐私噪声到达上限的训练比例轮次
    phase1_prv_noise_end = 0.95
    # Phase 1 哑隐私噪声相对于 prv_budget 的最大比例（降低上限，减小对消息收尾训练的干扰）
    phase1_prv_noise_max_ratio = 0.30
    # 哑隐私噪声 ramp 的幂次（>1 表示前期增长慢、后期增长快）
    phase1_prv_noise_power = 2.0
    # 哑隐私噪声平滑核大小（平均池化核，使噪声更自然）
    phase1_prv_noise_smooth_kernel = 7

    # 消息双分支损失的干净分支权重调度：Phase 1 内从 start 线性退火到 end。
    # 冷启动期干净分支占主导（0.8），避免含噪分支的噪声梯度稀释有效梯度（启动更快）；
    # 后期过渡到 0.5 均衡，让含噪分支充分学习抗噪；Phase 2 固定使用 end 值
    msg_clean_ratio_start = 0.80
    msg_clean_ratio_end = 0.50

    # 隐私双分支损失中含噪分支的权重（干净分支权重 = 1 - 该值）。
    # 含噪分支的梯度须穿过噪声层雅可比（quant 傅里叶近似震荡、jpeg 近似劣质），
    # 与干净分支等权（0.5）会让劣质梯度稀释一半有效梯度；
    # 0.3 让干净分支主导逃逸方向，含噪分支仅锚定传播场景鲁棒性
    privacy_noised_ratio = 0.3
    # Sprint 段的含噪分支权重：置 0，纯逃逸冲刺期间只用干净分支梯度
    # （劣质噪声雅可比会稀释本就微弱的逃逸信号；Balance 段恢复 0.3 锚定信道鲁棒性）
    privacy_noised_ratio_sprint = 0.0

    # Phase 2 逃逸冲刺段（Sprint）轮数：前 sprint_epochs 轮只训练 prv_head
    # （冻结 decoder/msg_head/shared，message 权重置 0，visual 轻锚），
    # 让隐私残差在无消息拉扯、方向一致的条件下学会泛化逃逸方向。
    # 依据：天花板探针证明在相同约束（96 低通/0.04 预算/掩码/双分支/视觉损失）下
    # 直接优化 60 步即可把 clip_sim 压到 0.64，而正常 Phase 2 的 20000 步停在 0.946——
    # 瓶颈是 Adam 在多目标、逐图方向不一致下的信噪比失效，而非约束集天花板。
    # Sprint 段 bit_acc 暂时下降是预期行为，由后续 Balance 段 + recovery 恢复
    # 40（原 20）：20 轮实测边际收益尚未衰减完（10→20 轮仍降 0.5pt），
    # 且新增结构（prv_trunk）从零初始化需要更长冲刺期收敛
    phase2_sprint_epochs = 40
    # Sprint 段 visual 锚权重（正常 Phase 2 为 15；仅防漂移，不追求视觉最优）
    phase2_sprint_visual = 1.5
    # Sprint 段 privacy 权重（与 Balance 末段一致，保持逃逸压力）
    phase2_sprint_privacy = 12.0
    # 终段精修轮数（Phase 2 最后 N 轮切换为 val/refine_encode.py 同配方目标）：
    # escape-only（redirect 关闭）+ 消息锚 10 + 视觉 15 + privacy 12，decoder 冻结，
    # fft/residual_penalty/noised 全部清零。动机：Sprint 在开头、Balance 在其后会把
    # 残差往"消息友好"方向回拉，最终模型停在折中态（实测 val clip_sim 卡 0.87）；
    # 终段精修让训练以"精修配方"收尾——该配方逐图验证可达 0.57 且 bit_acc 100%。
    # 默认 30：精修段是压 clip_sim 的核心机制，作为标准 Phase 2 流程的一部分；
    # 热启动纯精修 = --epochs-phase2 30（refine=N 全覆盖时无需显式传参）
    phase2_refine_epochs = 30
    # prv_head/prv_trunk 学习率倍率：绕开 Adam 信噪比瓶颈的最直接手段
    # （Adam 对梯度尺度不变，加大损失权重无效，只有提高 lr 能放大实际步长）
    prv_head_lr_mult = 5.0

    # ===== 方案 C：内环逐图优化蒸馏（Phase 2 启用）=====
    # 每个 train step 先冻结网络，把当前 batch 的低分辨率隐私残差场作为优化变量
    # 跑 distill_inner_steps 步 Adam（目标函数与逐图精修试点 val/refine_encode.py 一致：
    # escape + 消息双分支 BCE + 视觉 MSE/SSIM，约束链同构），得到教师残差；
    # 再用 MSE 把 prv_head 输出拉向教师残差，把探针级逃逸方向蒸进单次前向。
    # 试点实测（CPU，60 步）：clip_sim 0.94→0.55~0.65，bit_acc/PSNR/SSIM 均不劣化。
    # 代价：Phase 2 单步耗时约 2~4 倍；显存不足或过慢时可降为 5 或置 0 关闭
    distill_inner_steps = 10
    # 内环 Adam 学习率（与逐图精修验证一致）
    distill_lr = 0.1
    # 蒸馏损失权重（损失为按教师能量归一化的相对平方误差，量级 O(0.1~1)，
    # 与其他损失项同量级）。默认 25：蒸馏梯度是残差空间的直接回归（高信噪比），
    # escape 梯度须穿过整个 CLIP 网络（低信噪比），提权让干净信号主导更新方向
    distill_weight = 25.0
    # 教师目标函数各项权重（与 val/refine_encode.py 验证配方一致）
    distill_w_privacy = 12.0
    distill_w_message = 10.0
    distill_w_visual = 15.0

    # 编码期逐图精修的色度蒙层事后投影（深挖配方补丁，默认开启）。
    # 背景：实测残差能量 77% 落在色度通道、低频（8x8 块均值尺度）占其 44%，且总残差
    # 低频色度的 97% 来自 prv_residual——构成"发灰/蒙层"感知劣化主体（PSNR/SSIM 对其
    # 失明）。梯度惩罚方案实测无效（w=40~5000 均被逃逸梯度完全淹没，两者方向高度平行
    # ——CLIP 逃逸天然偏好色度通道），故改为结构化事后投影：精修收敛后把 prv_residual
    # 的 8x8 网格低频色度分量整支剔除。实测（test.jpg）：clip 0.438->0.467（仍在
    # margin 0.50 目标内）、bit 100%、PSNR +0.5dB、SSIM +0.001、低频色度 -91%。
    # 更粗尺度（16/16 以上网格）是逃逸承重结构（剔除后 clip 0.57/0.80），只动 8x8。
    # 门控：投影后 bit_acc 必须 100% 且 clip_sim ≤ (1-margin)，否则回退未投影结果。
    refine_chroma_projection = True

    # Phase 2 CLIP escape margin 渐进调度的起始值与终止值（目标 sim ≈ 1-margin）。
    # margin 是第一逃逸杠杆：二次 Hinge 在 sim < (1-margin) 后梯度归零，margin 即逃逸地板。
    # CPU 逐图精修实测（budget 0.08、60 步、bit_acc 100%）：margin 0.30/0.40/0.50
    # 对应 clip_sim 0.575/0.547/0.440（PSNR 39.9/39.4/39.0）。
    # 默认 0.50（深挖配方，目标 sim 0.50）：远低于实测 hard negative 均值 0.73，
    # CBIR top-1 易主余量充足；Sprint/Refine 段直接取该值，教师内环目标同源联动
    privacy_margin_start = 0.10
    privacy_margin_end = 0.50
    # Phase 2 DINOv2 escape margin 独立调度（DINO 视角下目标是脱离 near-duplicate 区 ~0.90，
    # 其 hard negative 极远（mean≈0.47），无需也无法压到其下）。
    # 默认 0.20（深挖配方，目标 sim 0.80）：精修实测代价很小，与 CLIP 逃逸不冲突
    dino_escape_margin_start = 0.06
    dino_escape_margin_end = 0.20

    # 是否启用 CLIP/DINO 特征重定向（CBIR 检索逃逸的核心机制）
    use_clip_redirect = True
    # CLIP memory bank 容量（用于存储历史特征作为 hard negative）
    clip_memory_size = 5000
    # 每次采样 hard negative 的数量
    clip_hard_negative_k = 5
    # CLIP escape loss 的 margin（遗留参数，redirect 目标现由 *_redirect_target_sim 控制）
    clip_redirect_margin = 0.35
    # 重定向目标相似度（按特征空间分别标定，依据实测 hard negative 分布上沿）：
    # CLIP 典型 hard negative sim≈0.73（p90≈0.83）。target 必须明显高于 neg 均值才有实质拉力
    # （target 0.80 时 relu(0.80-0.78)≈0.02，redirect 占隐私损失不到 1%，形同虚设）；
    # target 0.85 提供 ~0.07 的持续拉力，redirect 成为指路的第二引擎
    clip_redirect_target_sim = 0.85
    dino_redirect_target_sim = 0.60
    # hard negative 候选排除阈值（按空间标定，排除自身/近重复/同身份照片）：
    # CLIP p99≈0.92，DINO p99≈0.87
    clip_redirect_exclude_sim = 0.92
    dino_redirect_exclude_sim = 0.87
    # 重定向相似度控制上限（超过则施加惩罚，防止变成他人图像的"副本"；
    # 须保持 exclude > max > target 层次：0.92 > 0.90 > 0.85）
    clip_max_target_sim = 0.90
    # CLIP 重定向损失权重（Sprint 段的指路引擎：redirect 给出逐图明确的逃逸目标点，
    # 方向跨步一致，是提高 Adam 梯度信噪比的关键分量）
    clip_redirect_weight = 3.0
    # DINOv2 重定向损失权重
    dino_redirect_weight = 0.6
    # DINOv2 escape loss 的 margin（遗留参数）
    dino_redirect_margin = 0.3
    # DINOv2 重定向目标相似度上限
    dino_max_target_sim = 0.65

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
    # 默认开启 jpeg+resize+quant：溯源水印必须在失真信道上训练才具备实际鲁棒性
    noise_config = 'jpeg+resize(0.8,1.2)+quant'

    def is_phase2_refine_segment(self, epoch):
        # 判断 Phase 2 某轮是否处于终段精修区间（最后 phase2_refine_epochs 轮）。
        # 检查顺序上 refine 优先于 sprint：热启动纯精修（--epochs-phase2 N
        # --refine-epochs N）时全部轮次都应命中 refine 而非被 sprint 声明。
        # 参数：
        #   epoch: Phase 2 阶段内轮次（从 1 开始计数）
        refine = getattr(self, 'phase2_refine_epochs', 0)
        return refine > 0 and epoch > self.num_epochs_phase2 - refine

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
            # Phase 1 不计算隐私损失（privacy=0），不施加 FFT 惩罚（fft=0）。
            # noised 视觉权重恒为 0：L_visual_noised 会把信道失真（JPEG/resize 伪影）
            # 归咎于编码器残差，与"消息必须活着穿过噪声"的目标直接对抗，
            # 实测会导致编码器以零残差躺平、消息学习完全死锁；
            # 视觉质量只约束编码图本身（用户看到和传播的就是编码图）。
            # message 权重 15：噪声信道下消息学习本身偏慢，加大权重加速收敛
            return {
                'visual': 10.0,
                'message': 15.0,
                'adv': adv,
                'privacy': 0.0,
                'noised': 0.0,
                'residual_penalty': 0.5,
                'fft': 0.0
            }
        elif phase == 2:
            # Refine 终段精修（优先于 sprint 判定）：val/refine_encode.py 已验证配方——
            # escape-only privacy 12 + 消息锚 10（decoder 冻结，残差自适应既有解码器）
            # + visual 15。反作用力全部清零（同 sprint），margin 固定 end 值
            # （见 get_escape_margin 覆盖）。消息锚是该段与 sprint 的本质差异：
            # sprint 靠蒸馏教师间接保解码，refine 由学生目标直接保解码，
            # 因此终段可放心压 clip_sim 而不会把 bit_acc 打穿阈值
            if self.is_phase2_refine_segment(epoch):
                return {
                    'visual': self.distill_w_visual,
                    'message': self.distill_w_message,
                    'adv': 0.0,
                    'privacy': self.distill_w_privacy,
                    'noised': 0.0,
                    'residual_penalty': 0.0,
                    'fft': 0.0
                }
            # Sprint 段：纯逃逸冲刺，message 权重置 0、visual 仅轻锚。
            # 此段 decoder 被冻结（见 trainer._set_phase_freeze），
            # bit_acc 暂时下降是预期行为，Balance 段恢复
            if epoch <= self.phase2_sprint_epochs:
                # Sprint 段反作用力清零（P1 修复）：residual_penalty 是收缩残差的力，
                # fft 惩罚会压掉 CLIP 敏感的中频逃逸方向——两者在纯逃逸冲刺期
                # 直接对冲逃逸梯度，实测是 Sprint 推不动的次要原因之一；
                # Balance 段恢复（0.5 / lambda_fft）保证最终视觉与频谱约束
                return {
                    'visual': self.phase2_sprint_visual,
                    'message': 0.0,
                    'adv': 0.0,
                    'privacy': self.phase2_sprint_privacy,
                    'noised': 0.0,
                    'residual_penalty': 0.0,
                    'fft': 0.0
                }
            # Balance 段：将剩余轮数三等分，逐步提升隐私权重、降低消息权重。
            # 隐私权重 4/8/12：privacy 梯度须穿过整个 CLIP/DINO 网络才能到达残差
            # （∂sim/∂残差极小），而 visual/message 对残差是逐像素直接的（∂≈1），
            # 实测 1.5/2.5/3.5 的权重无法补偿 1~2 个数量级的雅可比差距，
            # 平衡点会被钉死在 sim≈0.97（privacy 占损失 88% 但 sim 几乎不降）
            balance_epochs = max(1, self.num_epochs_phase2 - self.phase2_sprint_epochs)
            balance_epoch = epoch - self.phase2_sprint_epochs
            third = max(1, balance_epochs // 3)
            if balance_epoch <= third:
                # 前 1/3：隐私权重较低，消息权重较高（优先恢复解码）
                privacy = 4.0
                message = 20.0
            elif balance_epoch <= 2 * third:
                # 中 1/3：隐私权重提升，消息权重略降
                privacy = 8.0
                message = 15.0
            else:
                # 后 1/3：隐私权重最高，消息权重最低
                privacy = 12.0
                message = 10.0
            # Phase 2 启用 FFT 高频惩罚，约束隐私残差频谱
            # visual 权重提升到 15：redirect 分担隐私目标后，把梯度预算让给视觉一致性；
            # noised 视觉权重同样恒为 0（理由见 Phase 1 注释）
            return {
                'visual': 15.0,
                'message': message,
                'adv': self.phase2_adv,
                'privacy': privacy,
                'noised': 0.0,
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
        # Refine 终段：目标直接取 end 值（与 val/refine_encode.py 精修配方一致）。
        # 热启动纯精修时轮次从 1 重新起算，若走正常 ramp 会从 start 值
        # 重新渐进——对一个已训好的模型等于先松后紧，浪费轮次且方向倒退
        # Sprint 段同样直接取 end 值：该段是 privacy 单目标的纯逃逸冲刺，
        # 不存在多目标突变风险，ramp 只会削弱二次 Hinge 梯度
        # （梯度 ∝ 50×(sim-target)，实测 40 轮 sprint 中 margin 仅 0.10→0.26，
        # 冲刺前期梯度被削弱 ~3 倍）
        if self.is_phase2_refine_segment(epoch) or epoch <= self.phase2_sprint_epochs:
            return self.privacy_margin_end
        # Phase 2 总轮数
        total = self.num_epochs_phase2
        # margin 在前 50% 轮次内到达 end 值，后半程目标稳定：
        # 若全程渐进，训练将一直追逐不断远去的 hinge 目标（实测 margin 到最后
        # 一个 epoch 才到达 end，模型从未在最终目标下稳定优化过）
        progress = min(1.0, (epoch - 1) / max(1, 0.5 * total - 1))
        # 线性插值得到当前 margin
        return self.privacy_margin_start + (self.privacy_margin_end - self.privacy_margin_start) * progress

    def get_dino_escape_margin(self, phase, epoch):
        # 计算 Phase 2 DINOv2 escape margin（独立于 CLIP，目标更宽松）。
        # 与 get_escape_margin 相同的线性调度形式。
        # 参数：
        #   phase: 训练阶段（1 或 2）
        #   epoch: 当前轮次（从 1 开始计数）
        # 返回：当前轮次的 DINO escape margin 标量
        if phase == 1:
            # Phase 1 不计算隐私损失，margin 恒为 0
            return 0.0
        # Refine 终段 / Sprint 段：与 CLIP margin 同理，直接取 end 值
        if self.is_phase2_refine_segment(epoch) or epoch <= self.phase2_sprint_epochs:
            return self.dino_escape_margin_end
        total = self.num_epochs_phase2
        # 与 get_escape_margin 一致：前 50% 轮次到达 end 值
        progress = min(1.0, (epoch - 1) / max(1, 0.5 * total - 1))
        return self.dino_escape_margin_start + (self.dino_escape_margin_end - self.dino_escape_margin_start) * progress

    def get_privacy_noised_ratio(self, phase, epoch):
        # 计算隐私双分支损失中含噪分支的权重（阶段感知）。
        # Sprint 段返回 privacy_noised_ratio_sprint（默认 0，纯逃逸冲刺只用干净分支梯度），
        # 其余阶段返回 privacy_noised_ratio（Balance 段锚定信道鲁棒性）。
        # 参数：
        #   phase: 训练阶段（1 或 2）
        #   epoch: 当前阶段内轮次（从 1 开始计数）
        # 返回：含噪分支权重标量
        if phase == 2 and epoch <= self.phase2_sprint_epochs:
            return getattr(self, 'privacy_noised_ratio_sprint', 0.0)
        # Refine 终段：同 Sprint 语义，只用干净分支梯度（val/refine_encode.py 精修
        # 只在干净图上算 escape；含噪分支劣质雅可比会稀释精修方向的信噪比）
        if phase == 2 and self.is_phase2_refine_segment(epoch):
            return getattr(self, 'privacy_noised_ratio_sprint', 0.0)
        return self.privacy_noised_ratio

    def get_msg_clean_ratio(self, phase, epoch):
        # 计算消息双分支损失中干净分支的权重。
        # Phase 1 内从 msg_clean_ratio_start 线性退火到 msg_clean_ratio_end；
        # Phase 2 固定使用 end 值（此时含噪分支应已具备抗噪能力）
        # 参数：
        #   phase: 训练阶段（1 或 2）
        #   epoch: 当前阶段内轮次（从 1 开始计数）
        # 返回：干净分支权重标量（含噪分支权重 = 1 - 该值）
        if phase == 1:
            total = max(1, self.num_epochs_phase1)
            progress = min(1.0, max(0.0, (epoch - 1) / max(1, total - 1)))
            return self.msg_clean_ratio_start + (self.msg_clean_ratio_end - self.msg_clean_ratio_start) * progress
        return self.msg_clean_ratio_end

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