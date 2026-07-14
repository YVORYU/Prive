# PrivSteg: 隐私保护图像隐写系统

基于 GAN 的隐私保护图像隐写框架，在载体图像中嵌入二进制消息的同时，对抗基于内容的图像检索（CBIR）系统，防止隐写图像被检索关联到原始载体图像。

## 核心特性

- **消息嵌入与提取**：在图像中嵌入二进制消息（默认 30 bit），解码端可准确提取
- **双残差编码架构**：编码器采用消息头（msg_head）和隐私头（prv_head）的双头设计，分别生成预算约束的消息残差和隐私残差，实现消息嵌入与隐私保护的解耦
- **纹理感知隐私扰动**：Phase 2 中隐私残差通过梯度驱动的纹理掩码加权，将扰动集中在纹理复杂区域，降低视觉感知
- **视觉不可感知性**：编码图像与原图在像素级（MSE）和结构级（SSIM）上保持高度一致
- **隐写分析对抗**：通过 GAN 对抗训练 + SpectralNorm 使隐写图像能逃逸隐写分析器的检测；判别器采用 Hinge Loss 并按准确率门控更新
- **频域正则化**：对隐私残差施加 FFT 高频能量惩罚，抑制易被频域分析器捕捉的高频痕迹
- **有向重定向隐私保护**：利用 CLIP + DINOv2 多特征空间进行有向重定向，将隐写图像特征偏移至语义相近但不同的硬负样本目标，防止被 CBIR 系统检索关联
- **渐进逃逸边界**：逃逸 margin 从 0.15 线性增长到 0.30，避免隐私损失突然激增导致训练崩溃
- **解码器恢复机制**：当 Phase 2 比特准确率下降时，在干净消息残差上重新训练解码器，确保消息提取能力不退化
- **噪声鲁棒性**：支持多种噪声层（裁剪、JPEG 压缩、量化等），保证消息在信道干扰下仍可提取

## 项目结构

```
PrivSteg/
├── config.py                  # 全局配置（超参数、损失权重、训练阶段等）
├── train.py                   # 训练入口（命令行参数解析、训练循环）
├── trainer.py                 # 训练器（模型初始化、训练步骤、验证、检查点管理）
├── validation_script.py       # 验证脚本（编码/解码/判别器/相似度判断）
├── noise_argparser.py         # 噪声层命令行参数解析
├── requirements.txt           # Python 依赖
├── models/
│   ├── encoder.py             # 编码器（双残差头 + 纹理掩码，预算约束）
│   ├── decoder.py             # 解码器（CNN + 全连接消息提取）
│   ├── steganalyzer.py        # 隐写分析器（判别器，SpectralNorm）
│   └── content_recognizer.py  # 内容识别器（冻结 CLIP ViT-B/32）
├── losses/
│   ├── ssim_loss.py           # SSIM 损失
│   └── advanced_privacy_loss.py  # 特征重定向损失（CLIP + DINOv2 多模态逃逸 + 有向重定向）
├── noise_layers/
│   ├── noiser.py              # 噪声层组合器（随机选择激活）
│   ├── identity.py            # 恒等层（无噪声）
│   ├── crop.py                # 裁剪噪声
│   ├── cropout.py             # 裁剪替换噪声
│   ├── dropout.py             # 随机丢弃噪声
│   ├── resize.py              # 缩放噪声
│   ├── jpeg_compression.py    # JPEG 压缩噪声（DCT 域量化）
│   └── quantization.py        # 傅里叶近似量化噪声
└── data/
    └── dataset.py             # 隐写数据集（图像加载 + 随机消息生成）
```

## 网络架构

### 编码器（Encoder）

- 16 层 ConvBNRelu（128 通道）特征提取 → 消息空间扩展拼接 → 双头输出
- 输入：载体图像 (B, 3, H, W) + 消息 (B, L) + 阶段参数
- 拼接通道：30(message) + 128(features) + 3(cover) = 161
- **消息头（msg_head）**：2 层 3×3 卷积 + 1 层 1×1 卷积 → `msg_budget × tanh` → 消息残差
- **隐私头（prv_head）**：2 层 3×3 卷积 + 1 层 1×1 卷积 → `prv_budget × tanh` → 隐私残差
- **纹理掩码**：对载体图像计算梯度幅值，经平滑和 sigmoid 后得到 [0.15, 1.0] 的空间权重，Phase 2 中乘到隐私残差上
- Phase 1：编码图 = 原图 + 消息残差 + 平滑归一化随机噪声（渐进引入）
- Phase 2：编码图 = 原图 + 消息残差 + 纹理掩码加权的隐私残差
- 输出：编码图像 (B, 3, H, W) + 残差惩罚项 + 消息残差 + 隐私残差

### 解码器（Decoder）

- 8 层 ConvBNRelu（64 通道）+ ConvBNRelu(64→30) + AdaptiveAvgPool + 全连接层
- 输入：编码图像（可含噪声）
- 输出：消息 logits (B, L)（配合 BCEWithLogitsLoss）

### 隐写分析器（Steganalyzer）

- 8 层卷积（SpectralNorm）+ 全局平均池化 + 双层全连接判别头
- 渐进式通道：64 → 64 → 128 → 128 → 256 → 256 → 512 → 512
- MaxPool 下采样：256 → 128 → 64 → 32 → 1
- LeakyReLU(0.2)
- FC 结构：512 → 256 → 1
- 输出：图像是否为隐写图像的 logits（正值为 stego，负值为 cover）

### 内容识别器（ContentRecognizer）

- 冻结的 CLIP ViT-B/32 图像编码器
- 输入预处理：Resize(224) → [-1,1]→[0,1] → ImageNet 归一化
- 提取 L2 归一化的 512 维图像特征
- 支持文本编码（encode_text 方法）

## 损失函数

| 损失            | 说明                                          |
| ------------- | ------------------------------------------- |
| L\_pixel      | 像素级 MSE 损失                                  |
| L\_ssim       | 结构相似性损失（11×11 高斯窗口）                        |
| L\_message    | 消息提取 BCEWithLogits 损失                       |
| L\_adv        | 生成器对抗损失（Softplus 形式 `softplus(-fake_logits)`）              |
| L\_D          | 判别器 Hinge Loss（margin=1.0，按准确率门控更新）               |
| L\_privacy    | 隐私保护损失（CLIP + DINOv2 特征逃逸，二次 Hinge + 重定向） |
| L\_residual   | 残差惩罚项（消息残差 + 隐私残差 MSE，控制隐写强度）          |
| L\_fft        | FFT 高频能量惩罚（仅 Phase 2，约束隐私残差的高频分量）          |

### 隐私损失详解

隐私损失由 `FeatureRedirectionLoss` 计算，在 CLIP (512d) 和 DINOv2 (384d) 特征空间中分别计算：

- **逃逸项**：组合二次 Hinge 与 Softplus 预热。`diff = cos_sim - (1 - margin)`，损失为 `25 × mean(relu(diff)²) + 0.1 × mean(softplus(diff))`，对违反约束的样本施加强二次梯度
- **重定向项**：Softplus Loss（`log1p(exp(target - sim_to_hardest))`）推动编码图像特征靠近硬负样本目标
- **排斥控制项**：Hinge Loss（`relu(sim_to_hardest - max_target_sim)`）防止过度接近导致特征坍塌
- **渐进逃逸边界**：逃逸 margin 从 `privacy_margin_start=0.15` 线性增长到 `privacy_margin_end=0.30`

## 两阶段训练

| 阶段      | Epoch | 目标               | 关键权重                                          |
| ------- | ----- | ---------------- | --------------------------------------------- |
| Phase 1 | 150   | 基础训练：视觉质量 + 消息提取 + 渐进对抗 | visual=10, message=10, adv 渐进(0→0.25→0.75), privacy=0       |
| Phase 2 | 100   | 隐私训练：冻结消息头 + 训练隐私头 + 隐私保护损失 + FFT 正则 | visual=10, message 渐进(20→15→10), adv=0.5, privacy 渐进(2→4→5), fft=10.0      |

**头部冻结策略**：
- Phase 1：消息头可训练，隐私头冻结，共享层可训练
- Phase 2 前段：消息头冻结，隐私头可训练，共享层可训练
- Phase 2 后段：消息头冻结，隐私头可训练，共享层冻结

**解码器恢复机制**：Phase 2 中若验证比特准确率低于 `bit_accuracy_threshold=0.90`，则在干净消息残差上重新训练解码器（`recovery_steps=10` 步）。

**记忆库预热**：Phase 1 开始前用 50 个 batch 的封面图像特征预填 CLIP 和 DINO 记忆库。

**阶段权重渐进调度**：
- Phase 1 内 `adv` 权重分三段渐进放大（0.0 → 0.25 → 0.75）
- Phase 2 内 `privacy` 权重分三段渐进放大（2.0 → 4.0 → 5.0）
- Phase 2 内 `message` 权重分三段渐进缩小（20.0 → 15.0 → 10.0）
- Phase 1 内隐私噪声比例 `prv_noise_ratio` 渐进引入（前 50% 为 0，50%-90% 幂律增长最高至 0.5，90% 后保持 0.5）

**判别器更新门控**：仅当对抗权重 `adv > 0` 且判别器准确率 `D_acc < disc_acc_high(0.70)` 时才更新判别器，防止判别器过强导致生成器梯度消失。

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA（推荐）

## 安装

```bash
git clone https://github.com/your-repo/PrivSteg.git
cd PrivSteg
pip install -r requirements.txt
```

> 注意：CLIP 需要从 GitHub 安装，`requirements.txt` 中已包含 `git+https://github.com/openai/CLIP.git`。DINOv2 模型权重需单独下载并放置到 `./models/` 目录（支持位置编码插值适配不同图像尺寸）。

## 数据准备

将训练和验证图像分别放置到以下目录结构：

```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── val/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

支持 `.png`、`.jpg`、`.jpeg`、`.bmp` 格式，数据集类会递归扫描子目录。

## 使用方法

### 训练

**启动新训练：**

```bash
python train.py new --data-dir ./dataset --name my_experiment
```

**常用训练参数：**

```bash
python train.py new \
    --data-dir ./dataset \
    --name my_experiment \
    --batch-size 8 \
    --size 256 \
    --message 30 \
    --epochs-phase1 150 \
    --epochs-phase2 100 \
    --lr-encoder 1e-3 \
    --lr-decoder 1e-3 \
    --lr-steganalyzer 1e-4 \
    --disc-acc-high 0.70 \
    --disc-hinge-margin 1.0 \
    --phase1-adv-start 0.0 \
    --phase1-adv-mid 0.25 \
    --phase1-adv-end 0.75 \
    --phase2-adv 0.5 \
    --lambda-pixel 1.0 \
    --lambda-ssim 1.0 \
    --noise "cropout((0.55, 0.6), (0.55, 0.6))+jpeg" \
    --seed 42
```

**从检查点继续训练：**

```bash
python train.py continue --folder ./experiments/my_experiment
```

**从指定检查点恢复新实验：**

```bash
python train.py new --data-dir ./dataset --name resumed_exp --continue-from-folder ./experiments/my_experiment
```

### 验证

**完整验证（编码 + 解码 + 判别器 + 相似度）：**

```bash
python validation_script.py \
    -c ./experiments/my_experiment/checkpoints/checkpoint.pth \
    -i ./test.jpg \
    -m "101000100110100101100011100010" \
    --mode full \
    -o output
```

**仅判别器预测：**

```bash
python validation_script.py \
    -c ./experiments/my_experiment/checkpoints/checkpoint.pth \
    -i ./encoded_image.png \
    --mode discriminator
```

**仅解码消息：**

```bash
python validation_script.py \
    -c ./experiments/my_experiment/checkpoints/checkpoint.pth \
    -i ./encoded_image.png \
    --mode decode
```

**相似度判断（需提供原图和隐写图）：**

```bash
python validation_script.py \
    -i ./encoded_image.png \
    --cover ./cover_image.png \
    --mode similarity
```

## 关键配置参数

| 参数                       | 默认值               | 说明                                  |
| ------------------------ | ----------------- | ----------------------------------- |
| `image_size`             | 256               | 图像尺寸                                |
| `message_length`         | 30                | 嵌入消息长度（bit）                         |
| `batch_size`             | 8                 | 批大小                                 |
| `lr_encoder`             | 1e-3              | 编码器学习率                              |
| `lr_decoder`             | 1e-3              | 解码器学习率                              |
| `lr_steganalyzer`        | 1e-4              | 隐写分析器学习率                            |
| `num_epochs_phase1`      | 150               | Phase 1 训练轮数                         |
| `num_epochs_phase2`      | 100               | Phase 2 训练轮数                         |
| `discriminator_loss_threshold` | 0.6         | 判别器损失阈值（遗留兼容参数）                    |
| `disc_acc_high`          | 0.70              | 判别器准确率达到此值时暂停更新                    |
| `disc_hinge_margin`      | 1.0               | 判别器 Hinge Loss 的 margin            |
| `lambda_pixel`           | 1.0               | MSE 像素损失权重                          |
| `lambda_ssim`            | 1.0               | SSIM 结构相似性损失权重                      |
| `lambda_fft`             | 10.0              | FFT 高频能量惩罚权重（Phase 2）               |
| `msg_budget`             | 0.04              | 消息残差预算（tanh 缩放系数）                  |
| `prv_budget`             | 0.04              | 隐私残差预算（tanh 缩放系数）                  |
| `texture_mask_strength`  | 3.0               | 纹理掩码 sigmoid 斜率                     |
| `texture_mask_smooth_kernel` | 5               | 纹理掩码梯度平滑核大小                        |
| `phase1_adv_start`       | 0.0               | Phase 1 前段对抗权重                      |
| `phase1_adv_mid`         | 0.25              | Phase 1 中段对抗权重                      |
| `phase1_adv_end`         | 0.75              | Phase 1 后段对抗权重                      |
| `phase2_adv`             | 0.5               | Phase 2 对抗权重                        |
| `phase1_prv_noise_start` | 0.50              | Phase 1 隐私噪声开始比例                    |
| `phase1_prv_noise_end`   | 0.90              | Phase 1 隐私噪声达到上限比例                  |
| `phase1_prv_noise_max_ratio` | 0.50            | Phase 1 隐私噪声最大比例                    |
| `phase1_prv_noise_power` | 2.0               | Phase 1 隐私噪声幂律指数                    |
| `phase1_prv_noise_smooth_kernel` | 7             | Phase 1 隐私噪声平滑核大小                   |
| `clip_weight`            | 1.0               | CLIP 特征逃逸损失权重                       |
| `dino_weight`            | 0.5               | DINOv2 特征逃逸损失权重                     |
| `privacy_margin_start`   | 0.15              | 逃逸边界起始值                             |
| `privacy_margin_end`     | 0.30              | 逃逸边界终止值                             |
| `use_clip_redirect`      | False             | 启用 CLIP/DINO 有向重定向                  |
| `clip_redirect_weight`   | 0.6               | CLIP 重定向损失权重                         |
| `dino_redirect_weight`   | 0.3               | DINO 重定向损失权重                         |
| `clip_redirect_margin`   | 0.35              | CLIP 重定向 margin                      |
| `dino_redirect_margin`   | 0.3               | DINO 重定向 margin                      |
| `clip_max_target_sim`    | 0.55              | CLIP 最大允许目标相似度上限                    |
| `dino_max_target_sim`    | 0.55              | DINO 最大允许目标相似度上限                    |
| `clip_memory_size`       | 5000              | CLIP 特征记忆库容量                        |
| `clip_hard_negative_k`   | 5                 | 硬负样本候选数                             |
| `bit_accuracy_threshold` | 0.90              | 触发解码器恢复的比特准确率阈值                    |
| `recovery_steps`         | 10                | 解码器恢复训练步数                           |
| `memory_warmup_batches`  | 50                | 记忆库预热批次数                            |
| `noise_config`           | identity          | 噪声层配置                               |

## 输出说明

训练过程会在 `experiments/<experiment_name>/` 目录下生成：

```
experiments/my_experiment/
├── config.json              # 训练配置
├── train.log                # 训练日志
├── checkpoints/
│   ├── checkpoint.pth       # 最新检查点
│   ├── checkpoint_phase{N}_epoch{M}.pth  # 阶段性检查点
│   └── visualizations/      # 验证可视化图像
└── tensorboard/             # TensorBoard 日志
```

可通过 TensorBoard 监控训练过程：

```bash
tensorboard --logdir ./experiments/my_experiment/tensorboard
```

## 技术原理

### 双残差编码架构

编码器采用消息头和隐私头的双头设计，将消息嵌入和隐私保护解耦为两个独立的残差路径：
- **消息残差**：由 `msg_head` 生成，经 `msg_budget × tanh` 约束幅度，确保消息信号在预算范围内嵌入
- **隐私残差**：由 `prv_head` 生成，经 `prv_budget × tanh` 约束幅度，在特征空间中偏移编码图像
- Phase 1 中隐私头冻结，仅注入经平滑与零均值单位方差归一化的随机噪声（幂律渐进引入），为 Phase 2 的隐私残差做铺垫
- Phase 2 中消息头冻结，仅训练隐私头和解码器，避免隐私训练破坏已学会的消息嵌入

### 纹理感知隐私扰动

Phase 2 中，隐私残差在叠加到图像前会乘以一个纹理掩码。该掩码通过对载体图像灰度图计算水平/垂直梯度幅值、经均值滤波平滑后通过 `sigmoid(grad × strength)` 生成，取值范围 [0.15, 1.0]。这使得隐私扰动集中在边缘和纹理丰富的区域——人眼对这些区域的像素变化不敏感——而平滑区域扰动被抑制，从而在相同的隐私预算下获得更好的视觉不可感知性。

### 频域正则化

Phase 2 对隐私残差施加 FFT 高频能量惩罚 `_fft_high_freq_loss`：对残差做二维正交 FFT 并 fftshift，统计中心半径 `min(H,W)//4` 以外的高频分量幅度均值的平方根。该损失乘以 `lambda_fft=10.0` 后加入总损失，抑制隐私扰动产生易被频域隐写分析器捕获的高频条纹。

### 隐私保护机制

传统隐写方法仅关注视觉不可感知性和消息鲁棒性，但忽略了基于内容的图像检索（CBIR）威胁——攻击者可通过特征相似度检索将隐写图像关联到原始载体。PrivSteg 通过以下机制解决此问题：

1. **多特征空间有向重定向**：同时在 CLIP（512d，语义理解）和 DINOv2（384d，细粒度纹理）特征空间中，使用二次 Hinge 逃逸损失推动隐写图像特征远离原图特征，并通过 Softplus Loss 将特征重定向至语义相近但不同的硬负样本目标
2. **硬负样本挖掘**：从记忆库（MoCo 风格循环缓冲区）中动态选取 top-k 最相似样本，再选最难的 1 个作为重定向目标，增强判别能力
3. **排斥控制**：通过 `max_target_sim` 限制重定向目标的最大允许相似度，防止编码图像被重定向到与原图过于相似的样本，保证隐私偏移的有效性
4. **渐进逃逸边界**：逃逸 margin 从 0.15 线性增长到 0.30，使隐私约束由松到紧，避免训练崩溃

### 对抗训练

判别器（隐写分析器）使用 Hinge Loss（`relu(margin - real) + relu(margin + fake)`，margin=1.0）替代传统 BCE，生成器使用 Softplus 形式的非饱和损失（`softplus(-fake_logits)`）。判别器更新受准确率门控：仅当 `D_acc < 0.70` 时才执行更新步，避免判别器过强导致生成器梯度消失。SpectralNorm 限制判别器满足 Lipschitz 连续性，进一步稳定对抗训练。

### 噪声鲁棒性

Noiser 模块在训练时随机选择一种噪声层施加于编码图像，模拟实际信道中的各种干扰，确保解码器在噪声环境下仍能准确提取消息。支持组合噪声配置，如 `cropout+jpeg`。

## 详细技术文档

完整的技术实现细节、算法原理、数据流图和参数说明请参阅 [TECHNICAL_DOC.md](TECHNICAL_DOC.md)。
