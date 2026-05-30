# PrivSteg: 隐私保护图像隐写系统

基于 GAN 的隐私保护图像隐写框架，在载体图像中嵌入秘密消息的同时，对抗基于内容的图像检索（CBIR）系统，防止隐写图像被检索关联到原始载体图像。

## 核心特性

- **消息嵌入与提取**：在图像中嵌入二进制消息（默认 30 bit），解码端可准确提取
- **双残差编码架构**：编码器采用消息头（msg_head）和隐私头（prv_head）的双头设计，分别生成预算约束的消息残差和隐私残差，实现消息嵌入与隐私保护的解耦
- **视觉不可感知性**：编码图像与原图在像素级（MSE）和结构级（SSIM）上保持高度一致
- **隐写分析对抗**：通过 GAN 对抗训练 + SpectralNorm 使隐写图像能逃逸隐写分析器的检测
- **有向重定向隐私保护**：利用 CLIP + DINOv2 多特征空间进行有向重定向，将隐写图像特征偏移至语义相近但不同的硬负样本目标，防止被 CBIR 系统检索关联
- **渐进逃逸边界**：逃逸 margin 从 0.05 线性增长到 0.30，避免隐私损失突然激增导致训练崩溃
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
│   ├── encoder.py             # 编码器（双残差头：msg_head + prv_head，预算约束）
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
- **消息头（msg_head）**：3 层卷积 → `msg_budget × tanh` → 消息残差
- **隐私头（prv_head）**：3 层卷积 → `prv_budget × tanh` → 隐私残差
- Phase 1：编码图 = 原图 + 消息残差 + 随机隐私噪声（渐进引入）
- Phase 2：编码图 = 原图 + 消息残差 + 隐私残差
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
- 输出：图像是否为隐写图像的 logits

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
| L\_adv        | GAN 对抗损失（非饱和 + SpectralNorm）               |
| L\_privacy    | 隐私保护损失（CLIP + DINOv2 有向重定向，Softplus 逃逸 + 重定向） |
| L\_residual   | 残差惩罚项（消息残差 + 隐私残差 MSE，控制隐写强度）          |

### 隐私损失详解

隐私损失由 `FeatureRedirectionLoss` 计算，在 CLIP (512d) 和 DINOv2 (384d) 特征空间中分别计算：

- **逃逸项**：Softplus Loss（`log1p(exp(sim - target))`）推动编码图像特征远离原图特征
- **重定向项**：Softplus Loss（`log1p(exp(target - sim_to_hardest))`）推动编码图像特征靠近硬负样本目标
- **排斥控制项**：Hinge Loss（`relu(sim - max_target_sim)`）防止过度接近导致特征坍塌
- **渐进逃逸边界**：逃逸 margin 从 `privacy_margin_start=0.05` 线性增长到 `privacy_margin_end=0.30`

## 两阶段训练

| 阶段      | Epoch | 目标               | 关键权重                                          |
| ------- | ----- | ---------------- | --------------------------------------------- |
| Phase 1 | 130   | 基础训练：视觉质量 + 消息提取 + 渐进对抗 | visual=10, message=10, adv 渐进, privacy=0       |
| Phase 2 | 150   | 隐私训练：冻结消息头 + 训练隐私头 + 隐私保护损失 | visual=1, message 渐进, adv=2, privacy 渐进      |

**头部冻结策略**：
- Phase 1：消息头可训练，隐私头冻结，共享层可训练
- Phase 2 前段：消息头冻结，隐私头可训练，共享层可训练
- Phase 2 后段：消息头冻结，隐私头可训练，共享层冻结

**解码器恢复机制**：Phase 2 中若验证比特准确率低于 `bit_accuracy_threshold=0.90`，则在干净消息残差上重新训练解码器（`recovery_steps=10` 步）。

**记忆库预热**：Phase 1 开始前用 50 个 batch 的封面图像特征预填 CLIP 和 DINO 记忆库。

**阶段权重渐进调度**：
- Phase 1 内 `adv` 权重分三段渐进放大（1× → 2× → 3×）
- Phase 2 内 `privacy` 权重分三段渐进放大（3.0 → 6.0 → 10.0）
- Phase 2 内 `message` 权重分三段渐进缩小（15.0 → 12.0 → 10.0）
- Phase 1 内隐私噪声比例 `prv_noise_ratio` 渐进引入（前 30% 为 0，30%-70% 线性增长，70% 后为 1）

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
    --epochs-phase1 130 \
    --epochs-phase2 150 \
    --lr-encoder 1e-3 \
    --lr-decoder 1e-3 \
    --lr-steganalyzer 1e-3 \
    --discriminator-loss-threshold 0.4 \
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
| `lr_steganalyzer`        | 1e-3              | 隐写分析器学习率                            |
| `num_epochs_phase1`      | 130               | Phase 1 训练轮数                         |
| `num_epochs_phase2`      | 150               | Phase 2 训练轮数                         |
| `discriminator_loss_threshold` | 0.4         | 判别器损失低于此值时暂停训练                    |
| `msg_budget`             | 0.04              | 消息残差预算（tanh 缩放系数）                  |
| `prv_budget`             | 0.06              | 隐私残差预算（tanh 缩放系数）                  |
| `clip_weight`            | 1.0               | CLIP 特征逃逸损失权重                       |
| `dino_weight`            | 1.0               | DINOv2 特征逃逸损失权重                     |
| `privacy_margin_start`   | 0.05              | 逃逸边界起始值                             |
| `privacy_margin_end`     | 0.30              | 逃逸边界终止值                             |
| `use_clip_redirect`      | True              | 启用 CLIP/DINO 有向重定向                  |
| `clip_redirect_weight`   | 0.3               | CLIP 重定向损失权重                         |
| `dino_redirect_weight`   | 0.4               | DINO 重定向损失权重                         |
| `clip_redirect_margin`   | 0.4               | CLIP 重定向 margin                      |
| `dino_redirect_margin`   | 0.4               | DINO 重定向 margin                      |
| `clip_max_target_sim`    | 0.65              | CLIP 最大允许目标相似度上限                    |
| `dino_max_target_sim`    | 0.65              | DINO 最大允许目标相似度上限                    |
| `clip_memory_size`       | 1000              | CLIP 特征记忆库容量                        |
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
│   ├── best_phase1.pth      # Phase 1 最佳模型
│   ├── best_phase2.pth      # Phase 2 最佳模型
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
- Phase 1 中隐私头冻结，仅注入随机噪声（渐进引入），为 Phase 2 的隐私残差做铺垫
- Phase 2 中消息头冻结，仅训练隐私头和解码器，避免隐私训练破坏已学会的消息嵌入

### 隐私保护机制

传统隐写方法仅关注视觉不可感知性和消息鲁棒性，但忽略了基于内容的图像检索（CBIR）威胁——攻击者可通过特征相似度检索将隐写图像关联到原始载体。PrivSteg 通过以下机制解决此问题：

1. **多特征空间有向重定向**：同时在 CLIP（512d，语义理解）和 DINOv2（384d，细粒度纹理）特征空间中，使用 Softplus Loss 推动隐写图像特征远离原图特征，并通过 Softplus Loss 将特征重定向至语义相近但不同的硬负样本目标，实现有方向的隐私偏移
2. **硬负样本挖掘**：从记忆库（MoCo 风格循环缓冲区）中动态选取 top-k 最相似样本，再选最难的 1 个作为重定向目标，增强判别能力
3. **排斥控制**：通过 `max_target_sim` 限制重定向目标的最大允许相似度，防止编码图像被重定向到与原图过于相似的样本，保证隐私偏移的有效性
4. **渐进逃逸边界**：逃逸 margin 从 0.05 线性增长到 0.30，使隐私约束由松到紧，避免训练崩溃

### 噪声鲁棒性

Noiser 模块在训练时随机选择一种噪声层施加于编码图像，模拟实际信道中的各种干扰，确保解码器在噪声环境下仍能准确提取消息。支持组合噪声配置，如 `cropout+jpeg`。

### Softplus Loss vs Hinge Loss

隐私损失中的逃逸项和重定向项均使用 `log1p(exp(x))`（Softplus）而非 `max(0, x)`（Hinge），优势在于：
- 处处可导，梯度更平滑
- 对已满足约束的样本仍有微小梯度，避免过早停止优化
- 数值稳定性更好

## 详细技术文档

完整的技术实现细节、算法原理、数据流图和参数说明请参阅 [TECHNICAL_DOC.md](TECHNICAL_DOC.md)。
