# PrivSteg: 隐私保护图像隐写系统

基于 GAN 的隐私保护图像隐写框架，在载体图像中嵌入秘密消息的同时，对抗基于内容的图像检索（CBIR）系统，防止隐写图像被检索关联到原始载体图像。

## 核心特性

- **消息嵌入与提取**：在图像中嵌入二进制消息（默认 30 bit），解码端可准确提取
- **视觉不可感知性**：编码图像与原图在像素级、结构级（SSIM）、感知级（VGG19 感知损失）上保持高度一致
- **隐写分析对抗**：通过 GAN 对抗训练使隐写图像能逃逸隐写分析器的检测
- **检索逃逸隐私保护**：利用 CLIP + DINOv2 多特征空间使隐写图像在语义特征层面与原图产生偏移，防止被 CBIR 系统检索关联
- **检索攻击模拟**：内置带记忆库的检索攻击模型（RetrievalAttackerWithMemory），通过硬负样本挖掘增强隐私鲁棒性
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
│   ├── encoder.py             # 编码器（消息融合 + 残差学习）
│   ├── decoder.py             # 解码器（CNN + 全连接消息提取）
│   ├── steganalyzer.py        # 隐写分析器（判别器，SpectralNorm）
│   ├── content_recognizer.py  # 内容识别器（冻结 CLIP ViT-B/32）
│   └── retrieval_attacker.py  # 检索攻击模型（带记忆库 + 硬负样本挖掘）
├── losses/
│   ├── ssim_loss.py           # SSIM 损失
│   ├── perceptual_loss.py     # VGG19 感知损失
│   ├── privacy_loss.py        # 基础隐私损失（CLIP 余弦相似度偏移）
│   └── advanced_privacy_loss.py  # 高级隐私损失（CLIP + DINOv2 + Triplet + 攻击损失）
├── noise_layers/
│   ├── noiser.py              # 噪声层组合器
│   ├── identity.py            # 恒等层（无噪声）
│   ├── crop.py                # 裁剪噪声
│   ├── cropout.py             # 中心裁剪噪声
│   ├── dropout.py             # 随机丢弃噪声
│   ├── resize.py              # 缩放噪声
│   ├── jpeg_compression.py    # JPEG 压缩噪声
│   ├── quantization.py        # 量化噪声
│   └── dct_filters.py         # DCT 滤波噪声
└── data/
    └── dataset.py             # 隐写数据集（图像加载 + 随机消息生成）
```

## 网络架构

### 编码器（Encoder）

- 8 层 ConvBNRelu 特征提取 → 消息空间扩展拼接 → 1×1 卷积输出
- 输入：载体图像 (B, 3, H, W) + 消息 (B, L)
- 输出：编码图像 (B, 3, H, W) + 残差惩罚项

### 解码器（Decoder）

- 8 层 ConvBNRelu + AdaptiveAvgPool + 全连接层
- 输入：编码图像（可含噪声）
- 输出：消息 logits (B, L)

### 隐写分析器（Steganalyzer）

- 8 层卷积（SpectralNorm）+ 全连接判别头
- 输出：图像是否为隐写图像的 logits

### 内容识别器（ContentRecognizer）

- 冻结的 CLIP ViT-B/32 图像编码器
- 提取 L2 归一化的 512 维图像特征

### 检索攻击模型（RetrievalAttackerWithMemory）

- 4 层卷积 + 注意力机制 + 记忆库
- 记忆库存储历史特征，支持硬负样本挖掘
- 模拟 CBIR 检索攻击，增强隐私训练的对抗性

## 损失函数

| 损失            | 说明                                          |
| ------------- | ------------------------------------------- |
| L\_pixel      | 像素级 MSE 损失                                  |
| L\_ssim       | 结构相似性损失                                     |
| L\_perceptual | VGG19 感知损失（L1）                              |
| L\_message    | 消息提取 BCEWithLogits 损失                       |
| L\_adv        | GAN 对抗损失                                    |
| L\_privacy    | 隐私保护损失（CLIP + DINOv2 特征偏移 + Triplet + 攻击损失） |
| L\_residual   | 残差惩罚项                                       |

## 三阶段训练

| 阶段      | Epoch | 目标               | 关键权重                                     |
| ------- | ----- | ---------------- | ---------------------------------------- |
| Phase 1 | 200   | 基础训练：视觉质量 + 消息提取 | visual=10, message=10, adv=0, privacy=0  |
| Phase 2 | 100   | 对抗训练：加入隐写分析器对抗   | visual=10, message=10, adv=5, privacy=0  |
| Phase 3 | 100   | 隐私训练：加入隐私保护损失    | visual=10, message=10, adv=3, privacy=10 |

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

> 注意：CLIP 需要从 GitHub 安装，`requirements.txt` 中已包含 `git+https://github.com/openai/CLIP.git`。DINOv2 模型权重需单独下载并放置到 `./models/` 目录。

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
    --epochs-phase1 200 \
    --epochs-phase2 100 \
    --epochs-phase3 100 \
    --lr-encoder 8e-4 \
    --lr-decoder 8e-4 \
    --lr-steganalyzer 5e-4 \
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
| `use_advanced_privacy`   | True              | 是否启用高级隐私损失（CLIP + DINOv2 + Triplet） |
| `privacy_mode`           | retrieval\_escape | 隐私模式                                |
| `clip_weight`            | 1.0               | CLIP 特征偏移损失权重                       |
| `dino_weight`            | 0.5               | DINOv2 特征偏移损失权重                     |
| `triplet_weight`         | 0.15              | Triplet 损失权重                        |
| `use_retrieval_attacker` | True              | 是否启用检索攻击模型                          |
| `attacker_feature_dim`   | 512               | 攻击模型特征维度                            |
| `attacker_memory_size`   | 1000              | 记忆库大小                               |
| `privacy_margin`         | 0.7               | 隐私损失间隔                              |
| `noise_config`           | identity          | 噪声层配置                               |

## 输出说明

训练过程会在 `experiments/<experiment_name>/` 目录下生成：

```
experiments/my_experiment/
├── config.json              # 训练配置
├── train.log                # 训练日志
├── checkpoints/
│   ├── checkpoint.pth       # 最新检查点
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

### 隐私保护机制

传统隐写方法仅关注视觉不可感知性和消息鲁棒性，但忽略了基于内容的图像检索（CBIR）威胁——攻击者可通过特征相似度检索将隐写图像关联到原始载体。PrivSteg 通过以下机制解决此问题：

1. **多特征空间偏移**：同时在 CLIP 和 DINOv2 特征空间中推动隐写图像特征远离原图特征，使余弦相似度低于安全阈值
2. **Triplet 损失引导**：利用硬负样本（记忆库中最相似的无关图像特征）引导隐写特征向无关方向偏移，避免特征坍缩
3. **对抗检索攻击**：训练检索攻击模型模拟 CBIR 攻击者，通过 InfoNCE 损失增强攻击者的检索能力，从而在对抗中提升隐私保护强度

### 噪声鲁棒性

Noiser 模块在训练时随机选择一种噪声层施加于编码图像，模拟实际信道中的各种干扰，确保解码器在噪声环境下仍能准确提取消息。支持组合噪声配置，如 `cropout+jpeg`。
