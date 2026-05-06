# PrivSteg 技术文档 —— 基于深度学习的隐私保护图像隐写系统

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统架构总览](#2-系统架构总览)
- [3. 项目文件结构](#3-项目文件结构)
- [4. 核心配置模块 (config.py)](#4-核心配置模块-configpy)
- [5. 数据处理模块 (data/)](#5-数据处理模块-data)
- [6. 模型架构模块 (models/)](#6-模型架构模块-models)
  - [6.1 编码器 (Encoder)](#61-编码器-encoder)
  - [6.2 解码器 (Decoder)](#62-解码器-decoder)
  - [6.3 隐写分析器 (Steganalyzer)](#63-隐写分析器-steganalyzer)
  - [6.4 内容识别器 (ContentRecognizer)](#64-内容识别器-contentrecognizer)
  - [6.5 检索攻击模型 (RetrievalAttacker)](#65-检索攻击模型-retrievalattacker)
- [7. 损失函数模块 (losses/)](#7-损失函数模块-losses)
  - [7.1 SSIM损失 (SSIMLoss)](#71-ssim损失-ssimloss)
  - [7.2 感知损失 (PerceptualLoss)](#72-感知损失-perceptualloss)
  - [7.3 基础隐私损失 (PrivacyLoss)](#73-基础隐私损失-privacyloss)
  - [7.4 特征重定向损失 (FeatureRedirectionLoss)](#74-特征重定向损失-featureredirectionloss)
  - [7.5 重定向三元组损失 (RedirectionTripletLoss)](#75-重定向三元组损失-redirectiontripletloss)
  - [7.6 检索逃逸损失 (RetrievalEscapeLoss)](#76-检索逃逸损失-retrievalescapeloss)
- [8. 噪声层模块 (noise_layers/)](#8-噪声层模块-noise_layers)
  - [8.1 噪声组合器 (Noiser)](#81-噪声组合器-noiser)
  - [8.2 Identity恒等映射](#82-identity恒等映射)
  - [8.3 随机裁剪 (Crop)](#83-随机裁剪-crop)
  - [8.4 裁剪填充 (Cropout)](#84-裁剪填充-cropout)
  - [8.5 像素丢弃 (Dropout)](#85-像素丢弃-dropout)
  - [8.6 缩放变换 (Resize)](#86-缩放变换-resize)
  - [8.7 JPEG压缩 (JpegCompression)](#87-jpeg压缩-jpegcompression)
  - [8.8 量化噪声 (Quantization)](#88-量化噪声-quantization)
  - [8.9 DCT滤波器生成器 (DctFilterGenerator)](#89-dct滤波器生成器-dctfiltergenerator)
- [9. 训练系统 (trainer.py & train.py)](#9-训练系统-trainerpy--trainpy)
  - [9.1 Trainer类详解](#91-trainer类详解)
  - [9.2 两阶段训练策略](#92-两阶段训练策略)
  - [9.3 训练流程 (train.py)](#93-训练流程-trainpy)
  - [9.4 记忆库预热与检索攻击者训练](#94-记忆库预热与检索攻击者训练)
- [10. 验证脚本 (validation_script.py)](#10-验证脚本-validation_scriptpy)
- [11. CLIP视觉模型](#11-clip视觉模型)
- [12. 核心算法原理深度剖析](#12-核心算法原理深度剖析)
  - [12.1 隐写编码原理](#121-隐写编码原理)
  - [12.2 对抗训练原理](#122-对抗训练原理)
  - [12.3 隐私保护的多模态特征逃逸机制](#123-隐私保护的多模态特征逃逸机制)
  - [12.4 记忆库驱动的硬负样本挖掘](#124-记忆库驱动的硬负样本挖掘)
- [13. 完整数据流图](#13-完整数据流图)
- [14. 使用指南](#14-使用指南)

---

## 1. 项目概述

**PrivSteg（Privacy-Preserving Steganography）** 是一个基于深度学习的**隐私保护图像隐写系统**。该项目的核心目标是在载体图像（Cover Image）中嵌入秘密消息（Secret Message），同时满足以下四个关键约束：

| 约束维度 | 技术手段 | 目标 |
|---------|---------|------|
| **不可感知性** | 像素级重建 + SSIM + VGG感知损失 | 编码后图像与原图视觉上无法区分 |
| **鲁棒性** | 多种噪声层模拟真实信道干扰 | 经过噪声攻击后仍能准确提取消息 |
| **抗检测性** | GAN对抗训练 + SpectralNorm隐写分析器 | 隐写图像能骗过隐写检测器 |
| **隐私保护** | CLIP/DINO特征逃逸 + 检索攻击防御 | 编码前后语义特征空间距离足够大 |

### 核心创新点

1. **两阶段渐进式训练**：先学习基础隐写能力 + 渐进对抗 → 再引入隐私保护
2. **多模态特征逃逸**：同时利用CLIP（ViT-B/32）和DINOv2（ViT-S/14）两种预训练模型的语义特征空间进行隐私约束
3. **CLIP/DINO 有向重定向**：不仅将编码图像推离原图，还通过记忆库硬负样本将其"拉向"其他无关图像，实现语义伪装
4. **检索攻击者建模**：引入带记忆库的检索攻击模型，通过对抗方式迫使编码器在检索特征空间中也实现"逃逸"
5. **硬负样本挖掘**：从记忆库中动态选取最难区分的负样本，增强三元组损失与有向逃逸损失的判别能力
6. **统一特征空间的三元组损失**：三元组损失在攻击者特征空间内统一计算（escape + redirect），避免跨特征空间梯度噪声

---

## 2. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PrivSteg 系统架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────┐           │
│   │ Cover    │────▶│   Encoder    │────▶│ Encoded      │           │
│   │ Image    │     │ (UNet-style) │     │ Image        │           │
│   └──────────┘     └──────┬───────┘     └──────┬───────┘           │
│                           ↑ message              │                   │
│                    (30-bit secret)               ▼                   │
│                                            ┌──────────┐            │
│                                            │  Noiser  │            │
│                                            │(噪声层)   │            │
│                                            └────┬─────┘            │
│                                                 │                   │
│                                    ┌────────────┼────────────┐      │
│                                    ▼            ▼            ▼      │
│                              ┌──────────┐ ┌────────┐ ┌─────────┐   │
│                              │ Decoder  │ │ Steg-   │ │Content  │   │
│                              │ (消息恢复)│ │ Analyzer│ │Recogniz.│   │
│                              └────┬─────┘ └────┬───┘ └────┬────┘   │
│                                   │             │          │        │
│                                   ▼             ▼          ▼        │
│                              Message_pred   Real/Fake   Feature    │
│                              (30 bits)     判定结果     Vectors     │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │                Privacy Protection Layer                   │     │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │     │
│   │  │ CLIP ViT-B/32│  │ DINOv2 ViT-S│  │ Retrieval       │  │     │
│   │  │ (冻结)        │  │ /14 (冻结)   │  │ Attacker+Memory │  │     │
│   │  └──────┬───────┘  └──────┬──────┘  └────────┬────────┘  │     │
│   │         │                 │                  │            │     │
│   │         └────────┬────────┘                  │            │     │
│   │                  ▼                            │            │     │
│   │         RetrievalEscapeLoss ◀─────────────────┘            │     │
│   │    (多模态特征逃逸 + 三元组 + 攻击者逃逸 + 有向重定向)        │     │
│   └──────────────────────────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 项目文件结构

```
PrivSteg/
├── config.py                     # 全局配置类，定义所有超参数和阶段权重
├── train.py                      # 主入口：命令行解析、实验管理、训练循环调度
├── trainer.py                    # Trainer核心类：模型初始化、前向传播、损失计算、优化
├── noise_argparser.py            # 噪声层命令行参数解析器
├── validation_script.py          # 验证脚本：编码/解码/检测/隐私评估
├── requirements.txt              # 依赖列表
│
├── data/
│   ├── __init__.py
│   └── dataset.py                # SteganographyDataset 数据集定义
│
├── models/
│   ├── __init__.py               # 模型统一导出
│   ├── encoder.py                # 编码器：UNet风格的消息嵌入网络
│   ├── decoder.py                # 解码器：CNN消息提取网络
│   ├── steganalyzer.py           # 隐写分析器：CNN+SpectralNorm二分类器
│   ├── content_recognizer.py     # 内容识别器：CLIP ViT-B/32 特征提取（冻结）
│   └── retrieval_attacker.py     # 检索攻击模型：CNN+注意力+记忆库
│
├── losses/
│   ├── __init__.py
│   ├── ssim_loss.py              # 结构相似性(SSIM)损失
│   ├── perceptual_loss.py        # VGG19感知损失
│   ├── privacy_loss.py           # 基础Softplus隐私损失
│   └── advanced_privacy_loss.py  # 高级隐私损失（特征重定向+三元组+检索逃逸）
│
├── noise_layers/
│   ├── noiser.py                 # 噪声层组合器（随机选择激活）
│   ├── identity.py               # 恒等映射（无操作）
│   ├── crop.py                   # 随机裁剪
│   ├── cropout.py                # 裁剪替换（局部区域用原图替换）
│   ├── dropout.py                # 随机像素丢弃
│   ├── resize.py                 # 随机缩放
│   ├── jpeg_compression.py       # JPEG压缩（DCT域量化）
│   ├── quantization.py           # 傅里叶四舍五入量化
│   └── dct_filters.py            # DCT/IDCT滤波器生成工具
│
└── output/                       # 验证输出目录（示例图像）
```

---

## 4. 核心配置模块 (`config.py`)

### 4.1 配置类定义

```python
class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

`Config` 类采用**类属性**方式集中管理所有超参数，避免全局变量污染。所有参数在类定义时直接赋值默认值。

### 4.2 图像与消息参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `image_size` | `256` | 输入图像分辨率（正方形，256×256） |
| `message_length` | `30` | 秘密消息比特长度（每个bit为0或1的浮点数） |
| `batch_size` | `8` | 训练批次大小 |

**设计考量**：
- `image_size=256`：平衡计算资源消耗与特征表达能力。UNet编码器的消息扩展操作需要将1×30的消息张量扩展到256×256的空间维度。
- `message_length=30`：约3.75字节的隐藏容量。对于图像隐写来说属于较低容量，但有利于保持高不可感知性和强隐私保护。

### 4.3 学习率配置

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `lr_encoder` | `1e-3` | 编码器学习率 |
| `lr_decoder` | `1e-3` | 解码器学习率 |
| `lr_steganalyzer` | `1e-3` | 隐写分析器学习率 |

**设计分析**：
- 编码器和解码器使用相同的学习率（`1e-3`），因为它们共享同一个Adam优化器（`optimizer_encoder_decoder`），作为生成器端协同训练。
- 隐写分析器使用相同的学习率（`1e-3`），因为其训练目标是独立的二分类任务。
- 检索攻击者使用编码器学习率的一半（`lr_encoder * 0.5 = 5e-4`），在Trainer中单独设置。

### 4.4 两阶段训练轮次

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `num_epochs_phase1` | `300` | Phase 1: 基础隐写 + 渐进对抗训练 |
| `num_epochs_phase2` | `100` | Phase 2: 隐私保护训练 |

总计 **400 epochs** 的训练过程。

### 4.5 视觉损失权重

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `lambda_pixel` | `1.0` | MSE像素损失权重 |
| `lambda_ssim` | `2.0` | SSIM结构相似性损失权重 |
| `lambda_perceptual` | `0.3` | VGG19感知损失权重 |
| `use_perceptual_loss` | `True` | 是否启用VGG感知损失 |
| `lambda_message` | `1.0` | 消息提取BCE损失权重 |
| `lambda_adv` | `0.1` | 对抗损失基础权重（Phase 1中渐进放大） |
| `lambda_privacy` | `0.3` | 隐私损失基础权重（Phase 2中渐进放大） |

**视觉损失组成公式**：

$$L_{visual} = \lambda_{pixel} \cdot L_{pixel} + \lambda_{ssim} \cdot L_{ssim} + \lambda_{perceptual} \cdot L_{perceptual}$$

代入默认值：

$$L_{visual} = 1.0 \cdot L_{MSE} + 2.0 \cdot L_{SSIM} + 0.3 \cdot L_{VGG}$$

三种视觉损失的权重设计体现了**多尺度图像质量评估**的思想，且SSIM权重较高以优先保证结构相似性：
- **MSE**：像素级绝对误差
- **SSIM**：结构级相似性（亮度、对比度、结构）
- **Perceptual**：高层语义级相似性（VGG特征空间）

### 4.6 隐私保护参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `privacy_margin` | `0.5` | 隐私余量边界 |
| `use_advanced_privacy` | `True` | 是否启用高级隐私保护 |
| `privacy_mode` | `'retrieval_escape'` | 隐私模式（检索逃逸） |
| `clip_weight` | `1.0` | CLIP特征逃逸损失权重 |
| `dino_weight` | `0.5` | DINOv2特征逃逸损失权重（Config默认值，Trainer中实际使用0.8） |
| `dino_model_path` | `'./models/dinov2_vits14_pretrain.pth'` | DINOv2模型权重路径 |
| `triplet_weight` | `0.15` | 三元组损失权重（Config默认值，Trainer中实际传入0.5） |
| `triplet_margin` | `0.5` | 三元组margin（Config默认值，Trainer中实际传入0.3） |

### 4.7 CLIP/DINO 有向重定向参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `use_clip_redirect` | `True` | 是否启用CLIP/DINO有向重定向 |
| `clip_memory_size` | `1000` | CLIP特征记忆库容量 |
| `clip_hard_negative_k` | `5` | CLIP硬负样本Top-K数量 |
| `clip_redirect_margin` | `0.6` | 重定向margin（要求与目标相似度 > 0.6） |
| `clip_max_target_sim` | `0.75` | 最大允许相似度上限（防止特征坍塌） |
| `clip_redirect_weight` | `0.5` | CLIP重定向损失权重 |
| `dino_redirect_weight` | `0.3` | DINO重定向损失权重 |

### 4.8 检索攻击者参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `use_retrieval_attacker` | `True` | 是否启用检索攻击者 |
| `attacker_feature_dim` | `512` | 攻击者输出特征维度 |
| `attacker_memory_size` | `1000` | 攻击者记忆库容量 |
| `attacker_temperature` | `0.05` | 相似度温度系数（用于InfoNCE/对比损失的温度缩放） |
| `attacker_weight` | `0.5` | 攻击者逃逸损失权重 |
| `use_hard_negatives` | `True` | 是否使用硬负样本 |
| `hard_negative_k` | `5` | 硬负样本数量 |
| `memory_warmup_batches` | `50` | 记忆库预热批次数 |

### 4.9 其他参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `data_dir` | `None` | 数据集根目录（命令行指定） |
| `train_dir` | `'train'` | 训练集子目录名 |
| `val_dir` | `'val'` | 验证集子目录名 |
| `checkpoint_dir` | `'checkpoints'` | 检查点保存目录 |
| `log_dir` | `'logs'` | 日志目录 |
| `num_workers` | `4` | 数据加载工作线程数 |
| `save_interval` | `10` | 检查点保存间隔（epoch数） |
| `noise_config` | `'identity'` | 噪声层配置字符串 |
| `discriminator_loss_threshold` | `0.1` | 判别器损失阈值，低于此值暂停判别器训练 |

### 4.10 阶段权重函数 `get_phase_weights(phase, epoch)`

这是整个系统的**核心调度逻辑**，决定了不同训练阶段的损失权重分配。与早期版本不同，当前实现采用**两阶段**训练，且部分权重在阶段内**渐进变化**。

#### Phase 1 — 基础隐写 + 渐进对抗训练（300 epochs）

```python
{
    'visual': 10.0,        # 高权重：优先学习图像重建
    'message': 10.0,       # 高权重：优先学习消息编解码
    'adv': 渐进,            # lambda_adv -> 2*lambda_adv -> 3*lambda_adv
    'privacy': 0.0,        # 关闭：不进行隐私保护
    'noised': 1.0,         # 含噪声图像的视觉损失权重
    'residual_penalty': 0.5 # 中等：限制修改幅度
}
```

**adv 渐进策略**：将Phase 1的300个epoch均分为3段：
- 第1-100 epoch：`adv = lambda_adv` (默认0.1)
- 第101-200 epoch：`adv = lambda_adv * 2` (默认0.2)
- 第201-300 epoch：`adv = lambda_adv * 3` (默认0.3)

**目标**：让编码器先学会将消息嵌入图像且保持视觉质量，解码器学会准确提取消息；在阶段后半段逐步引入轻量级对抗约束，为Phase 2做准备。

#### Phase 2 — 隐私保护训练（100 epochs）

```python
{
    'visual': 20.0,         # 进一步提高视觉质量权重
    'message': 10.0,        # 保持消息权重
    'adv': 5.0,             # 开启并固定对抗权重
    'privacy': 渐进,         # 1.0 -> 3.0 -> 10.0
    'noised': 0.5,          # 降低含噪声损失权重
    'residual_penalty': 0.3  # 降低残差惩罚
}
```

**privacy 渐进策略**：将Phase 2的100个epoch均分为3段：
- 第1-33 epoch：`privacy = 1.0`
- 第34-66 epoch：`privacy = 3.0`
- 第67-100 epoch：`privacy = 10.0`

**目标**：在保证隐写能力和对抗性的基础上，由浅入深地引入隐私保护约束，使编码图像在语义特征空间中逐渐远离原始图像，避免隐私损失突然激增导致训练崩溃。

---

## 5. 数据处理模块 (`data/`)

### 5.1 SteganographyDataset 类

**文件位置**: [dataset.py](data/dataset.py)

```python
class SteganographyDataset(Dataset):
    def __init__(self, root_dir, image_size=256, message_length=30, transform=None):
```

#### 5.1.1 数据集初始化流程

1. **图像路径收集**：递归遍历 `root_dir` 目录，筛选支持格式（`.png`, `.jpg`, `.jpeg`, `.bmp`）
2. **图像预处理Pipeline**：
   ```
   Resize(256×256) → ToTensor() → Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
   ```

   归一化到 `[-1, 1]` 区间，这是 `tanh` 激活函数输出的匹配范围。

3. **消息生成**：每个样本随机生成 `message_length`（30）维的二进制向量，每维为0或1的浮点数。

#### 5.1.2 `__getitem__` 返回格式

```python
return {
    'cover_image': image,      # shape: [3, 256, 256], range: [-1, 1]
    'message': message,        # shape: [30], values: {0.0, 1.0}
    'image_path': image_path   # str: 原始图像路径
}
```

**设计要点**：
- 消息是**每次访问时随机生成**的，而非预先固定。这意味着同一张图像在不同epoch会携带不同的秘密消息，增强了泛化能力。
- 使用 `torch.randint(0, 2, ...)` 保证消息的均匀分布。

---

## 6. 模型架构模块 (`models/`)

### 6.1 编码器 (Encoder)

**文件位置**: [encoder.py](models/encoder.py)

#### 6.1.1 架构设计

编码器采用**类UNet的全卷积架构**，核心思想是将秘密消息作为额外通道拼接到图像特征中：

```
Cover Image [B, 3, 256, 256]
       │
       ▼
┌──────────────────────┐
│ ConvBNRelu(3→64)     │  ← 第1层：初步特征提取
├──────────────────────┤
│ ConvBNRelu(64→64) ×7 │  ← 第2-8层：深层特征学习
├──────────────────────┤
│ Concat: [msg; feat; cover] │  ← 特征拼接
│ ConvBNRelu(97→64)    │  ← 融合层：(30+64+3)=97通道 → 64通道
├──────────────────────┤
│ Conv2d(64→3, k=1)    │  ← 1×1卷积恢复3通道
├──────────────────────┤
│ Tanh                 │  ← 输出范围[-1, 1]
└──────────────────────┘
       │
       ▼
Encoded Image [B, 3, 256, 256]
```

#### 6.1.2 详细组件解析

**ConvBNRelu 基础块**：

```python
class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
```

- **3×3卷积 + padding=1**：保持空间尺寸不变（same convolution）
- **bias=False + BatchNorm**：BatchNorm已包含偏置项，去掉冗余bias减少参数
- **ReLU(inplace=True)**：原地操作节省显存

**消息扩展机制**：

```python
expanded_message = message.unsqueeze(-1).unsqueeze(-1)  # [B, 30] → [B, 30, 1, 1]
expanded_message = expanded_message.expand(-1, -1, self.H, self.W)  # → [B, 30, 256, 256]
```

将一维消息向量（30维）**空间广播**为与图像相同分辨率的特征图。这种设计的含义是：消息的每一位信息被均匀地"涂抹"在整个图像空间上，类似于数字水印中的扩频技术。

**特征融合拼接**：

```python
concat = torch.cat([expanded_message, encoded_image, cover_image], dim=1)
# 拼接后通道数: 30(message) + 64(features) + 3(cover) = 97
im_w = self.after_concat_layer(concat)  # 97 → 64
```

拼接三种信息：
1. **expanded_message**：原始消息信号
2. **encoded_image**：经过8层卷积处理的图像特征
3. **cover_image**：原始载体图像（残差连接，保留低频信息）

**最终输出**：

```python
encoded_image = self.final_layer(im_w)        # 64 → 3 (1×1卷积)
encoded_image = torch.tanh(encoded_image)      # 映射到 [-1, 1]
residual_penalty = torch.mean((encoded_image - cover_image) ** 2)  # 残差惩罚
```

- **tanh激活**：将输出严格限制在 `[-1, 1]` 范围内，与输入图像归一化范围一致
- **residual_penalty**：编码图像与原始图像之间的MSE，作为辅助正则化项返回

#### 6.1.3 参数量估算

| 层 | 输入通道 | 输出通道 | 卷积核 | 参数量 |
|---|---------|---------|--------|--------|
| conv_layers[0] | 3 | 64 | 3×3 | 3×64×9 = 1,728 |
| conv_layers[1-7] | 64 | 64 | 3×3 | 7×64×64×9 = 258,048 |
| after_concat_layer | 97 | 64 | 3×3 | 97×64×9 = 55,872 |
| final_layer | 64 | 3 | 1×1 | 64×3×1 = 192 |
| **总计** | | | | **~315,840** |

---

### 6.2 解码器 (Decoder)

**文件位置**: [decoder.py](models/decoder.py)

#### 6.2.1 架构设计

解码器是一个**纯CNN特征提取 + 全连接分类**网络：

```
Encoded Image [B, 3, 256, 256]
       │
       ▼
┌──────────────────────┐
│ ConvBNRelu(3→64)     │  ← 第1层
├──────────────────────┤
│ ConvBNRelu(64→64) ×7 │  ← 第2-8层：深层特征
├──────────────────────┤
│ ConvBNRelu(64→30)    │  ← 特征降维到消息长度
├──────────────────────┤
│ AdaptiveAvgPool2d(1×1)│  ← 全局平均池化
├──────────────────────┤
│ Linear(30→30)        │  ← 全连接层（logits输出）
└──────────────────────┘
       │
       ▼
Message Prediction [B, 30]  (raw logits, 未经过sigmoid)
```

#### 6.2.2 关键设计细节

**全局平均池化（GAP）**：

```python
layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
```

将 `[B, 30, H, W]` 压缩为 `[B, 30, 1, 1]`，然后squeeze得到 `[B, 30]`。GAP的优势：
- 大幅减少全连接层参数量
- 具有平移不变性
- 对输入尺寸有一定适应性

**线性输出层**：

```python
self.linear = nn.Linear(message_length, message_length)
```

输出**原始logits**（非概率），配合 `BCEWithLogitsLoss` 使用，数值稳定性更好。

#### 6.2.3 与编码器的对称性

编码器和解码器都使用**8个ConvBNRelu块** + 64通道，形成对称结构。但解码器不需要消息拼接机制，因为它只需要从图像中"读取"信息，而不需要"写入"。

#### 6.2.4 参数量估算

| 层 | 输入通道 | 输出通道 | 卷积核 | 参数量 |
|---|---------|---------|--------|--------|
| conv_layers[0] | 3 | 64 | 3×3 | 1,728 |
| conv_layers[1-7] | 64 | 64 | 3×3 | 258,048 |
| 最后一层 ConvBNRelu | 64 | 30 | 3×3 | 17,280 |
| AdaptiveAvgPool2d | - | - | - | 0 |
| Linear | 30 | 30 | - | 930 |
| **总计** | | | | **~277,986** |

---

### 6.3 隐写分析器 (Steganalyzer)

**文件位置**: [steganalyzer.py](models/steganalyzer.py)

#### 6.3.1 架构设计

隐写分析器是一个**二分类CNN**，用于判断输入图像是否为隐写图像：

```
Input Image [B, 3, 256, 256]
       │
       ▼
┌─────────────────────────────────────┐
│ SpectralNorm(Conv2d(3→64, 3×3))     │  + BN + LeakyReLU(0.2)
│ SpectralNorm(Conv2d(64→64, 3×3))    │  + BN + LeakyReLU(0.2) + MaxPool(2)  → [128×128]
│ SpectralNorm(Conv2d(64→128, 3×3))   │  + BN + LeakyReLU(0.2)
│ SpectralNorm(Conv2d(128→128, 3×3))  │  + BN + LeakyReLU(0.2) + MaxPool(2)  → [64×64]
│ SpectralNorm(Conv2d(128→256, 3×3))  │  + BN + LeakyReLU(0.2)
│ SpectralNorm(Conv2d(256→256, 3×3))  │  + BN + LeakyReLU(0.2) + MaxPool(2)  → [32×32]
│ SpectralNorm(Conv2d(256→512, 3×3))  │  + BN + LeakyReLU(0.2)
│ SpectralNorm(Conv2d(512→512, 3×3))  │  + BN + LeakyReLU(0.2)
│ AdaptiveAvgPool2d(1×1)              │                                  → [1×1]
└─────────────────────────────────────┘
       │
       ▼
Flatten → [B, 512]
       │
       ▼
┌──────────────────────┐
│ Linear(512→1)        │  → logit output (注意：当前实现为单层线性输出)
└──────────────────────┘
       │
       ▼
Output: [B, 1]  (正值为stego, 负值为cover)
```

> **注**：当前 `steganalyzer.py` 的实现将原双层FC（512→256→1）简化为单层（512→1），以精简参数量。`validation_script.py` 中保留了旧权重的键映射逻辑，确保兼容性。

#### 6.3.2 Spectral Normalization（谱归一化）

```python
if use_spectral_norm:
    conv = lambda c_in, c_out, **kwargs: nn.utils.spectral_norm(
        nn.Conv2d(c_in, c_out, **kwargs))
```

**SpectralNorm的作用**：对卷积层的权重矩阵 $W$ 施加谱范数约束（即最大奇异值 $\sigma(W) \le 1$），从而满足Lipschitz连续性条件。这对GAN训练至关重要：

$$\|Wx\|_2 \le \|W\|_2 \cdot \|x\|_2 \le 1 \cdot \|x\|_2$$

**为什么隐写分析器需要SpectralNorm？**
- 隐写分析本质上是一个**敏感的二分类任务**（区分极其微小的像素差异）
- 没有SpectralNorm约束时，判别器可能通过放大梯度来"作弊"，导致训练不稳定
- SN限制了梯度的传播幅度，使对抗训练更加稳定

#### 6.3.3 激活函数选择：LeakyReLU

```python
nn.LeakyReLU(0.2, inplace=True)
```

使用LeakyReLU（负斜率0.2）而非标准ReLU的原因：
- 隐写分析需要捕获**微小异常信号**，这些信号可能分布在负值区域
- LeakyReLU允许负方向的小梯度流动，避免"死亡神经元"

#### 6.3.4 渐进式下采样

空间维度变化：`256 → 128 → 64 → 32 → 1×1`

每两个卷积块后接一个 `MaxPool2d(2)`，共3次下采样。这种渐进式设计使得网络能够：
- 浅层捕获局部纹理细节（对隐写痕迹敏感）
- 深层捕获全局统计特性

---

### 6.4 内容识别器 (ContentRecognizer)

**文件位置**: [content_recognizer.py](models/content_recognizer.py)

#### 6.4.1 架构概述

内容识别器是对 **OpenAI CLIP ViT-B/32** 模型的封装，用于提取图像的**高层语义特征向量**：

```python
class ContentRecognizer(nn.Module):
    def __init__(self, device='cuda', model_name='ViT-B/32'):
        self.clip_model, _ = clip.load(model_name, device=self._device)
        # 冻结所有CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
```

#### 6.4.2 前向传播流程

```python
def forward(self, image):
    # Step 1: 双三次插值调整到224×224（CLIP标准输入尺寸）
    image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    
    # Step 2: 从[-1,1]反归一化到[0,1]
    image_resized = (image_resized + 1) / 2
    
    # Step 3: CLIP标准化（ImageNet均值/标准差）
    image_resized = (image_resized - self.clip_mean) / self.clip_std
    # mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.275]
    
    # Step 4: CLIP图像编码器提取特征
    image_features = self.clip_model.encode_image(image_resized)
    
    # Step 5: L2归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features  # shape: [B, 512]
```

**输出特征维度**：ViT-B/32 的视觉编码器输出 **512维** 特征向量。

#### 6.4.3 为什么使用CLIP？

CLIP（Contrastive Language-Image Pre-training）通过4亿图文对的大规模对比学习训练，具有以下优势：
- **语义丰富性**：512维特征空间编码了丰富的语义信息（物体、场景、属性等）
- **通用性**：零样本迁移能力强，无需针对特定数据集微调
- **冻结使用**：无需训练，仅作特征提取器，不增加可训练参数

#### 6.4.4 设备迁移处理

```python
def to(self, *args, **kwargs):
    module = super().to(*args, **kwargs)
    self.clip_model = self.clip_model.to(self._device)
    return module
```

由于CLIP模型是外部实例化的，需要重写 `to()` 方法确保设备同步。

---

### 6.5 检索攻击模型 (RetrievalAttacker)

**文件位置**: [retrieval_attacker.py](models/retrieval_attacker.py)

该模块实现了三个层次的检索攻击模型，构成**对抗隐私保护的攻击方**。

#### 6.5.1 基础检索攻击器 RetrievalAttacker

```
Input Image [B, 3, 256, 256]
       │
       ▼
┌───────────────────────────────────────┐
│ Conv2d(3→64, k=7, s=2, p=3) + BN + ReLU │  → [B,64,128,128]
│ MaxPool2d(k=3, s=2, p=1)                │  → [B,64,64,64]
│ Conv2d(64→128, k=3, s=2, p=1) + BN + ReLU│  → [B,128,32,32]
│ Conv2d(128→256, k=3, s=2, p=1) + BN + ReLU│ → [B,256,16,16]
│ Conv2d(256→512, k=3, s=2, p=1) + BN + ReLU│ → [B,512,8,8]
├───────────────────────────────────────┤
│ [可选] Attention Module:               │
│   Conv2d(512→256, 1×1) + ReLU         │
│   Conv2d(256→512, 1×1) + Sigmoid      │  → 通道注意力mask
│   feat = feat * mask                  │  ← 注意力加权
├───────────────────────────────────────┤
│ AdaptiveAvgPool2d(1)                   │  → [B,512,1,1]
│ Flatten                                │  → [B,512]
│ Linear(512→feature_dim)               │  → [B,512]
│ L2 Normalize                           │  → 单位球面上的特征
└───────────────────────────────────────┘
       │
       ▼
Feature Vector [B, 512]  (||feat||₂ = 1)
```

**通道注意力机制（Channel Attention）**：

```python
self.attention = nn.Sequential(
    nn.Conv2d(512, 256, kernel_size=1),   # 降维
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 512, kernel_size=1),   # 升维
    nn.Sigmoid()                          # 产生[0,1]的注意力权重
)
# 应用: feat = feat * attention(feat)
```

这是一个轻量级的 **SE-Net（Squeeze-and-Excitation）** 风格注意力模块，让网络自适应地关注对检索更有判别力的特征通道。默认启用（`use_attention=True`）。

**权重初始化策略**：

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
```

- 卷积层：Kaiming初始化（适配ReLU激活）
- BatchNorm：标准初始化（weight=1, bias=0）
- 全连接层：小方差正态分布（std=0.01），适合L2归一化后的特征学习

#### 6.5.2 带记忆库的检索攻击器 RetrievalAttackerWithMemory

这是实际使用的攻击模型，在基础攻击器之上增加了**动态记忆库**机制：

```python
class RetrievalAttackerWithMemory(nn.Module):
    def __init__(self, feature_dim=512, memory_size=1000, temperature=0.07):
        self.attacker = RetrievalAttacker(feature_dim=feature_dim)
        
        # 注册为buffer（不参与梯度计算，但随模型移动设备）
        self.register_buffer('memory_bank', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))  # 写入指针
        self.register_buffer('memory_count', torch.zeros(1, dtype=torch.long))  # 当前存储数量
```

**记忆库更新机制**：

```python
@torch.no_grad()
def update_memory(self, features):
    features = F.normalize(features, p=2, dim=1)
    batch_size = features.shape[0]
    ptr = int(self.memory_ptr)
    
    if ptr + batch_size > self.memory_size:
        # 循环缓冲区：超出末尾则从头开始写入
        remaining = self.memory_size - ptr
        self.memory_bank[ptr:] = features[:remaining]
        self.memory_bank[:batch_size - remaining] = features[remaining:]
        self.memory_ptr[0] = batch_size - remaining
    else:
        self.memory_bank[ptr:ptr + batch_size] = features
        self.memory_ptr[0] = (ptr + batch_size) % self.memory_size
    
    self.memory_count[0] = min(self.memory_count[0] + batch_size, self.memory_size)
```

这是一个**循环队列（Ring Buffer）** 实现：
- 固定容量1000个特征向量
- 写入指针循环移动，新特征覆盖最旧的特征
- `@torch.no_grad()` 装饰器确保不产生计算图开销

**硬负样本挖掘**：

```python
def get_hard_negatives(self, features, k=5):
    valid_count = int(self.memory_count[0])
    if valid_count == 0:
        return None
    
    valid_memory = self.memory_bank[:valid_count]
    
    # 计算查询特征与所有记忆特征的点积相似度
    similarities = torch.mm(features, valid_memory.t())
    
    # 取top-k最相似的（即"最难区分"的负样本）
    actual_k = min(k, valid_count)
    _, indices = similarities.topk(actual_k, dim=1)
    hard_negatives = valid_memory[indices]
    return hard_negatives  # shape: [B, k, feature_dim] 或 None
```

**核心思想**：在特征空间中，与查询特征**最相似**的记忆样本是最具挑战性的负样本（Hard Negatives）。使用这些样本来训练三元组损失，可以迫使模型学习更具判别力的特征表示。

**前向传播**：

```python
def forward(self, x, return_similarities=False):
    features = self.attacker(x)
    
    if return_similarities:
        similarities = torch.mm(features, self.memory_bank.t()) / self.temperature
        return features, similarities
    
    return features
```

可选返回与记忆库的相似度矩阵，用于InfoNCE损失计算。

**温度系数**：`temperature=0.07`（构造函数默认值），用于控制相似度分布的锐利程度。较小的温度使相似度分布更加尖锐（接近one-hot），增大正负样本间的区分度。Config中配置为 `0.05`。

#### 6.5.3 对抗检索模块 AdversarialRetrievalModule（备用）

```python
class AdversarialRetrievalModule(nn.Module):
    def __init__(self, feature_dim=512, num_attack_heads=3):
        # 3个独立的检索攻击头
        self.attack_heads = nn.ModuleList([
            RetrievalAttacker(feature_dim=feature_dim) for _ in range(num_attack_heads)
        ])
        # 融合层：3×512 → 1024 → 512
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * num_attack_heads, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim)
        )
```

这是**多头对抗检索**设计，目前未在主训练流程中使用，预留用于增强攻击强度。

---

## 7. 损失函数模块 (`losses/`)

### 7.1 SSIM损失 (SSIMLoss)

**文件位置**: [ssim_loss.py](losses/ssim_loss.py)

#### 7.1.1 数学原理

SSIM（Structural Similarity Index）从**亮度(luminance)、对比度(contrast)、结构(structure)**三个维度衡量两幅图像的相似性：

$$SSIM(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

其中：
- $\mu_x, \mu_y$：局部均值
- $\sigma_x^2, \sigma_y^2$：局部方差
- $\sigma_{xy}$：局部协方差
- $C_1 = 0.01^2$, $C_2 = 0.03^2$：稳定常数，防止分母为零

**SSIM取值范围**：`[-1, 1]`，值越大表示越相似。

**SSIM Loss**：$L_{SSIM} = 1 - SSIM(x,y)$

#### 7.1.2 实现细节

**高斯窗口生成**：

```python
def gaussian(self, window_size, sigma):
    gauss = torch.Tensor([
        torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()
```

生成一维高斯核（$\sigma=1.5$），然后外积得到二维窗口：

```python
_2D_window = _1D_window.mm(_1D_window.t())  # 外积得到2D高斯
```

**滑动窗口卷积计算局部统计量**：

```python
mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

mu1_sq = mu1.pow(2)
mu2_sq = mu2.pow(2)
mu1_mu2 = mu1 * mu2

sigma1_sq = F.conv2d(img1*img1, window, ...) - mu1_sq  # E[X²] - E[X]²
theta2_sq = F.conv2d(img2*img2, window, ...) - mu2_sq
sigma12 = F.conv2d(img1*img2, window, ...) - mu1_mu2    # E[XY] - E[X]E[Y]
```

使用 `groups=channel` 对每个通道独立应用高斯窗口，高效计算局部统计量。

**参数设置**：
- `window_size=11`：11×11的高斯窗口，覆盖足够的邻域上下文
- `size_average=True`：对所有空间位置取平均，返回标量损失

---

### 7.2 感知损失 (PerceptualLoss)

**文件位置**: [perceptual_loss.py](losses/perceptual_loss.py)

#### 7.2.1 原理

感知损失利用**预训练VGG19网络**在高层次特征空间的距离作为图像相似度度量。人类视觉系统对高频细节变化不敏感，但对语义内容变化非常敏感——这正是VGG中间层特征所捕获的信息。

#### 7.2.2 实现架构

```python
vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:18])
```

截取VGG19的前18层（到 `conv4_2` 层之前），输出特征图形状为 `[B, 256, H/8, W/8]`。

**VGG19层级对应关系**：

| 层索引 | 层名 | 输出通道 | 感受野 |
|-------|------|---------|--------|
| 0-3 | conv1 (×2) | 64 | 低级边缘/颜色 |
| 4-8 | conv2 (×2) | 128 | 纹理 |
| 9-15 | conv3 (×3) | 256 | 模式/形状 |
| 16-17 | conv4 (前2层) | 256 | 语义对象部分 |

截取到第17层（`conv4_2`之前），保留了足够的语义信息同时控制计算量。

#### 7.2.3 ImageNet归一化

```python
def normalize_to_imagenet(self, img):
    img = (img + 1) / 2  # [-1,1] → [0,1]
    return (img - self.imagenet_mean) / self.imagenet_std
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

输入图像先从 `[-1,1]` 反归一化到 `[0,1]`，再按ImageNet标准归一化，确保VGG特征提取的输入分布正确。

#### 7.2.4 特征归一化策略

```python
def normalize_features_fn(self, feat):
    mean = feat.mean(dim=[2, 3], keepdim=True)
    std = feat.std(dim=[2, 3], keepdim=True) + 1e-8
    normalized = (feat - mean) / std
    
    # 对前3个通道做额外的通道级归一化
    if normalized.size(1) >= 3:
        channel_parts = []
        for c in range(normalized.size(1)):
            channel_feat = normalized[:, c:c+1]
            if c < 3:
                channel_mean = channel_feat.mean()
                channel_std = channel_feat.std() + 1e-8
                channel_parts.append((channel_feat - channel_mean) / channel_std)
            else:
                channel_parts.append(channel_feat)
        normalized = torch.cat(channel_parts, dim=1)
    return normalized
```

两层归一化：
1. **空间归一化**：对每个特征图的H×W维度做标准化（类似Instance Norm）
2. **通道级归一化**（前3通道）：进一步标准化每个通道的统计量

这确保了特征比较时不受绝对数值量级的影响，只关注相对分布模式。注意：`feat1` 通过 `torch.no_grad()` 提取并detach，仅 `feat2` 参与梯度计算。

#### 7.2.5 损失计算

```python
if self.use_l1:
    loss = self.l1_loss(feat1_normalized, feat2_normalized)  # L1距离
else:
    loss = self.mse_loss(feat1_normalized, feat2_normalized)  # L2距离
```

默认使用 **L1损失**，对异常值更鲁棒。

---

### 7.3 基础隐私损失 (PrivacyLoss)

**文件位置**: [privacy_loss.py](losses/privacy_loss.py)

#### 7.3.1 公式

$$L_{privacy} = \text{mean}\left(\text{log1p}\left(\exp\left(\text{cos\_sim}(f_{encoded}, f_{cover}) - (1 - margin)\right)\right)\right)$$

其中：
- $\text{cos\_sim}(a, b) = \frac{a \cdot b}{\|a\|\|b\|}$：余弦相似度，范围 `[-1, 1]`
- `margin`：隐私边界（默认0.5）
- $(1 - margin) = 0.5$：允许的最大相似度阈值
- $\text{log1p}(x) = \log(1 + x)$：数值稳定的softplus近似

#### 7.3.2 实现代码

```python
def forward(self, feat_cover, feat_encoded):
    sim = F.cosine_similarity(feat_encoded, feat_cover, dim=1)
    target = 1 - self.margin
    loss = torch.mean(torch.log1p(torch.exp(sim - target)))
    return loss
```

#### 7.3.3 直觉解释

该损失是一个 **Softplus Loss**（`log1p(exp(...))`），是Hinge Loss的平滑近似：
- 当 `cos_sim >> 0.5` 时：损失快速增长，强力推动特征分离
- 当 `cos_sim ≈ 0.5` 时：损失接近 `log(2) ≈ 0.693`
- 当 `cos_sim << 0.5` 时：损失趋近0，已经满足隐私要求

与Hinge Loss（`max(0, sim - target)`）相比，Softplus Loss处处可导，梯度更平滑，有利于优化收敛。

**目标**：让编码图像的特征向量在CLIP语义空间中与原始图像特征向量的夹角足够大（余弦相似度低于0.5），从而防止通过语义检索关联原始图像和隐写图像。

---

### 7.4 特征重定向损失 (FeatureRedirectionLoss)

**文件位置**: [advanced_privacy_loss.py](losses/advanced_privacy_loss.py)

#### 7.4.1 设计动机

单一模态的隐私保护可能存在盲区。FeatureRedirectionLoss支持**多模态联合特征逃逸**，并引入了**有向重定向（Directed Escape）**机制：

$$L_{total} = w_{clip} \cdot L_{escape}^{CLIP} + w_{dino} \cdot L_{escape}^{DINO}$$

其中每个逃逸损失支持两种模式：
- **纯逃逸**：仅使用Softplus loss将编码图像推离原图
- **有向重定向**：在推离原图的同时，通过硬负样本将编码图像"拉向"其他无关图像，实现语义伪装

#### 7.4.2 有向重定向损失公式

对于单一模态：

$$L_{directed} = L_{escape} + w_{redirect} \cdot \max(0, m_{redirect} - \text{sim}(f_{encoded}, f_{negative})) + 0.3 \cdot \max(0, \text{sim}(f_{encoded}, f_{negative}) - s_{max})$$

其中：
- $L_{escape}$：Softplus loss，推离原图
- $m_{redirect}=0.6$：重定向margin，要求与负样本相似度 > 0.6
- $s_{max}=0.75$：最大允许相似度，防止过度接近导致特征坍塌
- $w_{redirect}$：重定向权重（CLIP: 0.5, DINO: 0.3）

#### 7.4.3 模态互补性

| 模态 | 模型 | 特征维度 | 优势 |
|------|------|---------|------|
| CLIP | ViT-B/32 | 512 | 语义理解、图文对齐 |
| DINOv2 | ViT-S/14 | 384 | 自监督视觉特征、细粒度纹理 |

默认配置 `dino_weight=0.5`（Config中），Trainer中实际使用 `dino_weight=0.8`，与CLIP联合使用提供多模态隐私保护。

#### 7.4.4 返回值

该方法返回 `(total_loss, loss_dict)`，其中 `loss_dict` 包含 `clip_escape`、`clip_redirect`、`clip_control`、`dino_escape`、`dino_redirect`、`dino_control`、`total_privacy_loss` 等，便于训练监控。

---

### 7.5 重定向三元组损失 (RedirectionTripletLoss)

**文件位置**: [advanced_privacy_loss.py](losses/advanced_privacy_loss.py)

#### 7.5.1 公式

$$L_{triplet} = \underbrace{\text{mean}\left(\text{log1p}\left(\exp(\text{sim}(f_{encoded}^{att}, f_{cover}^{att}) - (1 - m_{escape}))\right)\right)}_{\text{逃逸项}} + \underbrace{\text{mean}\left(\text{relu}(m_{redirect} - \text{sim}(f_{encoded}^{att}, f_{target}^{att}))\right)}_{\text{吸引项}} + 0.3 \times \underbrace{\text{mean}\left(\text{relu}(\text{sim}(f_{encoded}^{att}, f_{target}^{att}) - s_{max})\right)}_{\text{排斥控制项}}$$

其中：
- $f_{cover}^{att}, f_{encoded}^{att}$：攻击者特征空间中的封面/编码图像特征
- $f_{target}^{att}$：硬负样本特征（从记忆库中获取的最难区分样本，同样在攻击者空间）
- $m_{escape}=0.3$（Trainer实际值）：逃逸margin，要求余弦相似度 < 0.7
- $m_{redirect}=0.6$：重定向margin，要求与目标相似度 > 0.6
- $s_{max}=0.75$：最大允许相似度

#### 7.5.2 直觉解释

**关键设计**：三元组损失**统一在攻击者特征空间内计算**，避免跨特征空间的梯度噪声问题：

1. **逃逸项**（Softplus Loss）：推动编码图像特征在攻击者空间中**远离**封面图像特征
2. **吸引项**（Hinge Loss）：推动编码图像特征**靠近**目标负样本（相似度 > 0.6），使其"伪装"成其他图像
3. **排斥控制项**（Hinge Loss）：防止过度接近（相似度 < 0.75），避免特征坍塌

如果未提供 `feat_target`，仅计算逃逸项（退化为标准Softplus loss）。

---

### 7.6 检索逃逸损失 (RetrievalEscapeLoss)

**文件位置**: [advanced_privacy_loss.py](losses/advanced_privacy_loss.py)

这是**最高级别的隐私损失聚合器**，整合了所有隐私保护子模块：

$$L_{retrieval\_escape} = \underbrace{L_{multi\_feature}}_{\text{CLIP/DINO有向逃逸}} + w_{triplet} \cdot \underbrace{L_{triplet}^{attacker}}_{\text{攻击者空间三元组(逃逸+重定向)}}$$

#### 7.6.1 各组成部分

| 组件 | 权重 | 来源 | 作用 |
|------|------|------|------|
| `multi_feature_loss` | `clip=1.0, dino=0.5/0.8` | FeatureRedirectionLoss | 在CLIP/DINO语义空间中执行推离+拉向 |
| `triplet_loss` | `triplet_weight` | RedirectionTripletLoss | 在攻击者特征空间中执行逃逸+重定向 |

**关键设计决策**：

三元组损失统一在攻击者特征空间内计算，输入为 `attacker_feat_cover`、`attacker_feat_encoded` 和 `feat_negatives`（均来自攻击者模型或其记忆库）。这避免了之前将CLIP特征与攻击者特征混合计算余弦相似度导致的跨空间梯度噪声问题。

攻击者逃逸损失已合并到三元组损失的逃逸项中，不再单独计算。

#### 7.6.2 Trainer中的实际参数

在Trainer初始化中，`RetrievalEscapeLoss` 的参数与Config默认值有所不同：

| 参数 | Config默认值 | Trainer实际值 | 说明 |
|------|------------|-------------|------|
| `dino_weight` | 0.5 | 0.8 | Trainer中覆盖为0.8 |
| `triplet_weight` | 0.15 | 0.5 | Trainer中覆盖为0.5 |
| `triplet_margin` | 0.5 | 0.3 | Trainer中覆盖为0.3 |
| `margin` | 0.5 | 0.5 | 与Config一致（privacy_margin） |

---

## 8. 噪声层模块 (`noise_layers/`)

噪声层用于**模拟真实世界中的信道干扰**，测试隐写系统的鲁棒性。训练时随机选择一种噪声施加于编码图像。

### 8.1 噪声组合器 (Noiser)

**文件位置**: [noiser.py](noise_layers/noiser.py)

```python
class Noiser(nn.Module):
    def forward(self, encoded_and_cover):
        idx = np.random.randint(0, len(self.noise_layers))
        return self.noise_layers[idx](encoded_and_cover)
```

**核心行为**：每次前向传播时，**均匀随机**选择一个噪声层执行。这相当于数据增强中的RandomAugment策略，迫使模型对所有类型的噪声都具有鲁棒性。

当没有配置任何噪声层时，自动添加Identity（恒等映射）层。

Noiser还支持字符串占位符：
- `'JpegPlaceholder'` → 自动替换为 `JpegCompression(device)`
- `'QuantizationPlaceholder'` → 自动替换为 `Quantization(device)`

### 8.2 Identity恒等映射

**文件位置**: [identity.py](noise_layers/identity.py)

```python
class Identity(nn.Module):
    def forward(self, noised_and_cover):
        return noised_and_cover  # 直接返回输入，不做任何修改
```

**用途**：作为默认/基准噪声层，或用于消融实验。

### 8.3 随机裁剪 (Crop)

**文件位置**: [crop.py](noise_layers/crop.py)

#### 8.3.1 算法流程

```
输入: noised_image [B,3,H,W], cover_image [B,3,H,W]
参数: height_ratio_range=(hmin, hmax), width_ratio_range=(wmin, wmax)

1. random_float(hmin, hmax) → 保留高度比例 rh ∈ [hmin, hmax]
2. random_float(wmin, wmax) → 保留宽度比例 rw ∈ [wmin, wmax]
3. remaining_h = round(rh * H), remaining_w = round(rw * W)
4. h_start = rand(0, H - remaining_h), w_start = rand(0, W - remaining_w)
5. 裁剪: image[:, :, h_start:h_end, w_start:w_end]

输出: (cropped_noised, cropped_cover)  # 两图同步裁剪
```

#### 8.3.2 示例

若 `height_ratio_range=(0.7, 0.9)`，`width_ratio_range=(0.7, 0.9)`，图像大小256×256：
- 可能保留 `179×205` 到 `230×230` 的区域
- 裁剪掉边缘部分，模拟图像被部分截断的场景

### 8.4 裁剪替换 (Cropout)

**文件位置**: [cropout.py](noise_layers/cropout.py)

#### 8.4.1 算法流程

```
1. 创建全零mask: cropout_mask = zeros_like(noised_image)
2. 随机选取矩形区域: (h_start, h_end, w_start, w_end)
3. 该区域内mask设为1
4. 合成: result = noised_image * mask + cover_image * (1 - mask)
```

**效果**：矩形区域保留编码图像，其余区域**还原为原始封面图像**。这模拟了部分区域被"修复"或"覆盖"的场景，比纯裁剪更温和。

### 8.5 像素丢弃 (Dropout)

**文件位置**: [dropout.py](noise_layers/dropout.py)

#### 8.5.1 算法流程

```
1. keep_percent ~ Uniform(keep_min, keep_max)  # 如 [0.9, 0.95]
2. 生成二值mask: P(1)=keep_percent, P(0)=1-keep_percent
3. 合成: result = noised * mask + cover * (1-mask)
```

**效果**：随机将一定比例的像素替换为原始图像的对应像素。这是一种**逐像素**的局部扰动，模拟传输过程中的丢包或像素损坏。

### 8.6 缩放变换 (Resize)

**文件位置**: [resize.py](noise_layers/resize.py)

#### 8.6.1 算法流程

```
1. ratio ~ Uniform(ratio_min, ratio_max)  # 如 [0.5, 1.0]
2. 下采样: image → [H*ratio, W*ratio]  （双线性插值）
3. 上采样回原尺寸: → [H, W]  （双线性插值）
```

**效果**：模拟图像被缩放后再恢复的过程，导致高频信息（包括隐写信号）丢失。两次插值的组合会产生平滑效应。

### 8.7 JPEG压缩 (JpegCompression)

**文件位置**: [jpeg_compression.py](noise_layers/jpeg_compression.py)

这是**最复杂也是最重要的噪声层**，完整模拟了JPEG有损压缩的DCT域量化过程。

#### 8.7.1 JPEG压缩流水线

```
RGB图像 [B,3,H,W]
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Step 1: ZeroPadding（补齐到8的倍数）              │
│   pad_h = (8 - H%8) % 8, pad_w = (8 - W%8) % 8   │
├─────────────────────────────────────────────────┤
│ Step 2: RGB → YUV色彩空间转换（BT.601标准）       │
│   Y  = 0.299R + 0.587G + 0.114B                 │
│   U  = -0.169R - 0.331G + 0.500B                │
│   V  = 0.500R - 0.419G - 0.081B                 │
├─────────────────────────────────────────────────┤
│ Step 3: 分块DCT变换（8×8块）                     │
│   使用预计算的64个DCT滤波器（8×8卷积核）          │
│   stride=8，无重叠                               │
├─────────────────────────────────────────────────┤
│ Step 4: 量化（JPEG核心步骤）                      │
│   使用Zigzag排序的掩码，只保留前N个AC系数         │
│   Y通道保留25个，U/V通道各保留9个                 │
│   掩码作用：image_dct * mask                     │
├─────────────────────────────────────────────────┤
│ Step 5: 分块IDCT逆变换                          │
│   使用预计算的64个IDCT滤波器                      │
├─────────────────────────────────────────────────┤
│ Step 6: YUV → RGB色彩空间逆转换                  │
├─────────────────────────────────────────────────┤
│ Step 7: 去除padding                             │
└─────────────────────────────────────────────────┘
    │
    ▼
压缩后图像 [B,3,H,W]
```

#### 8.7.2 DCT/IDCT滤波器生成

```python
def gen_filters(size_x, size_y, dct_or_idct_fun):
    filters = np.zeros((size_x * size_y, size_x, size_y))
    for k_y in range(size_y):
        for k_x in range(size_x):
            for n_y in range(size_y):
                for n_x in range(size_x):
                    filters[k_y*size_x+k_x, n_y, n_x] = \
                        dct_or_idct_fun(n_y, k_y, size_y) * dct_or_idct_fun(n_x, k_x, size_x)
    return filters
```

生成 **64个8×8的DCT基函数图像**（对应64个频率分量），排列成 `(64, 1, 8, 8)` 的卷积核张量。

**DCT系数公式**：

$$C(n,k,N) = \cos\left(\frac{\pi}{N}(n + 0.5)k\right)$$

**IDCT系数公式**：

$$IC(n,k,N) = \left(\delta(0,n)(-\frac{1}{2}) + \cos\left(\frac{\pi}{N}(k+0.5)n\right)\right) \times \sqrt{\frac{1}{2N}}$$

#### 8.7.3 DCT卷积实现

```python
def apply_conv(self, image, filter_type):
    filters = self.dct_conv_weights  # 或 idct_conv_weights
    
    image_conv_channels = []
    for channel in range(image.shape[1]):
        image_ch = image[:, channel, :, :].unsqueeze(1)  # [B, 1, H, W]
        image_conv = F.conv2d(image_ch, filters, stride=8)  # [B, 64, H/8, W/8]
        
        # 重排: 将64个通道的重排为8×8块结构
        image_conv = image_conv.permute(0, 2, 3, 1)  # [B, H/8, W/8, 64]
        image_conv = image_conv.view(B, H/8, W/8, 8, 8)
        image_conv = image_conv.permute(0, 1, 3, 2, 4)  # [B, H/8, 8, W/8, 8]
        image_conv = image_conv.contiguous().view(B, H, W)  # [B, H, W]
        
        image_conv_channels.append(image_conv)
    
    return torch.cat(image_conv_channels, dim=1)  # [B, 3, H, W]
```

**巧妙之处**：用 `stride=8` 的卷积实现分块DCT，避免了显式的循环。64个输出通道对应8×8块的64个DCT系数。后续的view/permute操作将系数重新排列为标准的块状布局。

#### 8.7.4 Zigzag量化掩码

```python
def get_jpeg_yuv_filter_mask(image_shape, window_size, keep_count):
    mask = np.zeros((window_size, window_size), dtype=np.uint8)
    
    # Zigzag排序：按频率从低到高排列
    index_order = sorted(
        ((x, y) for x in range(window_size) for y in range(window_size)),
        key=lambda p: (p[0]+p[1], -p[1] if (p[0]+p[1])%2 else p[1])
    )
    
    # 只保留前keep_count个最低频系数
    for i, j in index_order[0:keep_count]:
        mask[i, j] = 1
    
    return np.tile(mask, ...)  # 平铺到整个图像尺寸
```

**Zigzag扫描顺序**：从DC分量(0,0)开始，沿对角线方向遍历，频率逐渐升高。

**默认保留系数数量**：`yuv_keep_weights=(25, 9, 9)`
- Y（亮度）通道：保留25个系数（约39%）
- U/V（色度）通道：各保留9个（约14%，人眼对色度更不敏感）

这模拟了**中等质量（Q≈50-70）**的JPEG压缩效果。

**掩码缓存机制**：JpegCompression内部维护一个大型掩码（1000×1000），通过切片适配不同尺寸的输入图像，避免重复计算。

### 8.8 量化噪声 (Quantization)

**文件位置**: [quantization.py](noise_layers/quantization.py)

#### 8.8.1 傅里叶四舍五入量化

```python
def fourier_rounding(self, tensor):
    z = torch.mul(self.weights, torch.sin(torch.mul(tensor, self.scales)))
    z = torch.sum(z, dim=0)
    return tensor + z
```

这不是简单的四舍五入，而是基于**傅里叶级数的近似舍入**：

$$round(x) \approx x + \sum_{n=1}^{N} \frac{(-1)^{n+1}}{n\pi} \sin(2\pi(n+1)x)$$

其中 `N=10`（谐波数量）。傅里叶级数展开的阶跃函数（sign function）可以用来近似最近整数舍入。

**参数**：
- `weights`: 系数 $\frac{(-1)^{n+1}}{(n+1)\pi}$，形状 `[10]` → 扩展为 `[10,1,1,1,1]`
- `scales`: 频率 $2\pi(n+1)$，形状 `[10]` → 扩展为 `[10,1,1,1,1]`

#### 8.8.2 完整流程

```
1. transform(tensor, (0, 255)): 线性映射到 [0, 255]
2. clamp(min=0, max=255): 截断到有效范围
3. fourier_rounding(): 傅里叶近似舍入到整数
4. transform(tensor, (orig_min, orig_max)): 映射回原始范围
```

**效果**：模拟图像被保存为整数格式（如PNG/JPG的8-bit量化）时的精度损失。

### 8.9 DCT滤波器生成器 (DctFilterGenerator)

**文件位置**: [dct_filters.py](noise_layers/dct_filters.py)

这是一个工具类，提供DCT/IDCT滤波器和JPEG掩码的生成功能。主要供JpegCompression内部使用，也可独立调用。

与JpegCompression中的 `gen_filters` 不同，DctFilterGenerator 生成的滤波器排列方式为 `(channels, tile_x, tile_y, channels, tile_x*tile_y)`，适用于逐通道的DCT/IDCT操作。

---

## 9. 训练系统 (`trainer.py` & `train.py`)

### 9.1 Trainer类详解

**文件位置**: [trainer.py](trainer.py)

#### 9.1.1 初始化流程

Trainer的 `__init__` 方法完成了整个系统的组装：

```
1. 创建TensorBoard日志记录器
2. 初始化四大核心模型:
   ├── Encoder (可训练)
   ├── Decoder (可训练)
   ├── Steganalyzer (可训练)
   └── ContentRecognizer/CLIP (冻结)
3. 创建优化器:
   ├── optimizer_encoder_decoder: Adam(encoder+decoder参数)
   └── optimizer_steganalyzer: Adam(steganalyzer参数)
4. 创建损失函数集合:
   ├── SSIMLoss (可微分SSIM)
   ├── PerceptualLoss (VGG19特征)
   ├── PrivacyLoss / RetrievalEscapeLoss (隐私保护)
   ├── MSELoss (像素重建)
   ├── BCELoss / BCEWithLogitsLoss (消息/对抗)
5. 条件性创建高级组件:
   ├── DINOv2模型 (如 dino_weight > 0)
   ├── RetrievalAttackerWithMemory (如 use_retrieval_attacker)
   │   └── optimizer_attacker: Adam(attacker参数, lr=lr_encoder*0.5)
   └── FeatureMemoryBank (如 use_clip_redirect)
       ├── clip_memory_bank (CLIP特征)
       └── dino_memory_bank (DINO特征, 预留)
6. 解析并构建噪声层管道
```

#### 9.1.2 FeatureMemoryBank (通用特征记忆库)

**文件位置**: [trainer.py](trainer.py) 中的 `FeatureMemoryBank` 类

这是与 `RetrievalAttackerWithMemory` 并列的**通用特征记忆库**，用于存储CLIP/DINO等冻结模型的特征向量：

```python
class FeatureMemoryBank:
    def __init__(self, feature_dim, memory_size=1000, device='cuda'):
        self.memory_bank = torch.zeros(memory_size, feature_dim, device=device)
        self.memory_ptr = 0
        self.memory_count = 0
    
    @torch.no_grad()
    def update(self, features):
        features = F.normalize(features.float(), p=2, dim=1)
        # 循环缓冲区写入逻辑...
    
    def get_hard_negatives(self, features, k=5):
        # 点积相似度排序，取top-k
        # 两阶段选择：先top-k，再从中选最难的1个
```

**与 RetrievalAttackerWithMemory 的区别**：

| 特性 | FeatureMemoryBank | RetrievalAttackerWithMemory |
|------|-------------------|----------------------------|
| 用途 | 存储CLIP/DINO等**冻结模型**特征 | 存储**可训练攻击者**提取的特征 |
| 更新方式 | Python属性（非buffer） | PyTorch register_buffer |
| 状态保存 | 通过state_dict/load_state_dict | 通过模型state_dict自动保存 |
| 梯度 | 不参与任何梯度计算 | 攻击者特征参与攻击者训练梯度 |

在 `train_step` 中，CLIP记忆库的更新和硬负样本获取流程：

```python
if self.clip_memory_bank is not None:
    with torch.no_grad():
        clip_negatives = self.clip_memory_bank.get_hard_negatives(feat_cover, k=self.clip_hard_negative_k)
    if self.clip_memory_bank.memory_count > 0:
        with torch.no_grad():
            self.clip_memory_bank.update(feat_cover)
```

**注意**：记忆库更新使用**当前batch之前的记忆**来获取负样本（避免将当前batch样本作为负样本），更新操作在获取之后执行。

#### 9.1.3 DINOv2特征提取

```python
def extract_dino_features(self, image):
    if not hasattr(self, 'dino_model') or self.dino_model is None:
        return None
    with torch.no_grad():
        image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image_normalized = (image_resized + 1) / 2  # [-1,1] → [0,1]
        image_normalized = (image_normalized - self.dino_mean) / self.dino_std  # ImageNet标准化
        features = self.dino_model(image_normalized)  # DINOv2前向传播
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-7)  # L2归一化
    return features  # shape: [B, 384]
```

DINOv2 (`vit_small_patch14_dinov2.lvd142m`) 通过timm库加载，输出 **384维** 特征向量。

#### 9.1.4 DINOv2模型加载与位置编码插值

```python
self.dino_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=224)
if dino_model_path and os.path.exists(dino_model_path):
    state_dict = torch.load(dino_model_path, map_location=self.device)
    # 处理不同的state_dict格式
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    # 位置编码插值：适配不同图像尺寸
    model_pos_embed = self.dino_model.pos_embed.data
    ckpt_pos_embed = state_dict.get('pos_embed', None)
    if ckpt_pos_embed is not None and ckpt_pos_embed.shape != model_pos_embed.shape:
        cls_token = ckpt_pos_embed[:, :1, :]
        patch_embed = ckpt_pos_embed[:, 1:, :]
        embed_dim = patch_embed.shape[-1]
        orig_grid_size = int(patch_embed.shape[1] ** 0.5)
        new_grid_size = int((model_pos_embed.shape[1] - 1) ** 0.5)
        patch_embed = patch_embed.reshape(1, orig_grid_size, orig_grid_size, embed_dim).permute(0, 3, 1, 2)
        patch_embed = F.interpolate(patch_embed, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, embed_dim)
        state_dict['pos_embed'] = torch.cat([cls_token, patch_embed], dim=1)
    
    self.dino_model.load_state_dict(state_dict, strict=False)
```

**位置编码插值**：当预训练权重的位置编码尺寸与当前模型不匹配时（如预训练使用518×518，当前使用224×224），通过双三次插值将位置编码调整到正确的网格尺寸。保留CLS token不变，仅对patch embedding部分进行插值。

### 9.2 两阶段训练策略

#### 总体损失函数

$$L_{total} = w_{visual}(L_{visual} + w_{noised} \cdot L_{visual}^{noised}) + w_{message} \cdot L_{message} + w_{adv} \cdot L_{adv} + w_{privacy} \cdot L_{privacy} + w_{rp} \cdot L_{residual}$$

其中各组件详细说明如下：

#### 9.2.1 视觉重建损失

$$L_{visual} = 1.0 \cdot L_{MSE}(I_{enc}, I_{cov}) + 2.0 \cdot L_{SSIM}(I_{enc}, I_{cov}) + 0.3 \cdot L_{perc}(I_{enc}, I_{cov})$$

$$L_{visual}^{noised} = 1.0 \cdot L_{MSE}(I_{noised}, I_{cov}) + 2.0 \cdot L_{SSIM}(I_{noised}, I_{cov}) + 0.3 \cdot L_{perc}(I_{noised}, I_{cov})$$

**双重视觉损失的含义**：
- $L_{visual}$：无噪声情况下的重建质量（理想条件）
- $L_{visual}^{noised}$：经噪声层处理后的重建质量（真实条件）

两者的加权和确保模型不仅在干净条件下表现好，在有干扰时也能维持质量。

#### 9.2.2 消息提取损失

$$L_{message} = BCEWithLogitsLoss(\hat{m}, m)$$

其中 $\hat{m} = Decoder(I_{noised})$ 是解码器预测的原始logits，$m$ 是真实消息。

**BCEWithLogitsLoss** 内部执行 `sigmoid + binary_cross_entropy`，数值上比分开计算更稳定。

#### 9.2.3 对抗损失（GAN训练）

**判别器（Steganalyzer）训练**：

```python
# 判别器损失：cover=1, stego=0
L_D = (self.bce_with_logits_loss(cover_pred, torch.ones_like(cover_pred)) +
       self.bce_with_logits_loss(stego_pred, torch.zeros_like(stego_pred)))
```

标签分配：
- `cover_pred = Steganalyzer(noised_cover_image)` → **标签=1**（真实图像）
- `stego_pred = Steganalyzer(noised_encoded_image.detach())` → **标签=0**（隐写图像）

**判别器暂停机制**：

```python
disc_threshold = getattr(self.cfg, 'discriminator_loss_threshold', None)
skip_disc = (disc_threshold is not None and L_D.item() < disc_threshold)
```

当判别器损失低于阈值（默认0.1）时，暂停判别器训练，防止判别器过度强大导致生成器梯度消失。

**生成器（Encoder+Decoder）对抗损失**：

```python
stego_pred = self.steganalyzer(disc_stego_input)  # 注意：没有detach！
L_adv = self.bce_with_logits_loss(stego_pred, torch.ones_like(stego_pred))
# 标签=1：希望判别器认为隐写图像是真实的
```

这是标准的**非饱和GAN对抗损失**，生成器试图"欺骗"判别器。

**训练顺序**：
1. 先计算判别器损失并更新判别器参数（仅在 `adv > 0` 且未达到阈值时）
2. 再计算生成器对抗损失（使用更新后的判别器）
3. 最后连同其他损失一起更新生成器参数

#### 9.2.4 隐私保护损失

详见第7节的损失函数分析。在Phase 2中，`RetrievalEscapeLoss` 整合了以下子损失：

| 子损失 | 权重 | 含义 |
|--------|------|------|
| CLIP有向逃逸损失 | 1.0 | CLIP特征空间中的Softplus + 重定向 |
| DINO有向逃逸损失 | 0.8 (Trainer) | DINO特征空间中的Softplus + 重定向 |
| 攻击者空间三元组损失 | 0.5 (triplet_weight in Trainer) | 攻击者特征空间中的逃逸+重定向（统一计算） |

**CLIP/DINO 硬负样本选择流程**（Trainer中）：

```python
# 1. 从CLIP记忆库获取硬负样本
clip_negatives = self.clip_memory_bank.get_hard_negatives(feat_cover, k=5)

# 2. FeatureRedirectionLoss内部使用这些负样本计算有向重定向
#    escape + redirect + control
```

**攻击者硬负样本选择流程**：

```python
# 1. 获取top-k硬负样本
feat_negatives = self.retrieval_attacker.get_hard_negatives(attacker_feat_cover_for_negatives, k=5)

# 2. 从top-k中选择最难的1个
if feat_negatives.dim() == 3:  # [B, k, dim]
    similarities = F.cosine_similarity(
        attacker_feat_cover_for_negatives.unsqueeze(1),
        feat_negatives, dim=2
    )
    hardest_idx = similarities.argmax(dim=1)
    feat_negatives = feat_negatives[torch.arange(feat_negatives.size(0)), hardest_idx]
    # → [B, dim]
```

这种两阶段选择策略（先top-k再选最难的）比直接选top-1更稳定，因为top-k提供了候选池，避免了噪声样本的直接干扰。

#### 9.2.5 残差惩罚

$$L_{residual} = MSE(I_{enc}, I_{cov})$$

这个值由编码器的 `forward` 方法直接返回，作为额外的正则化项限制修改幅度。

### 9.3 训练流程 (`train.py`)

**文件位置**: [train.py](train.py)

#### 9.3.1 命令行接口

项目支持两种运行模式：

**模式1：新建实验 (`new`)**

```bash
python train.py new --data-dir ./dataset --name experiment1 \
    --batch-size 8 --size 256 --message 30 \
    --epochs-phase1 300 --epochs-phase2 100 \
    --noise "identity"
```

**模式2：继续训练 (`continue`)**

```bash
python train.py continue --folder ./experiments/experiment1
```

**模式3：从指定检查点恢复为新实验 (`new` + `--continue-from-folder`)**

```bash
python train.py new --data-dir ./dataset --name experiment2 \
    --continue-from-folder ./experiments/experiment1
```

这允许从已有实验的检查点继续训练，但保存到新的实验目录中（保留原始实验不变）。

#### 9.3.2 实验目录结构

```
experiments/{experiment_name}/
├── config.json              # 配置快照
├── train.log                # 训练日志
├── checkpoints/
│   ├── checkpoint.pth        # 最新检查点（含所有模块和优化器状态）
│   ├── checkpoint_phase{N}_epoch{M}.pth  # 阶段性检查点
│   ├── best_phase1.pth       # Phase 1 最佳检查点
│   ├── best_phase2.pth       # Phase 2 最佳检查点
│   └── visualizations/
│       ├── phase1_epoch10.png
│       └── ...
└── tensorboard/              # TensorBoard事件文件
```

#### 9.3.3 训练主循环 (`run_training_loop`)

```python
for phase in range(start_phase, 3):          # Phase 1, 2
    num_epochs = getattr(cfg, f'num_epochs_phase{phase}')
    phase_start_epoch = start_epoch if phase == start_phase else 1
    
    if phase == 1 and phase_start_epoch == 1:
        trainer.warmup_memory_bank(train_loader, num_batches=50)  # 预热记忆库
    
    for epoch in range(phase_start_epoch, num_epochs+1):
        for batch_idx, batch in enumerate(train_loader):
            losses = trainer.train_step(batch, phase, epoch)     # 单步训练
        
        val_losses = trainer.validate(val_loader, phase, epoch, save_vis)  # 验证
        
        if epoch % cfg.save_interval == 0:
            trainer.save_checkpoint(epoch, phase, is_best)           # 保存检查点
```

#### 9.3.4 检查点保存策略

```python
def save_checkpoint(self, epoch, phase, is_best=False):
    # 保存: encoder, decoder, steganalyzer状态字典
    # 保存: optimizer_encoder_decoder, optimizer_steganalyzer状态字典
    # 保存: 最佳损失记录
    # 条件保存: attacker优化器和retrieval_attacker状态字典
    # 条件保存: clip_memory_bank, dino_memory_bank状态字典
    
    # 保留策略: 每个phase最多保留5个最新检查点
    # 自动删除最旧的检查点
```

#### 9.3.5 随机种子设置

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

确保实验的可重复性。默认种子为42。

### 9.4 记忆库预热与检索攻击者训练

```python
def warmup_memory_bank(self, train_loader, num_batches=50):
    if self.retrieval_attacker is None and self.clip_memory_bank is None and self.dino_memory_bank is None:
        return
    
    print(f'Warming up Memory Banks with {num_batches} batches...')
    
    if self.retrieval_attacker is not None:
        self.retrieval_attacker.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            cover_image = batch['cover_image'].to(self.device)
            
            if self.retrieval_attacker is not None:
                features = self.retrieval_attacker(cover_image)
                self.retrieval_attacker.update_memory(features)
            
            if self.clip_memory_bank is not None:
                clip_feat = self.content_recognizer(cover_image)
                self.clip_memory_bank.update(clip_feat)
            
            if self.dino_memory_bank is not None:
                dino_feat = self.extract_dino_features(cover_image)
                if dino_feat is not None:
                    self.dino_memory_bank.update(dino_feat)
    
    if self.retrieval_attacker is not None:
        self.retrieval_attacker.train()
    
    # 打印各记忆库预热后的状态...
```

在Phase 1开始前，用50个batch的封面图像特征**预填**所有记忆库（攻击者、CLIP、DINO）。这确保在正式训练开始时，各记忆库已有足够的样本可供硬负样本挖掘。

**检索攻击者训练**：检索攻击者（RetrievalAttackerWithMemory）在Phase 2参与训练，其损失函数使用**InfoNCE（对比学习）损失**：

```python
if valid_count >= self.cfg.batch_size:
    valid_memory = self.retrieval_attacker.memory_bank[:valid_count].detach()
    
    # 正样本：编码图像与封面图像的相似度
    pos_sim = torch.sum(attacker_feat_encoded_new * attacker_feat_cover_new.detach(), dim=1, keepdim=True) / temperature
    # 负样本：编码图像与记忆库中所有特征的相似度
    neg_sim = torch.mm(attacker_feat_encoded_new, valid_memory.t()) / temperature
    
    # InfoNCE损失
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)  # 正样本在第0位
    L_attacker = F.cross_entropy(logits, labels)
else:
    # 记忆库不足时退化为MSE损失
    L_attacker = F.mse_loss(attacker_feat_encoded_new, attacker_feat_cover_new.detach())
```

**InfoNCE损失的含义**：训练攻击者将编码图像与对应的封面图像关联起来（正样本对），同时与记忆库中的其他图像区分开（负样本）。通过在对抗中训练攻击者，可以更真实地模拟CBIR系统的检索能力，从而提升隐私保护的鲁棒性。

训练完成后，更新记忆库：

```python
with torch.no_grad():
    self.retrieval_attacker.update_memory(attacker_feat_cover_new)
```

---

## 10. 验证脚本 (`validation_script.py`)

**文件位置**: [validation_script.py](validation_script.py)

验证脚本提供了四种工作模式：

### 10.1 完整验证模式 (`--mode full`)

执行完整的隐写流程并输出全面评估指标：

```python
def run_full_validation(self, image_path, message, output_dir=None):
    # 1. 加载并预处理图像
    # 2. 处理消息（补零/截断到30位）
    # 3. 编码：Encoder(cover, message) → encoded_image
    # 4. 解码：Decoder(encoded_image) → decoded_message
    # 5. 消息验证：计算比特准确率
    # 6. 判别器预测：分别对原图和隐写图进行隐写检测
    # 7. 隐私评估：计算余弦相似度、欧氏距离、SSIM、PSNR
    # 8. 保存可视化结果
```

**输出指标**：

| 指标 | 公式/方法 | 含义 |
|------|----------|------|
| 比特准确率 | `correct_bits / total_bits` | 消息提取正确率 |
| 隐写概率(原图) | `sigmoid(Steganalyzer(cover))` | 原图被判为隐写的概率（应接近0%） |
| 隐写概率(隐写图) | `sigmoid(Steganalyzer(encoded))` | 隐写图被判为隐写的概率（应接近50%，即成功欺骗） |
| 余弦相似度 | `cos(f_cover, f_encoded)` | CLIP特征空间相似度（越低隐私越好） |
| SSIM | `1 - SSIM_Loss` | 结构相似度（越高视觉质量越好） |
| PSNR | `10·log₁₀(4.0/MSE)` | 峰值信噪比（dB，越高越好） |
| 欧氏距离 | `‖f_encoded - f_cover‖₂` | 特征向量欧氏距离 |
| 相似概率 | `(cos_sim + 1) / 2` | 归一化到[0,1]的相似概率 |

### 10.2 判别器模式 (`--mode discriminator`)

仅对单张图像执行隐写检测：

```bash
python validation_script.py -c checkpoint.pth -i image.png --mode discriminator
```

### 10.3 解码模式 (`--mode decode`)

仅从隐写图像中提取隐藏消息：

```bash
python validation_script.py -c checkpoint.pth -i stego.png --mode decode
```

### 10.4 相似度判断模式 (`--mode similarity`)

仅计算两张图像之间的特征相似度（无需检查点）：

```bash
python validation_script.py -i encoded_image.png --cover cover_image.png --mode similarity
```

### 10.5 消息处理

验证脚本支持灵活的消息输入：
- 二进制字符串：直接输入如 `"101000100110100101100011100010"`
- 消息文件路径：指向包含二进制字符串的文件
- 自动补零/截断：消息长度不足30位时自动补0，超过30位时自动截断

---

## 11. CLIP视觉模型

本项目的CLIP模型通过 `pip install git+https://github.com/openai/CLIP.git` 安装，不依赖本地子模块。CLIP (Contrastive Language-Image Pre-Training) 由OpenAI开发，通过大规模图文对比学习训练。

### 11.1 CLIP模型简介

CLIP的核心思想是通过对比学习，让图像编码器和文本编码器在统一的特征空间中产生对齐的表示：

- **预训练规模**：约4亿图文对
- **架构**：图像端使用ViT-B/32（Transformer）或ResNet，文本端使用Transformer
- **特征维度**：512维L2归一化特征向量
- **使用方式**：本项目仅使用图像编码器（`clip.load` + `encode_image`），文本编码器不使用

### 11.2 在本项目中的作用

内容识别器（ContentRecognizer）使用冻结的CLIP ViT-B/32作为特征提取器：
- 将原始封面图像和编码图像分别通过CLIP编码，得到512维语义特征
- 计算两者的余弦相似度，用于隐私损失计算
- CLIP的语义理解能力使得即使像素级差异很小，语义级的"相似性泄露"也能被检测

### 11.3 为什么选择CLIP？

相比纯视觉特征提取器（如VGG、ResNet）：
- CLIP特征具有更强的**语义泛化能力**，不受特定数据集限制
- 训练过程中见过的图文对齐关系使其对场景、物体、属性的理解更加全面
- 冻结使用无需梯度计算，开销可控

---

## 12. 核心算法原理深度剖析

### 12.1 隐写编码原理

#### 传统隐写 vs 深度学习隐写

| 维度 | 传统隐写（LSB等） | 深度学习隐写（本系统） |
|------|------------------|---------------------|
| 嵌入方式 | 手工规则（修改最低有效位） | 神经网络学习最优嵌入 |
| 容量固定 | 是（如1bpp） | 可学习自适应 |
| 抗检测性 | 弱（易被统计分析发现） | 强（对抗训练） |
| 鲁棒性 | 差 | 强（噪声层训练） |
| 隐私保护 | 无 | 有（特征逃逸） |

#### 本系统的编码策略

编码器采用**全图像扩散嵌入**策略：

1. **消息空间化**：30位消息 → 30通道的256×256特征图
2. **特征融合**：消息 + 图像特征 + 原始图像 → 97通道联合表示
3. **残差学习**：最终输出 = 原始图像 + 微小残差（通过tanh限制在±1范围内）

残差的期望幅度很小（由MSE损失约束），这使得隐写信号分散在所有像素的所有通道中，类似于**扩频通信**中的扩频技术，极大提升了抗检测性。

### 12.2 对抗训练原理

本系统采用的是**非饱和GAN + 谱归一化**的训练范式：

```
┌──────────────┐         ┌──────────────────┐
│   Generator   │ encoded │   Discriminator  │
│ (Enc + Dec)  │────────▶│ (Steganalyzer)   │
│              │  image  │                  │
└──────┬───────┘         └────────┬─────────┘
       │                          │
       │ L_adv (fake→real)        │ L_D (real→1, fake→0)
       │                          │
       ▼                          ▼
  更新Gen参数                   更新Disc参数
```

**训练动力学**：
1. 判别器学会区分真实图像和隐写图像
2. 生成器学会制造更难被检测的隐写图像
3. 达到纳什均衡时，隐写图像的统计分布与真实图像无法区分

**判别器暂停机制**：当判别器损失低于阈值（0.1）时暂停训练，防止判别器过强导致生成器梯度消失，这是训练稳定性的重要保障。

**SpectralNorm的关键作用**：保证判别器满足1-Lipschitz约束，避免梯度爆炸/消失，使对抗训练稳定收敛。

### 12.3 隐私保护的多模态特征逃逸机制

#### 问题定义

传统隐写只关心"能否被发现"，但忽略了**语义隐私泄露**风险：
- 攻击者可以使用CLIP等语义模型提取图像特征
- 如果编码前后特征相似，攻击者可以通过**语义检索**建立联系
- 这构成了**推理攻击（Inference Attack）**的一种

#### 解决方案：特征空间逃逸 + 有向重定向

```
CLIP语义特征空间（512维单位超球面）

                    f_encoded (目标位置)
                       ●
                      /
                     /  ← 最大化距离(推离原图)
                    /
                   /
        f_cover ●─────────────────● f_other_image
                  \               /
                   \  ← 最小化距离(拉向负样本)
                    \           /
                     ● f_hard_negative
```

**逃逸策略**：
1. **推离**（Push Away）：最大化 `sim(f_encoded, f_cover)` 的距离（Softplus Loss）
2. **拉向**（Pull Toward）：最小化 `sim(f_encoded, f_hard_negative)` 的距离（有向重定向）
3. **对抗逃逸**：在攻击者的特征空间中也执行推离+拉向操作

#### 为什么多模态？

不同预训练模型捕获的语义信息有所差异：
- **CLIP**：强调图文对齐语义，对物体类别、场景类型敏感
- **DINOv2**：自监督学习，对纹理、细粒度结构更敏感

联合使用可以覆盖更多潜在的攻击向量。

#### 有向重定向的优势

相比纯"推离"策略，有向重定向引入了一个**语义目标**：
- 纯推离：编码图像可能飘向特征空间的任意方向，甚至可能靠近其他敏感区域
- 有向重定向：明确将编码图像"伪装"成某个无关图像，使攻击者更难以通过检索找到原图
- 控制项防止过度接近，保持特征多样性

### 12.4 记忆库驱动的硬负样本挖掘

#### 动机

普通的三元组损失使用batch内的随机样本作为负样本，但这些样本可能太容易区分（"简单负样本"），提供的训练信号较弱。

#### MoCo风格的记忆库

```
训练过程中:

Batch 1: [img_A, img_B, img_C, img_D]
    ↓ 提取特征并存入记忆库
Memory Bank: [f_A, f_B, f_C, f_D, ..., ?, ?, ...] (capacity=1000)

...

Batch N: 查询 img_X 的特征 f_X
    ↓ 在记忆库中搜索
Top-5 最相似: [f_P, f_Q, f_R, f_S, f_T]  ← 这些就是"硬负样本"
    ↓ 从中选最难的1个
Hardest: f_P
    ↓ 用作重定向目标
L_redirect = max(0, 0.6 - sim(f_encoded, f_P))
```

**多记忆库设计**：

| 记忆库 | 存储特征 | 用途 |
|--------|---------|------|
| `clip_memory_bank` | CLIP 512维特征 | CLIP有向重定向的负样本来源 |
| `dino_memory_bank` | DINOv2 384维特征 | DINO有向重定向的负样本来源（预留） |
| `retrieval_attacker.memory_bank` | 攻击者512维特征 | 攻击者三元组损失 + InfoNCE训练 |

**优势**：
- 记忆库积累了历史batch的特征，负样本池远大于当前batch
- 选择"最相似"的样本作为负样本，迫使模型学习更精细的区分能力
- 循环缓冲区设计保证特征分布的时效性
- 多记忆库覆盖多模态攻击向量

---

## 13. 完整数据流图

以下是单个训练step的完整数据流向：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Train Step 数据流 (Phase 2)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入                                                                     │
│  ┌─────────────────┐  ┌──────────────────┐                               │
│  │ cover_image      │  │ message           │                               │
│  │ [B,3,256,256]    │  │ [B,30]            │                               │
│  │ range:[-1,1]     │  │ values:{0,1}      │                               │
│  └────────┬────────┘  └────────┬─────────┘                               │
│           │                      │                                         │
│           ▼                      │                                         │
│  ┌───────────────────────────────┘                                         │
│  │                                                                        │
│  │  ENCODER                                                               │
│  │  ┌──────────────────────────────────────────────────────────┐          │
│  │  │ 1. message expand: [B,30] → [B,30,256,256]               │          │
│  │  │ 2. ConvBNRelu×8: cover → [B,64,256,256] features        │          │
│  │  │ 3. Concat: [msg(30); feat(64); cover(3)] → [B,97,256,256]│         │
│  │  │ 4. ConvBNRelu: [B,97] → [B,64,256,256]                  │          │
│  │  │ 5. Conv2d(1×1): [B,64] → [B,3,256,256]                  │          │
│  │  │ 6. Tanh: → encoded_image [B,3,256,256] range:[-1,1]     │          │
│  │  └──────────────────────────────────────────────────────────┘          │
│  │                      │                                                │
│  │                      ▼                                                │
│  │            encoded_image [B,3,256,256]                                │
│  │            residual_penalty (scalar)                                   │
│  │                                                                        │
│  ├───────────────────────────────────────────────────────────────────────┤
│  │                                                                        │
│  │  NOISER (随机选择一种噪声)                                              │
│  │  ┌──────────────────────────────────────────────────────────┐          │
│  │  │ Input: (encoded_image, cover_image)                       │          │
│  │  │ → 随机选择: Identity/Crop/Cropout/Dropout/Resize/JPEG/Quant │        │
│  │  │ Output: (noised_encoded, noised_cover)                    │          │
│  │  └──────────────────────────────────────────────────────────┘          │
│  │         │                    │                                        │
│  │         ▼                    ▼                                        │
│  │  noised_encoded          noised_cover                                 │
│  │  [B,3,256,256]           [B,3,256,256]                                │
│  │                                                                        │
│  ├───────────────────────────────────────────────────────────────────────┤
│  │                                                                        │
│  │  DECODER                                                               │
│  │  ┌──────────────────────────────────────────────────────────┐          │
│  │  │ ConvBNRelu×8 + ConvBNRelu(64→30) + AdaptiveAvgPool      │          │
│  │  │ + Linear(30→30)                                          │          │
│  │  │ → message_pred [B,30] (raw logits)                       │          │
│  │  └──────────────────────────────────────────────────────────┘          │
│  │                                                                        │
│  ├───────────────────────────────────────────────────────────────────────┤
│  │                                                                        │
│  │  LOSS COMPUTATIONS                                                     │
│  │                                                                        │
│  │  ├─ Visual Loss ──────────────────────────────────────────────────┐    │
│  │  │ L_pixel = MSE(encoded, cover)                                  │    │
│  │  │ L_ssim = 1 - SSIM(encoded, cover)                              │    │
│  │  │ L_perceptual = L1(VGG(encoded), VGG(cover))                   │    │
│  │  │ L_visual = 1.0*L_pixel + 2.0*L_ssim + 0.3*L_perceptual        │    │
│  │  │ L_visual_noised = 同上(noised_encoded vs cover)                │    │
│  │  └────────────────────────────────────────────────────────────────┘    │
│  │                                                                        │
│  │  ├─ Message Loss ─────────────────────────────────────────────────┐    │
│  │  │ L_message = BCEWithLogits(message_pred, message)               │    │
│  │  └────────────────────────────────────────────────────────────────┘    │
│  │                                                                        │
│  │  ├─ Adversarial Loss ─────────────────────────────────────────────┐    │
│  │  │ D_train: BCE(Stego(noised_cover)→1, Stego(noised_enc.detach)→0)│   │
│  │  │ L_adv = BCE(Stego(noised_enc)→1)  [生成器视角]                │    │
│  │  └────────────────────────────────────────────────────────────────┘    │
│  │                                                                        │
│  │  ├─ Privacy Loss (Phase 2 only) ──────────────────────────────────┐    │
│  │  │ 1. CLIP features:                                               │    │
│  │  │    f_cover = ContentRecognizer(cover).detach()  [B,512]        │    │
│  │  │    f_encoded = ContentRecognizer(encoded)       [B,512]        │    │
│  │  │ 2. DINO features:                                               │    │
│  │  │    dino_cover = DINOv2(cover)             [B,384]             │    │
│  │  │    dino_encoded = DINOv2(encoded)          [B,384]             │    │
│  │  │ 3. CLIP/DINO硬负样本 (from FeatureMemoryBank):                  │    │
│  │  │    clip_negatives = clip_memory_bank.get_hard_negatives(f_cover)│   │
│  │  │ 4. Attacker features (统一特征空间):                             │    │
│  │  │    att_cover = RetrievalAttacker(cover)      [B,512]           │    │
│  │  │    att_encoded = RetrievalAttacker(encoded)  [B,512]           │    │
│  │  │    feat_negatives = attacker.get_hard_negatives(att_cover)     │    │
│  │  │ 5. RetrievalEscapeLoss:                                         │    │
│  │  │    L_privacy = L_multi_feature + 0.5*L_triplet(att_space)      │    │
│  │  │    L_multi_feature = 1.0*L_clip_escape/redirect + 0.8*L_dino   │    │
│  │  │    L_triplet = escape(att) + redirect(att) + control(att)      │    │
│  │  └────────────────────────────────────────────────────────────────┘    │
│  │                                                                        │
│  │  ├─ Total Loss ────────────────────────────────────────────────────┐    │
│  │  │ L_total = 20*(L_visual + 0.5*L_visual_noised)                   │    │
│  │  │        + 10*L_message + 5*L_adv + privacy*L_privacy            │    │
│  │  │        + 0.3*L_residual                                         │    │
│  │  │  (privacy 渐进: 1.0→3.0→10.0)                                  │    │
│  │  └────────────────────────────────────────────────────────────────┘    │
│  │                                                                        │
│  ├───────────────────────────────────────────────────────────────────────┤
│  │                                                                        │
│  │  OPTIMIZER STEPS                                                       │
│  │  1. optimizer_encoder_decoder.zero_grad()                              │
│  │  2. L_total.backward()                                                │
│  │  3. clip_grad_norm(enc+dec params, max_norm=1.0)                      │
│  │  4. optimizer_encoder_decoder.step()                                   │
│  │  5. [If phase>=2]: Update attacker with InfoNCE loss                  │
│  │  6. Update memory banks (attacker + CLIP + DINO)                      │
│  │                                                                        │
│  └───────────────────────────────────────────────────────────────────────┘
│                                                                         │
│  输出                                                                     │
│  losses dict: {L_pixel, L_ssim, L_perceptual, L_visual, L_message,      │
│               L_adv, L_privacy, L_total, bit_accuracy, ...}             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 14. 使用指南

### 14.1 环境依赖

```
torch>=2.0.0
torchvision>=0.15.0
ftfy
regex
tqdm
Pillow
tensorboard
numpy
timm
git+https://github.com/openai/CLIP.git
```

### 14.2 启动训练

```bash
# 新建实验（基础训练，无噪声）
python train.py new --data-dir /path/to/dataset --name my_experiment

# 新建实验（带JPEG压缩噪声）
python train.py new --data-dir /path/to/dataset --name exp_with_noise \
    --noise "jpeg"

# 新建实验（组合多种噪声）
python train.py new --data-dir /path/to/dataset --name exp_combined \
    --noise "cropout((0.55,0.6),(0.55,0.6))+dropout((0.9,0.95))"

# 断点续训
python train.py continue --folder ./experiments/my_experiment
```

### 14.3 运行验证

```bash
# 完整验证（编码→解码→检测→隐私评估）
python validation_script.py \
    -c ./experiments/my_experiment/checkpoints/checkpoint.pth \
    -i ./test_image.jpg \
    -m "101000100110100101100011100010" \
    --mode full \
    -o ./output

# 仅判别器检测
python validation_script.py -c checkpoint.pth -i image.png --mode discriminator

# 仅解码消息
python validation_script.py -c checkpoint.pth -i stego.png --mode decode

# 相似度判断（需提供原图和隐写图）
python validation_script.py -i encoded_image.png --cover cover_image.png --mode similarity
```

### 14.4 监控训练

```bash
tensorboard --logdir experiments/my_experiment/tensorboard
```

TensorBoard中可查看的标量：
- `Phase{1,2}/Train/{L_total, L_visual, L_message, L_adv, L_privacy, ...}`
- `Phase{1,2}/Val/{L_total, L_visual, L_message, L_adv, L_privacy, ...}`

### 14.5 噪声层配置语法

| 命令 | 格式 | 说明 |
|------|------|------|
| identity | `identity` | 无操作 |
| cropout | `cropout((hmin,hmax),(wmin,wmax))` | 裁剪替换 |
| crop | `crop((hmin,hmax),(wmin,wmax))` | 随机裁剪 |
| dropout | `dropout((keep_min,keep_max))` | 像素丢弃 |
| resize | `resize((min_ratio,max_ratio))` | 随机缩放 |
| jpeg | `jpeg` | JPEG压缩 |
| quant | `quant` | 量化 |
| 组合 | `cmd1+cmd2+...` | 用+号连接，每次随机选一种 |

---

## 附录A：关键超参数速查表

| 类别 | 参数 | 默认值 | 作用 |
|------|------|--------|------|
| 图像 | image_size | 256 | 输入分辨率 |
| 消息 | message_length | 30 | 隐藏比特数 |
| 训练 | batch_size | 8 | 批次大小 |
| LR | lr_encoder/decoder | 1e-3 | 生成器学习率 |
| LR | lr_steganalyzer | 1e-3 | 判别器学习率 |
| LR | attacker (隐含) | 5e-4 | 攻击者学习率 (lr_encoder*0.5) |
| λ | lambda_pixel | 1.0 | MSE权重 |
| λ | lambda_ssim | 2.0 | SSIM权重 |
| λ | lambda_perceptual | 0.3 | VGG感知权重 |
| λ | lambda_message | 1.0 | 消息损失权重 |
| λ | lambda_adv | 0.1 | 对抗损失基础权重 |
| λ | lambda_privacy | 0.3 | 隐私损失基础权重 |
| 隐私 | privacy_margin | 0.5 | 隐私安全边界 |
| 隐私 | clip_weight | 1.0 | CLIP特征逃逸权重 |
| 隐私 | dino_weight | 0.5 (Config) / 0.8 (Trainer) | DINOv2特征逃逸权重 |
| 隐私 | triplet_weight | 0.15 (Config) / 0.5 (Trainer) | 三元组损失权重 |
| 隐私 | triplet_margin | 0.5 (Config) / 0.3 (Trainer) | 三元组margin |
| 重定向 | use_clip_redirect | True | 启用有向重定向 |
| 重定向 | clip_redirect_weight | 0.5 | CLIP重定向权重 |
| 重定向 | dino_redirect_weight | 0.3 | DINO重定向权重 |
| 重定向 | clip_redirect_margin | 0.6 | 重定向margin |
| 重定向 | clip_max_target_sim | 0.75 | 最大允许相似度 |
| 攻击 | attacker_feature_dim | 512 | 攻击特征维度 |
| 攻击 | attacker_memory_size | 1000 | 攻击者记忆库容量 |
| 攻击 | attacker_temperature | 0.05 | 温度系数 |
| 攻击 | attacker_weight | 0.5 | 攻击者逃逸权重 |
| 攻击 | hard_negative_k | 5 | 硬负样本数 |
| 攻击 | memory_warmup_batches | 50 | 记忆库预热批次数 |
| 阶段 | P1 epochs | 300 | 基础+渐进对抗训练 |
| 阶段 | P2 epochs | 100 | 隐私保护训练 |
| 其他 | discriminator_loss_threshold | 0.1 | 判别器暂停阈值 |
| 其他 | num_workers | 4 | 数据加载线程 |
| 其他 | save_interval | 10 | 检查点保存间隔 |
| 其他 | seed | 42 | 随机种子 |

## 附录B：模型参数量汇总

| 模型 | 近似参数量 | 可训练 |
|------|-----------|--------|
| Encoder (UNet-style) | ~316K | ✓ |
| Decoder (CNN) | ~278K | ✓ |
| Steganalyzer (CNN+SN) | ~2.5M | ✓ |
| ContentRecognizer (CLIP ViT-B/32) | ~150M | ✗ (冻结) |
| RetrievalAttacker | ~2.5M | ✓ |
| Memory Bank | ~2M (1000×512×4B) | buffer only |
| DINOv2 (可选) | ~22M | ✗ (冻结) |
| **总可训练** | **~5.6M** | |

## 附录C：术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| 隐写 | Steganography | 将秘密信息隐藏在载体中的技术 |
| 载体图像 | Cover Image | 用于承载秘密信息的原始图像 |
| 隐写图像 | Stego/Encoded Image | 嵌入了秘密信息的图像 |
| 编码器 | Encoder | 将消息嵌入载体图像的网络 |
| 解码器 | Decoder | 从图像中提取消息的网络 |
| 隐写分析器 | Steganalyzer | 检测图像是否含隐写信息的分类器 |
| 谱归一化 | SpectralNorm | 限制权重矩阵谱范数的正则化技术 |
| SSIM | Structural Similarity | 结构相似性指数 |
| 感知损失 | Perceptual Loss | 基于预训练网络特征的距离度量 |
| 特征逃逸 | Feature Escape | 在特征空间中拉开原始/编码图像的距离 |
| 有向重定向 | Directed Redirection | 将编码图像特征拉向无关负样本 |
| 硬负样本 | Hard Negative | 与查询样本最相似的负样本 |
| 记忆库 | Memory Bank | 存储历史特征向量的循环缓冲区 |
| Softplus Loss | | log1p(exp(x)) 形式的平滑损失函数 |
| Triplet Loss | | 锚点-正例-负例三元组约束损失 |
| InfoNCE | Information Noise Contrastive Estimation | 对比学习损失函数 |
| DCT | Discrete Cosine Transform | 离散余弦变换 |
| CLIP | Contrastive Language-Image Pre-training | OpenAI的图文对比学习模型 |
| DINOv2 | Self-distilled Vision Transformers | Meta的自监督视觉模型 |
| GAN | Generative Adversarial Network | 生成对抗网络 |
| Hinge Loss | | max(0, margin - score) 形式的损失函数 |

---

*本文档基于 PrivSteg 项目源码生成，涵盖全部模块的实现细节与技术原理。*
