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
- [7. 损失函数模块 (losses/)](#7-损失函数模块-losses)
  - [7.1 SSIM损失 (SSIMLoss)](#71-ssim损失-ssimloss)
  - [7.2 特征重定向损失 (FeatureRedirectionLoss)](#72-特征重定向损失-featureredirectionloss)
- [8. 噪声层模块 (noise_layers/)](#8-噪声层模块-noise_layers)
  - [8.1 噪声组合器 (Noiser)](#81-噪声组合器-noiser)
  - [8.2 Identity恒等映射](#82-identity恒等映射)
  - [8.3 随机裁剪 (Crop)](#83-随机裁剪-crop)
  - [8.4 裁剪填充 (Cropout)](#84-裁剪填充-cropout)
  - [8.5 像素丢弃 (Dropout)](#85-像素丢弃-dropout)
  - [8.6 缩放变换 (Resize)](#86-缩放变换-resize)
  - [8.7 JPEG压缩 (JpegCompression)](#87-jpeg压缩-jpegcompression)
  - [8.8 量化噪声 (Quantization)](#88-量化噪声-quantization)
- [9. 训练系统 (trainer.py & train.py)](#9-训练系统-trainerpy--trainpy)
  - [9.1 Trainer类详解](#91-trainer类详解)
  - [9.2 两阶段训练策略](#92-两阶段训练策略)
  - [9.3 编码器头部冻结策略](#93-编码器头部冻结策略)
  - [9.4 解码器恢复机制](#94-解码器恢复机制)
  - [9.5 训练流程 (train.py)](#95-训练流程-trainpy)
  - [9.6 记忆库预热](#96-记忆库预热)
- [10. 验证脚本 (validation_script.py)](#10-验证脚本-validation_scriptpy)
- [11. CLIP与DINOv2视觉模型](#11-clip与dinov2视觉模型)
- [12. 核心算法原理深度剖析](#12-核心算法原理深度剖析)
  - [12.1 双残差隐写编码原理](#121-双残差隐写编码原理)
  - [12.2 对抗训练原理](#122-对抗训练原理)
  - [12.3 隐私保护的多模态特征逃逸机制](#123-隐私保护的多模态特征逃逸机制)
  - [12.4 记忆库驱动的硬负样本挖掘](#124-记忆库驱动的硬负样本挖掘)
  - [12.5 渐进逃逸边界机制](#125-渐进逃逸边界机制)
- [13. 完整数据流图](#13-完整数据流图)
- [14. 使用指南](#14-使用指南)

---

## 1. 项目概述

**PrivSteg（Privacy-Preserving Steganography）** 是一个基于深度学习的**隐私保护图像隐写系统**。该项目的核心目标是在载体图像（Cover Image）中嵌入秘密消息（Secret Message），同时满足以下四个关键约束：

| 约束维度 | 技术手段 | 目标 |
|---------|---------|------|
| **不可感知性** | 像素级重建 + SSIM | 编码后图像与原图视觉上无法区分 |
| **鲁棒性** | 多种噪声层模拟真实信道干扰 | 经过噪声攻击后仍能准确提取消息 |
| **抗检测性** | GAN对抗训练 + SpectralNorm隐写分析器 | 隐写图像能骗过隐写检测器 |
| **隐私保护** | CLIP/DINO特征逃逸 + 有向重定向 | 编码前后语义特征空间距离足够大 |

### 核心创新点

1. **双残差编码器架构**：编码器采用消息头（msg_head）和隐私头（prv_head）的双头设计，分别生成预算约束的消息残差和隐私残差，实现消息嵌入与隐私保护的解耦
2. **两阶段渐进式训练**：先学习基础隐写能力 + 渐进对抗 → 再引入隐私保护，配合头部冻结策略确保阶段间能力不退化
3. **预算约束残差**：通过 `msg_budget` 和 `prv_budget` 参数配合 `tanh` 激活，严格控制残差幅度
4. **多模态特征逃逸**：同时利用CLIP（ViT-B/32）和DINOv2（ViT-S/14）两种预训练模型的语义特征空间进行隐私约束
5. **CLIP/DINO 有向重定向**：不仅将编码图像推离原图，还通过记忆库硬负样本将其"拉向"其他无关图像，实现语义伪装
6. **渐进逃逸边界**：逃逸margin从0.05线性增长到0.30，避免隐私损失突然激增导致训练崩溃
7. **解码器恢复机制**：当Phase 2比特准确率下降时，在干净消息残差上重新训练解码器，确保消息提取能力不退化

---

## 2. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PrivSteg 系统架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────┐           │
│   │ Cover    │────▶│   Encoder    │────▶│ Encoded      │           │
│   │ Image    │     │ (Dual-Head)  │     │ Image        │           │
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
│   │  ┌─────────────┐  ┌─────────────┐                        │     │
│   │  │ CLIP ViT-B/32│  │ DINOv2 ViT-S│                       │     │
│   │  │ (冻结)        │  │ /14 (冻结)   │                       │     │
│   │  └──────┬───────┘  └──────┬──────┘                        │     │
│   │         │                 │                                │     │
│   │         └────────┬────────┘                                │     │
│   │                  ▼                                         │     │
│   │         FeatureRedirectionLoss                             │     │
│   │    (多模态特征逃逸 + Softplus + 有向重定向)                  │     │
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
│   ├── __init__.py               # 模型统一导出：Encoder, Decoder, Steganalyzer, ContentRecognizer
│   ├── encoder.py                # 编码器：双残差头（msg_head + prv_head）消息/隐私嵌入网络
│   ├── decoder.py                # 解码器：CNN消息提取网络
│   ├── steganalyzer.py           # 隐写分析器：CNN+SpectralNorm二分类器
│   └── content_recognizer.py     # 内容识别器：CLIP ViT-B/32 特征提取（冻结）
│
├── losses/
│   ├── __init__.py               # 损失统一导出：SSIMLoss, FeatureRedirectionLoss
│   ├── ssim_loss.py              # 结构相似性(SSIM)损失
│   └── advanced_privacy_loss.py  # 特征重定向损失（多模态逃逸 + 有向重定向）
│
├── noise_layers/
│   ├── noiser.py                 # 噪声层组合器（随机选择激活）
│   ├── identity.py               # 恒等映射（无操作）
│   ├── crop.py                   # 随机裁剪
│   ├── cropout.py                # 裁剪替换（局部区域用原图替换）
│   ├── dropout.py                # 随机像素丢弃
│   ├── resize.py                 # 随机缩放
│   ├── jpeg_compression.py       # JPEG压缩（DCT域量化）
│   └── quantization.py           # 傅里叶四舍五入量化
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
- `image_size=256`：平衡计算资源消耗与特征表达能力。编码器的消息扩展操作需要将1×30的消息张量扩展到256×256的空间维度。
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

### 4.4 两阶段训练轮次

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `num_epochs_phase1` | `130` | Phase 1: 基础隐写 + 渐进对抗训练 |
| `num_epochs_phase2` | `150` | Phase 2: 隐私保护训练 |

总计 **280 epochs** 的训练过程。

### 4.5 损失权重参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `lambda_pixel` | `1.0` | MSE像素损失权重 |
| `lambda_ssim` | `2.0` | SSIM结构相似性损失权重 |
| `lambda_message` | `1.0` | 消息提取BCE损失权重 |
| `lambda_adv` | `0.1` | 对抗损失基础权重（Phase 1中渐进放大） |
| `lambda_privacy` | `0.3` | 隐私损失基础权重（Phase 2中渐进放大） |

**视觉损失组成公式**：

$$L_{visual} = \lambda_{pixel} \cdot L_{pixel} + \lambda_{ssim} \cdot L_{ssim}$$

代入默认值：

$$L_{visual} = 1.0 \cdot L_{MSE} + 2.0 \cdot L_{SSIM}$$

两种视觉损失的权重设计体现了**多尺度图像质量评估**的思想，且SSIM权重较高以优先保证结构相似性：
- **MSE**：像素级绝对误差
- **SSIM**：结构级相似性（亮度、对比度、结构）

### 4.6 残差预算参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `msg_budget` | `0.04` | 消息残差预算，控制消息嵌入的最大幅度 |
| `prv_budget` | `0.06` | 隐私残差预算，控制隐私扰动的最大幅度 |

**预算约束机制**：编码器的双头输出通过 `budget * tanh(raw_output)` 进行约束，`tanh` 将输出限制在 `[-1, 1]`，乘以预算系数后残差幅度被严格控制在 `[-budget, budget]` 范围内。

- `msg_budget=0.04`：消息残差最大幅度为像素值范围的4%（约±0.04），确保视觉不可感知
- `prv_budget=0.06`：隐私残差最大幅度为6%，略大于消息预算，因为隐私扰动需要更大的特征空间偏移

### 4.7 隐私保护参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `privacy_margin_start` | `0.05` | Phase 2起始逃逸margin |
| `privacy_margin_end` | `0.30` | Phase 2结束逃逸margin |
| `clip_weight` | `1.0` | CLIP特征逃逸损失权重 |
| `dino_weight` | `1.0` | DINOv2特征逃逸损失权重 |
| `dino_model_path` | `'./models/dinov2_vits14_pretrain.pth'` | DINOv2模型权重路径 |

**渐进逃逸边界**：`privacy_margin_start` 和 `privacy_margin_end` 定义了逃逸margin的线性插值范围，通过 `get_escape_margin()` 方法实现渐进式隐私约束。

### 4.8 CLIP/DINO 有向重定向参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `use_clip_redirect` | `True` | 是否启用CLIP/DINO有向重定向 |
| `clip_memory_size` | `1000` | CLIP特征记忆库容量 |
| `clip_hard_negative_k` | `5` | CLIP硬负样本Top-K数量 |
| `clip_redirect_margin` | `0.4` | CLIP重定向margin |
| `clip_max_target_sim` | `0.65` | CLIP最大允许相似度上限（防止特征坍塌） |
| `clip_redirect_weight` | `0.3` | CLIP重定向损失权重 |
| `dino_redirect_weight` | `0.4` | DINO重定向损失权重 |
| `dino_redirect_margin` | `0.4` | DINO重定向margin |
| `dino_max_target_sim` | `0.65` | DINO最大允许相似度上限 |

### 4.9 检索攻击者参数（预留）

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `use_retrieval_attacker` | `False` | 是否启用检索攻击者（当前已禁用） |
| `attacker_feature_dim` | `512` | 攻击者输出特征维度（预留） |
| `attacker_memory_size` | `1000` | 攻击者记忆库容量（预留） |
| `memory_warmup_batches` | `50` | 记忆库预热批次数 |

### 4.10 恢复机制参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `bit_accuracy_threshold` | `0.90` | 比特准确率阈值，低于此值触发解码器恢复 |
| `recovery_steps` | `10` | 恢复训练的批次数 |

### 4.11 其他参数

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
| `discriminator_loss_threshold` | `0.4` | 判别器损失阈值，低于此值暂停判别器训练 |

### 4.12 阶段权重函数 `get_phase_weights(phase, epoch)`

这是整个系统的**核心调度逻辑**，决定了不同训练阶段的损失权重分配。与早期版本不同，当前实现采用**两阶段**训练，且部分权重在阶段内**渐进变化**。

#### Phase 1 — 基础隐写 + 渐进对抗训练（130 epochs）

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

**adv 渐进策略**：将Phase 1的130个epoch均分为3段：
- 第1-43 epoch：`adv = lambda_adv` (默认0.1)
- 第44-86 epoch：`adv = lambda_adv * 2` (默认0.2)
- 第87-130 epoch：`adv = lambda_adv * 3` (默认0.3)

**目标**：让编码器先学会将消息嵌入图像且保持视觉质量，解码器学会准确提取消息；在阶段后半段逐步引入轻量级对抗约束，为Phase 2做准备。

#### Phase 2 — 隐私保护训练（150 epochs）

```python
{
    'visual': 1.0,          # 降低视觉权重（Phase 1已充分学习）
    'message': 渐进,         # 15.0 -> 12.0 -> 10.0
    'adv': 2.0,             # 固定对抗权重
    'privacy': 渐进,         # 3.0 -> 6.0 -> 10.0
    'noised': 0.3,          # 降低含噪声损失权重
    'residual_penalty': 0.2  # 降低残差惩罚
}
```

**message 渐进策略**：将Phase 2的150个epoch均分为3段：
- 第1-50 epoch：`message = 15.0`（高权重保护消息能力）
- 第51-100 epoch：`message = 12.0`
- 第101-150 epoch：`message = 10.0`

**privacy 渐进策略**：
- 第1-50 epoch：`privacy = 3.0`
- 第51-100 epoch：`privacy = 6.0`
- 第101-150 epoch：`privacy = 10.0`

**目标**：在保证隐写能力和对抗性的基础上，由浅入深地引入隐私保护约束，使编码图像在语义特征空间中逐渐远离原始图像，避免隐私损失突然激增导致训练崩溃。消息权重从15.0逐步降至10.0，与隐私权重从3.0升至10.0形成此消彼长的关系，确保在增强隐私保护时消息提取能力不退化。

### 4.13 渐进逃逸边界函数 `get_escape_margin(phase, epoch)`

```python
def get_escape_margin(self, phase, epoch):
    if phase == 1:
        return 0.0
    total = self.num_epochs_phase2
    progress = min(1.0, (epoch - 1) / max(1, total - 1))
    return self.privacy_margin_start + (self.privacy_margin_end - self.privacy_margin_start) * progress
```

- **Phase 1**：返回 `0.0`，不进行隐私逃逸约束
- **Phase 2**：线性插值从 `privacy_margin_start`（0.05）到 `privacy_margin_end`（0.30）

**直觉解释**：逃逸margin决定了编码图像特征需要与原图特征保持的最小距离。较小的margin意味着更严格的逃逸要求（需要更低的相似度）。从0.05开始，模型只需要轻微地推离原图特征；随着训练推进，margin增大到0.30，要求特征之间保持更大的距离。这种渐进策略避免了训练初期过强的隐私约束导致视觉质量崩塌。

### 4.14 隐私噪声比例函数 `get_prv_noise_ratio(epoch)`

```python
def get_prv_noise_ratio(self, epoch):
    total = self.num_epochs_phase1
    warmup_end = total * 0.3
    ramp_end = total * 0.7
    if epoch <= warmup_end:
        return 0.0
    elif epoch <= ramp_end:
        return (epoch - warmup_end) / (ramp_end - warmup_end)
    else:
        return 1.0
```

此函数仅在 **Phase 1** 中使用，控制编码器中虚拟隐私噪声（dummy_prv）的比例：

- **前30% epochs**（epoch 1-39）：`ratio = 0.0`，不添加虚拟隐私噪声
- **30%-70% epochs**（epoch 40-91）：`ratio` 从0线性增长到1
- **后30% epochs**（epoch 92-130）：`ratio = 1.0`，全量虚拟隐私噪声

**设计目的**：Phase 1中隐私头（prv_head）被冻结，但为了让编码器在Phase 2开始时不至于对隐私残差完全陌生，通过添加与 `prv_budget` 同尺度的随机噪声来"预适应"共享特征层，使Phase 2解冻prv_head时过渡更平滑。

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

   归一化到 `[-1, 1]` 区间，与编码器输出范围一致。

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

编码器采用**双残差头（Dual-Residual Head）架构**，核心思想是将消息嵌入和隐私保护解耦为两个独立的残差路径：

```
Cover Image [B, 3, 256, 256]     Message [B, 30]
       │                              │
       ▼                              ▼
┌──────────────────────┐     message expand: [B,30] → [B,30,256,256]
│ ConvBNRelu(3→128)    │  ← 第1层：初步特征提取
├──────────────────────┤
│ ConvBNRelu(128→128)×15│ ← 第2-16层：深层特征学习（128通道）
├──────────────────────┤
│ Concat: [msg; feat; cover] │  ← 特征拼接
│ ConvBNRelu(161→128)  │  ← 融合层：(30+128+3)=161通道 → 128通道
├──────────────────────┤
│        shared features [B, 128, 256, 256]                        │
├──────────┬───────────┤
│  msg_head│  prv_head │  ← 双头分支
│  Conv(128→128,k=3)  │  Conv(128→128,k=3)
│  +BN+ReLU           │  +BN+ReLU
│  Conv(128→128,k=3)  │  Conv(128→128,k=3)
│  +BN+ReLU           │  +BN+ReLU
│  Conv(128→3,k=1)    │  Conv(128→3,k=1)
├──────────┴───────────┤
│ msg_residual = 0.04*tanh(msg_head)  ← 消息残差（预算约束）       │
│ prv_residual = 0.06*tanh(prv_head)  ← 隐私残差（预算约束）       │
├──────────────────────┤
│ Phase 1: encoded = cover + msg_residual + dummy_prv_noise        │
│ Phase 2: encoded = cover + msg_residual + prv_residual           │
├──────────────────────┤
│ torch.clamp(encoded, -1, 1)  ← 输出范围[-1, 1]                  │
└──────────────────────┘
       │
       ▼
Encoded Image [B, 3, 256, 256]
Returns: (encoded_image, residual_penalty, msg_residual, prv_residual)
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
concat = torch.cat([expanded_message, features, cover_image], dim=1)
# 拼接后通道数: 30(message) + 128(features) + 3(cover) = 161
shared = self.after_concat_layer(concat)  # 161 → 128
```

拼接三种信息：
1. **expanded_message**：原始消息信号
2. **features**：经过16层卷积处理的图像特征（128通道）
3. **cover_image**：原始载体图像（残差连接，保留低频信息）

**双头架构**：

```python
self.msg_head = nn.Sequential(
    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 3, kernel_size=1)
)

self.prv_head = nn.Sequential(
    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 3, kernel_size=1)
)
```

两个头结构完全对称，各包含2个3×3卷积层和1个1×1输出层。关键区别在于：
- **msg_head**：生成消息嵌入残差，Phase 1解冻训练，Phase 2冻结
- **prv_head**：生成隐私扰动残差，Phase 1冻结，Phase 2解冻训练

**预算约束残差**：

```python
msg_residual = self.msg_budget * torch.tanh(msg_head(shared))  # budget=0.04
prv_residual = self.prv_budget * torch.tanh(prv_head(shared))  # budget=0.06
```

`tanh` 将原始输出限制在 `[-1, 1]`，乘以预算系数后：
- `msg_residual ∈ [-0.04, 0.04]`
- `prv_residual ∈ [-0.06, 0.06]`

这种设计确保了残差幅度有严格上界，不会因为训练过程中的梯度更新而无限增长。

**阶段感知前向传播**：

```python
def forward(self, cover_image, message, phase=1, prv_noise_ratio=0.0):
    # ... 共享层计算 ...
    
    if phase == 2 and not self.msg_head[0].weight.requires_grad:
        with torch.no_grad():
            msg_residual = self.msg_head(shared)
            msg_residual = self.msg_budget * torch.tanh(msg_residual)
    else:
        msg_residual = self.msg_head(shared)
        msg_residual = self.msg_budget * torch.tanh(msg_residual)
    
    if phase == 1:
        dummy_prv = self.prv_budget * prv_noise_ratio * torch.randn_like(cover_image)
        encoded_image = cover_image + msg_residual + dummy_prv
        residual_penalty = torch.mean(msg_residual ** 2) + torch.mean(dummy_prv ** 2)
    else:
        prv_residual = self.prv_head(shared)
        prv_residual = self.prv_budget * torch.tanh(prv_residual)
        encoded_image = cover_image + msg_residual + prv_residual
        residual_penalty = torch.mean(msg_residual ** 2) + torch.mean(prv_residual ** 2)
    
    encoded_image = torch.clamp(encoded_image, -1, 1)
    return encoded_image, residual_penalty, msg_residual, prv_residual
```

**关键设计点**：
1. **Phase 1**：prv_head被冻结，使用随机高斯噪声代替隐私残差（`dummy_prv`），其幅度由 `prv_noise_ratio` 控制
2. **Phase 2**：当msg_head被冻结时，使用 `torch.no_grad()` 计算消息残差，避免不必要的梯度计算
3. **输出裁剪**：使用 `torch.clamp` 而非 `tanh` 进行最终范围约束，因为残差已经通过预算约束限制了幅度
4. **返回值**：除了编码图像和残差惩罚外，还返回 `msg_residual` 和 `prv_residual`，供解码器恢复机制使用

#### 6.1.3 参数量估算

| 层 | 输入通道 | 输出通道 | 卷积核 | 参数量 |
|---|---------|---------|--------|--------|
| conv_layers[0] | 3 | 128 | 3×3 | 3×128×9 = 3,456 |
| conv_layers[1-15] | 128 | 128 | 3×3 | 15×128×128×9 = 2,211,840 |
| after_concat_layer | 161 | 128 | 3×3 | 161×128×9 = 185,472 |
| msg_head Conv1 | 128 | 128 | 3×3 | 128×128×9 = 147,456 |
| msg_head Conv2 | 128 | 128 | 3×3 | 128×128×9 = 147,456 |
| msg_head Conv3 | 128 | 3 | 1×1 | 128×3×1 = 384 |
| prv_head Conv1 | 128 | 128 | 3×3 | 128×128×9 = 147,456 |
| prv_head Conv2 | 128 | 128 | 3×3 | 128×128×9 = 147,456 |
| prv_head Conv3 | 128 | 3 | 1×1 | 128×3×1 = 384 |
| **总计** | | | | **~2,991,360** |

> 注：以上未计入BatchNorm参数（每个BN层有2个可训练参数：weight和bias）。

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

#### 6.2.3 参数量估算

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
│ Linear(512→256)      │
│ LeakyReLU(0.2)       │
│ Linear(256→1)        │  → logit output
└──────────────────────┘
       │
       ▼
Output: [B, 1]  (正值为stego, 负值为cover)
```

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

#### 6.3.5 双层全连接头

```python
self.fc_layers = nn.Sequential(
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 1)
)
```

相比单层线性输出，双层FC提供了更强的非线性判别能力，中间的LeakyReLU激活使网络能学习更复杂的决策边界。

---

### 6.4 内容识别器 (ContentRecognizer)

**文件位置**: [content_recognizer.py](models/content_recognizer.py)

#### 6.4.1 架构概述

内容识别器是对 **OpenAI CLIP ViT-B/32** 模型的封装，用于提取图像的**高层语义特征向量**：

```python
class ContentRecognizer(nn.Module):
    def __init__(self, device='cuda', model_name='ViT-B/32'):
        self.clip_model, _ = clip.load(model_name, device=self._device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
```

#### 6.4.2 前向传播流程

```python
def forward(self, image):
    image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    image_resized = (image_resized + 1) / 2
    image_resized = (image_resized - self.clip_mean) / self.clip_std
    image_features = self.clip_model.encode_image(image_resized)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features  # shape: [B, 512]
```

**输出特征维度**：ViT-B/32 的视觉编码器输出 **512维** 特征向量。

#### 6.4.3 文本编码方法

```python
def encode_text(self, text):
    tokens = clip.tokenize(text, truncate=True).to(self._device)
    with torch.no_grad():
        text_features = self.clip_model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features
```

`encode_text()` 方法提供了CLIP文本编码器的访问接口，可用于图文跨模态相似度评估等扩展任务。

#### 6.4.4 为什么使用CLIP？

CLIP（Contrastive Language-Image Pre-training）通过4亿图文对的大规模对比学习训练，具有以下优势：
- **语义丰富性**：512维特征空间编码了丰富的语义信息（物体、场景、属性等）
- **通用性**：零样本迁移能力强，无需针对特定数据集微调
- **冻结使用**：无需训练，仅作特征提取器，不增加可训练参数

#### 6.4.5 设备迁移处理

```python
def to(self, *args, **kwargs):
    module = super().to(*args, **kwargs)
    if len(args) > 0:
        self._device = args[0] if isinstance(args[0], torch.device) else torch.device(args[0])
    elif 'device' in kwargs:
        self._device = kwargs['device'] if isinstance(kwargs['device'], torch.device) else torch.device(kwargs['device'])
    self.clip_model = self.clip_model.to(self._device)
    return module
```

由于CLIP模型是外部实例化的，需要重写 `to()` 方法确保设备同步。

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
_2D_window = _1D_window.mm(_1D_window.t())
```

**滑动窗口卷积计算局部统计量**：

```python
mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

sigma1_sq = F.conv2d(img1*img1, window, ...) - mu1_sq
sigma2_sq = F.conv2d(img2*img2, window, ...) - mu2_sq
sigma12 = F.conv2d(img1*img2, window, ...) - mu1_mu2
```

使用 `groups=channel` 对每个通道独立应用高斯窗口，高效计算局部统计量。

**参数设置**：
- `window_size=11`：11×11的高斯窗口，覆盖足够的邻域上下文
- `size_average=True`：对所有空间位置取平均，返回标量损失

---

### 7.2 特征重定向损失 (FeatureRedirectionLoss)

**文件位置**: [advanced_privacy_loss.py](losses/advanced_privacy_loss.py)

这是系统中**唯一的隐私损失类**，整合了多模态特征逃逸和有向重定向功能。

#### 7.2.1 设计动机

单一模态的隐私保护可能存在盲区。FeatureRedirectionLoss支持**多模态联合特征逃逸**，并引入了**有向重定向（Directed Escape）**机制：

$$L_{total} = w_{clip} \cdot L_{escape}^{CLIP} + w_{redirect}^{clip} \cdot L_{redirect}^{CLIP} + w_{dino} \cdot L_{escape}^{DINO} + w_{redirect}^{dino} \cdot L_{redirect}^{DINO}$$

其中每个模态支持两种损失：
- **逃逸损失**（escape_loss）：使用Softplus loss将编码图像推离原图
- **重定向损失**（redirect_loss）：通过硬负样本将编码图像"拉向"其他无关图像

#### 7.2.2 类定义

```python
class FeatureRedirectionLoss(nn.Module):
    def __init__(self, clip_weight=1.0, dino_weight=0.5, escape_margin=0.5,
                 clip_redirect_weight=0.3, dino_redirect_weight=0.2,
                 clip_redirect_margin=0.3, dino_redirect_margin=0.3,
                 clip_max_target_sim=0.55, dino_max_target_sim=0.55):
```

**Trainer中的实际初始化参数**：

| 参数 | 类默认值 | Trainer实际值 | 说明 |
|------|---------|-------------|------|
| `clip_weight` | 1.0 | `cfg.clip_weight` (1.0) | CLIP逃逸损失权重 |
| `dino_weight` | 0.5 | `cfg.dino_weight` (1.0) | DINO逃逸损失权重 |
| `escape_margin` | 0.5 | `cfg.privacy_margin_start` (0.05) | 初始逃逸margin |
| `clip_redirect_weight` | 0.3 | `cfg.clip_redirect_weight` (0.3) | CLIP重定向权重 |
| `dino_redirect_weight` | 0.2 | `cfg.dino_redirect_weight` (0.4) | DINO重定向权重 |
| `clip_redirect_margin` | 0.3 | `cfg.clip_redirect_margin` (0.4) | CLIP重定向margin |
| `dino_redirect_margin` | 0.3 | `cfg.dino_redirect_margin` (0.4) | DINO重定向margin |
| `clip_max_target_sim` | 0.55 | `cfg.clip_max_target_sim` (0.65) | CLIP最大相似度 |
| `dino_max_target_sim` | 0.55 | `cfg.dino_max_target_sim` (0.65) | DINO最大相似度 |

#### 7.2.3 逃逸损失 (escape_loss)

$$L_{escape} = \text{mean}\left(\text{log1p}\left(\exp\left(\text{cos\_sim}(f_{encoded}, f_{cover}) - (1 - margin)\right)\right)\right)$$

```python
def escape_loss(self, feat_cover, feat_encoded, margin):
    sim = F.cosine_similarity(feat_encoded, feat_cover, dim=1)
    target = 1 - margin
    loss = torch.mean(torch.log1p(torch.exp(sim - target)))
    return loss, sim.mean().item()
```

**直觉解释**：
- 当 `cos_sim >> (1-margin)` 时：损失快速增长，强力推动特征分离
- 当 `cos_sim ≈ (1-margin)` 时：损失接近 `log(2) ≈ 0.693`
- 当 `cos_sim << (1-margin)` 时：损失趋近0，已经满足隐私要求

与Hinge Loss（`max(0, sim - target)`）相比，Softplus Loss处处可导，梯度更平滑，有利于优化收敛。

#### 7.2.4 重定向损失 (redirect_loss)

$$L_{redirect} = \text{mean}\left(\text{log1p}\left(\exp\left((1 - m_{redirect}) - \text{sim}(f_{encoded}, f_{hardest\_neg})\right)\right)\right)$$

$$L_{control} = \text{mean}\left(\text{relu}\left(\text{sim}(f_{encoded}, f_{hardest\_neg}) - s_{max}\right)\right)$$

```python
def redirect_loss(self, feat_encoded, hard_negatives, margin, max_target_sim):
    sim_to_neg = F.cosine_similarity(feat_encoded.unsqueeze(1), hard_negatives, dim=2)
    hardest_idx = sim_to_neg.argmax(dim=1)
    hardest_neg = hard_negatives[torch.arange(feat_encoded.size(0)), hardest_idx]
    sim_to_hardest = F.cosine_similarity(feat_encoded, hardest_neg, dim=1)
    redirect_target = 1 - margin
    redirect_loss = torch.mean(torch.log1p(torch.exp(redirect_target - sim_to_hardest)))
    control_loss = torch.mean(F.relu(sim_to_hardest - max_target_sim))
    return redirect_loss, control_loss
```

**关键设计**：
1. **硬负样本选择**：从Top-K负样本中选择与编码特征最相似的1个（hardest negative），而非随机选择
2. **重定向项**（Softplus Loss）：推动编码特征**靠近**最难的负样本（相似度 > `1-margin`）
3. **控制项**（Hinge Loss）：防止过度接近（相似度 < `max_target_sim`），避免特征坍塌

**注意**：重定向项也使用Softplus Loss而非Hinge Loss，确保梯度处处可导。

#### 7.2.5 前向传播

```python
def forward(self, clip_cover, clip_encoded, dino_cover=None, dino_encoded=None,
            clip_hard_negatives=None, dino_hard_negatives=None, escape_margin=None):
```

计算流程：
1. 计算CLIP逃逸损失 → 加权加入总损失
2. 如有CLIP硬负样本 → 计算CLIP重定向损失 + 控制损失 → 加权加入总损失
3. 如有DINO特征 → 计算DINO逃逸损失 → 加权加入总损失
4. 如有DINO硬负样本 → 计算DINO重定向损失 + 控制损失 → 加权加入总损失

返回 `(total_loss, loss_dict)`，其中 `loss_dict` 包含 `clip_escape`、`clip_redirect`、`clip_control`、`dino_escape`、`dino_redirect`、`dino_control`、`total_privacy_loss` 等，便于训练监控。

#### 7.2.6 模态互补性

| 模态 | 模型 | 特征维度 | 优势 |
|------|------|---------|------|
| CLIP | ViT-B/32 | 512 | 语义理解、图文对齐 |
| DINOv2 | ViT-S/14 | 384 | 自监督视觉特征、细粒度纹理 |

默认配置 `clip_weight=1.0`、`dino_weight=1.0`，两种模态等权联合使用提供多模态隐私保护。

#### 7.2.7 set_epoch 方法

```python
def set_epoch(self, epoch):
    self.current_epoch = epoch
```

每个epoch开始时由训练循环调用，记录当前epoch信息，可用于未来实现epoch感知的损失调度。

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
        return noised_and_cover
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
    filters = self.dct_conv_weights
    image_conv_channels = []
    for channel in range(image.shape[1]):
        image_ch = image[:, channel, :, :].unsqueeze(1)
        image_conv = F.conv2d(image_ch, filters, stride=8)
        image_conv = image_conv.permute(0, 2, 3, 1)
        image_conv = image_conv.view(B, H/8, W/8, 8, 8)
        image_conv = image_conv.permute(0, 1, 3, 2, 4)
        image_conv = image_conv.contiguous().view(B, H, W)
        image_conv_channels.append(image_conv)
    return torch.cat(image_conv_channels, dim=1)
```

**巧妙之处**：用 `stride=8` 的卷积实现分块DCT，避免了显式的循环。64个输出通道对应8×8块的64个DCT系数。后续的view/permute操作将系数重新排列为标准的块状布局。

#### 8.7.4 Zigzag量化掩码

```python
def get_jpeg_yuv_filter_mask(image_shape, window_size, keep_count):
    mask = np.zeros((window_size, window_size), dtype=np.uint8)
    index_order = sorted(
        ((x, y) for x in range(window_size) for y in range(window_size)),
        key=lambda p: (p[0]+p[1], -p[1] if (p[0]+p[1])%2 else p[1])
    )
    for i, j in index_order[0:keep_count]:
        mask[i, j] = 1
    return np.tile(mask, ...)
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

---

## 9. 训练系统 (`trainer.py` & `train.py`)

### 9.1 Trainer类详解

**文件位置**: [trainer.py](trainer.py)

#### 9.1.1 初始化流程

Trainer的 `__init__` 方法完成了整个系统的组装：

```
1. 创建TensorBoard日志记录器
2. 初始化四大核心模型:
   ├── Encoder (可训练, msg_budget=0.04, prv_budget=0.06)
   ├── Decoder (可训练)
   ├── Steganalyzer (可训练)
   └── ContentRecognizer/CLIP (冻结)
3. 创建优化器:
   ├── optimizer_encoder_decoder: Adam(encoder+decoder参数)
   └── optimizer_steganalyzer: Adam(steganalyzer参数)
4. 创建损失函数集合:
   ├── SSIMLoss (可微分SSIM)
   ├── MSELoss (像素重建)
   ├── BCEWithLogitsLoss (消息/对抗)
   └── FeatureRedirectionLoss (隐私保护)
5. 条件性创建高级组件:
   ├── DINOv2模型 (如 dino_weight > 0)
   ├── FeatureMemoryBank for CLIP (如 use_clip_redirect)
   │   └── clip_memory_bank (CLIP 512维特征)
   └── FeatureMemoryBank for DINO (如 use_clip_redirect && dino_weight > 0)
       └── dino_memory_bank (DINO 384维特征)
6. 解析并构建噪声层管道
```

#### 9.1.2 FeatureMemoryBank (通用特征记忆库)

**文件位置**: [trainer.py](trainer.py) 中的 `FeatureMemoryBank` 类

这是用于存储CLIP/DINO等冻结模型特征向量的**通用特征记忆库**：

```python
class FeatureMemoryBank:
    def __init__(self, feature_dim, memory_size=1000, device='cuda'):
        self.memory_bank = torch.zeros(memory_size, feature_dim, device=device)
        self.memory_ptr = 0
        self.memory_count = 0
```

**记忆库更新机制**：

```python
@torch.no_grad()
def update(self, features):
    features = F.normalize(features.float(), p=2, dim=1)
    batch_size = features.shape[0]
    ptr = self.memory_ptr
    if ptr + batch_size > self.memory_size:
        remaining = self.memory_size - ptr
        self.memory_bank[ptr:] = features[:remaining]
        self.memory_bank[:batch_size - remaining] = features[remaining:]
        self.memory_ptr = batch_size - remaining
    else:
        self.memory_bank[ptr:ptr + batch_size] = features
        self.memory_ptr = (ptr + batch_size) % self.memory_size
    self.memory_count = min(self.memory_count + batch_size, self.memory_size)
```

这是一个**循环队列（Ring Buffer）** 实现：
- 固定容量1000个特征向量
- 写入指针循环移动，新特征覆盖最旧的特征
- `@torch.no_grad()` 装饰器确保不产生计算图开销

**硬负样本挖掘**：

```python
def get_hard_negatives(self, features, k=5):
    if self.memory_count == 0:
        return None
    features = features.float()
    valid_memory = self.memory_bank[:self.memory_count]
    similarities = torch.mm(features, valid_memory.t())
    actual_k = min(k, self.memory_count)
    _, indices = similarities.topk(actual_k, dim=1)
    hard_negatives = valid_memory[indices]
    return hard_negatives  # shape: [B, k, feature_dim] 或 None
```

返回Top-K最相似的记忆特征作为硬负样本候选。

**状态保存/加载**：

```python
def state_dict(self):
    return {
        'memory_bank': self.memory_bank.cpu(),
        'memory_ptr': self.memory_ptr,
        'memory_count': self.memory_count
    }

def load_state_dict(self, state_dict):
    self.memory_bank = state_dict['memory_bank'].to(self.device)
    self.memory_ptr = state_dict['memory_ptr']
    self.memory_count = state_dict['memory_count']
```

#### 9.1.3 DINOv2特征提取

```python
def extract_dino_features(self, image):
    if not hasattr(self, 'dino_model') or self.dino_model is None:
        return None
    with torch.no_grad():
        image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image_normalized = (image_resized + 1) / 2
        image_normalized = (image_normalized - self.dino_mean) / self.dino_std
        features = self.dino_model(image_normalized)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-7)
    return features  # shape: [B, 384]
```

DINOv2 (`vit_small_patch14_dinov2.lvd142m`) 通过timm库加载，输出 **384维** 特征向量。

#### 9.1.4 DINOv2模型加载与位置编码插值

```python
self.dino_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=224)
if dino_model_path and os.path.exists(dino_model_path):
    state_dict = torch.load(dino_model_path, map_location=self.device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

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

$$L_{visual} = \lambda_{pixel} \cdot L_{MSE}(I_{enc}, I_{cov}) + \lambda_{ssim} \cdot L_{SSIM}(I_{enc}, I_{cov})$$

$$L_{visual}^{noised} = \lambda_{pixel} \cdot L_{MSE}(I_{noised}, I_{cov}) + \lambda_{ssim} \cdot L_{SSIM}(I_{noised}, I_{cov})$$

代入默认值：

$$L_{visual} = 1.0 \cdot L_{MSE} + 2.0 \cdot L_{SSIM}$$

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
L_D = (self.bce_with_logits_loss(cover_pred, torch.ones_like(cover_pred)) +
       self.bce_with_logits_loss(stego_pred, torch.zeros_like(stego_pred)))
```

标签分配：
- `cover_pred = Steganalyzer(cover_image)` → **标签=1**（真实图像）
- `stego_pred = Steganalyzer(noised_encoded_image.detach())` → **标签=0**（隐写图像）

**判别器暂停机制**：

```python
disc_threshold = getattr(self.cfg, 'discriminator_loss_threshold', None)
skip_disc = (disc_threshold is not None and L_D.item() < disc_threshold)
```

当判别器损失低于阈值（默认0.4）时，暂停判别器训练，防止判别器过度强大导致生成器梯度消失。

**生成器（Encoder+Decoder）对抗损失**：

```python
stego_pred = self.steganalyzer(disc_stego_input)
L_adv = self.bce_with_logits_loss(stego_pred, torch.ones_like(stego_pred))
```

这是标准的**非饱和GAN对抗损失**，生成器试图"欺骗"判别器。

**训练顺序**：
1. 先计算判别器损失并更新判别器参数（仅在 `adv > 0` 且未达到阈值时）
2. 再计算生成器对抗损失（使用更新后的判别器）
3. 最后连同其他损失一起更新生成器参数

#### 9.2.4 隐私保护损失

隐私损失由 `FeatureRedirectionLoss` 直接计算，详见第7.2节。在Phase 2中，隐私损失整合了以下子损失：

| 子损失 | 权重 | 含义 |
|--------|------|------|
| CLIP逃逸损失 | `clip_weight=1.0` | CLIP特征空间中的Softplus逃逸 |
| CLIP重定向损失 | `clip_redirect_weight=0.3` | CLIP特征空间中的有向重定向 |
| DINO逃逸损失 | `dino_weight=1.0` | DINO特征空间中的Softplus逃逸 |
| DINO重定向损失 | `dino_redirect_weight=0.4` | DINO特征空间中的有向重定向 |

**CLIP/DINO 记忆库更新与硬负样本获取流程**（Trainer中）：

```python
if self.use_clip_redirect and self.clip_memory_bank is not None:
    self.clip_memory_bank.update(feat_cover.detach())
    clip_hard_negatives = self.clip_memory_bank.get_hard_negatives(
        feat_encoded.detach(), k=self.clip_hard_negative_k
    )

if self.use_clip_redirect and self.dino_memory_bank is not None and dino_cover is not None:
    self.dino_memory_bank.update(dino_cover.detach())
    dino_hard_negatives = self.dino_memory_bank.get_hard_negatives(
        dino_encoded.detach(), k=self.clip_hard_negative_k
    )
```

**关键设计**：记忆库**先更新**封面图像特征（feat_cover），**再获取**硬负样本时使用编码图像特征（feat_encoded）作为查询。这确保：
1. 当前batch的封面特征立即可供后续batch使用
2. 查询使用的是编码图像特征，与重定向目标一致

#### 9.2.5 残差惩罚

$$L_{residual} = \text{mean}(msg\_residual^2) + \text{mean}(prv\_residual^2)$$

这个值由编码器的 `forward` 方法直接返回，作为额外的正则化项限制修改幅度。Phase 1中 `prv_residual` 为 `dummy_prv` 随机噪声。

#### 9.2.6 梯度清零保护

```python
if phase == 2 and self.phase2_msg_head_frozen:
    for param in self.encoder.msg_head.parameters():
        param.grad = None
```

在Phase 2中，即使msg_head的参数被冻结（`requires_grad=False`），由于共享层的梯度回传，msg_head参数仍可能累积梯度。通过在backward后显式清零msg_head梯度，确保msg_head不会被意外更新。

### 9.3 编码器头部冻结策略

编码器的双头架构需要精心的冻结策略，确保各阶段训练目标不互相干扰：

```python
def _set_phase_freeze(self, phase, epoch):
    if phase == 1:
        self.unfreeze_msg_head()
        self.freeze_prv_head()
        self.unfreeze_shared_layers()
    else:
        self.freeze_msg_head()
        self.unfreeze_prv_head()
        if epoch <= self.cfg.num_epochs_phase2 // 3:
            self.unfreeze_shared_layers()
        else:
            self.freeze_shared_layers()
```

#### Phase 1 冻结策略

| 组件 | 状态 | 原因 |
|------|------|------|
| msg_head | ✅ 解冻 | 需要学习消息嵌入 |
| prv_head | ❄️ 冻结 | Phase 1不进行隐私保护训练 |
| 共享层（conv_layers + after_concat_layer） | ✅ 解冻 | 需要学习基础特征表示 |

#### Phase 2 冻结策略

| 组件 | 前1/3 epochs | 后2/3 epochs | 原因 |
|------|-------------|-------------|------|
| msg_head | ❄️ 冻结 | ❄️ 冻结 | 保护已学习的消息嵌入能力 |
| prv_head | ✅ 解冻 | ✅ 解冻 | 需要学习隐私扰动 |
| 共享层 | ✅ 解冻 | ❄️ 冻结 | 前期微调共享特征适配prv_head，后期冻结防止破坏msg能力 |

**设计思路**：
- Phase 2开始时msg_head立即冻结，防止隐私训练破坏消息嵌入能力
- 共享层在Phase 2前1/3解冻，允许prv_head通过微调共享特征来适配隐私任务
- Phase 2后2/3冻结共享层，防止过度调整影响消息提取的稳定性
- 配合解码器恢复机制，确保消息提取准确率始终维持在阈值以上

### 9.4 解码器恢复机制

当Phase 2中验证比特准确率低于阈值时，触发解码器恢复训练：

```python
def recovery_step(self, train_loader, num_batches=5):
    self.encoder.eval()
    self.decoder.train()
    self.freeze_prv_head()

    temp_optimizer = optim.Adam(self.decoder.parameters(), lr=self.cfg.lr_decoder * 0.5)

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        cover_image = batch['cover_image'].to(self.device)
        message = batch['message'].to(self.device)

        with torch.no_grad():
            _, _, msg_residual, _ = self.encoder(
                cover_image, message, phase=1, prv_noise_ratio=0.0
            )
            clean_encoded = cover_image + msg_residual
            clean_encoded = torch.clamp(clean_encoded, -1, 1)

        message_pred = self.decoder(clean_encoded)
        L_msg = self.bce_with_logits_loss(message_pred, message)

        temp_optimizer.zero_grad()
        L_msg.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        temp_optimizer.step()

    self.unfreeze_prv_head()
    self.encoder.train()
```

**关键设计**：
1. **编码器设为eval模式**：不更新编码器参数
2. **使用phase=1编码**：仅生成消息残差（不含隐私残差），在干净的编码图像上训练解码器
3. **临时优化器**：使用半学习率（`lr_decoder * 0.5`）的独立Adam优化器，避免影响主优化器状态
4. **冻结prv_head**：防止恢复训练影响隐私头
5. **有限步数**：默认10个batch，快速恢复不过度训练

**触发条件**（在train.py中）：

```python
if phase == 2:
    val_bit_acc = val_losses.get('bit_accuracy', 1.0)
    threshold = getattr(cfg, 'bit_accuracy_threshold', 0.90)
    if val_bit_acc < threshold:
        recovery_steps = getattr(cfg, 'recovery_steps', 10)
        trainer.recovery_step(train_loader, num_batches=recovery_steps)
```

### 9.5 训练流程 (`train.py`)

**文件位置**: [train.py](train.py)

#### 9.5.1 命令行接口

项目支持两种运行模式：

**模式1：新建实验 (`new`)**

```bash
python train.py new --data-dir ./dataset --name experiment1 \
    --batch-size 8 --size 256 --message 30 \
    --epochs-phase1 130 --epochs-phase2 150 \
    --discriminator-loss-threshold 0.4 \
    --lambda-pixel 1.0 --lambda-ssim 2.0 \
    --lambda-message 1.0 --lambda-adv 0.1 --lambda-privacy 0.3 \
    --num-workers 4 --save-interval 10 \
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

#### 9.5.2 CLI参数完整列表

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--data-dir` | 必需 | 数据集根目录 |
| `--train-dir` | `train` | 训练集子目录 |
| `--val-dir` | `val` | 验证集子目录 |
| `--batch-size` | 8 | 批次大小 |
| `--name` | `default` | 实验名称 |
| `--size` | 256 | 图像分辨率 |
| `--message` | 30 | 消息比特长度 |
| `--continue-from-folder` | `''` | 恢复检查点目录 |
| `--tensorboard` | False | 启用TensorBoard |
| `--enable-fp16` | False | 启用混合精度 |
| `--noise` | `identity` | 噪声层配置 |
| `--seed` | 42 | 随机种子 |
| `--epochs-phase1` | 130 | Phase 1轮次 |
| `--epochs-phase2` | 150 | Phase 2轮次 |
| `--discriminator-loss-threshold` | 0.4 | 判别器暂停阈值 |
| `--lr-encoder` | 1e-3 | 编码器学习率 |
| `--lr-decoder` | 1e-3 | 解码器学习率 |
| `--lr-steganalyzer` | 1e-3 | 隐写分析器学习率 |
| `--lambda-pixel` | 1.0 | 像素损失权重 |
| `--lambda-ssim` | 2.0 | SSIM损失权重 |
| `--lambda-message` | 1.0 | 消息损失权重 |
| `--lambda-adv` | 0.1 | 对抗损失权重 |
| `--lambda-privacy` | 0.3 | 隐私损失权重 |
| `--num-workers` | 4 | 数据加载线程 |
| `--save-interval` | 10 | 检查点保存间隔 |

#### 9.5.3 实验目录结构

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

#### 9.5.4 训练主循环 (`run_training_loop`)

```python
for phase in range(start_phase, 3):
    num_epochs = getattr(cfg, f'num_epochs_phase{phase}')
    phase_start_epoch = start_epoch if phase == start_phase else 1

    if phase == 1 and phase_start_epoch == 1:
        trainer.warmup_memory_bank(train_loader, num_batches=50)

    if phase == 2 and not trainer.phase2_msg_head_frozen:
        trainer.freeze_msg_head()

    for epoch in range(phase_start_epoch, num_epochs+1):
        for batch_idx, batch in enumerate(train_loader):
            losses = trainer.train_step(batch, phase, epoch)

        val_losses = trainer.validate(val_loader, phase, epoch, save_vis)

        if phase == 2:
            val_bit_acc = val_losses.get('bit_accuracy', 1.0)
            if val_bit_acc < cfg.bit_accuracy_threshold:
                trainer.recovery_step(train_loader, num_batches=cfg.recovery_steps)

        if epoch % cfg.save_interval == 0:
            trainer.save_checkpoint(epoch, phase, is_best)
```

#### 9.5.5 检查点保存策略

```python
def save_checkpoint(self, epoch, phase, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'encoder_state_dict': self.encoder.state_dict(),
        'decoder_state_dict': self.decoder.state_dict(),
        'steganalyzer_state_dict': self.steganalyzer.state_dict(),
        'optimizer_encoder_decoder': self.optimizer_encoder_decoder.state_dict(),
        'optimizer_steganalyzer': self.optimizer_steganalyzer.state_dict(),
        'best_loss': self.best_loss,
        'phase2_msg_head_frozen': self.phase2_msg_head_frozen
    }

    if self.clip_memory_bank is not None:
        checkpoint['clip_memory_bank'] = self.clip_memory_bank.state_dict()
    if self.dino_memory_bank is not None:
        checkpoint['dino_memory_bank'] = self.dino_memory_bank.state_dict()
```

**保留策略**：每个phase最多保留5个最新检查点，自动删除最旧的检查点。

#### 9.5.6 随机种子设置

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

### 9.6 记忆库预热

```python
def warmup_memory_bank(self, train_loader, num_batches=50):
    if self.clip_memory_bank is None and self.dino_memory_bank is None:
        return

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            cover_image = batch['cover_image'].to(self.device)

            if self.clip_memory_bank is not None:
                clip_feat = self.content_recognizer(cover_image)
                self.clip_memory_bank.update(clip_feat)

            if self.dino_memory_bank is not None:
                dino_feat = self.extract_dino_features(cover_image)
                if dino_feat is not None:
                    self.dino_memory_bank.update(dino_feat)
```

在Phase 1开始前，用50个batch的封面图像特征**预填**CLIP和DINO记忆库。这确保在正式训练开始时，记忆库已有足够的样本可供硬负样本挖掘。

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
    # 3. 编码：Encoder(cover, message, phase=2, prv_noise_ratio=0.0) → encoded_image
    # 4. 解码：Decoder(encoded_image) → decoded_message
    # 5. 消息验证：计算比特准确率
    # 6. 判别器预测：分别对原图和隐写图进行隐写检测
    # 7. 隐私评估：计算CLIP/DINO余弦相似度、欧氏距离、SSIM、PSNR
    # 8. 保存可视化结果
```

**编码器调用方式**：

```python
encoded_image, residual_penalty, msg_residual, prv_residual = self.encoder(
    cover_image, message, phase=2, prv_noise_ratio=0.0
)
```

验证时使用 `phase=2` 和 `prv_noise_ratio=0.0`，确保同时应用消息残差和隐私残差。

**输出指标**：

| 指标 | 公式/方法 | 含义 |
|------|----------|------|
| 比特准确率 | `correct_bits / total_bits` | 消息提取正确率 |
| 隐写概率(原图) | `sigmoid(Steganalyzer(cover))` | 原图被判为隐写的概率（应接近0%） |
| 隐写概率(隐写图) | `sigmoid(Steganalyzer(encoded))` | 隐写图被判为隐写的概率（应接近50%，即成功欺骗） |
| CLIP余弦相似度 | `cos(f_cover, f_encoded)` | CLIP特征空间相似度（越低隐私越好） |
| DINO余弦相似度 | `cos(f_cover_dino, f_encoded_dino)` | DINO特征空间相似度（越低隐私越好） |
| SSIM | `1 - SSIM_Loss` | 结构相似度（越高视觉质量越好） |
| PSNR | `10·log₁₀(4.0/MSE)` | 峰值信噪比（dB，越高越好） |
| CLIP欧氏距离 | `‖f_encoded - f_cover‖₂` | CLIP特征向量欧氏距离 |
| DINO欧氏距离 | `‖f_encoded_dino - f_cover_dino‖₂` | DINO特征向量欧氏距离 |

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

### 10.6 DINOv2相似度评估

验证脚本集成了DINOv2相似度评估功能，在隐私相似度判断中同时输出CLIP和DINO两种模态的指标：

```python
if self.dino_model is not None:
    feat_cover_dino = self.extract_dino_features(cover_image)
    feat_encoded_dino = self.extract_dino_features(encoded_image)
    dino_sim = F.cosine_similarity(feat_cover_dino, feat_encoded_dino, dim=1).item()
    dino_euclidean_dist = torch.norm(feat_encoded_dino - feat_cover_dino, p=2, dim=1).item()
```

---

## 11. CLIP与DINOv2视觉模型

### 11.1 CLIP模型

本项目的CLIP模型通过 `pip install git+https://github.com/openai/CLIP.git` 安装。CLIP (Contrastive Language-Image Pre-Training) 由OpenAI开发，通过大规模图文对比学习训练。

**核心特性**：
- **预训练规模**：约4亿图文对
- **架构**：图像端使用ViT-B/32（Transformer）
- **特征维度**：512维L2归一化特征向量
- **使用方式**：本项目仅使用图像编码器（`clip.load` + `encode_image`），文本编码器通过 `encode_text()` 方法可选使用

**在本项目中的作用**：
- 将原始封面图像和编码图像分别通过CLIP编码，得到512维语义特征
- 计算两者的余弦相似度，用于隐私损失计算
- CLIP的语义理解能力使得即使像素级差异很小，语义级的"相似性泄露"也能被检测

### 11.2 DINOv2模型

DINOv2（Self-distilled Vision Transformers）由Meta开发，通过自蒸馏方式训练。

**核心特性**：
- **架构**：ViT-S/14（Small variant，patch size 14）
- **特征维度**：384维L2归一化特征向量
- **加载方式**：通过timm库加载（`vit_small_patch14_dinov2.lvd142m`）
- **权重路径**：`./models/dinov2_vits14_pretrain.pth`

**在本项目中的作用**：
- 提供与CLIP互补的视觉特征表示
- CLIP侧重图文对齐语义，DINOv2侧重自监督视觉特征和细粒度纹理
- 联合使用覆盖更多潜在的攻击向量

**与CLIP的互补性**：

| 特性 | CLIP ViT-B/32 | DINOv2 ViT-S/14 |
|------|--------------|-----------------|
| 训练方式 | 图文对比学习 | 自蒸馏 |
| 特征维度 | 512 | 384 |
| 语义侧重 | 物体类别、场景类型 | 纹理、细粒度结构 |
| 隐私权重 | 1.0 | 1.0 |

---

## 12. 核心算法原理深度剖析

### 12.1 双残差隐写编码原理

#### 传统隐写 vs 双残差隐写

| 维度 | 传统隐写（LSB等） | 单残差隐写 | 双残差隐写（本系统） |
|------|------------------|-----------|-------------------|
| 嵌入方式 | 手工规则（修改最低有效位） | 单一神经网络残差 | 解耦的消息残差 + 隐私残差 |
| 容量控制 | 固定 | 隐式学习 | 预算约束（msg_budget, prv_budget） |
| 隐私保护 | 无 | 与消息嵌入耦合 | 独立隐私头，阶段解耦训练 |
| 阶段适配 | 无 | 无 | Phase 1虚拟噪声预适应 |

#### 本系统的编码策略

编码器采用**双残差预算约束嵌入**策略：

1. **消息空间化**：30位消息 → 30通道的256×256特征图
2. **特征融合**：消息 + 128通道图像特征 + 原始图像 → 161通道联合表示
3. **共享特征提取**：161通道 → 128通道共享特征
4. **双头分支**：
   - msg_head：128通道 → 3通道消息残差，`msg_budget * tanh` 约束
   - prv_head：128通道 → 3通道隐私残差，`prv_budget * tanh` 约束
5. **残差叠加**：`encoded = cover + msg_residual + prv_residual`

**预算约束的数学含义**：

$$msg\_residual = \beta_{msg} \cdot \tanh(f_{msg}(h_{shared}))$$

其中 $\beta_{msg} = 0.04$，$\tanh$ 输出范围 $[-1, 1]$，因此 $msg\_residual \in [-0.04, 0.04]$。

$$prv\_residual = \beta_{prv} \cdot \tanh(f_{prv}(h_{shared}))$$

其中 $\beta_{prv} = 0.06$，因此 $prv\_residual \in [-0.06, 0.06]$。

总修改幅度上界：$|msg\_residual| + |prv\_residual| \le 0.04 + 0.06 = 0.10$

这保证了编码图像与原图的像素差异不超过10%的动态范围，确保视觉不可感知性。

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

**判别器暂停机制**：当判别器损失低于阈值（0.4）时暂停训练，防止判别器过强导致生成器梯度消失，这是训练稳定性的重要保障。

**SpectralNorm的关键作用**：保证判别器满足1-Lipschitz约束，避免梯度爆炸/消失，使对抗训练稳定收敛。

### 12.3 隐私保护的多模态特征逃逸机制

#### 问题定义

传统隐写只关心"能否被发现"，但忽略了**语义隐私泄露**风险：
- 攻击者可以使用CLIP等语义模型提取图像特征
- 如果编码前后特征相似，攻击者可以通过**语义检索**建立联系
- 这构成了**推理攻击（Inference Attack）**的一种

#### 解决方案：特征空间逃逸 + 有向重定向

```
CLIP/DINO语义特征空间（单位超球面）

                    f_encoded (目标位置)
                       ●
                      /
                     /  ← 最大化距离(推离原图) — escape_loss
                    /
                   /
        f_cover ●─────────────────● f_other_image
                  \               /
                   \  ← 最小化距离(拉向负样本) — redirect_loss
                    \           /
                     ● f_hard_negative
                    ↑
              control_loss 防止过度接近
```

**逃逸策略**：
1. **推离**（Push Away）：最大化 `sim(f_encoded, f_cover)` 的距离（Softplus Loss）
2. **拉向**（Pull Toward）：最小化 `sim(f_encoded, f_hard_negative)` 的距离（有向重定向，Softplus Loss）
3. **控制**（Control）：防止过度接近负样本（Hinge Loss）

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

普通损失函数使用batch内的随机样本作为负样本，但这些样本可能太容易区分（"简单负样本"），提供的训练信号较弱。

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
    ↓ FeatureRedirectionLoss内部选最难的1个
Hardest: f_P
    ↓ 用作重定向目标
L_redirect = mean(log1p(exp((1-margin) - sim(f_encoded, f_P))))
L_control = mean(relu(sim(f_encoded, f_P) - max_target_sim))
```

**多记忆库设计**：

| 记忆库 | 存储特征 | 特征维度 | 用途 |
|--------|---------|---------|------|
| `clip_memory_bank` | CLIP特征 | 512 | CLIP有向重定向的负样本来源 |
| `dino_memory_bank` | DINOv2特征 | 384 | DINO有向重定向的负样本来源 |

**优势**：
- 记忆库积累了历史batch的特征，负样本池远大于当前batch
- 选择"最相似"的样本作为负样本，迫使模型学习更精细的区分能力
- 循环缓冲区设计保证特征分布的时效性
- 多记忆库覆盖多模态攻击向量

### 12.5 渐进逃逸边界机制

#### 问题

如果在Phase 2一开始就使用较大的逃逸margin，隐私损失会产生巨大的梯度，可能导致：
- 编码器过度修改图像，视觉质量严重下降
- 消息提取准确率急剧下降
- 训练过程不稳定甚至崩溃

#### 解决方案

通过 `get_escape_margin()` 实现逃逸margin的线性增长：

$$margin(epoch) = margin_{start} + (margin_{end} - margin_{start}) \times \frac{epoch - 1}{total\_epochs - 1}$$

代入默认值：

$$margin(epoch) = 0.05 + 0.25 \times \frac{epoch - 1}{149}$$

| Phase 2 Epoch | margin | 逃逸目标 (1-margin) | 含义 |
|--------------|--------|-------------------|------|
| 1 | 0.050 | 0.950 | 编码特征需与原图相似度 < 0.95 |
| 50 | 0.133 | 0.867 | 相似度需 < 0.87 |
| 100 | 0.217 | 0.783 | 相似度需 < 0.78 |
| 150 | 0.300 | 0.700 | 相似度需 < 0.70 |

**渐进策略的效果**：
- 训练初期：margin小，逃逸要求宽松，模型可以平稳过渡
- 训练后期：margin大，逃逸要求严格，隐私保护充分
- 整个过程平滑过渡，避免梯度突变

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
│  │  ENCODER (Dual-Head)                                                   │
│  │  ┌──────────────────────────────────────────────────────────┐          │
│  │  │ 1. message expand: [B,30] → [B,30,256,256]               │          │
│  │  │ 2. ConvBNRelu×16: cover → [B,128,256,256] features      │          │
│  │  │ 3. Concat: [msg(30); feat(128); cover(3)] → [B,161,...]  │         │
│  │  │ 4. ConvBNRelu: [B,161] → [B,128,256,256] (shared)       │          │
│  │  │ 5. msg_head: shared → [B,3,256,256] → 0.04*tanh → msg_r │         │
│  │  │ 6. prv_head: shared → [B,3,256,256] → 0.06*tanh → prv_r │         │
│  │  │ 7. encoded = cover + msg_r + prv_r                        │         │
│  │  │ 8. clamp(encoded, -1, 1)                                  │         │
│  │  └──────────────────────────────────────────────────────────┘          │
│  │                      │                                                │
│  │                      ▼                                                │
│  │            encoded_image [B,3,256,256]                                │
│  │            residual_penalty (scalar)                                   │
│  │            msg_residual [B,3,256,256]                                  │
│  │            prv_residual [B,3,256,256]                                  │
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
│  │  │ L_visual = 1.0*L_pixel + 2.0*L_ssim                            │    │
│  │  │ L_visual_noised = 同上(noised_encoded vs cover)                │    │
│  │  └────────────────────────────────────────────────────────────────┘    │
│  │                                                                        │
│  │  ├─ Message Loss ─────────────────────────────────────────────────┐    │
│  │  │ L_message = BCEWithLogits(message_pred, message)               │    │
│  │  └────────────────────────────────────────────────────────────────┘    │
│  │                                                                        │
│  │  ├─ Adversarial Loss ─────────────────────────────────────────────┐    │
│  │  │ D_train: BCE(Stego(cover)→1, Stego(noised_enc.detach)→0)       │    │
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
│  │  │ 3. Memory bank update + hard negatives:                         │    │
│  │  │    clip_mem.update(f_cover)                                     │    │
│  │  │    clip_neg = clip_mem.get_hard_negatives(f_encoded, k=5)      │    │
│  │  │    dino_mem.update(dino_cover)                                  │    │
│  │  │    dino_neg = dino_mem.get_hard_negatives(dino_encoded, k=5)   │    │
│  │  │ 4. FeatureRedirectionLoss:                                      │    │
│  │  │    escape_margin = get_escape_margin(phase, epoch)              │    │
│  │  │    L_privacy = 1.0*L_clip_escape + 0.3*L_clip_redirect         │    │
│  │  │             + 1.0*L_dino_escape + 0.4*L_dino_redirect          │    │
│  │  └────────────────────────────────────────────────────────────────┘    │
│  │                                                                        │
│  │  ├─ Total Loss ────────────────────────────────────────────────────┐    │
│  │  │ L_total = 1.0*(L_visual + 0.3*L_visual_noised)                 │    │
│  │  │        + 15.0*L_message + 2.0*L_adv                            │    │
│  │  │        + 3.0*L_privacy + 0.2*L_residual                        │    │
│  │  │  (Phase 2 前1/3: message=15.0, privacy=3.0)                    │    │
│  │  │  (Phase 2 中1/3: message=12.0, privacy=6.0)                    │    │
│  │  │  (Phase 2 后1/3: message=10.0, privacy=10.0)                   │    │
│  │  └────────────────────────────────────────────────────────────────┘    │
│  │                                                                        │
│  ├───────────────────────────────────────────────────────────────────────┤
│  │                                                                        │
│  │  OPTIMIZER STEPS                                                       │
│  │  1. optimizer_encoder_decoder.zero_grad()                              │
│  │  2. L_total.backward()                                                │
│  │  3. [If phase==2 && msg_head_frozen]: zero msg_head grads             │
│  │  4. clip_grad_norm(enc+dec params, max_norm=1.0)                      │
│  │  5. optimizer_encoder_decoder.step()                                   │
│  │                                                                        │
│  └───────────────────────────────────────────────────────────────────────┘
│                                                                         │
│  输出                                                                     │
│  losses dict: {L_pixel, L_ssim, L_visual, L_message, L_adv,            │
│               L_privacy, L_total, bit_accuracy, escape_margin,          │
│               prv_noise_ratio, privacy_clip_escape, ...}                │
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

# 自定义超参数
python train.py new --data-dir /path/to/dataset --name custom_exp \
    --epochs-phase1 130 --epochs-phase2 150 \
    --discriminator-loss-threshold 0.4 \
    --lambda-pixel 1.0 --lambda-ssim 2.0 \
    --lambda-message 1.0 --lambda-adv 0.1 --lambda-privacy 0.3

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
- `Phase{1,2}/Train/{L_total, L_visual, L_message, L_adv, L_privacy, bit_accuracy, escape_margin, ...}`
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
| LR | lr_encoder | 1e-3 | 编码器学习率 |
| LR | lr_decoder | 1e-3 | 解码器学习率 |
| LR | lr_steganalyzer | 1e-3 | 判别器学习率 |
| λ | lambda_pixel | 1.0 | MSE权重 |
| λ | lambda_ssim | 2.0 | SSIM权重 |
| λ | lambda_message | 1.0 | 消息损失权重 |
| λ | lambda_adv | 0.1 | 对抗损失基础权重 |
| λ | lambda_privacy | 0.3 | 隐私损失基础权重 |
| 预算 | msg_budget | 0.04 | 消息残差预算 |
| 预算 | prv_budget | 0.06 | 隐私残差预算 |
| 隐私 | privacy_margin_start | 0.05 | 起始逃逸margin |
| 隐私 | privacy_margin_end | 0.30 | 结束逃逸margin |
| 隐私 | clip_weight | 1.0 | CLIP特征逃逸权重 |
| 隐私 | dino_weight | 1.0 | DINOv2特征逃逸权重 |
| 重定向 | use_clip_redirect | True | 启用有向重定向 |
| 重定向 | clip_redirect_weight | 0.3 | CLIP重定向权重 |
| 重定向 | dino_redirect_weight | 0.4 | DINO重定向权重 |
| 重定向 | clip_redirect_margin | 0.4 | CLIP重定向margin |
| 重定向 | clip_max_target_sim | 0.65 | CLIP最大允许相似度 |
| 重定向 | dino_redirect_margin | 0.4 | DINO重定向margin |
| 重定向 | dino_max_target_sim | 0.65 | DINO最大允许相似度 |
| 记忆库 | clip_memory_size | 1000 | 记忆库容量 |
| 记忆库 | clip_hard_negative_k | 5 | 硬负样本数 |
| 记忆库 | memory_warmup_batches | 50 | 预热批次数 |
| 恢复 | bit_accuracy_threshold | 0.90 | 恢复触发阈值 |
| 恢复 | recovery_steps | 10 | 恢复训练批次数 |
| 阶段 | P1 epochs | 130 | 基础+渐进对抗训练 |
| 阶段 | P2 epochs | 150 | 隐私保护训练 |
| 其他 | discriminator_loss_threshold | 0.4 | 判别器暂停阈值 |
| 其他 | num_workers | 4 | 数据加载线程 |
| 其他 | save_interval | 10 | 检查点保存间隔 |
| 其他 | seed | 42 | 随机种子 |

## 附录B：模型参数量汇总

| 模型 | 近似参数量 | 可训练 |
|------|-----------|--------|
| Encoder (Dual-Head, 128ch, 16blocks) | ~2.99M | ✓ (阶段性冻结部分头) |
| Decoder (CNN, 64ch, 8blocks) | ~278K | ✓ |
| Steganalyzer (CNN+SN, 双层FC) | ~4.7M | ✓ |
| ContentRecognizer (CLIP ViT-B/32) | ~150M | ✗ (冻结) |
| DINOv2 (ViT-S/14) | ~22M | ✗ (冻结) |
| CLIP Memory Bank | ~2M (1000×512×4B) | buffer only |
| DINO Memory Bank | ~1.5M (1000×384×4B) | buffer only |
| **总可训练** | **~7.97M** | |

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
| 特征逃逸 | Feature Escape | 在特征空间中拉开原始/编码图像的距离 |
| 有向重定向 | Directed Redirection | 将编码图像特征拉向无关负样本 |
| 硬负样本 | Hard Negative | 与查询样本最相似的负样本 |
| 记忆库 | Memory Bank | 存储历史特征向量的循环缓冲区 |
| Softplus Loss | | log1p(exp(x)) 形式的平滑损失函数 |
| 预算约束 | Budget Constraint | 通过tanh×budget限制残差幅度的机制 |
| 双残差头 | Dual-Residual Head | 编码器中消息头和隐私头的双分支架构 |
| 头部冻结 | Head Freezing | 训练中冻结特定头部参数的策略 |
| 解码器恢复 | Decoder Recovery | 比特准确率下降时重新训练解码器的机制 |
| 渐进逃逸边界 | Progressive Escape Margin | 逃逸margin从0.05线性增长到0.30的策略 |
| DCT | Discrete Cosine Transform | 离散余弦变换 |
| CLIP | Contrastive Language-Image Pre-training | OpenAI的图文对比学习模型 |
| DINOv2 | Self-distilled Vision Transformers | Meta的自监督视觉模型 |
| GAN | Generative Adversarial Network | 生成对抗网络 |
| Hinge Loss | | max(0, margin - score) 形式的损失函数 |

---

*本文档基于 PrivSteg 项目源码生成，涵盖全部模块的实现细节与技术原理。*
