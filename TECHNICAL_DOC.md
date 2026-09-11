# PrivSteg 技术文档 —— 基于深度学习的隐私保护图像隐写系统

> 本文档与当前代码实现同步（2026-09-11 重写版）。当文档与代码不一致时，以代码为准。
> 快速上手请看 [README.md](README.md)；仓库工作约定见 [AGENTS.md](AGENTS.md)；
> 论文写作素材（研究背景、相关工作综述、设计理念叙事、指标形式化、实验设计与参考文献）见 [docs/PAPER_MATERIAL.md](docs/PAPER_MATERIAL.md)。

## 目录

1. [项目概述](#1-项目概述)
2. [研究背景与威胁模型](#2-研究背景与威胁模型)
3. [系统架构总览](#3-系统架构总览)
4. [项目文件结构](#4-项目文件结构)
5. [核心配置模块 (`config.py`)](#4-核心配置模块-configpy)
6. [数据处理模块 (`data/`)](#5-数据处理模块-data)
7. [模型架构模块 (`models/`)](#6-模型架构模块-models)
8. [损失函数模块 (`losses/`)](#7-损失函数模块-losses)
9. [噪声层模块 (`noise_layers/`)](#8-噪声层模块-noise_layers)
10. [训练系统 (`trainer.py` & `train.py`)](#9-训练系统-trainerpy--trainpy)
11. [验证与评估工具链 (`val/`)](#10-验证与评估工具链-val)
12. [核心算法原理](#11-核心算法原理)
13. [使用指南](#12-使用指南)
14. [附录A：关键超参数速查表](#附录a关键超参数速查表)
15. [附录B：输出文件清单](#附录b输出文件清单)
16. [附录C：术语表](#附录c术语表)

---

## 1. 项目概述

### 1.1 问题定义

PrivSteg 解决的是**发布侧隐私保护隐写**问题——一张图像同时要完成三个互相牵制的目标：

- **功能目标（可解码）**：在载体图像中嵌入 30 bit 二进制消息（水印/溯源），消息在 uint8 往返（PNG 保存重载）与信道噪声（JPEG 压缩 / 缩放 / 量化）下可靠解码，比特准确率 ≥ 0.90 为硬约束。
- **隐私目标（不可检索）**：使编码图像在 CLIP ViT-B/32 与 DINOv2 ViT-S/14 特征空间中偏离原载体，令攻击者通过**以图搜图（CBIR 内容检索）**无法把发布图关联回原始载体。
- **视觉目标（不可感知）**：扰动幅度受预算约束（深挖配方下 PSNR ~39 dB、SSIM ~0.95），肉眼不可辨。

三目标的张力是本项目的核心矛盾：消息嵌入要求残差承载可解码结构，隐私逃逸要求特征大幅偏离，感知约束又把两个残差的总能量钉死在极小预算内。

### 1.2 三层系统分工

系统由三层组成，各层解决"逐图优化"这一非凸问题在不同成本预算下的近似：

| 层 | 机制 | 成本 | CLIP sim（相对原载体） |
| --- | --- | --- | --- |
| **网络** | 一次前向完成编码 | 零额外成本 | ~0.85-0.95（摊销求解器的固有鸿沟） |
| **训练内蒸馏** | 每个 train step 内环跑 10 步逐图优化得到教师残差，蒸进权重 | 训练期 2~4× 单步耗时 | 缩小摊销鸿沟 |
| **编码期逐图精修** | 部署时对单图跑 30~60 步 Adam 直接求解 | CPU 每图 ~1 分钟 / GPU 秒级 | 0.44~0.55（深挖配方） |

精修发生在**编码端**（一次性成本），解码端零成本——这是"编码慢/解码快"架构下的标准选择：发布者对图像付出的成本可以摊销在发布流程里，接收方解码仍是一次前向。

### 1.3 核心创新点

1. **双残差解耦架构**：消息头（`msg_head`）与隐私头（`prv_head`）分离，消息嵌入与隐私逃逸互不干扰；热启动重训隐私支路时消息通路完整复用。
2. **隐私独立支路（`prv_trunk`）**：空洞卷积支路（dilation 1/3/9/27，感受野 81×81）直接从载体图像提特征，绕开只为消息目标训练且全程冻结的共享层（信息瓶颈），修复小感受野算不出空间结构化逐图攻击模式的容量瓶颈。
3. **低分辨率残差约束**：隐私残差先池化到 96×96 再双三次上采样回全分辨率——96 网格 ≈ 2.7px/格，恰好覆盖 CLIP ViT-B/32（patch 16×16）最敏感的 patch 尺度中频段，同时结构性消除高频条纹。
4. **纹理感知扰动**：逐图分位数归一化纹理掩码（q05/q95 → [0.15, 1.0]），扰动集中在人眼不敏感的纹理复杂区。
5. **Phase 2 三段调度**：Sprint（纯逃逸冲刺）→ Balance（多目标平衡）→ Refine（终段精修配方），配合分阶段冻结与判别器门控；针对"Adam 在多目标、逐图方向不一致下梯度信噪比失效"这一核心瓶颈。
6. **内环逐图优化蒸馏**：把探针级逃逸方向（逐图精修 60 步 clip_sim 0.94→0.55~0.65）蒸进网络单次前向。
7. **部署期逐图精修不劣化保证**：以网络原始输出为初始候选，只在比特 100% 正确的状态里选最优，隐私逃逸不以牺牲解码为代价。
8. **色度低频事后投影**：精修收敛后剔除隐私残差的 8×8 网格低频色度分量——针对 PSNR/SSIM 失明的"发灰/蒙层"感知劣化的结构化修复（见 11.8）。

### 1.4 实测效果摘要

- **逐图精修杠杆**（同一 checkpoint、60 步、bit_acc 全程 100%）：margin 是第一杠杆——0.30/0.40/0.50（budget 0.08）对应 CLIP sim 0.575/0.547/0.440；预算是第二杠杆——budget 0.04/0.06/0.08（margin 0.30）对应 0.634/0.603/0.575；视觉权重 15→8 无影响。margin 0.50 + budget 0.08 时 PSNR 38.95 / SSIM 0.957。
- **检索逃逸**（20 图库、深挖配方）：基线 top-1 命中 100%（攻击者一击必中），精修后逃逸率 66.7%。
- **色度投影**（单图实测）：clip_sim 0.438→0.467（仍在 margin 0.50 目标内）、低频色度 −91%、PSNR +0.5 dB、SSIM +0.001、bit 100%。
- **热启动复用**：直入 Phase 2 首轮 val bit_acc 即 0.925（消息通路完整复用）。

---

## 2. 研究背景与威胁模型

### 2.1 应用场景

可溯源图像发布：机构或个人需要在公开分享的图像中嵌入水印/溯源载荷（版权、真伪、泄漏溯源），同时不希望该图像被**以图搜图**类内容检索（Google Lens、百度识图、电商平台同款检索、社交平台近重复去重）与原始载体或其他已发布图像关联。

### 2.2 威胁模型（CBIR 攻击者）

- **攻击者能力**：掌握现代通用特征提取器（本文以 CLIP ViT-B/32 为代表；DINOv2 ViT-S/14 作为第二种自监督特征对冲），对图像提取 L2 归一化特征后与图库做余弦相似度排序（以图搜图）。
- **攻击者目标**：给定发布图（编码图），在图库中找回原始载体（top-k 命中）。
- **防御者目标**：编码图对原载体的特征相似度足够低，低到与图库中最强干扰物的相似度之下（攻击者排错第一名），且这一偏离要能抵抗 JPEG/缩放/量化信道（图像上传过程必然发生转码）。
- **防御者约束**：消息必须可被授权解码器恢复（uint8 往返 + 信道噪声下 bit_acc ≥ 0.90）；扰动必须不可感知（预算 + 纹理掩码 + 频段结构化约束）。

两点威胁建模决策值得强调：

1. **攻击面选在特征空间而非像素空间**——CBIR 系统的现实实现是深度特征检索，逐像素哈希只在完全一致的复制场景生效；因此防御的有效判据是特征余弦相似度下降与检索排名恶化，而非 PSNR 类像素指标。
2. **双特征空间对冲**——攻击者可能改用自监督特征（DINOv2）做近重复检测，单逃 CLIP 不够；CLIP（语义对齐）与 DINOv2（细粒度纹理）的判别机理不同，需分别设 margin 与重定向标定。

### 2.3 相关工作简述

完整综述与参考文献（含 DOI）见 [docs/PAPER_MATERIAL.md](docs/PAPER_MATERIAL.md)。简述：

- **深度学习隐写/水印**（HiDDeN ECCV 2018、StegaStamp CVPR 2020、MBRS ACM MM 2021 及 DL 隐写综述）：解决"消息如何在失真信道下可靠嵌入/提取"，不约束编码图在检索系统中的可检索性。
- **特征反演**（Mahendran & Vedaldi CVPR 2015、Dosovitskiy & Brox CVPR 2016）：证明深度特征可被反演回图像——即"特征相似 = 内容可关联"的隐私风险有理论依据。
- **防御性对抗扰动**（Fawkes USENIX Security 2020 等 cloaking 工作）：在数据侧加对抗扰动保护隐私，但目标通常是单个人脸识别模型，且不要求嵌入可解码消息。
- **检索系统对抗攻击**（AP-GAN 2020、CIKM 2021 deep-hashing 攻击）：证明 CBIR/哈希检索可被对抗样本欺骗，但均为单向攻击视角，未与"消息仍可解码"的多目标兼容。
- **摊销优化**（Amos, Foundations and Trends in ML 2023）：本系统"网络一次前向 + 逐图精修"正是 amortize-then-refine 结构，训练内蒸馏对应"教师指导摊销器学习"。

**本工作的差异化定位**：据我们所知，同时满足"消息可靠解码 + 信道鲁棒 + 不可感知 + CBIR 检索逃逸"四约束的可训练图像隐写系统此前未被研究；本系统给出的双残差架构、摊销-精修三层结构与特征空间标定的逃逸配方是该组合问题的首个完整解。

---

## 3. 系统架构总览

```
                        ┌──────────────────────────────────────────┐
                        │                Encoder                   │
 cover ──────┬─────────►│ shared(16×ConvBNRelu,128) ─┐            │
 (B,3,H,W)  │          │   │                         │ msg_head  │
 [-1,1]      │          │   ▼ 融合(161→128)            │   ↓       │
             │          │ cat[msg⊕feat⊕cover]          │ msg_budget×tanh
             │          │   │                         └────┬──────┘
             │          │   │              prv_trunk(dil 1/3/9/27)◄── cover
             │          │   ▼                         ┌────┴──────┐
             │          │ prv_head(cat[shared,trunk]) │           │
             │          │   ↓ 96×96低通→上采样→tanh×budget→×纹理掩码      │
             │          └──────────┬──────────────────┘           │
             │                     │ msg_residual + prv_residual │
             └─────────────────────►+ ──► clamp[-1,1] ──► encoded_image
                                   │
        ┌──────────────────────────┼──────────────────────────────┐
        ▼                          ▼                              ▼
   Decoder(BCE)            Noiser(随机选一层)                CLIP/DINOv2(冻结)
   消息 logits             jpeg/resize/quant/...             特征余弦相似度
        │                          │                              │
        ▼                          ▼                              ▼
   bit_accuracy              含噪分支(escape锚定)            escape+redirect
                              └──────────┬───────────────────────┘
                                         ▼
                                FeatureRedirectionLoss
                         (二次Hinge逃逸 + 硬负样本有向重定向)
```

训练数据流（Phase 2）：

1. `SteganographyDataset` 返回 `{cover_image, message, image_path}`。
2. `Encoder` 输出 `encoded_image`、`residual_penalty`、`msg_residual`、`prv_residual`。
3. **消息双分支**：干净编码图与含噪编码图分别过 decoder，干净分支权重由 `get_msg_clean_ratio` 调度。
4. **隐私双分支**：干净分支计算 escape+redirect（主力），含噪分支仅 escape（锚定传播场景），按 `get_privacy_noised_ratio` 加权。
5. 视觉损失只算干净分支（含噪视觉权重恒为 0，防止把信道失真归咎于残差）。
6. Phase 2 每个 train step 额外执行**内环蒸馏**：冻结网络 → 10 步 Adam 优化 96×96 教师残差 → 能量归一化 MSE 蒸给学生。
7. 验证时记录干净/含噪两套指标（`bit_accuracy` / `bit_accuracy_clean`、`clip_sim(_noised)`、`dino_sim(_noised)`），best checkpoint 用含噪指标判定（传播场景优先）。

---

## 4. 项目文件结构

```
PrivSteg/
├── config.py                  # 全局配置（超参数、损失权重、阶段调度函数）
├── train.py                   # 训练入口（new/continue、热启动、CLI、checkpoint 兼容加载）
├── trainer.py                 # 训练器（阶段冻结、内环蒸馏、recovery、记忆库、验证）
├── noise_argparser.py         # 噪声层命令行参数解析（argparse Action）
├── README.md                  # 项目说明（快速上手）
├── AGENTS.md                  # 仓库工作约定（供 AI 编码助手）
├── TECHNICAL_DOC.md           # 本文档
├── docs/                      # 补充文档（PAPER_MATERIAL.md 论文写作素材）
├── val/                        # 验证与评估脚本目录（从项目根运行）
│   ├── validation_script.py   # 单图验证（编码+精修+解码+判别器+相似度+可视化报告）
│   ├── refine_encode.py       # 独立逐图精修（不训练网络，试探逃逸天花板）
│   ├── baseline_validation.py # 单图基线验证（网络直出，无精修，无 report.html）
│   ├── batch_eval.py          # 批量定量评估（精修版：统计/鲁棒性/uint8 往返，断点续跑）
│   ├── batch_baseline.py      # 批量基线评估（网络直出，速度约 20~30 倍，断点续跑）
│   ├── retrieval_eval.py      # CBIR 检索逃逸评估（精修版，以图搜图攻击测试）
│   ├── retrieval_baseline.py  # CBIR 检索逃逸基线评估（网络直出）
│   ├── run_full_validation.py # 全链路一键验证编排（冒烟→单图→基线→精修→汇总）
│   ├── visual_report.py       # 单图验证可视化报告生成器
│   └── eval_visualizer.py     # 批量/检索评估可视化报告生成器
├── models/
│   ├── encoder.py             # 编码器（双残差头 + prv_trunk 独立支路 + 纹理掩码）
│   ├── decoder.py             # 解码器（CNN + 全连接消息提取）
│   ├── steganalyzer.py        # 隐写分析器（判别器，SpectralNorm）
│   └── content_recognizer.py  # 内容识别器（冻结 CLIP ViT-B/32）
├── losses/
│   ├── ssim_loss.py           # SSIM 损失
│   └── advanced_privacy_loss.py  # 特征逃逸 + 有向重定向损失（CLIP + DINOv2）
├── noise_layers/
│   ├── noiser.py              # 噪声层组合器（每次 forward 随机选择激活）
│   ├── identity.py            # 恒等映射（空配置兜底）
│   ├── crop.py / cropout.py    # 裁剪类噪声
│   ├── dropout.py / resize.py  # 丢弃/缩放噪声
│   ├── jpeg_compression.py    # JPEG 压缩噪声（DCT 域量化，卷积实现可微）
│   └── quantization.py        # 傅里叶近似量化噪声
├── data/
│   └── dataset.py             # 隐写数据集（图像加载 + 随机消息生成）
└── CLIP/                      # vendored OpenAI CLIP 包（独立 setup 与测试）
```

注意：`val/` 是**验证脚本目录**，与数据集根下的 `train/`/`val/` 数据子目录是两回事。所有 val 脚本启动时会自动把项目根加入 `sys.path` 并把工作目录切到项目根，因此从任何目录启动，相对路径（checkpoint、`./models/dinov2_vits14_pretrain.pth`、输出目录）行为都一致。val 脚本间的相互导入（如 `from validation_script import ValidationScript`）依赖**脚本模式**（`python val/xxx.py`），不支持包式导入 `import val.xxx`。

---

## 5. 核心配置模块 (`config.py`)

`Config` 类用类属性集中管理全部配置。以下按功能分组列出**当前默认值**（深挖配方已全部设为代码默认值）。

### 5.1 设备、数据与 IO

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `device` | cuda 可用则 cuda，否则 cpu | 计算设备 |
| `data_dir` | `None` | 数据集根目录（CLI `--data-dir`/`-d` 默认 `./dataset1`） |
| `train_dir` / `val_dir` | `'train'` / `'val'` | 数据子目录名 |
| `num_workers` | `4` | DataLoader 进程数 |
| `save_interval` | `10` | 每 N 轮保存 checkpoint |
| `checkpoint_dir` / `log_dir` | `'checkpoints'` / `'logs'` | 训练时改为实验绝对路径 |

### 5.2 图像与消息

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `image_size` | `256` | 图像 resize 尺寸 |
| `message_length` | `30` | 消息位数 |
| `batch_size` | `35` | 批大小（A100 口径；本地 CPU 冒烟请显式传 `--batch-size 1`） |
| `lr_encoder` / `lr_decoder` | `1e-3` | 编码器/解码器学习率 |
| `lr_steganalyzer` | `1e-4` | 判别器学习率 |
| `prv_head_lr_mult` | `5.0` | 隐私支路参数组学习率倍增 |

### 5.3 两阶段训练轮数与恢复

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `num_epochs_phase1` | `150` | Phase 1 轮数 |
| `num_epochs_phase2` | `100` | Phase 2 轮数 |
| `bit_accuracy_threshold` | `0.90` | 触发 decoder recovery / best 判定的阈值 |
| `recovery_steps` | `10` | recovery 重训 decoder 的 batch 数 |

### 5.4 对抗（判别器）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `disc_acc_high` | `0.70` | 判别器准确率门控上限（超过则暂停更新） |
| `disc_hinge_margin` | `1.0` | 判别器 Hinge Loss margin |
| `discriminator_loss_threshold` | `0.6` | 遗留兼容参数 |
| `phase1_adv_start/mid/end` | 均 `0.0` | Phase 1 三段对抗权重 |
| `phase2_adv` | `0.0` | Phase 2 对抗权重 |

> 当前场景不依赖隐写分析对抗，各阶段 adv 权重均为 0：判别器仅作监控指标，不向编码器回传梯度。判别器失效时 L_adv 只会向残差注入额外高频伪影，需要时可重新调高。

### 5.5 视觉与残差预算

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `lambda_pixel` / `lambda_ssim` | `1.0` / `1.0` | MSE / SSIM 权重 |
| `lambda_fft` | `50.0` | FFT 高频能量惩罚权重（仅 Balance 段生效） |
| `msg_budget` | `0.04` | 消息残差预算（tanh 幅度） |
| `prv_budget` | `0.08` | 隐私残差预算（深挖配方；CLI `--prv-budget` 可调） |
| `prv_lowres_size` | `96` | 隐私残差低分辨率约束尺寸 |
| `fft_high_freq_radius_ratio` | `0.25` | FFT 高频判定半径（奈奎斯特半径的比例；256 图约 r=32） |

`prv_budget=0.08` 的依据：逃逸深度近似随预算平方增长（cos 下降 ∝ 特征位移²），0.04→0.08 实测精修 clip_sim 0.634→0.575，PSNR 42.8→39.9 仍在视觉高质量区间。

### 5.6 纹理掩码

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `texture_mask_smooth_kernel` | `5` | 掩码梯度图平滑核 |
| `texture_mask_strength` | `3.0` | 遗留兼容参数，不参与计算（掩码由分位数归一化决定） |

掩码实际映射：`mask = 0.15 + 0.85 × clip((grad − q05)/(q95 − q05), 0, 1)`，即 [0.15, 1.0]。下限 0.15 的依据：探针实测下限 0.05 时 CLIP 逃逸天花板 sim≈0.72，贴边目标 0.70 无余量。

### 5.7 Phase 1 哑隐私噪声（预适应）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `phase1_prv_noise_start` | `0.80` | 前 80% 轮次不注入噪声（随机噪声会淹没 msg 信号、阻碍消息通路冷启动） |
| `phase1_prv_noise_end` | `0.95` | 噪声达到最大比例的轮次比例 |
| `phase1_prv_noise_max_ratio` | `0.30` | 最大注入比例（相对 prv_budget） |
| `phase1_prv_noise_power` | `2.0` | ramp 幂次 |
| `phase1_prv_noise_smooth_kernel` | `7` | 噪声平滑核 |

### 5.8 消息/隐私双分支

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `msg_clean_ratio_start` / `msg_clean_ratio_end` | `0.80` / `0.50` | Phase 1 干净分支权重线性退火区间，Phase 2 固定 end |
| `privacy_noised_ratio` | `0.3` | 隐私含噪分支权重（Phase 1 / Balance 段） |
| `privacy_noised_ratio_sprint` | `0.0` | Sprint / Refine 段含噪分支权重（只用干净分支梯度） |

### 5.9 Phase 2 三段调度

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `phase2_sprint_epochs` | `40` | Sprint 段轮数（Phase 2 开头；CLI `--sprint-epochs`） |
| `phase2_sprint_visual` | `1.5` | Sprint 视觉轻锚权重 |
| `phase2_sprint_privacy` | `12.0` | Sprint 隐私权重 |
| `phase2_refine_epochs` | `30` | Refine 终段轮数（Phase 2 末尾；CLI `--refine-epochs`） |

Sprint 段 40（原 20）的依据：20 轮实测边际收益尚未衰减完（10→20 轮仍降 0.5pt），且新增结构（prv_trunk）从零初始化需要更长冲刺期收敛。

### 5.10 内环蒸馏

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `distill_inner_steps` | `10` | 内环教师优化步数（0 关闭；CLI `--distill-steps`） |
| `distill_lr` | `0.1` | 内环 Adam 学习率 |
| `distill_weight` | `25.0` | 蒸馏损失权重（默认 25：蒸馏梯度是残差空间的直接回归、高信噪比，escape 梯度须穿过整个 CLIP 网络且信噪比低，提权让干净信号主导更新方向） |
| `distill_w_privacy` / `distill_w_message` / `distill_w_visual` | `12.0` / `10.0` / `15.0` | 教师目标各项权重（与 `val/refine_encode.py` 验证配方一致；Refine 段同时作为学生目标权重） |

### 5.11 Escape Margin（逃逸边界）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `privacy_margin_start` / `privacy_margin_end` | `0.10` / `0.50` | CLIP escape margin 渐进区间（CLI `--privacy-margin-end`；深挖配方 0.50） |
| `dino_escape_margin_start` / `dino_escape_margin_end` | `0.06` / `0.20` | DINO escape margin 渐进区间（CLI `--dino-margin-end`；深挖配方 0.20） |

margin 是第一逃逸杠杆：二次 Hinge 在 sim < (1−margin) 后梯度归零，margin 即逃逸地板。CPU 逐图精修实测（budget 0.08、60 步、bit_acc 100%）：margin 0.30/0.40/0.50 对应 clip_sim 0.575/0.547/0.440（PSNR 39.9/39.4/39.0）。0.50 远低于实测 hard negative 均值 0.73，CBIR top-1 易主余量充足；Sprint/Refine 段直接取该值，教师内环目标同源联动。DINO 视角下目标是脱离 near-duplicate 区 ~0.90，其 hard negative 极远（mean≈0.47），目标 sim 0.80（margin 0.20）精修代价很小、与 CLIP 逃逸不冲突。

### 5.12 CLIP / DINO 有向重定向

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `use_clip_redirect` | `True` | 启用记忆库硬负样本重定向 |
| `clip_memory_size` | `5000` | 记忆库容量（CLIP 与 DINO 各一） |
| `clip_hard_negative_k` | `5` | 每次采样 hard negative 数 |
| `clip_redirect_target_sim` / `dino_redirect_target_sim` | `0.85` / `0.60` | 重定向目标相似度 |
| `clip_max_target_sim` / `dino_max_target_sim` | `0.90` / `0.65` | 重定向相似度上限（control loss） |
| `clip_redirect_exclude_sim` / `dino_redirect_exclude_sim` | `0.92` / `0.87` | hard negative 排除阈值（自身/近重复；CLIP p99≈0.92、DINO p99≈0.87） |
| `clip_redirect_weight` / `dino_redirect_weight` | `3.0` / `0.6` | 重定向损失权重 |
| `clip_weight` / `dino_weight` | `1.0` / `0.5` | CLIP / DINO 逃逸总权重 |
| `clip_redirect_margin` / `dino_redirect_margin` | `0.35` / `0.3` | 遗留参数（redirect 目标现由 `*_redirect_target_sim` 控制） |
| `dino_model_path` | `'./models/dinov2_vits14_pretrain.pth'` | DINOv2 权重路径 |

> redirect 目标按特征空间分别标定，必须明显高于该空间典型 hard negative 相似度（CLIP≈0.73，p90≈0.83；DINO≈0.47）才有实质拉力——target 0.80 时 relu(0.80−0.78)≈0.02，redirect 占隐私损失不到 1%，形同虚设；target 0.85 提供 ~0.07 的持续拉力，redirect 成为指路的第二引擎。target < exclude > max 的层次必须保持（0.85 < 0.90 < 0.92）。

### 5.13 色度低频投影（精修事后处理）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `refine_chroma_projection` | `True` | 精修收敛后剔除 prv 低频色度蒙层（CLI `--refine-chroma-projection 1/0`） |

动机与机制见 11.8 节。所有含精修的 val 脚本均以 `--refine-chroma-projection`（或 `--chroma-projection`）同口径透传。

### 5.14 噪声与记忆库

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `noise_config` | `'jpeg+resize(0.8,1.2)+quant'` | 噪声层配置（CLI `--noise` 覆盖；`'identity'` 表示无噪声） |
| `memory_warmup_batches` | `50` | Phase 1 开始时记忆库预热批次数 |

### 5.15 阶段权重函数

**`get_phase_weights(phase, epoch)`** —— 返回 `{visual, message, adv, privacy, noised, residual_penalty, fft}` 权重字典（`epoch` 为**阶段内相对轮次**）：

- **Phase 1**：`visual=10, message=15, adv=三段(0/0/0), privacy=0, noised=0, residual_penalty=0.5, fft=0`。
- **Phase 2 Refine 终段**（优先判定）：`visual=distill_w_visual(15), message=distill_w_message(10), privacy=distill_w_privacy(12), adv=0, noised=0, residual_penalty=0, fft=0`。
- **Phase 2 Sprint 段**（`epoch <= phase2_sprint_epochs`）：`visual=1.5, message=0, privacy=12, adv=0, noised=0, residual_penalty=0, fft=0`。
- **Phase 2 Balance 段**（其余轮次，按剩余轮数三等分）：
  - 前 1/3：`privacy=4, message=20`
  - 中 1/3：`privacy=8, message=15`
  - 后 1/3：`privacy=12, message=10`
  - 固定：`visual=15, adv=0, noised=0, residual_penalty=0.5, fft=50`

privacy 权重 4/8/12 的依据：privacy 梯度须穿过整个 CLIP/DINO 网络才能到达残差（∂sim/∂残差极小），而 visual/message 对残差是逐像素直接的（∂≈1），实测 1.5/2.5/3.5 的权重无法补偿 1~2 个数量级的雅可比差距，平衡点会被钉死在 sim≈0.97（privacy 占损失 88% 但 sim 几乎不降）。

**`get_escape_margin(phase, epoch)` / `get_dino_escape_margin(phase, epoch)`**：Phase 1 恒 0；Phase 2 的 Refine/Sprint 段直接取 end 值；其余（Balance）在前 50% 轮次内从 start 线性渐进到 end，后半程稳定在 end（若全程渐进，训练将一直追逐不断远去的 hinge 目标——实测 margin 到最后一个 epoch 才到达 end，模型从未在最终目标下稳定优化过）。

**`get_privacy_noised_ratio(phase, epoch)`**：Sprint 与 Refine 段返回 `privacy_noised_ratio_sprint`（0），其余返回 `privacy_noised_ratio`（0.3）。

**`get_msg_clean_ratio(phase, epoch)`**：Phase 1 从 0.80 线性退火到 0.50；Phase 2 固定 0.50。

**`get_prv_noise_ratio(epoch)`**：Phase 1 哑噪声注入比例。前 80% 轮次为 0；80%~95% 轮次按幂次 2.0 渐进到最大比例 0.30；之后恒 0.30。

**`is_phase2_refine_segment(epoch)`**：`refine > 0 and epoch > num_epochs_phase2 - refine`。判定优先级 Refine > Sprint > Balance——热启动纯精修（`--epochs-phase2 N --refine-epochs N`）时全部轮次命中 Refine 而非被 Sprint 声明。

---

## 6. 数据处理模块 (`data/`)

`data/dataset.py` 定义 `SteganographyDataset(root_dir, image_size=256, message_length=30)`：

- 用 `os.walk` 递归扫描 `root_dir`，支持 `.png` / `.jpg` / `.jpeg` / `.bmp`。
- 默认 transform：`Resize((image_size, image_size))` → `ToTensor()` → `Normalize([0.5]*3, [0.5]*3)`（映射到 `[-1, 1]`）。
- `__getitem__` 返回字典：

```python
{
    'cover_image': Tensor[3, H, W],          # [-1, 1]
    'message':     Tensor[message_length],   # torch.randint(0, 2, ...).float()
    'image_path':  str,
}
```

---

## 7. 模型架构模块 (`models/`)

### 7.1 编码器 (Encoder)

编码器实现**双残差**架构，`forward(cover_image, message, phase, prv_noise_ratio)` 返回 `(encoded_image, residual_penalty, msg_residual, prv_residual)`。

#### 共享层

- 16 个 `ConvBNRelu`（3→128 → 15×(128→128)），每个为 `Conv2d(3×3, bias=False) + BatchNorm2d + ReLU`。
- 消息空间扩展：`message (B, L)` → `unsqueeze` 后 `expand` 到 `(B, L, H, W)`。
- 融合层：`cat[expanded_message, features, cover_image]`（通道 30+128+3=161）→ `ConvBNRelu(161, 128)`。

#### 消息头（msg_head）

`ConvBNRelu(128→128) → ConvBNRelu(128→128) → Conv2d(1×1, 128→3)`，输出经 `msg_budget × tanh` 得 `msg_residual`。Phase 2 起全程冻结。

#### 隐私独立支路（prv_trunk）

```python
ConvBNRelu(3, 128)                       # dilation 1
ConvBNRelu(128, 128, dilation=3)
ConvBNRelu(128, 128, dilation=9)
ConvBNRelu(128, 128, dilation=27)
```

- 直接以载体图像为输入（不经过共享层），输出 128 通道特征与共享特征拼接。
- 感受野 81×81，解决 5×5 小感受野算不出空间结构化逐图攻击模式的容量瓶颈。
- 绕开只为消息目标训练且全程冻结的共享层（信息瓶颈）。
- Phase 1 与 `prv_head` 一起冻结；Phase 2 解冻，与 `prv_head` 共用 `prv_head_lr_mult`（×5）加速参数组。

#### 隐私头（prv_head）

输入 `cat[shared_features, prv_trunk(cover)]`（128+128=256 通道）：`ConvBNRelu(256→128) → ConvBNRelu(128→128) → Conv2d(1×1, 128→3)`。

> 兼容性提示：旧结构 checkpoint 的 `prv_head` 首层输入为 128 通道，新增 `prv_trunk` 后为 256 通道，形状不匹配。`train.py` 的 `load_encoder_state_compatible` 与 `val/validation_script.py` 会自动跳过不兼容权重（隐私支路由 Phase 2 重训，消息通路完整复用）。

#### 隐私残差约束链（顺序固定）

```
prv_head 输出 (B,3,H,W)
  → F.adaptive_avg_pool2d 到 (96, 96)          # 结构性低通
  → F.interpolate 双三次上采样回 (H, W)
  → prv_budget × tanh                           # 预算约束
  → × texture_mask                              # 纹理掩码加权
prv_residual
```

编码图像：`clamp(cover + msg_residual + prv_residual, -1, 1)`。

96×96 的依据：CLIP ViT-B/32 patch 为 16×16 像素，对 patch 尺度中频结构（32~96 周期/图）最敏感；48×48 低通（<24 周期）恰好把残差约束在 CLIP 最不敏感的频段（实测 privacy 损失占 88% 但 sim 几乎不降）；96 网格 ≈ 2.7px/格，上采样后仍结构性消除 >96 周期的高频条纹，FFT 惩罚兜底不变。

#### 纹理掩码（不参与反向传播）

1. `cover` 转灰度，计算垂直/水平梯度；
2. 以 `texture_mask_smooth_kernel`（默认 5）均值池化平滑；
3. **逐图**计算分位数 `q05` / `q95`，归一化 `(grad − q05)/(q95 − q05)` 并 clamp 到 [0, 1]；
4. 映射 `mask = 0.15 + 0.85 × g_norm`，输出 `(B, 1, H, W)`，取值 [0.15, 1.0]。

#### Phase 1 哑隐私噪声（预适应）

Phase 1 中 `prv_residual = None`，注入经平滑与归一化的随机噪声：

```
randn → avg_pool 平滑(核7) → 逐样本零均值 → 逐样本单位方差
     → prv_budget × prv_noise_ratio × tanh(noise)
```

编码图像 = `cover + msg_residual + dummy_prv`。注入比例由 `get_prv_noise_ratio(epoch)` 调度（前 80% 轮次为 0，之后渐进到 0.30）。设计意图：让消息通路在"图像上存在与隐私残差同分布的干扰"这一前提下预适应，Phase 2 换成真实隐私残差时解码通路不发生分布突变。

### 7.2 解码器 (Decoder)

`ConvBNRelu(3→64) → 7×ConvBNRelu(64→64) → ConvBNRelu(64→30) → AdaptiveAvgPool2d(1,1) → Linear(30, 30)`。

输出消息 logits `(B, message_length)`，**未经 sigmoid**——训练损失使用 `BCEWithLogitsLoss`（sigmoid 内置以提升数值稳定性），计算损失前不要额外加 sigmoid。

### 7.3 隐写分析器 (Steganalyzer)

判别器式二分类器，默认启用 SpectralNorm：

```
Conv(3→64)  → Conv(64→64)  → MaxPool(2)
Conv(64→128) → Conv(128→128) → MaxPool(2)
Conv(128→256) → Conv(256→256) → MaxPool(2)
Conv(256→512) → Conv(512→512) → AdaptiveAvgPool(1,1)
FC: Linear(512→256) + LeakyReLU(0.2) + Linear(256→1)
```

输出 logits `(B, 1)`，配合 Hinge Loss（margin=1.0）使用。当前配置（各阶段 adv 权重 0）仅作监控。

### 7.4 内容识别器 (ContentRecognizer) 与 DINOv2

**CLIP ViT-B/32**（`content_recognizer.py`）：

- `clip.load('ViT-B/32')`，全部参数冻结，`eval()` 模式。
- 前向：双线性插值到 224×224 → `[-1,1]` 映射到 `[0,1]` → CLIP 均值 `[0.481, 0.458, 0.408]` / 标准差 `[0.269, 0.261, 0.275]` 归一化 → `encode_image` → L2 归一化。
- 输出 512 维归一化特征；`encode_text` 同样输出 L2 归一化文本特征。

**DINOv2 ViT-S/14**（`trainer.py` / `val/validation_script.py` 中加载）：

- `timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=224)`。
- 权重从 `Config.dino_model_path`（`./models/dinov2_vits14_pretrain.pth`）加载；`pos_embed` 形状不一致时做 2D 插值后以 `strict=False` 加载；**权重文件缺失时警告并使用随机初始化**。
- 冻结、eval；推理均值 `[0.485, 0.456, 0.406]` / 标准差 `[0.229, 0.224, 0.225]`，输出 384 维特征。
- 仅当 `dino_weight > 0` 时启用。

---

## 8. 损失函数模块 (`losses/`)

### 8.1 SSIM 损失 (`ssim_loss.py`)

标准结构相似性损失，与 MSE 共同构成视觉损失 `L_visual = lambda_pixel × MSE + lambda_ssim × (1 − SSIM)`。

### 8.2 特征重定向损失 (`advanced_privacy_loss.py`)

`FeatureRedirectionLoss` 组合三类损失，CLIP 与 DINO 各自独立计算：

**Escape（逃逸）损失** —— 主项为二次 Hinge，预热项为 Softplus：

```python
sim   = cosine(feat_encoded, feat_cover)          # 逐样本
diff  = sim - (1 - margin)
loss  = mean(relu(diff) ** 2) * 25.0             # 二次 Hinge（放大 25×）
      + mean(softplus(diff)) * 0.1               # Softplus 预热（小偏离也有梯度）
```

margin 越大，逃逸目标 `sim < 1 − margin` 越激进；`sim` 低于目标后梯度归零（这是 margin 成为第一杠杆的结构原因，见 11.9）。

**Redirect（有向重定向）损失** —— 记忆库硬负样本方向：

```python
sim_to_neg    = cosine(feat_encoded, hard_negatives)      # (B, k)
hardest_neg   = 每样本取最相似负样本
sim_to_hardest = cosine(feat_encoded, hardest_neg)
redirect_loss = mean(relu(target_sim - sim_to_hardest))     # 线性 Hinge 拉升
control_loss  = mean(relu(sim_to_hardest - max_target_sim)) # 防过头
```

把特征推向语义相近但不同的目标，防止逃逸后落入其他敏感区域；`control_loss` 计入总损失。

**总损失组装**：`total = clip_weight × clip_escape (+ clip_redirect_weight × (clip_redirect + clip_control)) + dino_weight × dino_escape (+ dino_redirect_weight × (dino_redirect + dino_control))`，返回 `(loss, loss_dict)`。escape margin 由训练侧通过 `Config.get_escape_margin` / `get_dino_escape_margin` 每步传入（构造函数默认值仅为兜底）。

### 8.3 训练侧损失组装（`trainer.py`）

**隐私双分支**（噪声层雅可比会毒化 CLIP 梯度——quant 傅里叶近似梯度在 1~21 倍震荡、jpeg 近似劣质，只在含噪图上算会完全推不动）：

- 干净分支：escape + redirect（主力方向）；
- 含噪分支：仅 escape（锚定传播场景鲁棒性）；
- 融合：`L_privacy = (1 − noised_ratio) × L_clean + noised_ratio × L_noised`。

**消息双分支**：干净编码图 + 含噪编码图分别过 decoder，`L_message = clean_ratio × BCE(clean) + (1 − clean_ratio) × BCE(noised)`；`clean_ratio` 由 `get_msg_clean_ratio` 调度（Phase 1 从 0.8 退火到 0.5 保冷启动——仅含噪单分支会让随机初始化的微弱信号被信道摧毁、L_message 锁死在 ln2）。

**含噪视觉权重恒为 0**（所有阶段）：含噪视觉损失会把信道失真归咎于残差，与消息目标直接对抗，同样导致躺平。

**FFT 高频能量惩罚**（仅 Balance 段）：

```python
fft        = torch.fft.fft2(residual, norm='ortho')
amplitude  = |fft| → fftshift                     # 零频移到中心
radius     = max(1, int((min(H,W)//2) × 0.25))    # 256 图约 r=32
L_fft      = mean(amplitude × (dist > radius))    # 半径外线性幅值均值 × lambda_fft(50)
```

实现说明（代码中的修复点）：原半径 `min(H,W)//4` 只覆盖 r>64 的极高频，周期 4~8px 的可见条纹（r=32~64）完全豁免，现半径由 `fft_high_freq_radius_ratio` 控制；原 `sqrt(mean(A))` 形态梯度随违规增大而递减（病态），现改为 `mean(A)` 线性形式，梯度恒定、与违规程度成正比。

**内环蒸馏损失**：`L_distill = mse(prv_residual, teacher) / (teacher.pow(2).mean() + 1e-7)`——能量归一化 MSE。依据：原始残差空间 MSE 量级 ~1e-5（残差 RMS ~0.015），加权后完全不可见；归一化后师生相对平方误差 ~0.4，加权 25 后与其他损失项同量级且以高信噪比的直接回归信号主导残差更新方向。

---

## 9. 噪声层模块 (`noise_layers/`)

`noise_argparser.py` 提供自定义 argparse Action，接受 `+` 分隔的组合字符串：

```
crop((hmin,hmax),(wmin,wmax))    随机裁剪（对编码图与原图施加相同窗口）
cropout((hmin,hmax),(wmin,wmax)) 窗口外用原图像素替换
dropout(keep_min,keep_max)        逐像素二值掩码，0 处用原图替换
resize(min,max)                  先缩放再缩放回（双线性，可微）
jpeg                             DCT 域量化压缩
quant                            傅里叶级数近似 round 量化
```

`Noiser` 每次前向**随机选择一个**激活噪声层（空配置时退化为 `Identity`）。所有层接收并返回 `(noised_image, cover_image)` 二元组。

| 噪声层 | 可微性 | 实现要点 |
| --- | --- | --- |
| Identity | 可微 | 恒等映射 |
| Crop / Cropout / Dropout | 不可微（离散选择） | 随机窗口/掩码，混合原图像素 |
| Resize | 可微 | `F.interpolate` 双线性 |
| JpegCompression | 可微 | RGB→YUV → 卷积实现 DCT → Zigzag 保留低频（`yuv_keep_weights=(25,9,9)`）→ IDCT → RGB |
| Quantization | 可微 | 傅里叶级数近似 round（`N=10` 项，`weights[n]=(−1)^(n+1)/(π(n+1))`） |

默认配置 `'jpeg+resize(0.8,1.2)+quant'`：溯源水印必须在失真信道上训练才具备实际鲁棒性。

---

## 10. 训练系统 (`trainer.py` & `train.py`)

### 10.1 Trainer 初始化

- 模型：`Encoder` / `Decoder` / `Steganalyzer`（SpectralNorm）/ `ContentRecognizer`（冻结 CLIP）；`dino_weight > 0` 时额外加载 DINOv2。
- **单个 Adam 优化器三个参数组**：
  1. 编码器非隐私参数（共享层、msg_head）——`lr_encoder`；
  2. `prv_head` / `prv_trunk` 参数——`lr_encoder × prv_head_lr_mult`（×5）；
  3. decoder 参数——`lr_decoder`。
- 判别器独立 Adam（`lr_steganalyzer`）。
- `use_clip_redirect=True` 时创建 CLIP 记忆库（512 维）；DINO 启用时另建 384 维记忆库（容量均 `clip_memory_size=5000`）。

### 10.2 Phase 1（默认 150 轮）：基础隐写

- 冻结：`prv_head` + `prv_trunk`；训练：共享层 + `msg_head` + decoder。
- 编码图 = 原图 + 消息残差 + 渐进哑隐私噪声（见 7.1）。
- 权重：`message=15, visual=10, privacy=0, adv=0, residual_penalty=0.5, fft=0`。
- 判别器仅在 `adv > 0` 且 `D_acc < disc_acc_high(0.70)` 时更新；adv=0 时纯监控。
- 开始时记忆库预热 `memory_warmup_batches=50` 个 batch（只入库不训练），避免训练初期记忆库为空导致 hard negative mining 失效。

### 10.3 Phase 2（默认 100 轮）：三段调度

`msg_head` 与共享层**全程冻结**（不再按 epoch 分段解冻）。三段判定优先级：**Refine > Sprint > Balance**（均以**本次运行的阶段内相对轮次** `sched_epoch` 计算）。

| 段 | 轮次（默认） | decoder | message | visual | privacy | fft / residual_penalty | 含噪分支 | margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Sprint** | 第 1~40 轮 | 冻结 | 0 | 1.5 | 12 | 清零 | 0 | 直接取 end 值 |
| **Balance** | 第 41~70 轮 | 解冻 | 20/15/10 | 15 | 4/8/12 | 50 / 0.5 | 0.3 | 前 50% 轮渐进到 end |
| **Refine** | 第 71~100 轮 | 冻结 | 10 | 15 | 12 | 清零 | 0 | 直接取 end 值 |

**Sprint 段**（纯逃逸冲刺）设计依据：天花板探针证明同约束下直接优化 60 步即可把 clip_sim 压到 0.64，而正常 Phase 2 的 20000 步停在 0.946——瓶颈是 Adam 在多目标、逐图方向不一致下的信噪比失效（Adam 对梯度尺度不变，加大损失权重无效）。反作用力全部清零：`residual_penalty`（收缩残差直接对冲逃逸梯度）、`fft`（压中频削弱 CLIP 敏感的逃逸方向）、`message`（纯逃逸）、含噪分支（劣质噪声雅可比会稀释本就微弱的逃逸信号）。Sprint 段 bit_acc 暂时下降是预期行为，不触发 recovery、不参与 best 判定惩罚。

**Balance 段**（多目标平衡）：恢复正常组合，decoder 解冻恢复解码，privacy 权重三段递增 4/8/12（补偿雅可比失衡），recovery 机制兜底。

**Refine 终段**：切换为 `val/refine_encode.py` 已验证配方目标（visual 15 / message 10 / privacy 12，反作用力清零），margin 固定 end 值，redirect 关闭（redirect 与 escape 方向对冲）。与 Sprint 的本质差异：Sprint 靠蒸馏教师间接保解码（message=0），Refine 由学生目标直接保解码（message=10，decoder 冻结——残差自适应既有解码器而非解码器迁就残差），因此终段可放心压 clip_sim 而不会把 bit_acc 打穿阈值。热启动纯精修 = `--epochs-phase2 30`（refine=N 全覆盖时无需显式传 `--refine-epochs`）。

### 10.4 冻结策略汇总

| 阶段/段 | 共享层 | msg_head | decoder | prv_head / prv_trunk |
| --- | --- | --- | --- | --- |
| Phase 1 | 解冻 | 解冻 | 解冻 | 冻结 |
| Phase 2 Sprint | 冻结 | 冻结 | 冻结 | 解冻 |
| Phase 2 Balance | 冻结 | 冻结 | 解冻 | 解冻 |
| Phase 2 Refine | 冻结 | 冻结 | 冻结 | 解冻 |

### 10.5 内环逐图优化蒸馏（`_refine_teacher_residual`）

Phase 2 全程启用（`distill_inner_steps > 0` 时；可用 `--distill-steps 0` 关闭）：

1. 冻结全部网络参数；
2. **优化变量**：96×96 低分辨率残差场 `delta`，从网络当前输出热启动——
   `ratio = (prv_residual / (budget × texture_mask + ε)).clamp(−0.95, 0.95)`，`delta = avg_pool(atanh(ratio), 96)`；
3. **教师目标**（与逐图精修脚本验证配方一致）：
   - escape-only 隐私损失（无 redirect），margin 固定取 `privacy_margin_end` / `dino_escape_margin_end`；
   - 消息双分支 BCE（0.5 干净 + 0.5 含噪）；
   - 视觉 `MSE + SSIM`；
   - 加权 `distill_w_privacy × L_priv + distill_w_message × L_msg + distill_w_visual × L_vis`；
   - 约束链与训练同构：`interpolate(delta → H×W) → budget × tanh → × texture_mask → clamp`；
4. 跑 `distill_inner_steps`（10）步 Adam（`distill_lr=0.1`）得到教师残差；
5. 学生损失：`distill_weight(25) × 能量归一化 MSE(prv_residual, teacher)`，把探针级逃逸方向蒸进单次前向。

代价：Phase 2 单步耗时约 2~4 倍。Sprint 段蒸馏教师含消息 BCE，起到保解码锚的作用。

### 10.6 Decoder Recovery

- **触发**：仅 Balance 段（Sprint/Refine 明确跳过并打日志），当验证 `bit_accuracy < bit_accuracy_threshold`（0.90）。
- **恢复**：冻结编码器与隐私支路，用临时 Adam（`lr_decoder × 0.5`）在**含隐私残差的完整编码图 + 信道噪声**上重训 decoder `recovery_steps`（10）个 batch——与实际解码分布一致。

### 10.7 记忆库（FeatureMemoryBank）

- **先查询后入库**：每个 train step 先用干净分支编码特征查询 hard negatives，计算完隐私损失后再把当前 batch 的载体特征入库（避免检索到自身）。
- 检索 top-`clip_hard_negative_k`（5）个，按空间分别用 `clip/dino_redirect_exclude_sim`（0.92/0.87）排除相似度过高的近重复候选。
- Refine 段关闭 redirect（不查询、不入库相关目标）。

### 10.8 Best Checkpoint 判定

| 阶段 | 准则 |
| --- | --- |
| Phase 1 | 验证 `L_message` 最小 |
| Phase 2 | `clip_sim_noised + dino_sim_noised` 最小（无含噪指标时回退干净指标）；`bit_accuracy < 0.90` 时加 10.0 惩罚分 |

命中时额外保存 `checkpoints/checkpoint_best.pth`。两阶段准则数值不可比（L_message 收敛后 ~0.1-0.3 而 sim 和 ~1.4-1.9），检测到准则所属阶段切换时重置 `best_loss`（`Trainer.best_loss_phase` 记录当前最优的归属阶段，随 checkpoint 持久化；`continue` 续训同阶段则保留历史最优，旧格式 checkpoint 缺该字段时首轮按当前阶段准则重判）——否则从头训练时 Phase 2 的 best 分数永远高于 Phase 1 遗留值，best 会冻结在 Phase 1 整个 Phase 2 不更新。

### 10.9 轮次语义：相对轮次 vs 绝对轮号

- 训练循环把**阶段内相对轮次**（本次运行从第 1 轮起算）传给 `train_step` / `validate` / 权重调度函数（`train.py run_training_loop` 的 `sched_epoch`）——从旧 checkpoint 热启动重训 Phase 2 时 Sprint/Balance/Refine 能正确触发。
- checkpoint 保存与日志使用**绝对轮号**；`continue` 子命令恢复时从 `epoch + 1` 续跑。

### 10.10 热启动与断点续训

- `new --continue-from-folder`：**纯权重初始化**。训练计划由本次 CLI 参数决定，从 Phase 1 第 1 轮起算；`best_loss` 不继承、`phase2_msg_head_frozen` 复位。绝不携带 checkpoint 的绝对轮号（否则旧 Phase 2 epoch 100 会使 Phase 2 循环为空）。
- `load_encoder_state_compatible`：跳过与新结构形状不兼容的编码器权重（如旧 `prv_head` 128→256 通道变化），消息通路完整复用，隐私支路重训。
- 直入 Phase 2 时若 checkpoint 未携带记忆库则自动补预热。
- 断点续训（携带绝对轮号）使用 `continue` 子命令。

### 10.11 判别器训练

- 损失：Hinge Loss（margin=`disc_hinge_margin`=1.0）；生成器侧为 Softplus 非饱和损失。
- 门控：仅 `adv > 0` 且 `D_acc < disc_acc_high`（0.70）时更新；梯度裁剪 `clip_grad_norm_(1.0)`。
- 当前所有阶段 `adv=0`：判别器仅监控，不回传梯度。

---

## 11. 验证与评估工具链 (`val/`)

所有脚本统一从项目根运行：`python val/<脚本名>.py ...`。脚本启动时自动把项目根加入 `sys.path` 并切换工作目录到项目根，相对路径行为与脚本在项目根时完全一致。图表与 HTML 报告文本统一使用英文（matplotlib 默认 DejaVu Sans 含完整英文字形，任何平台无需额外字体文件）。

### 11.1 单图验证 (`validation_script.py`)

```bash
python val/validation_script.py -c ./experiments/exp/checkpoints/checkpoint.pth \
  -i ./test.jpg -m "101000100110100101100011100010" --mode full -o output
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--mode` | `full` | `full` / `decode` / `discriminator` / `similarity` |
| `--refine-steps` | 30 | 逐图精修步数（0 关闭） |
| `--refine-lr` | 0.1 | 精修学习率 |
| `--refine-budget` | 0.08 | 精修残差预算（深挖配方默认） |
| `--refine-margin` | 0.50 | 精修逃逸目标（深挖配方默认） |
| `--refine-chroma-projection` | None(取 Config) | 精修收敛后剔除 prv 低频色度（1 开 / 0 关） |

**full 模式流程**：加载 → 编码 → 逐图精修 →（解码/判别器/相似度/保存全部基于精修后图像）→ 自动调用 `visual_report.py` 生成 `report.html`。

**精修配方**（`refine_encoded_image`，与 `refine_encode.py` 同源）：冻结全部网络；优化变量为 96×96 低分辨率隐私残差场（从网络原始残差经 `atanh` 变换热启动，即**初始候选 = 网络原始输出，保证不劣化**）；目标 `w_privacy × escape(margin) + w_message × 双分支BCE + w_visual × (MSE+SSIM)`（12/10/15）；约束链 `interpolate → tanh × budget → × texture_mask → clamp`；**只在 `bit_acc ≥ 1.0` 的状态里选 clip_sim 最低者**为最终结果。

**色度投影**（`_project_out_chroma_lowfreq`，精修收敛后默认执行）：剔除隐私残差 8×8 网格低频色度分量（BT.601 YCbCr 变换，Y 通道不变），门控保证投影后 `bit_acc=100%` 且 `clip_sim ≤ (1−margin)` 否则回退未投影结果。机制与实测见 11.8 节。

其他模式：`decode` 仅解码消息；`discriminator` 仅判别器预测；`similarity` 需 `--cover` 提供，计算原图与隐写图的 CLIP/DINO/SSIM/PSNR 相似度。

### 11.2 单图基线验证 (`baseline_validation.py`)

网络直出（无逐图精修、不生成 report.html），复用 `ValidationScript` 的全部组件与流程：

```bash
python val/baseline_validation.py -c ./other/checkpoint.pth -i ./other/test.jpg \
  -m "101000100110100101100011100010" --device cpu -o other/baseline_output
```

参数：`-c` / `-i` / `-m`（默认全 0 串）/ `-o` / `--device`。输出编码图与全指标，汇总格式与 full 模式同口径，用作"网络单独 vs 网络+编码期精修"的对照基线（amortization gap 分析）或 checkpoint 原生质量的快速检查。

### 11.3 独立逐图精修 (`refine_encode.py`)

不训练网络，对任意 checkpoint 的**单张或多张**图像（`-i` 接受多路径）直接优化隐私残差：

```bash
python val/refine_encode.py -c ./exp/checkpoints/checkpoint.pth \
  -i ./test.jpg -m "101..." --steps 30 --margin-end 0.50 --prv-budget 0.08 -o other/refined
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--steps` | 60 | 优化步数 |
| `--lr` | 0.1 | 学习率 |
| `--w-privacy` / `--w-message` / `--w-visual` | 12 / 10 / 15 | 目标各项权重 |
| `--prv-budget` / `--margin-end` | 0.08 / 0.50 | 覆盖约束链与逃逸目标（**训练前试探预算/margin 组合的逃逸天花板**） |
| `--chroma-projection` | None(取 Config) | 收敛后色度投影（1/0） |
| `--noise-eval` | 8 | 信道鲁棒性采样次数 |

每图输出 `metrics.json`，末尾汇总 `summary.json`。

### 11.4 批量定量评估 (`batch_eval.py`，精修版)

```bash
python val/batch_eval.py -c ./other/checkpoint.pth --val-dir ./dataset1/val \
  --num-images 200 --refine-steps 30 -o other/batch_eval --device cpu
```

- 从 `--val-dir` 递归采样 `--num-images`（默认 200）张，每张随机消息，逐一编码+精修（深挖配方默认 budget 0.08 / margin 0.50 / 色度投影取 Config）。
- 输出基线/精修的 CLIP/DINO/bit_acc/PSNR/SSIM 统计（均值±标准差、分位数）、信道鲁棒性（`--noise-eval` 默认 8 次随机噪声采样的解码率与相似度）、uint8 往返（内存内模拟 PNG 8bit 量化）。
- `--save-images`（默认开）/ `--no-save-images`；`results.json` **增量保存支持断点续跑**（中断后原命令重跑自动跳过已完成部分；从头重跑须换输出目录或删除旧 `results.json`）；`--report-only` 不加载模型仅重出报告。
- 输出：`results.json` / `summary.json` / `details.csv`（逐图明细）/ `report.html` + 4 张 PNG 面板。

### 11.5 批量基线评估 (`batch_baseline.py`，网络直出)

复用 `batch_eval.py` 的采样/噪声/往返公共函数（同 seed 同 val-dir 时评估同一批图像，与含精修版直接对照），网络单次前向后计算全指标+信道鲁棒性+uint8 往返；单条件记录（无 base/refined 冗余结构），**速度约为含精修版 20~30 倍**，适合大规模快速评估 checkpoint 原生质量或论文 amortization gap 分析的基线列。

```bash
python val/batch_baseline.py -c other/checkpoint.pth --num-images 200 -o other/batch_baseline --device cpu
```

输出同样为 `results.json` / `summary.json` / `details.csv` / 可视化报告（断点续跑）；默认不保存编码图（`--save-images` 开启）。

### 11.6 CBIR 检索逃逸评估 (`retrieval_eval.py`，精修版)

```bash
python val/retrieval_eval.py -c ./other/checkpoint.pth --val-dir ./dataset1/val \
  --num-queries 100 --gallery-size 1000 --refine-steps 30 \
  --refine-margin 0.50 --refine-budget 0.08 -o other/retrieval_eval --device cpu
```

- 攻击者视角测试：图库从 `--val-dir` 采样（`--extra-gallery-dir` + `--extra-gallery-size` 可从 train 集扩充干扰池），**查询集为图库子集且原图保留在图库内**（检索才有"命中"可言）。
- 编码图（基线）与精修图分别作为查询对全图库做 CLIP 余弦检索，报告 top-1/5/10 命中率、逃逸率（=1−top1）、平均排名、中位排名、MRR 与"原图相似度 vs 最强干扰相似度"竞争余量。
- **自检**：原图自查询应 100% top-1（特征链路正确性校验）。
- 图库特征缓存至 `gallery_feats.pt`（路径集合命中缓存即复用）；`eval_meta.json` 记录图库规模等元信息；`results.json` 断点续跑；`--report-only` 仅重出报告。
- 实测（20 图库、深挖配方）：基线 top-1 命中 100%、精修后逃逸率 66.7%。

### 11.7 CBIR 检索逃逸基线评估 (`retrieval_baseline.py`，网络直出)

复用 `retrieval_eval.py` 的图库构造/特征缓存/检索函数，编码图单次前向后直接作查询；图库特征缓存 `gallery_feats.pt` 与精修版共用（同图库配置直接命中，无需重新提取）；单条件记录，报告 top-k 命中率、逃逸率、MRR、自检；断点续跑 + 可视化报告。

```bash
python val/retrieval_baseline.py -c other/checkpoint.pth --num-queries 100 --gallery-size 1000 -o other/retrieval_baseline --device cpu
```

### 11.8 全链路一键验证 (`run_full_validation.py`)

编排 val/ 下五个评估脚本（快阶段先行、失败早暴露），各阶段断点续跑：

```bash
python val/run_full_validation.py -c ./experiments/full_run/checkpoints/checkpoint_best.pth \
  --val-dir ./dataset/test -o ./other/full_validation
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-c` | 必填 | checkpoint 路径 |
| `--val-dir` | `./dataset/test` | 测试集目录 |
| `--num-images` | 1000 | 批量评估规模 |
| `--num-queries` / `--gallery-size` | 100 / 1000 | 检索评估规模 |
| `--extra-gallery-dir` / `--extra-gallery-size` | None / 0 | 图库扩充 |
| `--refine-steps/-lr/-budget/-margin/-chroma-projection` | 30 / 0.1 / 0.08 / 0.50 / None | 精修配方（深挖默认） |
| `--noise-eval` | 8 | 信道鲁棒性采样 |
| `--message` | 30 bit 默认串 | 单图全链路验证消息 |
| `--seed` | 42 | 全局种子（同 seed 同 val-dir 保证基线与精修评估同一批图像） |
| `--no-save-images` | 否 | 批量精修评估不保存精修编码图 |
| `--stages` | `all` | 阶段子集：`smoke,single,batch_baseline,retrieval_baseline,batch_eval,retrieval_eval,summary` |
| `--report-only` | 否 | 从已有 results.json 重出各子报告与汇总 |
| `--dry-run` | 否 | 只打印将执行的命令 |

**阶段顺序**（STAGE_ORDER）：`smoke → single → batch_baseline → retrieval_baseline → batch_eval → retrieval_eval → summary`。

- **冒烟门控（smoke）**：`_smoke` 临时目录，每次全量重跑。校验检索自检（原图自查询）top-1=100%、直出 bit_acc≥0.5、精修链路可用后才开始全量——特征链路或模型损坏能最快暴露。
- **缓存复用**：检索两脚本图库配置一致时，把基线阶段的 `gallery_feats.pt` 复制给精修版（缓存按路径列表自校验）。
- **汇总（summary）**：读取四个 `summary.json` 生成 `full_summary.json` + 英文 `full_summary.md`（网络直出/精修三列对照表，可直接进论文），并做两路直出结果一致性交叉检查（两路直出 clip_sim 均值差 >0.02 或检索 top-1 差 >5% 时警告 seed/配置不一致）。
- 子脚本自带断点续跑：任一阶段中断后重跑同一条命令自动续；从头重跑需换 `--output-root` 或删除对应子目录的 `results.json`。

### 11.9 可视化报告生成器

**`visual_report.py`**（单图验证，full 模式自动调用）：自包含 `report.html`（base64 内嵌图表）+ 独立 PNG 面板。内容：核心指标卡片（bit_acc/CLIP·DINO 隐私保护/PSNR/SSIM/判别器，按阈值标色）、原图-编码图-残差对比、精修逃逸轨迹（相似度与 bit_acc/PSNR 随步数变化）、隐私逃逸对比条形图（附逃逸目标线）、消息比特核验面板（逐比特正确性+解码置信度）、精修前后对照表。

**`eval_visualizer.py`**（批量/检索评估，评估结束自动调用，可 `--report-only`）：批量报告含相似度分布对比、逐图改善哑铃图、质量-隐私权衡散点、鲁棒性对比、全指标对照表；检索报告含排名分布/CDF、逐查询排名散点（log-log）、相似度竞争散点（逃逸机制）、全指标对照表；基线版（`generate_batch_baseline_report`/`generate_retrieval_baseline_report`）为单条件报告。

两模块在 matplotlib 缺失或生成失败时自动跳过，不影响评估流程本身。

### 11.10 评估工作流建议

1. 正式评估前先小规模冒烟：`--num-images 3` / `--num-queries 3 --gallery-size 20`（各约 2-3 分钟）。
2. 确认检索输出末尾"自检（原图自查询 top-1 命中率）"为 100% 且基线 top-1 命中接近 100% 后再跑正式规模。
3. 正式规模耗时参考：批量 200 张约 2~2.5 小时 CPU；检索 100 查询/1000 图库约 1.5~2 小时 CPU；A100 分钟级。
4. 双配方对照（深挖 margin 0.50/budget 0.08 默认 vs 标准 margin 0.30/budget 0.04 显式指定 `--refine-budget 0.04 --refine-margin 0.30` 各跑一遍）构成"视觉质量-隐私保护"权衡表。
5. 精修步数 30 与 60 的逃逸效果基本相同（逃逸主要发生在前 30 步）——30 步砍掉一半时间几乎不损失效果。
6. 一键全链路：直接用 `run_full_validation.py`（内置冒烟门控与汇总对照表）。

---

## 12. 核心算法原理

### 12.1 双残差编码

消息残差与隐私残差分离产生、独立预算约束、独立冻结调度：

- **解耦**：消息目标（高频、逐比特可靠）与隐私目标（中频、patch 尺度结构化）在频段与梯度方向上都不同，单残差会互相牵制。
- **复用**：热启动重训隐私支路时消息通路完整复用（直入 Phase 2 首轮 val bit_acc 0.925）。
- **保解码**：精修不动 `msg_residual`，只在比特 100% 正确状态里选优。

### 12.2 多特征空间逃逸

CLIP（语义）+ DINOv2（细粒度纹理）双空间二次 Hinge 逃逸，margin 独立调度：

- 攻击者可能改用自监督特征做近重复检测，单逃 CLIP 不够；
- CLIP/DINO 特征空间的典型 hard negative 相似度不同（≈0.73 vs ≈0.47），redirect 目标与 margin 必须分别标定。

### 12.3 频段设计与纹理掩码

- 96×96 低通把残差约束在 CLIP 敏感、人眼不敏感的中频段（2.7px/格，覆盖 32~96 周期/图）；48×48 低通实测逃逸失效（privacy 占损失 88% 但 sim 几乎不降）；
- 纹理掩码把幅度集中到梯度复杂区（q05/q95 → [0.15, 1.0]，下限 0.15 保证逃逸余量）；
- 两者合起来使逃逸与视觉质量不冲突（margin 0.50 + budget 0.08 下仍 PSNR 38.95 / SSIM 0.957）。

### 12.4 记忆库硬负样本有向重定向

逃逸只约束"离开原图"，不约束"去向哪里"——无约束逃逸可能落入其他敏感区域（例如检索到另一张同场景图）。Redirect 从记忆库检索与当前编码特征最相似的**其他**载体特征（排除近重复），以线性 Hinge 把相似度拉升到 `target_sim` 即停（CLIP 0.85 / DINO 0.60），并用 `control_loss` 封顶（0.90 / 0.65）防止过头变成他人图像的"副本"。

### 12.5 渐进逃逸边界

Balance 段 margin 在前 50% 轮次内从 start 渐进到 end：二次 Hinge 的梯度 `∝ 2×(sim − target)`，margin 突变会造成目标跳变与优化震荡；渐进让网络先在温和目标下建立方向，再逐步收紧。Sprint/Refine 段直接取 end 值（纯逃逸阶段无需渐进——实测 40 轮 sprint 中若走 ramp，margin 仅 0.10→0.26，冲刺前期梯度被削弱 ~3 倍）。

### 12.6 蒸馏与部署精修（摊销鸿沟）

网络是"摊销求解器"——用一次前向近似逐图优化问题，存在不可消除的摊销鸿沟。三层递进：

1. **内环蒸馏**（训练期）：教师残差是当前 batch 的逐图最优解近似，学生通过归一化 MSE 学习教师的**方向与能量分布**（而非绝对值——残差绝对值量级太小）。
2. **部署精修**（编码期）：对单图真正求解。以网络输出为初始候选（保证不劣化），30~60 步 Adam。
3. **消息锚硬保证**：所有时刻只在 bit_acc=100% 的可行域内选优。

### 12.7 色度低频投影（发灰/蒙层修复）

**问题**：实测隐私残差能量 77% 落在色度通道、低频（8×8 块均值尺度）占其 44%，且总残差低频色度的 97% 来自 prv_residual——构成"发灰/蒙层"感知劣化主体，而 PSNR/SSIM 对此失明。

**为什么不用梯度惩罚**：实测 w=40~5000 的色度惩罚均被逃逸梯度完全淹没——逃逸方向与色度方向高度平行（CLIP 逃逸天然偏好色度通道），惩罚项与主目标对冲，形同虚设。

**方案（结构化事后投影）**：精修收敛后，把 `prv_residual` 的 8×8 网格低频色度分量整支剔除：

```
YCbCr 变换（BT.601）→ Cb/Cr 各做 8×8 自适应均值池化 → 双线性上采样回全分辨率
→ 按逆变换系数从 RGB 残差中扣除（Y 通道不动）
```

**尺度选择**：更粗尺度（16/16 以上网格）的低频是逃逸承重结构（剔除后 clip_sim 回升到 0.57/0.80），只动 8×8 网格。

**门控回退**：投影后 bit_acc 必须 100% 且 clip_sim ≤ (1−margin)，否则回退未投影结果。

**实测**（test.jpg）：clip 0.438→0.467（仍在 margin 0.50 目标内）、低频色度 −91%、PSNR +0.5 dB、SSIM +0.001、bit 100%。

### 12.8 逃逸杠杆与实测（CPU 逐图精修，60 步，bit_acc 全程 100%）

| 杠杆 | 取值 | CLIP sim |
| --- | --- | --- |
| margin（budget 0.08） | 0.30 / 0.40 / 0.50 | 0.575 / 0.547 / 0.440 |
| budget（margin 0.30） | 0.04 / 0.06 / 0.08 | 0.634 / 0.603 / 0.575 |
| 视觉权重 w_visual | 15 → 8 | 无影响（幅度由 margin 驱动的 hinge 平衡决定） |

margin 0.50 + budget 0.08 时 PSNR 38.95 / SSIM 0.957，仍属视觉高质量区间。margin 排第一的原因：二次 Hinge 在 sim < (1−margin) 后梯度归零，margin 直接决定 hinge 平衡点；budget 排第二：逃逸深度 ∝ 特征位移²，位移受 budget × 掩码能量约束。

### 12.9 失败模式汇编（工程经验，可作为论文 discussion）

以下每条均为实测验证过的反直觉结论（"问题 → 现象 → 归因"）：

| 失败模式 | 现象 | 归因与修复 |
| --- | --- | --- |
| 消息分支锁死 | Phase 1 仅用含噪单分支时 L_message 锁死在 ln2 | 随机初始化的微弱 msg 信号被信道摧毁；干净分支权重 0.8 冷启动 |
| 编码器躺平 | 加大 noised 视觉权重后 sim 不降、视觉不变 | 含噪视觉损失把信道失真归咎于残差；各阶段 noised 恒 0 |
| 多目标 SNR 失效 | 正常 Phase 2 20000 步停在 clip_sim 0.946，加大 privacy 权重无效 | Adam 对梯度尺度不变，多目标逐图方向不一致时信噪比失效；Sprint 段单目标 + lr×5 |
| 雅可比毒化 | 隐私损失只在含噪图上算时完全推不动 | quant 傅里叶近似梯度 1~21 倍震荡、jpeg 近似劣质；干净分支主力 + 含噪分支仅锚定 |
| redirect 形同虚设 | target 仅高于 hard negative 均值 0.02 时 redirect 占损失 <1% | 目标必须明显高于该空间典型 hard negative（CLIP 0.73→target 0.85；DINO 0.47→0.60） |
| 频段错配 | 48×48 低通时 privacy 占损失 88% 但 sim 几乎不降 | 48 网格 <24 周期把残差压进 CLIP 最不敏感频段；96 网格覆盖 patch 尺度 |
| 纹理下限过低 | 掩码下限 0.05 时逃逸天花板 sim≈0.72 贴边无余量 | 下限提至 0.15 |
| 色度惩罚无效 | w=40~5000 色度梯度惩罚全部失败 | 逃逸方向与色度方向高度平行；改为结构化事后投影 |
| 多目标回拉 | Sprint 开头 + Balance 之后模型停在折中态（val clip_sim 卡 0.87） | Balance 会把残差往"消息友好"方向回拉；终段 Refine 以精修配方收尾 |
| best 冻结 | 从头训练时 Phase 2 的 best 永远不更新 | 两阶段准则数值不可比；按 `best_loss_phase` 检测准则阶段切换并重置 |

---

## 13. 使用指南

### 13.1 环境依赖

- Python >= 3.8，PyTorch >= 2.0.0，torchvision >= 0.15.0
- CUDA（训练推荐）/ CPU（验证与评估可用）
- DINOv2 权重：`./models/dinov2_vits14_pretrain.pth`（缺失时警告并随机初始化）

依赖安装（仓库已 vendored `CLIP/`，从项目根运行时 `import clip` 直接命中，无需额外安装；也可 `pip install git+https://github.com/openai/CLIP.git`）：

```bash
pip install torch torchvision timm pillow numpy matplotlib tensorboard
```

### 13.2 训练

```bash
# 从头训练（深挖配方已全部设为代码默认值；数据集默认 ./dataset1）
python train.py new --name full_run

# 指定数据集与常用参数
python train.py new --data-dir ./dataset --name my_experiment \
  --batch-size 35 --size 256 --message 30 \
  --epochs-phase1 150 --epochs-phase2 100 \
  --noise "jpeg+resize(0.8,1.2)+quant" --seed 42

# 冒烟测试（约 1 轮 Phase 1；本地 CPU 显式传 --batch-size 1）
python train.py new --data-dir ./dataset --name smoke \
  --epochs-phase1 1 --epochs-phase2 0 --batch-size 1 --num-workers 0 --save-interval 1

# 断点续训（携带绝对轮号）
python train.py continue --folder ./experiments/my_experiment

# 热启动重训隐私支路（纯权重初始化，训练计划由本次 CLI 决定）
python train.py new --data-dir ./dataset --name privacy_v2 \
  --continue-from-folder ./experiments/old_exp \
  --epochs-phase1 0 --epochs-phase2 100
```

### 13.3 隐私深挖配方（更大预算与 margin，现为默认值，显式写出便于回溯）

```bash
python train.py new --name deep_escape \
  --continue-from-folder ./experiments/warm \
  --epochs-phase1 0 --epochs-phase2 100 --sprint-epochs 20 --refine-epochs 80 \
  --prv-budget 0.08 --privacy-margin-end 0.50 --dino-margin-end 0.20
```

### 13.4 验证与评估

见[第 11 章](#11-验证与评估工具链-val)各节命令示例。常用：

```bash
# 单图完整验证（默认 30 步精修 + 色度投影 + 可视化报告）
python val/validation_script.py -c ./experiments/exp/checkpoints/checkpoint.pth \
  -i ./test.jpg -m "101000100110100101100011100010" --mode full -o output

# 关闭精修 / 回落标准配方
python val/validation_script.py -c ... -i ./test.jpg -m "101..." -o output --refine-steps 0
python val/validation_script.py -c ... -i ./test.jpg -m "101..." -o output \
  --refine-budget 0.04 --refine-margin 0.30

# 单图基线（网络直出）
python val/baseline_validation.py -c ./other/checkpoint.pth -i ./other/test.jpg \
  -m "101000100110100101100011100010" --device cpu -o other/baseline_output

# 批量 / 检索评估（精修版与基线版）
python val/batch_eval.py -c ./other/checkpoint.pth --num-images 200 -o other/batch_eval --device cpu
python val/batch_baseline.py -c other/checkpoint.pth --num-images 200 -o other/batch_baseline --device cpu
python val/retrieval_eval.py -c ./other/checkpoint.pth --num-queries 100 --gallery-size 1000 \
  -o other/retrieval_eval --device cpu
python val/retrieval_baseline.py -c other/checkpoint.pth --num-queries 100 --gallery-size 1000 \
  -o other/retrieval_baseline --device cpu

# 全链路一键验证（冒烟门控 + 断点续跑 + 汇总对照表）
python val/run_full_validation.py -c ./experiments/full_run/checkpoints/checkpoint_best.pth \
  --val-dir ./dataset/test -o ./other/full_validation

# 仅重出报告（不加载模型）
python val/batch_eval.py --report-only -o other/batch_eval
python val/retrieval_eval.py --report-only -o other/retrieval_eval
python val/batch_baseline.py --report-only -o other/batch_baseline
python val/retrieval_baseline.py --report-only -o other/retrieval_baseline
python val/run_full_validation.py --report-only -o other/full_validation
```

### 13.5 监控训练

启用 `--tensorboard` 后：

```bash
tensorboard --logdir experiments/my_experiment/tensorboard
```

### 13.6 噪声层配置语法

```bash
--noise "jpeg"                                  # 单层
--noise "jpeg+resize(0.8,1.2)"                  # 组合（+ 分隔）
--noise "cropout((0.55,0.6),(0.55,0.6))+jpeg"   # 带参数
```

### 13.7 CLIP 一致性测试（vendored 包）

```bash
python -m pytest CLIP/tests/test_consistency.py
```

（需要 pytest；当前仓库根目录没有项目级 lint、format 或主测试配置。）

---

## 附录A：关键超参数速查表

| 参数 | 默认值 | CLI 覆盖 |
| --- | --- | --- |
| `image_size` | 256 | `--size` |
| `message_length` | 30 | `--message` |
| `batch_size` | 35（A100 口径） | `--batch-size` |
| `num_epochs_phase1` / `num_epochs_phase2` | 150 / 100 | `--epochs-phase1` / `--epochs-phase2` |
| `msg_budget` / `prv_budget` | 0.04 / 0.08 | `--prv-budget` |
| `privacy_margin_end` | 0.50 | `--privacy-margin-end` |
| `dino_escape_margin_end` | 0.20 | `--dino-margin-end` |
| `prv_lowres_size` | 96 | — |
| `prv_head_lr_mult` | 5.0 | — |
| `phase2_sprint_epochs` | 40 | `--sprint-epochs` |
| `phase2_refine_epochs` | 30 | `--refine-epochs` |
| `distill_inner_steps` | 10 | `--distill-steps` |
| `distill_weight` | 25.0 | `--distill-weight` |
| `lambda_fft` | 50.0 | — |
| `privacy_noised_ratio` | 0.3 | — |
| `use_clip_redirect` | True | — |
| `clip_redirect_target_sim` / `dino_redirect_target_sim` | 0.85 / 0.60 | — |
| `clip/dino_redirect_exclude_sim` | 0.92 / 0.87 | — |
| `clip/dino_max_target_sim` | 0.90 / 0.65 | — |
| `dino_weight` | 0.5 | — |
| `refine_chroma_projection` | True | `--refine-chroma-projection`（val 脚本） |
| `noise_config` | jpeg+resize(0.8,1.2)+quant | `--noise` |
| `bit_accuracy_threshold` | 0.90 | — |
| `recovery_steps` | 10 | — |
| `lr_encoder` / `lr_decoder` / `lr_steganalyzer` | 1e-3 / 1e-3 / 1e-4 | `--lr-*` |
| `memory_warmup_batches` | 50 | — |
| 各阶段 adv / noised 权重 | 0.0 | `--phase1-adv-*` / `--phase2-adv` |
| 纹理掩码下限 | 0.15 | — |

## 附录B：输出文件清单

**训练（`experiments/<name>/`）**：

```
config.json              # 训练配置（continue 子命令恢复用）
train.log                # 训练日志
checkpoints/
  ├── checkpoint.pth                 # 最新（含绝对轮号、优化器、记忆库、best_loss_phase）
  ├── checkpoint_best.pth            # 多准则最优
  └── checkpoint_phase{N}_epoch{M}.pth
  └── visualizations/phase{N}_epoch{M}.png   # 验证时保存的原图-编码图对照
tensorboard/             # TensorBoard 事件日志
```

**单图验证（`-o` 输出目录）**：`cover_image.png` / `encoded_image.png` / `report.html` / `panel_images.png` / `panel_trajectory.png` / `panel_privacy.png` / `panel_message.png`。

**单图基线（`baseline_validation.py`）**：`cover_image.png` / `encoded_image.png` / 指标输出（无 report.html）。

**批量评估（精修版与基线版）**：`results.json`（断点续跑）/ `summary.json` / `details.csv` / `report.html` / `panel_dist.png` / `panel_dumbbell.png` / `panel_pareto.png` / `panel_robustness.png`（基线版默认不含精修图，`--save-images` 可开）。

**检索评估（精修版与基线版）**：`results.json`（断点续跑）/ `summary.json` / `eval_meta.json` / `gallery_feats.pt`（图库特征缓存，两脚本共用）/ `report.html` / `panel_rankhist.png` / `panel_rankcdf.png` / `panel_rankscatter.png` / `panel_sims.png`。

**独立精修（`refine_encode.py`）**：每图 `metrics.json` + 汇总 `summary.json`。

**全链路（`run_full_validation.py`）**：输出根目录下五个子目录（`_smoke/`、`single/`、`batch_baseline/`、`retrieval_baseline/`、`batch_eval/`、`retrieval_eval/`）+ 顶层 `full_summary.json` / `full_summary.md`（三列对照汇总，可直接进论文）。

## 附录C：术语表

| 术语 | 含义 |
| --- | --- |
| Cover / 载体图像 | 原始未修改图像 |
| Encoded / 编码图像 | 嵌入消息与隐私残差后的图像 |
| msg_residual | 消息残差（`msg_budget × tanh` 约束） |
| prv_residual | 隐私残差（低通 + `prv_budget × tanh` + 纹理掩码约束） |
| Escape / 逃逸 | 使编码图特征偏离原载体（CLIP/DINO 余弦相似度下降） |
| Redirect / 有向重定向 | 把特征推向记忆库硬负样本方向，约束逃逸去向 |
| Hard negative | 记忆库中与当前编码特征最相似的其他载体特征 |
| Escape margin | 二次 Hinge 逃逸目标的边界（目标 sim < 1 − margin） |
| Sprint / Balance / Refine | Phase 2 三段调度：纯逃逸冲刺 / 多目标平衡 / 终段精修配方 |
| 内环蒸馏 | 每个 train step 内跑 N 步逐图优化得到教师残差并蒸进网络 |
| 逐图精修 | 部署时冻结网络、仅优化单图隐私残差场的编码期优化 |
| 不劣化保证 | 精修只在 bit_acc=100% 的可行状态内选优，初始候选为网络原始输出 |
| 色度投影 | 精修收敛后剔除 prv 残差 8×8 网格低频色度分量的事后处理（门控回退） |
| 摊销鸿沟 | 一次前向（摊销求解）与逐图最优解之间的性能差距 |
| CBIR | 基于内容的图像检索（以图搜图），本系统的隐私威胁模型 |
| 自检 | 检索评估中原图自查询应 100% top-1 的特征链路校验 |
| 冒烟门控 | run_full_validation 在全量评估前的小规模自检关卡 |
| 图库特征缓存 | `gallery_feats.pt` 按路径列表缓存图库 CLIP 特征，基线/精修检索脚本共用 |
| uint8 往返 | 内存内模拟 PNG 8bit 量化的保存-重载循环 |
| 断点续跑 | 评估脚本按 `results.json` 已完成记录自动跳过的增量执行 |
| Recovery | bit_acc 低于阈值时在含噪完整编码图上重训 decoder 的机制 |
| 热启动 | `new --continue-from-folder` 纯权重初始化（不携带绝对轮号） |
