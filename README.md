# PrivSteg: 隐私保护图像隐写系统

基于深度学习的隐私保护图像隐写框架：在载体图像中嵌入二进制消息（水印/溯源），同时通过 CLIP/DINOv2 特征偏移使编码图像逃逸基于内容的图像检索（CBIR），防止被以图搜图关联到原始载体。

系统由三层组成：**网络**（一次前向完成编码，具备泛化逃逸能力）+ **训练内蒸馏**（把逐图最优逃逸方向蒸进网络权重）+ **编码期逐图精修**（部署时对单图深挖逃逸，CLIP 相似度可压至 0.44~0.55，同时比特准确率与视觉质量不劣化）。深挖配方（prv_budget=0.08、margin=0.50、色度投影）已全部设为代码默认值。

## 核心特性

- **消息嵌入与提取**：30 bit 二进制消息，uint8 往返与信道噪声（JPEG/缩放/量化）下可靠解码
- **双残差编码架构**：消息头（msg_head）与隐私头（prv_head）分离，消息嵌入与隐私逃逸解耦
- **隐私独立支路（prv_trunk）**：空洞卷积（dilation 1/3/9/27，感受野 81×81）直接从载体图像提特征，绕开只为消息目标训练且全程冻结的共享层，解决信息瓶颈与容量瓶颈
- **低分辨率残差约束**：隐私残差先池化到 96×96 再双三次上采样，结构性消除高频条纹（96 网格 ≈ 2.7px/格，恰好覆盖 CLIP ViT-B/32 最敏感的 patch 尺度中频段）
- **纹理感知扰动**：逐图分位数归一化纹理掩码（q05/q95 → [0.15, 1.0]），扰动集中在人眼不敏感的纹理复杂区
- **三段式 Phase 2 调度**：Sprint（纯逃逸冲刺）→ Balance（多目标平衡）→ Refine（终段精修配方），配合判别器/消息头/共享层/解码器分阶段冻结
- **内环逐图优化蒸馏**：每个训练步对当前 batch 跑 10 步逐图优化得到教师残差，蒸进网络单次前向
- **编码期逐图精修**：`val/validation_script.py` full 模式默认开启，`val/refine_encode.py` 可独立使用；冻结网络仅优化 96×96 残差场 30 步，只在比特 100% 正确的状态里选最优（不劣化保证）；收敛后默认执行色度低频投影（消"发灰/蒙层"感知劣化，PSNR/SSIM 失明项）
- **完整评估工具链**：单图验证（含基线版）、批量定量评估（含基线版，20~30 倍速）、CBIR 检索逃逸评估（攻击者视角以图搜图实测）、全链路一键编排（冒烟门控 + 汇总对照表）
- **噪声鲁棒性**：支持裁剪、JPEG、量化、缩放等可微噪声层随机组合训练

## 项目结构

```
PrivSteg/
├── config.py                  # 全局配置（超参数、损失权重、阶段调度）
├── train.py                   # 训练入口（new/continue、热启动、CLI 参数）
├── trainer.py                 # 训练器（阶段冻结、内环蒸馏、recovery、记忆库）
├── noise_argparser.py         # 噪声层命令行参数解析
├── val/                        # 验证与评估脚本目录（从项目根运行，如 python val/batch_eval.py）
│   ├── validation_script.py   # 单图验证（编码+精修+解码+判别器+相似度+可视化报告）
│   ├── baseline_validation.py # 单图基线验证（网络直出，无精修）
│   ├── refine_encode.py       # 独立逐图精修（不训练网络，试探逃逸天花板）
│   ├── batch_eval.py          # 批量定量评估（精修版：统计/鲁棒性/uint8 往返，断点续跑）
│   ├── batch_baseline.py      # 批量基线评估（网络直出，速度约 20~30 倍）
│   ├── retrieval_eval.py      # CBIR 检索逃逸评估（精修版，断点续跑）
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
│   ├── noiser.py              # 噪声层组合器（随机选择激活）
│   ├── crop.py / cropout.py    # 裁剪类噪声
│   ├── dropout.py / resize.py  # 丢弃/缩放噪声
│   ├── jpeg_compression.py    # JPEG 压缩噪声（DCT 域量化）
│   └── quantization.py        # 傅里叶近似量化噪声
├── data/
│   └── dataset.py             # 隐写数据集（图像加载 + 随机消息生成）
├── docs/
│   └── PAPER_MATERIAL.md      # 论文写作素材（背景/相关工作/设计理念/实验设计）
└── CLIP/                      # vendored OpenAI CLIP 包
```

## 网络架构

### 编码器（Encoder）

- 共享层：16 个 ConvBNRelu（128 通道，3×3）→ 消息空间扩展拼接 → 融合层（30+128+3=161 → 128）
- **消息头（msg_head）**：2 层 3×3 ConvBNRelu + 1×1 卷积 → `msg_budget × tanh` → 消息残差
- **隐私独立支路（prv_trunk）**：4 层 ConvBNRelu，空洞率 1/3/9/27，感受野 81×81，直接从载体图像提特征；Phase 1 冻结，Phase 2 与 prv_head 一起训练（共用 ×5 学习率加速参数组）
- **隐私头（prv_head）**：输入为 `cat([共享特征, prv_trunk(cover)])`（256 通道），2 层 3×3 ConvBNRelu + 1×1 卷积 → 96×96 低分辨率场 → 双三次上采样 → `prv_budget × tanh` → 纹理掩码加权 → 隐私残差
- Phase 1：编码图 = 原图 + 消息残差 + 渐进引入的平滑随机噪声（隐私支路预适应）
- Phase 2：编码图 = 原图 + 消息残差 + 纹理掩码加权的隐私残差

### 解码器（Decoder）

- 8 层 ConvBNRelu（64 通道）+ ConvBNRelu(64→30) + AdaptiveAvgPool + 全连接层
- 输出消息 logits (B, L)，配合 BCEWithLogitsLoss

### 隐写分析器（Steganalyzer）

- 8 层 SpectralNorm 卷积（64→512 渐进通道）+ 全局平均池化 + 双层全连接判别头
- 当前配置（各阶段 adv 权重均为 0）仅作监控，不向编码器回传梯度

### 特征提取器

- **CLIP ViT-B/32**（冻结）：512 维 L2 归一化图像特征，主逃逸目标
- **DINOv2 ViT-S/14**（冻结，`timm` 加载，权重 `./models/dinov2_vits14_pretrain.pth`）：384 维特征，鲁棒性对冲（防攻击者改用自监督特征做近重复检测）

## 损失函数

| 损失            | 说明                                          |
| ------------- | ------------------------------------------- |
| L_pixel / L_ssim | 视觉重建损失（MSE + SSIM）                    |
| L_message     | 消息 BCEWithLogits 损失（干净 + 含噪双分支）      |
| L_adv         | 生成器对抗损失（Softplus 非饱和形式；当前权重 0）   |
| L_D           | 判别器 Hinge Loss（margin=1.0，准确率门控更新）   |
| L_privacy     | CLIP + DINOv2 特征逃逸（二次 Hinge + Softplus 预热）+ 可选 hard-negative 有向重定向 |
| L_distill     | 内环蒸馏损失（学生残差对教师残差的能量归一化 MSE，权重 25）  |
| L_residual    | 残差惩罚（控制残差幅度）                        |
| L_fft         | FFT 高频能量惩罚（仅 Balance 段；线性幅值，半径 0.25 奈奎斯特比例） |

隐私损失双分支设计：干净编码图分支承载逃逸主力（噪声层雅可比会毒化 CLIP 梯度），含噪分支仅锚定传播场景鲁棒性，按 0.3 加权融合。

## 两阶段训练

### Phase 1（默认 150 epochs）：基础隐写

- 训练消息通路（共享层 + msg_head + decoder），隐私支路冻结并注入渐进随机噪声做预适应
- 权重：message 15、visual 10、adv 0（当前配置）、privacy 0

### Phase 2（默认 100 epochs）：隐私逃逸

msg_head 与共享层全程冻结，采用三段调度（`sprint-epochs` / `refine-epochs` 可调）：

| 段          | 轮数    | decoder | message | visual | privacy | fft / residual | 说明 |
| ---------- | ------ | ------- | ------- | ------ | ------- | -------------- | ---- |
| Sprint     | 前 40   | 冻结    | 0       | 1.5    | 12      | 清零           | 纯逃逸冲刺，只用干净分支梯度 |
| Balance    | 中段    | 解冻    | 20/15/10 | 15    | 4/8/12  | 50 / 0.5      | 多目标平衡，recovery 兜底 |
| Refine     | 末 30   | 冻结    | 10      | 15     | 12      | 清零           | refine_encode.py 同配方目标（可用 `--refine-epochs 0` 关闭） |

**内环逐图优化蒸馏**（Phase 2 全程，`distill-steps 0` 可关闭）：每个训练步冻结网络，对当前 batch 的低分辨率残差场跑 10 步 Adam（目标与逐图精修同配方），得到教师残差后以 `distill_weight(25) × 能量归一化 MSE` 蒸进单次前向。代价：单步耗时约 2~4 倍。

**best checkpoint 判定**：Phase 1 按验证 L_message；Phase 2 在 `bit_accuracy ≥ 0.90` 约束下最小化 `clip_sim_noised + dino_sim_noised`（含噪传播场景指标优先，未达标加 10.0 惩罚分）。

### 热启动重训隐私支路（推荐工作流）

`new --continue-from-folder` 是纯权重初始化（不携带绝对轮号），消息通路完整复用、隐私支路重训：

```bash
python train.py new --data-dir ./dataset --name privacy_v2 \
  --continue-from-folder ./experiments/old_exp \
  --epochs-phase1 0 --epochs-phase2 100
```

### 隐私深挖配方（现已设为代码默认值；显式写出便于回溯）

```bash
python train.py new --name deep_escape \
  --continue-from-folder ./experiments/warm \
  --epochs-phase1 0 --epochs-phase2 100 --sprint-epochs 20 --refine-epochs 80 \
  --prv-budget 0.08 --privacy-margin-end 0.50 --dino-margin-end 0.20
```

**逃逸杠杆优先级**（CPU 逐图精修实测，同一 checkpoint、60 步、bit_acc 全程 100%）：

- **margin 是第一杠杆**：二次 Hinge 在 sim < (1−margin) 后梯度归零，margin 0.30/0.40/0.50（budget 0.08）对应 CLIP sim 0.575/0.547/0.440
- **预算是第二杠杆**：budget 0.04/0.06/0.08（margin 0.30）对应 0.634/0.603/0.575
- 视觉权重 15→8 无影响（幅度由 margin 驱动的 hinge 平衡决定）
- margin 0.50 + budget 0.08 时 PSNR 38.95 / SSIM 0.957，仍属高质量区间

## 环境要求

- Python >= 3.8，PyTorch >= 2.0.0，torchvision >= 0.15.0
- CUDA（训练推荐）/ CPU（验证与评估可用）
- DINOv2 权重：`./models/dinov2_vits14_pretrain.pth`（缺失时警告并随机初始化）
- 依赖：torch、torchvision、timm、pillow、numpy、matplotlib（可视化报告）、tensorboard（可选）

## 安装

```bash
git clone https://github.com/your-repo/PrivSteg.git
cd PrivSteg
pip install torch torchvision timm pillow numpy matplotlib tensorboard
```

仓库已 vendored `CLIP/` 包：从项目根运行时 `import clip` 直接命中，无需额外安装（也可 `pip install git+https://github.com/openai/CLIP.git`）。

## 数据准备

```
dataset/                  # 训练数据集根目录（CLI --data-dir，默认 ./dataset1）
├── train/
│   ├── image1.jpg
│   └── ...
└── val/
    ├── image1.jpg
    └── ...
```

支持 `.png`、`.jpg`、`.jpeg`、`.bmp`，递归扫描子目录。

## 使用方法

### 训练

```bash
# 从头训练（深挖配方已全部设为代码默认值；数据集默认 ./dataset1）
python train.py new --name my_experiment

# 常用参数（均与 Config 默认一致的可省略）
python train.py new --data-dir ./dataset --name my_experiment \
  --batch-size 35 --size 256 --message 30 \
  --epochs-phase1 150 --epochs-phase2 100 \
  --noise "jpeg+resize(0.8,1.2)+quant" --seed 42

# 冒烟测试（本地 CPU 显式传 --batch-size 1）
python train.py new --data-dir ./dataset --name smoke \
  --epochs-phase1 1 --epochs-phase2 0 --batch-size 1 --num-workers 0 --save-interval 1

# 断点续训（携带绝对轮号）
python train.py continue --folder ./experiments/my_experiment

# 热启动新实验（纯权重初始化，训练计划由本次 CLI 决定）
python train.py new --data-dir ./dataset --name resumed_exp \
  --continue-from-folder ./experiments/my_experiment
```

### 单图验证（full 模式默认含 30 步逐图精修 + 色度投影 + 自动可视化报告）

```bash
python val/validation_script.py \
  -c ./experiments/my_experiment/checkpoints/checkpoint.pth \
  -i ./test.jpg \
  -m "101000100110100101100011100010" \
  --mode full \
  -o output
```

编码后自动执行逐图精修，后续解码/判别器/相似度/保存全部基于精修后图像；输出目录生成 `report.html`（自包含可视化报告：指标卡片、原图-编码图-残差对比、精修逃逸轨迹、消息比特核验）+ 4 张 PNG 面板。

```bash
# 关闭精修
python val/validation_script.py -c ... -i ./test.jpg -m "101..." -o output --refine-steps 0

# 回落标准配方对照（深挖默认之外显式覆盖）
python val/validation_script.py -c ... -i ./test.jpg -m "101..." -o output \
  --refine-budget 0.04 --refine-margin 0.30
```

### 单图基线验证（网络直出，无精修）

```bash
python val/baseline_validation.py -c ./other/checkpoint.pth -i ./other/test.jpg \
  -m "101000100110100101100011100010" --device cpu -o other/baseline_output
```

网络单次前向直接输出编码图与全指标，不生成 report.html；用作"网络单独 vs 网络+精修"的对照基线（amortization gap 分析）。

### 独立逐图精修（不训练网络）

```bash
python val/refine_encode.py -c ./experiments/my_experiment/checkpoints/checkpoint.pth \
  -i ./test.jpg -m "101000100110100101100011100010" \
  --steps 30 --margin-end 0.50 --prv-budget 0.08 -o other/refined
```

用于训练前试探预算/margin 组合的逃逸天花板（`--steps 60` 默认），或对任意 checkpoint 的单张/多张图像直接优化隐私残差。

### 批量定量评估（精修版 / 基线版）

```bash
# 精修版（默认深挖配方，含逐图精修）
python val/batch_eval.py -c ./other/checkpoint.pth --val-dir ./dataset1/val \
  --num-images 200 --refine-steps 30 -o other/batch_eval --device cpu

# 基线版（网络直出，速度约 20~30 倍）
python val/batch_baseline.py -c other/checkpoint.pth --num-images 200 -o other/batch_baseline --device cpu
```

val 集采样 N 张逐一评估，输出基线/精修的 CLIP/DINO/bit_acc/PSNR/SSIM 统计（均值±标准差、分位数）、信道鲁棒性（JPEG/resize/quant 随机采样 k 次的解码率）、uint8 往返（模拟 PNG 8bit 量化）；生成 `results.json`（断点续跑）/ `summary.json` / `details.csv` / `report.html` 可视化报告。同 seed 同 val-dir 时两版评估同一批图像，可直接对照。

### CBIR 检索逃逸评估（以图搜图攻击测试，精修版 / 基线版）

```bash
# 精修版（查询图经逐图精修）
python val/retrieval_eval.py -c ./other/checkpoint.pth --val-dir ./dataset1/val \
  --num-queries 100 --gallery-size 1000 --refine-steps 30 \
  --refine-margin 0.50 --refine-budget 0.08 \
  -o other/retrieval_eval --device cpu

# 基线版（网络直出查询，与精修版共用图库特征缓存 gallery_feats.pt）
python val/retrieval_baseline.py -c other/checkpoint.pth --num-queries 100 \
  --gallery-size 1000 -o other/retrieval_baseline --device cpu
```

从 val 集建图库（CLIP 特征磁盘缓存，`--extra-gallery-dir` 可从 train 集扩充干扰池），查询集为图库子集；编码图与精修图分别作为查询做余弦检索，报告 top-1/5/10 命中率、逃逸率、平均排名、MRR 与"原图相似度 vs 最强干扰相似度"竞争余量；含原图自查询自检（应 100% top-1）；生成可视化报告（排名分布/CDF、逐查询排名散点、逃逸机制散点、全指标对照表）。

实测参考（小规模演示、深挖配方）：基线 top-1 命中 100%（攻击者一击必中）、精修后逃逸率 66.7%，CLIP sim 从 0.93 压到 0.47、跌破最强干扰相似度。

### 全链路一键验证（推荐工作流）

```bash
python val/run_full_validation.py -c ./experiments/full_run/checkpoints/checkpoint_best.pth \
  --val-dir ./dataset/test -o ./other/full_validation
```

编排 val/ 下评估脚本：冒烟自检（校验特征链路与精修可用后才开始全量）→ 单图 full → 批量/检索的基线版与精修版 → 汇总三列对照表 `full_summary.json` / 英文 `full_summary.md`（网络直出/精修对照，可直接进论文）。快阶段先行失败早暴露、各阶段断点续跑；支持 `--stages` 阶段子集、`--report-only` 仅重出报告、`--dry-run` 只打印命令。

### 仅重出可视化报告（不加载模型）

```bash
python val/batch_eval.py --report-only -o other/batch_eval
python val/batch_baseline.py --report-only -o other/batch_baseline
python val/retrieval_eval.py --report-only -o other/retrieval_eval
python val/retrieval_baseline.py --report-only -o other/retrieval_baseline
python val/run_full_validation.py --report-only -o other/full_validation
```

### 其他验证模式

```bash
python val/validation_script.py -c <checkpoint> -i ./encoded_image.png --mode decode
python val/validation_script.py -c <checkpoint> -i ./encoded_image.png --mode discriminator
python val/validation_script.py -i ./encoded_image.png --cover ./cover_image.png --mode similarity
```

## 关键配置参数

| 参数                       | 默认值               | 说明                                  |
| ------------------------ | ----------------- | ----------------------------------- |
| `image_size`             | 256               | 图像尺寸                                |
| `message_length`         | 30                | 嵌入消息长度（bit）                         |
| `batch_size`             | 35（A100 口径）     | 本地 CPU 冒烟显式传 `--batch-size 1`        |
| `num_epochs_phase1/2`    | 150 / 100         | 两阶段训练轮数                             |
| `msg_budget`             | 0.04              | 消息残差预算                              |
| `prv_budget`             | 0.08              | 隐私残差预算（深挖配方默认；CLI `--prv-budget` 可调） |
| `privacy_margin_end`     | 0.50              | CLIP 逃逸边界（CLI `--privacy-margin-end`；Sprint/Refine 段直接取该值） |
| `dino_escape_margin_end` | 0.20              | DINO 逃逸边界（CLI `--dino-margin-end`）    |
| `prv_lowres_size`        | 96                | 隐私残差低分辨率约束尺寸                        |
| `dino_weight`            | 0.5               | DINOv2 逃逸损失权重                       |
| `lambda_fft`             | 50.0              | FFT 高频能量惩罚（Sprint/Refine 段清零）         |
| `phase2_sprint_epochs`   | 40                | Sprint 段轮数（CLI `--sprint-epochs`）     |
| `phase2_refine_epochs`   | 30                | Refine 段轮数（CLI `--refine-epochs`）      |
| `distill_inner_steps`    | 10                | 内环蒸馏教师优化步数（CLI `--distill-steps`）     |
| `distill_weight`         | 25.0              | 蒸馏损失权重（CLI `--distill-weight`）         |
| `prv_head_lr_mult`       | 5.0               | 隐私支路学习率倍增                            |
| `privacy_noised_ratio`   | 0.3               | 隐私含噪分支权重（Sprint/Refine 段清零）          |
| `use_clip_redirect`      | True              | 启用 CLIP/DINO 有向重定向                   |
| `clip_redirect_target_sim` | 0.85            | CLIP 重定向目标相似度                        |
| `clip_max_target_sim`    | 0.90              | CLIP 重定向相似度上限                        |
| `refine_chroma_projection` | True            | 精修收敛后色度低频投影（val 脚本 `--refine-chroma-projection`） |
| `noise_config`           | jpeg+resize(0.8,1.2)+quant | 噪声层配置                          |
| `phase1_adv_*` / `phase2_adv` | 0.0          | 对抗权重（当前场景不依赖隐写分析对抗）                  |
| `bit_accuracy_threshold` | 0.90              | 触发解码器恢复的比特准确率阈值                    |
| 纹理掩码下限             | 0.15              | 纹理掩码分位数归一化下限                    |

## 输出说明

训练会在 `experiments/<name>/` 下生成：

```
experiments/my_experiment/
├── config.json              # 训练配置
├── train.log                # 训练日志
├── checkpoints/
│   ├── checkpoint.pth       # 最新检查点
│   ├── checkpoint_best.pth  # Phase 2 多准则最优
│   └── checkpoint_phase{N}_epoch{M}.pth
└── tensorboard/             # TensorBoard 日志
```

评估脚本在 `-o` 输出目录生成 `results.json`（断点续跑）/ `summary.json` / `details.csv`（批量）/ `eval_meta.json` / `gallery_feats.pt`（检索图库缓存）/ `report.html`（可视化）/ 独立 PNG 面板。

## 技术原理

### 三层分工：网络 / 蒸馏 / 部署精修

网络是"摊销求解器"——用一次前向近似逐图优化问题，存在不可消除的摊销鸿沟（网络单独输出 CLIP sim ~0.85）。训练内蒸馏把教师的逐图最优方向蒸进权重（缩小鸿沟），部署精修在编码端对单图真正求解（CLIP sim 0.44~0.55）。精修发生在编码端（一次性成本，CPU 每图约 1 分钟、GPU 秒级），解码端零成本——这是编码慢/解码快问题的标准架构选择。

### 隐私逃逸机制

1. **多特征空间逃逸**：CLIP（语义）+ DINOv2（细粒度纹理）双空间二次 Hinge 逃逸，margin 分别调度
2. **有向重定向**：记忆库 hard-negative 重定向，把特征推向语义相近但不同的目标，防止逃逸后落入其他敏感区域
3. **纹理掩码 + 96 低通约束**：扰动集中在纹理区且结构性消除高频条纹，逃逸与视觉质量不冲突
4. **色度低频投影**：精修后剔除 8×8 网格低频色度（"发灰/蒙层"感知劣化主体，PSNR/SSIM 失明项），门控回退保证不劣化
5. **消息锚硬保证**：精修只在比特 100% 正确的状态里选优，隐私逃逸不以牺牲解码为代价

### 详细文档

- 完整技术细节（架构、损失、调度、蒸馏、评估工具链、超参数速查）：[TECHNICAL_DOC.md](TECHNICAL_DOC.md)
- 论文写作素材（研究背景、相关工作综述、设计理念叙事、指标形式化、实验设计蓝图、参考文献）：[docs/PAPER_MATERIAL.md](docs/PAPER_MATERIAL.md)
