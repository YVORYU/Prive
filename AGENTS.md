# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概览

PrivSteg 是一个基于 PyTorch 的图像隐写研究项目。它训练一个类似 GAN 的系统，在载体图像中嵌入二进制消息，同时保持视觉质量、对抗隐写分析器，并可选地通过 CLIP/DINOv2 特征偏移来降低 CBIR 图像检索关联风险。

## 环境与依赖

- Python >= 3.8，PyTorch >= 2.0.0，torchvision >= 0.15.0。
- 安装项目依赖：
  ```bash
  pip install -r requirements.txt
  ```
- `requirements.txt` 会从 GitHub 安装 OpenAI CLIP。仓库中也包含一个 vendored `CLIP/` 包，带有独立的 setup 和测试。
- DINOv2 通过 `timm` 加载；当 `Config.dino_weight > 0` 时，代码期望权重位于 `./models/dinov2_vits14_pretrain.pth`。如果权重不存在，当前代码会警告并使用随机初始化。
- 训练数据应放在数据集根目录下的 `train/` 和 `val/` 子目录中，支持 `.png`、`.jpg`、`.jpeg`、`.bmp`。图像会被 resize 到 `Config.image_size` 并归一化到 `[-1, 1]`。

## 常用命令

安装依赖：
```bash
pip install -r requirements.txt
```

启动新的训练：
```bash
python train.py new --data-dir ./dataset --name my_experiment
```

在数据集上运行短训练 smoke test：
```bash
python train.py new --data-dir ./dataset --name smoke --epochs-phase1 1 --epochs-phase2 0 --batch-size 1 --num-workers 0 --save-interval 1
```

继续已有实验：
```bash
python train.py continue --folder ./experiments/my_experiment
```

从另一个实验 checkpoint 初始化新实验：
```bash
python train.py new --data-dir ./dataset --name resumed_exp --continue-from-folder ./experiments/my_experiment
```

使用显式噪声配置训练：
```bash
python train.py new --data-dir ./dataset --name noisy --noise "cropout((0.55,0.6),(0.55,0.6))+jpeg"
```

对单张图片运行完整验证：
```bash
python validation_script.py -c ./experiments/my_experiment/checkpoints/checkpoint.pth -i ./test.jpg -m "101000100110100101100011100010" --mode full -o output
```

验证模式：
```bash
python validation_script.py -c ./experiments/my_experiment/checkpoints/checkpoint.pth -i ./encoded_image.png --mode decode
python validation_script.py -c ./experiments/my_experiment/checkpoints/checkpoint.pth -i ./encoded_image.png --mode discriminator
python validation_script.py -i ./encoded_image.png --cover ./cover_image.png --mode similarity
```

运行 vendored CLIP 一致性测试：
```bash
python -m pytest CLIP/tests/test_consistency.py
```

运行单个 CLIP 测试用例：
```bash
python -m pytest CLIP/tests/test_consistency.py::test_consistency
```

当前仓库根目录没有项目级 lint、format 或主测试配置。

## 高层架构

- `config.py` 定义 `Config`，使用类属性集中管理训练、验证、模型构造和损失调度所需配置。当 README 与当前实现不一致时，以当前代码为准。
- `train.py` 是 CLI 入口。它解析 `new` 和 `continue` 命令，在 `experiments/<name>/` 下创建实验目录，写入 `config.json` 和 `train.log`，初始化 `Trainer`，按需加载 checkpoint，并运行阶段训练循环。
- `trainer.py` 负责模型初始化、优化器、dataloader、训练/验证 step、checkpoint、TensorBoard 日志、CLIP/DINO memory bank，以及 Phase 2 冻结 `msg_head`、bit accuracy 下降时执行 decoder recovery 等阶段行为。
- `data/dataset.py` 递归加载图像文件，并返回包含 `cover_image`、随机二进制 `message` 和 `image_path` 的 batch 字典。
- `models/encoder.py` 实现双残差编码器。它将 message 扩展到空间维度，提取图像/消息共享特征，然后输出受预算约束的 `msg_residual`；在 Phase 2 还会输出受预算约束的 `prv_residual`，并乘以梯度驱动的纹理掩码将扰动集中在纹理复杂区域。Phase 1 中 `prv_head` 冻结，注入经平滑和零均值单位方差归一化的随机噪声作为预适应。编码图像会被 clamp 到 `[-1, 1]`。
- `models/decoder.py` 从编码图像预测 message logits；训练损失使用 `BCEWithLogitsLoss`，因此计算训练损失前不要额外加 sigmoid。
- `models/steganalyzer.py` 是判别器式二分类器，使用 spectral-normalized 卷积层，输出 logits。
- `models/content_recognizer.py` 封装冻结的 CLIP ViT-B/32 图像/文本特征提取，并对输出做归一化以用于 cosine similarity。
- `losses/advanced_privacy_loss.py` 定义 `FeatureRedirectionLoss`，组合 CLIP/DINO escape loss（二次 Hinge + Softplus 预热），以及可选的 memory bank hard-negative redirection。
- `trainer.py` 中的对抗训练采用 Hinge Loss（判别器，margin=1.0）+ Softplus 非饱和损失（生成器），并通过判别器准确率门控（`disc_acc_high=0.70`）控制更新。Phase 2 还会对隐私残差施加 FFT 高频能量惩罚（`lambda_fft=10.0`）。
- `noise_layers/` 包含可微或随机信道噪声模块。`noise_argparser.py` 接受 `crop((hmin,hmax),(wmin,wmax))`、`cropout((hmin,hmax),(wmin,wmax))`、`dropout(keep_min,keep_max)`、`resize(min,max)`、`jpeg`、`quant` 等字符串；`trainer.py` 用 `+` 拆分组合配置，`Noiser` 每次 forward 随机选择一个激活噪声层。
- `validation_script.py` 加载 checkpoint，并支持单张图像的编码、解码、判别器预测、CLIP/DINO/SSIM/PSNR 相似度检查。

## 训练流程细节

当前代码使用两个阶段：

1. Phase 1 训练基础隐写能力和对抗能力。编码器使用 `msg_residual` 加上由 `Config.get_prv_noise_ratio` 渐进引入的随机隐私噪声；privacy loss 权重为 0。
2. Phase 2 冻结 `encoder.msg_head`，通过 `prv_head` 训练隐私扰动，使用 `Config.get_escape_margin` 提供的渐进 privacy margin，并在验证 bit accuracy 低于 `Config.bit_accuracy_threshold` 时触发 decoder recovery。

当前 `config.py` 中的重要默认值包括：`image_size=256`、`message_length=30`、`batch_size=8`、`num_epochs_phase1=150`、`num_epochs_phase2=100`、`msg_budget=0.04`、`prv_budget=0.04`、`use_clip_redirect=False`、`dino_weight=0.5`、`noise_config='identity'`。

## 生成输出

训练会在 `experiments/<name>/` 下生成：

- `config.json`
- `train.log`
- `checkpoints/checkpoint.pth` 以及周期性/best checkpoint
- `tensorboard/` event logs

验证脚本在使用会保存 artifact 的模式时，会将编码图像和相关输出写入 `-o/---output` 指定的目录。
