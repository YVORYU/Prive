# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概览

PrivSteg 是一个基于 PyTorch 的图像隐写研究项目。它训练一个类似 GAN 的系统，在载体图像中嵌入二进制消息，同时保持视觉质量、对抗隐写分析器，并可选地通过 CLIP/DINOv2 特征偏移来降低 CBIR 图像检索关联风险。

## 环境与依赖

- Python >= 3.8，PyTorch >= 2.0.0，torchvision >= 0.15.0。
- 安装项目依赖（仓库未带 requirements.txt，直接 pip 安装；CLIP 为仓库内 vendored 包，从项目根运行时 `import clip` 直接命中，无需额外安装）：
  ```bash
  pip install torch torchvision timm pillow numpy matplotlib tensorboard
  ```
- DINOv2 通过 `timm` 加载；当 `Config.dino_weight > 0` 时，代码期望权重位于 `./models/dinov2_vits14_pretrain.pth`。如果权重不存在，当前代码会警告并使用随机初始化。
- 训练数据应放在数据集根目录下的 `train/` 和 `val/` 子目录中，支持 `.png`、`.jpg`、`.jpeg`、`.bmp`。图像会被 resize 到 `Config.image_size` 并归一化到 `[-1, 1]`。
- **验证与评估脚本统一位于项目根的 `val/` 目录**（`validation_script.py`、`refine_encode.py`、`batch_eval.py`、`batch_baseline.py`、`retrieval_eval.py`、`retrieval_baseline.py`、`baseline_validation.py`、`visual_report.py`、`eval_visualizer.py`、`run_full_validation.py`），从项目根以 `python val/<脚本名>.py ...` 方式运行；脚本启动时会自动把项目根加入 `sys.path` 并把工作目录切到项目根，因此相对路径（checkpoint、`./models/dinov2_vits14_pretrain.pth`、输出目录）行为与脚本在项目根时完全一致，从任何目录启动均可。注意：此 `val/` 是验证脚本目录，与数据集根下的 `train/`/`val/` 数据子目录是两回事。
- **图表与报告文本统一使用英文**：matplotlib 默认 DejaVu Sans 不含 CJK 字形（Linux 服务器上会渲染为方块并报 Glyph missing 警告；这与 UTF-8 编码无关，是字体缺字形）。所有可视化面板（chart 标题/标签/图例）与 HTML 报告内容一律写英文，任何平台（Windows/Linux 服务器）无需额外字体文件，`plt.rcParams['axes.unicode_minus'] = False` 处理负号字形。控制台输出与代码注释不受此约束。

## 常用命令

安装依赖：
```bash
pip install torch torchvision timm pillow numpy matplotlib tensorboard
```

启动新的训练：
```bash
python train.py new --data-dir ./dataset --name my_experiment
```

**从头完整训练（深挖配方已全部设为代码默认值，最小命令）**：Phase 1 150 轮 + Phase 2 100 轮（Sprint 40 / Balance 30 / Refine 30），默认 `batch_size=35`（A100 口径；本地 CPU 冒烟显式传 `--batch-size 1`）、`prv_budget=0.08`、`privacy_margin_end=0.50`、`dino_escape_margin_end=0.20`、`distill_weight=25`、`distill_inner_steps=10`、`noise='jpeg+resize(0.8,1.2)+quant'`、数据集默认 `./dataset1`（可用 `-d` 覆盖）：
```bash
python train.py new --name full_run
```

训练完成后直接用 val/ 下的验证评估脚本验证 `experiments/full_run/checkpoints/checkpoint_best.pth`（多目标准则最优：bit_acc≥0.90 约束下 clip_sim+dino_sim 最小；`checkpoint.pth` 为最后一轮，两者均可用）：
```bash
python val/validation_script.py -c ./experiments/full_run/checkpoints/checkpoint_best.pth -i ./test.jpg -m "101000100110100101100011100010" -o other/output
python val/batch_eval.py -c ./experiments/full_run/checkpoints/checkpoint_best.pth --num-images 200 -o other/batch_eval
python val/retrieval_eval.py -c ./experiments/full_run/checkpoints/checkpoint_best.pth --num-queries 100 --gallery-size 1000 -o other/retrieval_eval
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

使用更大逃逸预算与 margin 从旧 checkpoint 热启动重训隐私支路（参数即深挖配方，现为代码默认值，显式写出便于回溯）：
```bash
python train.py new --name deep_escape --continue-from-folder ./experiments/warm \
  --epochs-phase1 0 --epochs-phase2 100 --sprint-epochs 20 --refine-epochs 80
```

对单张图片运行完整验证（full 模式默认在编码后执行 30 步逐图精修，精修默认即深挖配方 budget 0.08/margin 0.50，后续解码/判别器/相似度/保存全部基于精修后的图像；结束后自动生成可视化报告 report.html）：
```bash
python val/validation_script.py -c ./experiments/my_experiment/checkpoints/checkpoint.pth -i ./test.jpg -m "101000100110100101100011100010" --mode full -o output
```

关闭精修或回落标准配方（`--refine-steps 0` 关闭；`--refine-budget 0.04 --refine-margin 0.30` 回落到标准配方用于对照）：
```bash
python val/validation_script.py -c ./checkpoints/checkpoint.pth -i ./test.jpg -m "101000..." -o output --refine-steps 0
python val/validation_script.py -c ./checkpoints/checkpoint.pth -i ./test.jpg -m "101000..." -o output --refine-budget 0.04 --refine-margin 0.30
```

单图基线验证（网络直出，无逐图精修、不生成 report.html，直接输出编码图与全指标；汇总格式与 full 模式同口径，用作"网络单独 vs 网络+精修"对照）：
```bash
python val/baseline_validation.py -c ./other/checkpoint.pth -i ./other/test.jpg -m "101000100110100101100011100010" --device cpu -o other/baseline_output
```

批量基线评估（网络直出无精修，val 集采样 N 张：编码+全指标+信道鲁棒性+uint8 往返，速度约为含精修版 20~30 倍；单条件记录无 base/refined 冗余，输出 results.json/summary.json/details.csv，断点续跑，同 seed 同 val-dir 时与 batch_eval.py 评估同一批图像可直接对照；结束后自动生成可视化报告）：
```bash
python val/batch_baseline.py -c other/checkpoint.pth --num-images 200 -o other/batch_baseline --device cpu
```

CBIR 检索逃逸基线评估（网络直出无精修的以图搜图攻击测试：统计编码图查询的原图 top-k 命中率、逃逸率、MRR；图库特征缓存 gallery_feats.pt 与 retrieval_eval.py 共用，同图库配置直接命中；断点续跑；结束后自动生成可视化报告）：
```bash
python val/retrieval_baseline.py -c other/checkpoint.pth --num-queries 100 --gallery-size 1000 -o other/retrieval_baseline --device cpu
```

全链路一键验证（编排 val/ 下五个评估脚本：冒烟自检→单图 full→批量/检索的基线（网络直出）与精修版→汇总对照表；快阶段先行失败早暴露，各阶段断点续跑；默认测试集 `./dataset/test`、批量 1000 张、检索 100 查询/1000 图库、seed 42、device 自动、精修深挖配方；冒烟阶段校验自检 top-1=100% 与解码质量通过后才开始全量；检索精修版自动复用基线版提取的 gallery_feats.pt；结束后读四个 summary.json 汇总成 `full_summary.json` + 英文 `full_summary.md`（网络直出/精修对照表，可直接进论文），并做两路直出结果一致性交叉检查；`--stages` 选阶段子集、`--report-only` 仅重出报告、`--dry-run` 只打印命令）：
```bash
python val/run_full_validation.py -c ./experiments/full_run/checkpoints/checkpoint_best.pth --val-dir ./dataset/test -o ./other/full_validation
```

批量定量评估（val 集采样 N 张：编码+精修+信道鲁棒性+uint8 往返，精修默认深挖配方、默认保存精修图，输出 results.json/summary.json/details.csv，断点续跑；结束后自动生成可视化报告 report.html）：
```bash
python val/batch_eval.py --num-images 200 -o other/batch_eval --device cpu
```

CBIR 检索逃逸评估（以图搜图攻击测试：查询编码图在图库中检索，统计原图 top-k 命中率；精修默认深挖配方，图库特征缓存，断点续跑；结束后自动生成可视化报告）：
```bash
python val/retrieval_eval.py --num-queries 100 --gallery-size 1000 -o other/retrieval_eval --device cpu
```

仅从已有 results.json 重新生成可视化报告（不加载模型、不重跑评估，改图后重出报告用）：
```bash
python val/batch_eval.py --report-only -o other/batch_eval
python val/retrieval_eval.py --report-only -o other/retrieval_eval
python val/batch_baseline.py --report-only -o other/batch_baseline
python val/retrieval_baseline.py --report-only -o other/retrieval_baseline
python val/run_full_validation.py --report-only -o other/full_validation
```

**评估工作流建议**：正式评估前先小规模冒烟（`--num-images 3` / `--num-queries 3 --gallery-size 20`，各约 2-3 分钟），确认检索输出末尾"自检（原图自查询 top-1 命中率）"为 100% 且基线 top-1 命中接近 100% 后再跑正式规模（批量 200 张约 2~2.5 小时 CPU、检索 100 查询/1000 图库约 1.5~2 小时 CPU，A100 分钟级；两脚本均断点续跑——中断后原命令重跑自动跳过已完成部分）。注意"从头重跑"时须换输出目录或删除旧 `results.json`（否则续跑旧记录）。结果查看：输出目录 `report.html`（浏览器直接打开，含指标卡片/分布图/逐图改善/权衡散点/全指标对照表）+ `summary.json` + `details.csv`（仅批量）+ `images/` 精修图 + 4 张 PNG 面板（可直接进论文）。双配方对照（深挖 margin 0.50/budget 0.08 默认 vs 标准 margin 0.30/budget 0.04 显式指定 `--refine-budget 0.04 --refine-margin 0.30` 各跑一遍）构成论文的"视觉质量-隐私保护"权衡表。精修步数 30 与 60 的逃逸效果基本相同（深挖配方下 sim 都压到 0.46~0.49，逃逸率一致），逃逸主要发生在前 30 步——30 步砍掉一半时间几乎不损失效果。

验证模式：
```bash
python val/validation_script.py -c ./experiments/my_experiment/checkpoints/checkpoint.pth -i ./encoded_image.png --mode decode
python val/validation_script.py -c ./experiments/my_experiment/checkpoints/checkpoint.pth -i ./encoded_image.png --mode discriminator
python val/validation_script.py -i ./encoded_image.png --cover ./cover_image.png --mode similarity
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
- `models/encoder.py` 实现双残差编码器。它将 message 扩展到空间维度，提取图像/消息共享特征，然后输出受预算约束的 `msg_residual`；在 Phase 2 还会输出受预算约束的 `prv_residual`，并乘以梯度驱动的纹理掩码将扰动集中在纹理复杂区域。`prv_residual` 由 `prv_head(cat([共享特征, prv_trunk(cover)]))` 产生：`prv_trunk` 是独立的空洞卷积支路（dilation 1/3/9/27，感受野 81×81），直接从载体图像提特征，绕开只为消息目标训练且全程冻结的共享层（信息瓶颈），并修复原 5×5 感受野算不出空间结构化逐图攻击模式的容量瓶颈；`prv_trunk` 与 `prv_head` 共用 `prv_head_lr_mult` 加速参数组，Phase 1 一起冻结。`prv_residual` 会先池化到 `Config.prv_lowres_size`（默认 96×96，依据：CLIP ViT-B/32 patch 为 16×16 像素，对 patch 尺度中频结构最敏感，48×48 低通会把残差约束在 CLIP 不敏感的极低频段导致逃逸失效；96 网格≈2.7px/格，仍结构性消除高频条纹）再双三次上采样回全分辨率，随后才做 tanh 预算约束。纹理掩码使用逐图分位数归一化（q05/q95 → [0.15, 1.0]；下限 0.15 的依据：探针实测 0.05 时 CLIP 逃逸天花板 sim≈0.72 贴边目标 0.70 无余量），`texture_mask_strength` 为遗留兼容参数不再参与计算。Phase 1 中 `prv_head`/`prv_trunk` 冻结，注入经平滑和零均值单位方差归一化的随机噪声作为预适应。编码图像会被 clamp 到 `[-1, 1]`。注意：旧结构 checkpoint 的 `prv_head` 首层权重（128→256 输入通道变化）形状不兼容，`train.py load_encoder_state_compatible` 与 `val/validation_script.py` 会自动跳过（隐私支路由 Phase 2 重训，消息通路完整复用）。
- `models/decoder.py` 从编码图像预测 message logits；训练损失使用 `BCEWithLogitsLoss`，因此计算训练损失前不要额外加 sigmoid。
- `models/steganalyzer.py` 是判别器式二分类器，使用 spectral-normalized 卷积层，输出 logits。
- `models/content_recognizer.py` 封装冻结的 CLIP ViT-B/32 图像/文本特征提取，并对输出做归一化以用于 cosine similarity。
- `losses/advanced_privacy_loss.py` 定义 `FeatureRedirectionLoss`，组合 CLIP/DINO escape loss（二次 Hinge + Softplus 预热），以及可选的 memory bank hard-negative redirection。CLIP 与 DINO 使用独立的 escape margin（`Config.get_escape_margin` / `Config.get_dino_escape_margin`，均在 Phase 2 前 50% 轮次内渐进到达 end 值，后半程目标稳定）。redirect 目标按特征空间分别标定（`clip_redirect_target_sim=0.85` / `dino_redirect_target_sim=0.60`，线性 Hinge 拉升到位即停；取值必须明显高于该空间典型 hard negative 相似度——CLIP≈0.73、DINO≈0.47——才有实质拉力，仅高 0.02 时 redirect 会形同虚设），上限为 `clip/dino_max_target_sim`（0.90/0.65），control loss 计入总损失。
- `trainer.py` 中的对抗训练采用 Hinge Loss（判别器，margin=1.0）+ Softplus 非饱和损失（生成器），并通过判别器准确率门控（`disc_acc_high=0.70`）控制更新；默认各阶段 adv 权重均为 0（本场景不依赖隐写分析对抗），此时判别器仅作监控，不向编码器回传梯度。Phase 2 还会对隐私残差施加 FFT 高频能量惩罚（`lambda_fft=50.0`，线性幅值形式，半径由 `fft_high_freq_radius_ratio=0.25` 奈奎斯特比例控制）。
- 隐私损失采用双分支：干净编码图分支承载 escape+redirect 主力（噪声层雅可比会毒化 CLIP 梯度——quant 傅里叶近似梯度在 1~21 倍震荡、jpeg 近似劣质，只在含噪图上算会完全推不动），含噪编码图分支仅承载 escape 以锚定传播场景鲁棒性；按 `privacy_noised_ratio=0.3` 加权融合（干净分支主导逃逸方向，等权会让劣质梯度稀释一半有效梯度）。Phase 2 隐私权重三段 4/8/12（补偿 CLIP 特征对残差的雅可比比逐像素损失小 1~2 个数量级的失衡）。memory bank 检索 hard negative 时先查询后入库，按空间分别用 `clip/dino_redirect_exclude_sim`（0.92/0.87）排除自身/近重复候选。
- 消息损失同样为双分支：干净编码图 + 含噪编码图分别过 decoder，干净分支权重按 `Config.get_msg_clean_ratio` 从 0.8（Phase 1 初期，保冷启动）线性退火到 0.5；仅含噪单分支会导致随机初始化的微弱 msg 信号被信道摧毁、L_message 锁死在 ln2。各阶段 `noised` 视觉损失权重恒为 0（含噪视觉损失会把信道失真归咎于残差，与消息目标对抗，同样导致躺平）。
- `noise_layers/` 包含可微或随机信道噪声模块。`noise_argparser.py` 接受 `crop((hmin,hmax),(wmin,wmax))`、`cropout((hmin,hmax),(wmin,wmax))`、`dropout(keep_min,keep_max)`、`resize(min,max)`、`jpeg`、`quant` 等字符串；`trainer.py` 用 `+` 拆分组合配置，`Noiser` 每次 forward 随机选择一个激活噪声层。
- `val/validation_script.py` 加载 checkpoint，并支持单张图像的编码、解码、判别器预测、CLIP/DINO/SSIM/PSNR 相似度检查。full 模式默认在编码后执行逐图精修（与 `val/refine_encode.py` 同配方，`ValidationScript.refine_encoded_image`）：冻结全部网络、仅优化 96×96 低分辨率隐私残差场 30 步 Adam（`--refine-steps 0` 关闭），只在比特 100% 正确的状态里挑 clip_sim 最低者且以网络原始残差为初始候选（精修不劣化保证）；后续解码/判别器/相似度/保存全部基于精修后图像，`--refine-budget`/`--refine-margin` 可覆盖精修的约束链与逃逸目标。精修收敛后默认执行色度蒙层事后投影（`_project_out_chroma_lowfreq`，`Config.refine_chroma_projection=True`，`--refine-chroma-projection 0` 关闭）：剔除隐私残差 8×8 网格低频色度分量——"发灰/蒙层"感知劣化主体（残差 77% 能量在色度通道、低频占其 44%，PSNR/SSIM 对其失明；梯度惩罚实测无效，逃逸方向与色度方向高度平行，只能结构化剔除），门控保证投影后 bit_acc=100% 且 clip_sim≤(1-margin) 否则回退未投影结果；实测 clip 0.438→0.467（仍在 0.50 目标内）、低频色度 −91%、PSNR +0.5dB。`batch_eval.py`/`retrieval_eval.py`/`run_full_validation.py`/`refine_encode.py` 均以 `--refine-chroma-projection`（或 `--chroma-projection`）同口径透传。
- `val/baseline_validation.py` 单图基线验证脚本（网络直出，无逐图精修）：复用 `ValidationScript` 的全部组件与流程（编码→解码验证→判别器→CLIP/DINO/SSIM/PSNR→保存编码图），但跳过逐图精修与 report.html 生成，直接输出网络单次前向的编码图像与指标；汇总格式与 full 模式同口径，用作"网络单独 vs 网络+编码期精修"的对照基线（amortization gap 分析）或 checkpoint 原生质量的快速检查。
- `val/visual_report.py` 单图验证可视化报告生成器（`val/validation_script.py` full 模式结束时自动调用）：从验证数据生成自包含 `report.html`（base64 内嵌图表，浏览器直接打开）+ 独立 PNG 面板。内容含核心指标卡片（比特准确率/CLIP·DINO 隐私保护/PSNR/SSIM/判别器，按阈值标色）、原图-编码图-残差对比面板、精修逃逸轨迹（CLIP/DINO 相似度与 bit_acc/PSNR 随步数变化）、隐私逃逸对比条形图（基线 vs 精修，附逃逸目标线）、消息比特核验面板（逐比特正确性 + 解码置信度）、精修前后对照表。matplotlib 缺失或生成失败时自动跳过，不影响验证本身。
- `val/batch_eval.py` 批量定量评估：val 集采样 N 张逐一编码+精修，输出基线/精修的 CLIP/DINO/bit_acc/PSNR/SSIM 统计（均值±标准差、分位数）、信道鲁棒性（噪声 k 次采样的解码率与相似度）、uint8 往返（内存内模拟 PNG 8bit 量化）；`results.json` 增量保存支持断点续跑，`details.csv` 为逐图明细。
- `val/batch_baseline.py` 批量基线评估（网络直出，无逐图精修）：复用 `batch_eval.py` 的采样/噪声/往返公共函数（同 seed 同 val-dir 评估同一批图像，与含精修版直接对照），网络单次前向后计算全指标+信道鲁棒性+uint8 往返；单条件记录（无 base/refined 冗余结构），速度约为含精修版 20~30 倍，适合大规模快速评估 checkpoint 原生质量或论文 amortization gap 分析的基线列；输出 results.json/summary.json/details.csv（断点续跑）+ 可视化报告。
- `val/retrieval_eval.py` CBIR 检索逃逸评估（以图搜图攻击测试）：查询集为图库子集（原图保留在图库），编码图与精修图分别作为查询对全图库做 CLIP 余弦检索，报告 top-1/top-5/top-10 命中率、逃逸率、平均排名、MRR 与"原图相似度 vs 最强干扰相似度"竞争余量；含原图自查询自检（应 100% top-1）；图库特征缓存至 `gallery_feats.pt`，`--extra-gallery-dir` 可从 train 集扩充干扰池提升任务难度；`results.json` 增量保存支持断点续跑。实测（20 图库、深挖配方）：基线 top-1 命中 100%、精修后逃逸率 66.7%。
- `val/retrieval_baseline.py` CBIR 检索逃逸基线评估（网络直出，无逐图精修）：复用 `retrieval_eval.py` 的图库构造/特征缓存/检索函数，编码图单次前向后直接作查询；图库特征缓存 `gallery_feats.pt` 与精修版共用（同图库配置直接命中，无需重新提取）；单条件记录，报告 top-k 命中率、逃逸率、MRR、自检；`results.json` 增量保存支持断点续跑 + 可视化报告。
- `val/eval_visualizer.py` 评估可视化报告生成器（`val/batch_eval.py`/`val/retrieval_eval.py`/`val/batch_baseline.py`/`val/retrieval_baseline.py` 评估结束后自动调用，`--report-only` 可仅重出报告）：从 `results.json` 生成自包含 `report.html`（base64 内嵌图表，浏览器直接打开）+ 独立 PNG 面板。批量报告含相似度分布对比、逐图改善哑铃图、质量-隐私权衡散点、鲁棒性对比、全指标对照表；检索报告含排名分布/CDF、逐查询排名散点（log-log）、相似度竞争散点（逃逸机制）、全指标对照表；基线版（`generate_batch_baseline_report`/`generate_retrieval_baseline_report`）为单条件报告——相似度分布、权衡散点、鲁棒性条形图、排名分布/CDF、逃逸机制散点，指标卡片与表格均单列。matplotlib 缺失时自动跳过，不影响评估本身。
- `val/run_full_validation.py` 全链路验证编排脚本：按"冒烟自检 → 单图 full → 批量/检索基线（网络直出）→ 批量/检索精修"顺序用 subprocess 调用 val/ 下五个评估脚本（快阶段先行、失败早暴露），同 seed 同 val-dir 保证基线与精修评估同一批图像；冒烟阶段（`_smoke` 临时目录，每次全量重跑）校验检索自检 top-1=100%、直出 bit_acc≥0.5、精修链路可用后才放行全量；检索两脚本图库配置一致时把基线阶段的 `gallery_feats.pt` 复制给精修版复用（缓存按路径列表自校验）；结束读取四个 `summary.json` 汇总成 `full_summary.json` + 英文 `full_summary.md`（网络直出/精修三列对照表 + 一致性交叉检查：两路直出 clip_sim 均值差>0.02 或检索 top-1 差>5% 时警告 seed/配置不一致）。支持 `--stages` 阶段子集、`--report-only` 仅重出报告与汇总、`--dry-run` 只打印命令；子脚本自带断点续跑，中断后重跑同一条命令自动续。

## 训练流程细节

当前代码使用两个阶段：

1. Phase 1 训练基础隐写能力和对抗能力。编码器使用 `msg_residual` 加上由 `Config.get_prv_noise_ratio` 渐进引入的随机隐私噪声；privacy loss 权重为 0。
2. Phase 2 冻结 `encoder.msg_head` 与共享层（全程冻结，不再按 epoch 分段解冻），采用**三段式调度**：
   - **Sprint 段**（前 `phase2_sprint_epochs=40` 轮）：纯逃逸冲刺——decoder 也冻结，`message` 权重 0、`visual` 轻锚 1.5、`privacy` 12，只训 prv_head+prv_trunk。依据：天花板探针证明同约束下直接优化 60 步即可把 clip_sim 压到 0.64，而正常 Phase 2 停在 0.946 的瓶颈是 Adam 在多目标、逐图方向不一致下的信噪比失效（加大损失权重无效，Adam 对梯度尺度不变）。Sprint 段 `residual_penalty`/`fft` 权重清零（收缩残差与压中频都会直接对冲逃逸梯度），`privacy_noised_ratio` 置 0 只用干净分支梯度（`Config.get_privacy_noised_ratio` 阶段感知），Balance 段全部恢复。Sprint 段 bit_acc 暂时下降是预期行为，不参与 best 判定（bit_acc 惩罚自动排除）、不触发 recovery。
   - **Balance 段**（中段）：恢复正常组合（message 20/15/10、privacy 4/8/12、visual 15、residual_penalty 0.5、fft 50），decoder 解冻恢复解码，recovery 机制兜底（recovery 使用含隐私残差的完整编码图 + 信道噪声，与实际解码分布一致）。
   - **Refine 段**（最后 `phase2_refine_epochs=30` 轮，判定优先于 Sprint）：切换为 `val/refine_encode.py` 已验证的精修配方目标——escape-only privacy 12（`distill_w_privacy`）+ 消息锚 10（`distill_w_message`）+ visual 15（`distill_w_visual`），`residual_penalty`/`fft`/`noised` 全部清零，escape margin 固定取 end 值，redirect 关闭。decoder 冻结（残差自适应既有解码器；recovery 跳过——重训 decoder 会使消息锚目标漂移，bit_acc 由学生目标自身的 message 权重维持）。消息锚是该段与 Sprint 的本质差异：Sprint 靠蒸馏教师间接保解码，Refine 由学生目标直接保解码，因此终段可放心压 clip_sim 而不会把 bit_acc 打穿阈值。可用 `--refine-epochs 0` 关闭。
   - 此外 `prv_head`/`prv_trunk` 使用独立参数组、学习率为 `lr_encoder × prv_head_lr_mult`（默认 ×5），直接放大其实际更新步长。
   - **内环逐图优化蒸馏**（方案 C，`trainer._refine_teacher_residual`，Phase 2 全程启用，`distill_inner_steps>0` 时）：每个 train step 冻结网络，把当前 batch 的低分辨率隐私残差场作为优化变量跑 `distill_inner_steps`（默认 10）步 Adam——目标函数与逐图精修脚本 `val/refine_encode.py` 验证配方一致（escape margin 取 end 值 + 消息双分支 BCE + 视觉 MSE/SSIM，约束链低通→tanh×预算→纹理掩码→clamp 同构）——得到教师残差后，用 `distill_weight × MSE(prv_residual, 教师残差)` 把探针级逃逸方向蒸进单次前向（实测逐图精修 60 步 clip_sim 0.94→0.55~0.65 且 bit_acc/PSNR/SSIM 不劣化）。代价：Phase 2 单步耗时约 2~4 倍，可用 `--distill-steps 0` 关闭。
   - 注意：训练循环把**阶段内相对轮次**（本次运行从第 1 轮起算）传给 `train_step`/`validate` 做权重调度（`train.py run_training_loop` 的 `sched_epoch`），因此从旧 checkpoint 热启动重训 Phase 2 时 Sprint/Balance/Refine 三段能正确触发；checkpoint 保存与日志仍用绝对轮号。
   - **热启动语义**：`new --continue-from-folder` 是纯权重初始化（训练计划由本次 CLI 参数决定，从 Phase 1 第 1 轮起算；`best_loss` 不继承），绝不携带 checkpoint 的绝对轮号（否则旧 Phase 2 epoch 100 会使 Phase 2 循环为空）。从已训完 Phase 2 的旧 checkpoint 重训隐私支路的推荐命令：`python train.py new --data-dir <数据集> --name <新实验> --continue-from-folder <旧实验目录> --epochs-phase1 0 --epochs-phase2 100`——消息通路（含 decoder）完整复用（实测直入 Phase 2 首轮 val bit_acc 即 0.925），隐私支路（prv_head 首层 128→256 通道不兼容）由 `load_encoder_state_compatible` 跳过重训；直入 Phase 2 时若 checkpoint 未携带 memory bank 则自动补预热。断点续训（携带绝对轮号）使用 `continue` 子命令。

训练循环中的 best checkpoint 判定为多目标准则：Phase 1 按验证 `L_message`；Phase 2 在 `bit_accuracy ≥ bit_accuracy_threshold` 约束下最小化 `clip_sim + dino_sim`（未达标加惩罚分），命中时额外保存 `checkpoints/checkpoint_best.pth`。两阶段准则数值不可比（L_message 收敛后 ~0.1-0.3 而 sim 和 ~1.4-1.9），检测到准则所属阶段切换时重置 `best_loss`（`Trainer.best_loss_phase` 记录当前最优的归属阶段，随 checkpoint 持久化；`continue` 续训同阶段则保留历史最优，旧格式 checkpoint 缺该字段时首轮按当前阶段准则重判）——否则从头训练时 Phase 2 的 best 分数永远高于 Phase 1 遗留值，best 会冻结在 Phase 1 整个 Phase 2 不更新。

当前 `config.py` 中的重要默认值包括：`image_size=256`、`message_length=30`、`batch_size=35`（A100 口径）、`num_epochs_phase1=150`、`num_epochs_phase2=100`、`msg_budget=0.04`、`prv_budget=0.08`（深挖配方，CLI `--prv-budget` 可调）、`use_clip_redirect=True`、`dino_weight=0.5`、`noise_config='jpeg+resize(0.8,1.2)+quant'`、`privacy_margin_end=0.50`（深挖配方；Sprint/Refine 段直接取该值，教师内环目标同源联动）、`dino_escape_margin_end=0.20`（CLI `--dino-margin-end`）、`clip/dino_redirect_target_sim=0.85/0.60`、`clip/dino_redirect_exclude_sim=0.92/0.87`、`clip_redirect_weight=3.0`、`privacy_noised_ratio=0.3`（Sprint 段经 `privacy_noised_ratio_sprint=0.0` 清零）、`prv_lowres_size=96`、`lambda_fft=50.0`（Sprint 段清零）、`fft_high_freq_radius_ratio=0.25`、Phase 1 `message` 权重 15、Phase 2 三段式（Sprint 40 轮纯逃逸 + Balance 三段 privacy 4/8/12 + Refine 终段 30 轮精修配方，`phase2_refine_epochs=30`）+ 内环蒸馏（`distill_inner_steps=10`、`distill_lr=0.1`、`distill_weight=25.0`、教师目标 `distill_w_privacy/w_message/w_visual=12/10/15`）、`prv_head_lr_mult=5.0`、纹理掩码下限 0.15、各阶段 `adv`/`noised` 视觉权重均为 0。`val/refine_encode.py` 是独立的编码期逐图精修脚本（不训练网络），可对任意 checkpoint 的单张/多张图像直接优化隐私残差并输出与 validation_script 同口径的指标；支持 `--prv-budget`/`--margin-end` 覆盖约束链与逃逸目标，用于训练前试探预算/margin 组合的逃逸天花板。

**逃逸杠杆优先级**（CPU 逐图精修实测，同一 checkpoint、60 步、bit_acc 全程 100%）：margin 是第一杠杆——二次 Hinge 在 sim < (1-margin) 后梯度归零，margin 0.30/0.40/0.50（budget 0.08）对应 clip_sim 0.575/0.547/0.440；预算是第二杠杆——budget 0.04/0.06/0.08（margin 0.30）对应 0.634/0.603/0.575；视觉权重 w_visual 15→8 无影响（幅度由 margin 驱动的 hinge 平衡决定，不受视觉权重约束）。margin 0.50 + budget 0.08 时 PSNR 38.95/SSIM 0.957，仍属视觉高质量区间。

## 生成输出

训练会在 `experiments/<name>/` 下生成：

- `config.json`
- `train.log`
- `checkpoints/checkpoint.pth` 以及周期性/best checkpoint
- `tensorboard/` event logs

验证脚本在使用会保存 artifact 的模式时，会将编码图像和相关输出写入 `-o/---output` 指定的目录。
