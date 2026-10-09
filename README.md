# PrivSteg: Privacy-Preserving Image Steganography

A deep-learning framework that embeds a binary message (watermark / provenance tag) into a
cover image while shifting the encoded image away from its cover in CLIP and DINOv2 feature
space, so that content-based image retrieval (CBIR) can no longer link a published image back
to its source photograph.

The system has three layers: a **network** (a single forward pass produces an encoded image
with generalised evasion ability) + **in-training distillation** (per-image optimal evasion
directions are distilled into the network weights) + **publication-time per-image refinement**
(evasion is deepened for the individual image being released, driving CLIP similarity down to
0.44–0.55 without degrading bit accuracy or visual quality). The deep-escape recipe
(`prv_budget=0.08`, `margin=0.50`, chroma projection) is the code default.

## Checkpoint

The trained model behind every number in the paper is attached to the
[latest release](../../releases/latest). Download `checkpoint_best.pth` and either drop it into
`checkpoints/` or pass its path to any validation script with `-c`. Model weights are not stored
in git because of their size.

## Key features

- **Message embedding and extraction** — a 30-bit binary message, decodable after a uint8
  round trip and under channel noise (JPEG / resizing / quantization)
- **Dual-residual encoder** — separate message head (`msg_head`) and privacy head (`prv_head`)
  so that payload embedding and privacy evasion no longer compete for one capacity budget
- **Independent privacy branch (`prv_trunk`)** — dilated convolutions (rates 1/3/9/27, 81×81
  receptive field) extract features straight from the cover, bypassing the shared trunk that is
  trained only for the message objective and frozen throughout Phase 2; this removes both the
  information bottleneck and the capacity bottleneck
- **Low-resolution residual constraint** — the privacy residual is pooled to 96×96 and then
  bicubically upsampled, which removes high-frequency stripes structurally (96 cells ≈ 2.7 px
  per cell, covering exactly the patch-scale mid-frequency band that CLIP ViT-B/32 is most
  sensitive to)
- **Texture-aware perturbation** — a per-image quantile-normalised texture mask
  (q05/q95 → [0.15, 1.0]) concentrates the perturbation in visually complex regions
- **Three-stage Phase 2 schedule** — Sprint (pure evasion) → Balance (multi-objective) →
  Refine (final-stage refinement recipe), with staged freezing of the discriminator, message
  head, shared trunk and decoder
- **Inner-loop distillation** — every training step runs 10 per-image optimisation steps on the
  current batch to obtain a teacher residual, which is distilled into the single forward pass
- **Publication-time refinement** — enabled by default in `val/validation_script.py --mode full`
  and available standalone as `val/refine_encode.py`; all network weights are frozen and only a
  96×96 residual field is optimised for 30 steps, selecting only among states with 100% bit
  accuracy (non-degradation guarantee). On convergence it applies a gated low-frequency chroma
  projection that removes the "grey veil" artefact that PSNR and SSIM are blind to
- **Complete evaluation toolchain** — single-image validation (plus a baseline variant), batch
  quantitative evaluation (plus a baseline variant, 20–30× faster), CBIR retrieval-evasion
  evaluation (the attacker's view, measured by image search), and one-command end-to-end
  orchestration (smoke gate + summary tables)
- **Noise robustness** — differentiable noise layers (crop, JPEG, quantization, resizing) in
  randomly combined training configurations

## Repository layout

```
PrivSteg/
├── config.py                   # Global configuration (hyperparameters, loss weights, schedule)
├── train.py                    # Training entry point (new/continue, warm start, CLI flags)
├── trainer.py                  # Trainer (stage freezing, inner-loop distillation, recovery, memory bank)
├── noise_argparser.py          # Command-line parsing for noise-layer strings
├── requirements.txt            # Python dependencies
├── val/                        # Validation and evaluation scripts (run from the repo root)
│   ├── validation_script.py    # Single image: encode + refine + decode + discriminator + similarity + report
│   ├── baseline_validation.py  # Single image, network output only (no refinement)
│   ├── refine_encode.py        # Standalone per-image refinement (no training; probes the evasion ceiling)
│   ├── batch_eval.py           # Batch quantitative evaluation, refined (statistics / robustness / uint8, resumable)
│   ├── batch_baseline.py       # Batch evaluation, network output only (≈20–30× faster)
│   ├── retrieval_eval.py       # CBIR retrieval-evasion evaluation, refined (resumable)
│   ├── retrieval_baseline.py   # CBIR retrieval-evasion evaluation, network output only
│   ├── channel_retrieval_eval.py # Retrieval after the transmission channel (C2 × C4)
│   ├── transfer_eval.py        # Retrieval in uncalibrated third-party feature spaces
│   ├── external_baseline.py    # Cross-system comparison against TrustMark
│   ├── run_full_validation.py  # End-to-end orchestration (smoke → single → baseline → refined → summary)
│   ├── visual_report.py        # Single-image visual report generator
│   └── eval_visualizer.py      # Batch / retrieval visual report generator
├── models/
│   ├── encoder.py              # Encoder (dual residual heads + independent prv_trunk + texture mask)
│   ├── decoder.py              # Decoder (CNN + fully connected message extraction)
│   ├── steganalyzer.py         # Steganalysis discriminator (SpectralNorm)
│   └── content_recognizer.py   # Content recogniser (frozen CLIP ViT-B/32)
├── losses/
│   ├── ssim_loss.py            # SSIM loss
│   └── advanced_privacy_loss.py  # Feature escape + directed redirection (CLIP + DINOv2)
├── noise_layers/
│   ├── noiser.py               # Noise-layer combiner (random activation)
│   ├── crop.py / cropout.py    # Cropping noise
│   ├── dropout.py / resize.py  # Dropout / resizing noise
│   ├── jpeg_compression.py     # JPEG compression noise (DCT-domain quantization)
│   └── quantization.py         # Fourier-approximation quantization noise
└── data/
    └── dataset.py              # Steganography dataset (image loading + random message generation)
```

`CLIP/` (OpenAI's CLIP package) is deliberately **not** vendored here; `requirements.txt`
installs it from source. See [Installation](#installation).

## Architecture

### Encoder

- Shared trunk: 16 ConvBNRelu blocks (128 channels, 3×3) → message broadcast and concatenation
  → fusion layer (30+128+3 = 161 → 128)
- **Message head (`msg_head`)** — 2 × 3×3 ConvBNRelu + 1×1 convolution → `msg_budget × tanh` →
  message residual
- **Independent privacy branch (`prv_trunk`)** — 4 ConvBNRelu blocks with dilation rates
  1/3/9/27 (81×81 receptive field) reading directly from the cover image; frozen in Phase 1,
  trained together with `prv_head` in Phase 2 (shared ×5 learning-rate multiplier)
- **Privacy head (`prv_head`)** — input is `cat([shared features, prv_trunk(cover)])`
  (256 channels); 2 × 3×3 ConvBNRelu + 1×1 convolution → 96×96 low-resolution field → bicubic
  upsampling → `prv_budget × tanh` → texture-mask weighting → privacy residual
- Phase 1: encoded image = cover + message residual + progressively introduced smoothed random
  noise (pre-adaptation of the privacy branch)
- Phase 2: encoded image = cover + message residual + texture-mask-weighted privacy residual

### Decoder

- 8 ConvBNRelu blocks (64 channels) + ConvBNRelu(64→30) + AdaptiveAvgPool + fully connected layer
- Outputs message logits `(B, L)`; trained with `BCEWithLogitsLoss`

### Steganalysis discriminator

- 8 SpectralNorm convolutions (64→512, progressively wider channels) + global average pooling +
  a two-layer classification head
- With the current configuration (all adversarial weights set to 0 in every stage) it only
  monitors; no gradient is propagated back to the encoder

### Feature extractors

- **CLIP ViT-B/32** (frozen) — 512-dimensional L2-normalised image features; the primary
  evasion target
- **DINOv2 ViT-S/14** (frozen, loaded through `timm` from
  `./models/dinov2_vits14_pretrain.pth`) — 384-dimensional features; robustness hedge against an
  attacker switching to self-supervised features for near-duplicate detection

## Losses

| Loss | Description |
| --- | --- |
| `L_pixel` / `L_ssim` | Visual reconstruction (MSE + SSIM) |
| `L_message` | Message BCEWithLogits loss (dual branch: clean + noised) |
| `L_adv` | Generator adversarial loss (non-saturating Softplus; weight 0 in the current configuration) |
| `L_D` | Discriminator hinge loss (margin = 1.0, accuracy-gated updates) |
| `L_privacy` | CLIP + DINOv2 feature escape (quadratic hinge + Softplus warm-up) plus optional hard-negative directed redirection |
| `L_distill` | Inner-loop distillation (energy-normalised MSE between student and teacher residuals, weight 25) |
| `L_residual` | Residual-norm penalty |
| `L_fft` | FFT high-frequency energy penalty (Balance stage only; linear amplitude, radius 0.25 of Nyquist) |

The privacy loss is dual-branch by design: the clean-encoding branch carries the escape
objective (noise-layer Jacobians poison the CLIP gradient), while the noised branch only anchors
robustness in the propagation scenario; the two are combined with a 0.3 weight.

## Two-phase training

### Phase 1 (default 150 epochs): basic steganography

- Trains the message path (shared trunk + `msg_head` + decoder); the privacy branch is frozen and
  receives progressively introduced random noise as pre-adaptation
- Weights: message 15, visual 10, adv 0 (current configuration), privacy 0

### Phase 2 (default 100 epochs): privacy evasion

`msg_head` and the shared trunk stay frozen for the whole phase, which uses a three-stage
schedule (`--sprint-epochs` / `--refine-epochs` adjust the boundaries):

| Stage | Epochs | Decoder | Message | Visual | Privacy | fft / residual | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sprint | first 40 | frozen | 0 | 1.5 | 12 | zeroed | Pure evasion; clean-branch gradients only |
| Balance | middle | unfrozen | 20/15/10 | 15 | 4/8/12 | 50 / 0.5 | Multi-objective balance; recovery fallback |
| Refine | last 30 | frozen | 10 | 15 | 12 | zeroed | Same recipe as `refine_encode.py` (disable with `--refine-epochs 0`) |

**Inner-loop distillation** (throughout Phase 2, disable with `--distill-steps 0`): at every
training step the network is frozen, 10 Adam steps are run on the low-resolution residual field
of the current batch (same objective as per-image refinement) to obtain a teacher residual, and
that residual is distilled into the single forward pass with
`distill_weight(25) × energy-normalised MSE`. Cost: roughly 2–4× slower per step.

**Best-checkpoint criterion**: Phase 1 selects on validation `L_message`; Phase 2 minimises
`clip_sim_noised + dino_sim_noised` subject to `bit_accuracy ≥ 0.90` (noised-propagation metrics
first; a 10.0 penalty is added when the constraint is missed).

### Warm-start retraining of the privacy branch (recommended workflow)

`new --continue-from-folder` is pure weight initialisation (it does not carry absolute epoch
numbers): the message path is reused intact and the privacy branch is retrained.

```bash
python train.py new --data-dir ./dataset --name privacy_v2 \
  --continue-from-folder ./experiments/old_exp \
  --epochs-phase1 0 --epochs-phase2 100
```

### Deep-escape recipe (now the code default; written out explicitly for traceability)

```bash
python train.py new --name deep_escape \
  --continue-from-folder ./experiments/warm \
  --epochs-phase1 0 --epochs-phase2 100 --sprint-epochs 20 --refine-epochs 80 \
  --prv-budget 0.08 --privacy-margin-end 0.50 --dino-margin-end 0.20
```

**Escape levers, in order of strength** (CPU per-image refinement, same checkpoint, 60 steps,
bit accuracy 100% throughout):

- **Margin is the first lever** — the quadratic hinge has zero gradient below
  `sim < (1 − margin)`; margins 0.30/0.40/0.50 at budget 0.08 give CLIP similarities
  0.575/0.547/0.440
- **Budget is the second lever** — budgets 0.04/0.06/0.08 at margin 0.30 give 0.634/0.603/0.575
- Visual weight 15→8 has no measurable effect (the amplitude is set by the margin-driven hinge
  equilibrium)

## Requirements

- Python >= 3.8, PyTorch >= 2.0.0, torchvision >= 0.15.0
- CUDA (recommended for training) / CPU (sufficient for validation and evaluation)
- DINOv2 weights at `./models/dinov2_vits14_pretrain.pth` (a warning is issued and random
  initialisation is used when missing)
- Dependencies: torch, torchvision, timm, pillow, numpy, matplotlib (visual reports),
  tensorboard (training logs), and CLIP (image/text features — installed from source, it is not
  on PyPI)

## Installation

```bash
git clone https://github.com/your-repo/PrivSteg.git
cd PrivSteg
pip install -r requirements.txt
```

`requirements.txt` includes the CLIP source install
(`clip @ git+https://github.com/openai/CLIP.git`): `models/content_recognizer.py` calls it
through `import clip`, and the package is not published on PyPI, so it must be installed from
GitHub. To install it on its own:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Data preparation

```
dataset/                  # Dataset root (CLI --data-dir; defaults to ./dataset1)
├── train/
│   ├── image1.jpg
│   └── ...
└── val/
    ├── image1.jpg
    └── ...
```

`.png`, `.jpg`, `.jpeg` and `.bmp` are supported; subdirectories are scanned recursively.
ImageNet and MS COCO are the datasets used in the paper; both must be downloaded from their
original sources and must not be redistributed.

## Usage

### Training

```bash
# Train from scratch (the deep-escape recipe is the code default; data defaults to ./dataset1)
python train.py new --name my_experiment

# Common flags (all equal to the Config defaults and therefore optional)
python train.py new --data-dir ./dataset --name my_experiment \
  --batch-size 35 --size 256 --message 30 \
  --epochs-phase1 150 --epochs-phase2 100 \
  --noise "jpeg+resize(0.8,1.2)+quant" --seed 42

# Smoke test (pass --batch-size 1 explicitly on CPU)
python train.py new --data-dir ./dataset --name smoke \
  --epochs-phase1 1 --epochs-phase2 0 --batch-size 1 --num-workers 0 --save-interval 1

# Resume (carries absolute epoch numbers)
python train.py continue --folder ./experiments/my_experiment

# Warm-start a new experiment (pure weight initialisation; the schedule comes from this CLI)
python train.py new --data-dir ./dataset --name resumed_exp \
  --continue-from-folder ./experiments/my_experiment
```

### Single-image validation (full mode: 30-step refinement + chroma projection + report)

```bash
python val/validation_script.py \
  -c ./experiments/my_experiment/checkpoints/checkpoint.pth \
  -i ./test.jpg \
  -m "101000100110100101100011100010" \
  --mode full \
  -o output
```

Refinement runs automatically after encoding, and decoding, discriminator scoring, similarity and
saving all use the refined image. The output directory receives `report.html` (a self-contained
report: metric cards, cover/encoded/residual comparison, refinement trajectory, message-bit
verification) plus four PNG panels.

```bash
# Disable refinement
python val/validation_script.py -c ... -i ./test.jpg -m "101..." -o output --refine-steps 0

# Fall back to the standard recipe for comparison
python val/validation_script.py -c ... -i ./test.jpg -m "101..." -o output \
  --refine-budget 0.04 --refine-margin 0.30
```

### Single-image baseline validation (network output only)

```bash
python val/baseline_validation.py -c ./other/checkpoint.pth -i ./other/test.jpg \
  -m "101000100110100101100011100010" --device cpu -o other/baseline_output
```

A single forward pass produces the encoded image and every metric, with no `report.html`. Use it
as the "network alone vs network + refinement" baseline for amortization-gap analysis.

### Standalone per-image refinement (no training)

```bash
python val/refine_encode.py -c ./experiments/my_experiment/checkpoints/checkpoint.pth \
  -i ./test.jpg -m "101000100110100101100011100010" \
  --steps 30 --margin-end 0.50 --prv-budget 0.08 -o other/refined
```

Use it to probe the evasion ceiling of a budget/margin combination before committing to
training (default `--steps 60`), or to optimise the privacy residual for any number of images
against any checkpoint.

### Batch quantitative evaluation (refined / baseline)

```bash
# Refined (deep-escape recipe by default, per-image refinement included)
python val/batch_eval.py -c ./other/checkpoint.pth --val-dir ./dataset1/val \
  --num-images 200 --refine-steps 30 -o other/batch_eval --device cpu

# Baseline (network output only, ≈20–30× faster)
python val/batch_baseline.py -c other/checkpoint.pth --num-images 200 -o other/batch_baseline --device cpu
```

Samples N images from the validation set and reports CLIP / DINOv2 / bit accuracy / PSNR / SSIM
statistics (mean ± std, percentiles), channel robustness (decode rate over k random JPEG /
resize / quant samples) and a uint8 round trip (in-memory simulation of 8-bit PNG
quantization). Outputs `results.json` (resumable), `summary.json`, `details.csv` and a
`report.html`. With the same seed and `--val-dir`, both variants evaluate the same images and can
be compared directly.

### CBIR retrieval-evasion evaluation (refined / baseline)

```bash
# Refined (queries go through per-image refinement)
python val/retrieval_eval.py -c ./other/checkpoint.pth --val-dir ./dataset1/val \
  --num-queries 100 --gallery-size 1000 --refine-steps 30 \
  --refine-margin 0.50 --refine-budget 0.08 \
  -o other/retrieval_eval --device cpu

# Baseline (queries are network outputs; shares gallery_feats.pt with the refined run)
python val/retrieval_baseline.py -c other/checkpoint.pth --num-queries 100 \
  --gallery-size 1000 -o other/retrieval_baseline --device cpu
```

The gallery is built from the validation set (CLIP features cached to disk;
`--extra-gallery-dir` enlarges the distractor pool from the training set) and the query set is a
subset of the gallery. Encoded and refined images are used as queries against the full gallery by
cosine similarity, reporting top-1/5/10 hit rates, evasion rate, mean rank, MRR and the
competitive margin between "similarity to cover" and "similarity to the strongest distractor".
A self-check queries each unmodified cover against itself and must return top-1 in 100% of
cases. A visual report is generated (rank distribution / CDF, per-query rank scatter, evasion
mechanism scatter, full metric table).

Reference measurement (small-scale demo, deep-escape recipe): baseline top-1 hit rate 100% (the
attacker succeeds on every query); after refinement the evasion rate is 66.7%, with CLIP
similarity driven from 0.93 to 0.47, below the strongest distractor similarity.

### End-to-end validation (recommended workflow)

```bash
python val/run_full_validation.py -c ./experiments/full_run/checkpoints/checkpoint_best.pth \
  --val-dir ./dataset/test -o ./other/full_validation
```

Orchestrates the `val/` scripts: smoke self-check (the feature pipeline and refinement must work
before the full run starts) → single-image full → batch/retrieval baseline and refined →
summary tables `full_summary.json` and `full_summary.md` (network-output vs refined, ready for
the paper). Fast stages run first so failures surface early, and every stage is resumable.
Supports `--stages` (subset), `--report-only` (regenerate reports only) and `--dry-run` (print
the commands).

### Regenerating reports only (no model loaded)

```bash
python val/batch_eval.py --report-only -o other/batch_eval
python val/batch_baseline.py --report-only -o other/batch_baseline
python val/retrieval_eval.py --report-only -o other/retrieval_eval
python val/retrieval_baseline.py --report-only -o other/retrieval_baseline
python val/run_full_validation.py --report-only -o other/full_validation
```

### Other validation modes

```bash
python val/validation_script.py -c <checkpoint> -i ./encoded_image.png --mode decode
python val/validation_script.py -c <checkpoint> -i ./encoded_image.png --mode discriminator
python val/validation_script.py -i ./encoded_image.png --cover ./cover_image.png --mode similarity
```

## Key configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `image_size` | 256 | Image size |
| `message_length` | 30 | Embedded message length (bits) |
| `batch_size` | 35 (A100) | Pass `--batch-size 1` explicitly for a CPU smoke test |
| `num_epochs_phase1/2` | 150 / 100 | Epochs per phase |
| `msg_budget` | 0.04 | Message residual budget |
| `prv_budget` | 0.08 | Privacy residual budget (deep-escape default; CLI `--prv-budget`) |
| `privacy_margin_end` | 0.50 | CLIP escape margin (CLI `--privacy-margin-end`; Sprint/Refine use it directly) |
| `dino_escape_margin_end` | 0.20 | DINOv2 escape margin (CLI `--dino-margin-end`) |
| `prv_lowres_size` | 96 | Low-resolution constraint size for the privacy residual |
| `dino_weight` | 0.5 | DINOv2 escape loss weight |
| `lambda_fft` | 50.0 | FFT high-frequency penalty (zeroed in Sprint/Refine) |
| `phase2_sprint_epochs` | 40 | Sprint stage length (CLI `--sprint-epochs`) |
| `phase2_refine_epochs` | 30 | Refine stage length (CLI `--refine-epochs`) |
| `distill_inner_steps` | 10 | Teacher optimisation steps in inner-loop distillation (CLI `--distill-steps`) |
| `distill_weight` | 25.0 | Distillation loss weight (CLI `--distill-weight`) |
| `prv_head_lr_mult` | 5.0 | Learning-rate multiplier for the privacy branch |
| `privacy_noised_ratio` | 0.3 | Weight of the noised privacy branch (zeroed in Sprint/Refine) |
| `use_clip_redirect` | True | Enable CLIP/DINOv2 directed redirection |
| `clip_redirect_target_sim` | 0.85 | CLIP redirection target similarity |
| `clip_max_target_sim` | 0.90 | CLIP redirection similarity ceiling |
| `refine_chroma_projection` | True | Low-frequency chroma projection after refinement (`--refine-chroma-projection` in the val scripts) |
| `noise_config` | jpeg+resize(0.8,1.2)+quant | Noise-layer configuration |
| `phase1_adv_*` / `phase2_adv` | 0.0 | Adversarial weights (this setting does not rely on steganalysis resistance) |
| `bit_accuracy_threshold` | 0.90 | Bit-accuracy threshold that triggers decoder recovery |
| Texture-mask floor | 0.15 | Lower bound of the quantile-normalised texture mask |

## Outputs

Training writes into `experiments/<name>/`:

```
experiments/my_experiment/
├── config.json              # Training configuration
├── train.log                # Training log
├── checkpoints/
│   ├── checkpoint.pth       # Latest checkpoint
│   ├── checkpoint_best.pth  # Phase 2 multi-criteria best
│   └── checkpoint_phase{N}_epoch{M}.pth
└── tensorboard/             # TensorBoard logs
```

Evaluation scripts write into the `-o` directory: `results.json` (resumable), `summary.json`,
`details.csv` (batch), `eval_meta.json`, `gallery_feats.pt` (retrieval gallery cache),
`report.html` and standalone PNG panels.

## Technical notes

### Three layers: network / distillation / deployment-time refinement

The network is an *amortized solver*: one forward pass approximates the per-image optimisation
problem, and an irreducible amortization gap remains (the network alone reaches CLIP similarity
≈0.85). In-training distillation pushes the teacher's per-image optimal directions into the
weights (narrowing the gap), and deployment-time refinement actually solves the per-image
problem at the encoding end (CLIP similarity 0.44–0.55). Refinement sits on the encoding side —
a one-off cost of roughly a minute per image on CPU and seconds on GPU — while decoding costs
nothing extra, which is the standard architectural choice for the slow-encode / fast-decode
setting.

### How evasion works

1. **Multi-space escape** — quadratic hinge escape in both CLIP (semantic) and DINOv2
   (fine-grained texture) feature space, with separately scheduled margins
2. **Directed redirection** — memory-bank hard-negative redirection pushes features towards
   semantically nearby but distinct targets, preventing an escaped image from landing next to
   another sensitive image
3. **Texture mask + 96 low-pass constraint** — the perturbation stays in textured regions while
   high-frequency stripes are removed structurally, so evasion and visual quality do not conflict
4. **Low-frequency chroma projection** — after refinement, low-frequency chroma on an 8×8 grid
   (the main perceptual artefact, invisible to PSNR and SSIM) is projected out, with a gate that
   falls back whenever the result would degrade
5. **Hard message anchor** — refinement only ever selects states with 100% bit accuracy, so
   evasion is never paid for with decoding reliability

### Scope

The evasion guarantee is calibrated against two frozen encoders (CLIP ViT-B/32 and DINOv2
ViT-S/14) in a white-box setting, and it does not transfer to retrieval backbones outside that
calibration. Adversarial steganalysis is outside the scope of this code release.

## Documentation

The repository ships code only. The published article is the citable record for the method,
the evaluation protocol and all reported numbers.

A Chinese translation of this README is available at [README.zh-CN.md](README.zh-CN.md).
