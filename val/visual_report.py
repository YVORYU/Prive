# 模块名: visual_report.py
# 模块职责: 验证结果可视化报告生成器（供 validation_script.py 的 full 模式调用）。
# 生成自包含 HTML 报告 report.html，内容包括：
#   1. 核心指标卡片（比特准确率/CLIP/DINO 隐私保护/PSNR/SSIM/判别器）
#   2. 图像对比面板（原图 / 编码图 / 归一化残差）
#   3. 逐图精修逃逸轨迹（CLIP/DINO 相似度 + bit_acc/PSNR 随步数变化）
#   4. 隐私逃逸对比条形图（基线 vs 精修后，附逃逸目标线）
#   5. 消息比特核验面板（逐比特正确性 + 解码置信度）
#   6. 精修前后指标对照表
# 所有图表以 base64 PNG 内嵌，单 HTML 文件可直接用浏览器打开；
# 同时将各面板 PNG 保存到输出目录，便于论文/报告直接引用。
# matplotlib 未安装或生成失败时由调用方降级跳过，不影响验证流程本身。

import base64
import io
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 图表与报告文本统一使用英文：matplotlib 默认 DejaVu Sans 含完整英文字形，
# 任何平台（Windows/Linux 服务器）无需额外字体文件
plt.rcParams['axes.unicode_minus'] = False

_CSS = """
body { font-family: 'Segoe UI', sans-serif; background: #f5f6f8;
       color: #24292f; margin: 0; padding: 26px; }
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px 0; }
h2 { font-size: 16px; margin: 0 0 12px 0; color: #444; }
.meta { background: #fff; border-radius: 10px; padding: 12px 18px; box-shadow: 0 1px 3px rgba(0,0,0,.08);
        font-size: 13px; color: #57606a; }
.meta table { border-collapse: collapse; }
.meta td { padding: 2px 14px 2px 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); gap: 12px; margin-top: 16px; }
.card { background: #fff; border-radius: 10px; padding: 15px 8px; text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); border-top: 3px solid #9fb3c8; }
.card.good { border-top-color: #2e9e5b; }
.card.warn { border-top-color: #e8a13c; }
.card.bad  { border-top-color: #d64545; }
.card .val { font-size: 25px; font-weight: 700; }
.card .name { font-size: 13px; color: #57606a; margin-top: 4px; }
.card .sub { font-size: 11px; color: #8b949e; margin-top: 2px; }
section { background: #fff; border-radius: 10px; padding: 18px 20px; margin-top: 18px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08); }
img.chart { width: 100%; height: auto; display: block; }
table.cmp { border-collapse: collapse; width: 100%; font-size: 13px; }
table.cmp th, table.cmp td { border: 1px solid #e1e4e8; padding: 6px 12px; text-align: center; }
table.cmp th { background: #f6f8fa; }
table.cmp td:first-child { text-align: left; }
.foot { color: #8b949e; font-size: 12px; margin-top: 16px; line-height: 1.7; }
"""


def _tensor_to_rgb(t):
    # [-1,1] (1,3,H,W) 张量 -> [0,1] (H,W,3) numpy
    arr = t.detach().squeeze(0).cpu().numpy()
    return np.clip((arr + 1.0) / 2.0, 0.0, 1.0).transpose(1, 2, 0)


def _save_fig(fig, path=None, dpi=110):
    # 图表输出为 base64 字符串；path 非空时同时保存 PNG 文件
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    png = buf.getvalue()
    if path is not None:
        with open(path, 'wb') as f:
            f.write(png)
    return base64.b64encode(png).decode()


def _panel_images(cover, encoded, path=None):
    # 图像对比面板: 原图 / 编码图 / 残差（归一化到 [-1,1] 的发散色图）
    diff = (encoded - cover).detach().squeeze(0).cpu().numpy().mean(axis=0)
    dmax = max(float(diff.max()), float(-diff.min()), 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    axes[0].imshow(_tensor_to_rgb(cover))
    axes[0].set_title('Cover')
    axes[1].imshow(_tensor_to_rgb(encoded))
    axes[1].set_title('Encoded')
    im = axes[2].imshow(diff / dmax, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    axes[2].set_title(f'Total residual (normalized, peak {dmax:.4f})')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return _save_fig(fig, path)


def _panel_trajectory(log, target_clip=None, target_dino=None, path=None):
    # 精修轨迹面板: 左侧特征相似度逃逸曲线，右侧解码率/视觉质量曲线
    steps = [e['step'] for e in log]
    clip = [e['clip_sim'] for e in log]
    dino = [e['dino_sim'] if e.get('dino_sim') is not None else np.nan for e in log]
    bit = [e['bit_acc'] * 100 for e in log]
    psnr = [e['psnr'] for e in log]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    ax1.plot(steps, clip, 'o-', color='#3b6fd4', label='CLIP sim')
    ax1.plot(steps, dino, 's--', color='#8a5fd4', label='DINO sim')
    if target_clip is not None:
        ax1.axhline(target_clip, ls=':', color='#3b6fd4', alpha=0.75, label=f'CLIP target {target_clip:.2f}')
    if target_dino is not None:
        ax1.axhline(target_dino, ls=':', color='#8a5fd4', alpha=0.75, label=f'DINO target {target_dino:.2f}')
    ax1.set_xlabel('step')
    ax1.set_ylabel('Feature similarity to cover')
    ax1.set_title('Refinement escape trajectory (lower = better)')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot(steps, bit, '^-', color='#2e9e5b', label='bit acc (%)')
    ax2.plot(steps, psnr, 'v--', color='#e8a13c', label='PSNR (dB)')
    ax2.set_xlabel('step')
    ax2.set_title('Decoding & visual quality')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, path)


def _panel_privacy(rows, target_clip=None, target_dino=None, path=None):
    # 隐私逃逸对比条形图: rows = [(名称, 相似度, 颜色), ...]
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(8.8, 0.72 * len(rows) + 1.1))
    bars = ax.barh(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2,
                f'{v:.3f} (protection {(1 - v) * 100:.1f}%)', va='center', fontsize=9)
    if target_clip is not None:
        ax.axvline(target_clip, color='#3b6fd4', ls=':', lw=1.2)
    if target_dino is not None:
        ax.axvline(target_dino, color='#8a5fd4', ls=':', lw=1.2)
    ax.set_xlim(0, 1.18)
    ax.set_title('Feature similarity comparison (lower = harder to associate; dashed = escape target)')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, path)


def _panel_message(orig, decoded, probs, path=None):
    # 消息比特面板: 上排逐比特正确性（绿=对/红=错，格内为解码值），下排解码置信度
    n = len(orig)
    correct = orig == decoded
    cmap = ListedColormap(['#d64545', '#2e9e5b'])
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11.5, 3.4),
                                   gridspec_kw={'height_ratios': [1, 1.5]})
    ax0.imshow(correct.reshape(1, -1), cmap=cmap, aspect='auto', vmin=0, vmax=1)
    for i in range(n):
        ax0.text(i, 0, str(int(decoded[i])), ha='center', va='center',
                 color='white', fontsize=8, fontweight='bold')
    ax0.set_yticks([])
    ax0.set_xticks(range(n))
    ax0.set_xticklabels(range(1, n + 1), fontsize=7)
    ax0.set_title(f'Decoded bit sequence (green = correct, red = wrong, accuracy {correct.mean() * 100:.1f}%)')
    ax1.bar(range(n), probs, color=['#2e9e5b' if c else '#d64545' for c in correct])
    ax1.axhline(0.5, color='gray', ls='--', lw=1)
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(range(1, n + 1), fontsize=7)
    ax1.set_ylabel('Sigmoid confidence')
    ax1.set_title('Per-bit decoding confidence (dashed = decision threshold 0.5)')
    fig.tight_layout()
    return _save_fig(fig, path)


def _card(val, name, level='good', sub=''):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ''
    return f'<div class="card {level}"><div class="val">{val}</div><div class="name">{name}</div>{sub_html}</div>'


def _level(v, good, warn):
    return 'good' if v >= good else ('warn' if v >= warn else 'bad')


def generate_html_report(output_dir, *, cover_image, encoded_image, message_bits,
                         decoded_bits, decoded_probs, bit_accuracy, correct_bits,
                         total_bits, cover_prob, stego_prob, privacy_results,
                         refine_report=None, meta=None):
    # 组装并写出 report.html，返回其路径
    meta = meta or {}

    # ---- 指标提取 ----
    clip_sim = privacy_results['similarity_probability']
    dino_sim = privacy_results.get('dino_similarity_probability')
    psnr = privacy_results['psnr']
    ssim = privacy_results['ssim']

    # ---- 各面板生成（同时落盘 PNG）----
    img_panel = _panel_images(cover_image, encoded_image,
                              path=os.path.join(output_dir, 'panel_images.png'))

    target_clip = 1.0 - (refine_report['margin'] if refine_report else meta.get('margin', 0.30))
    target_dino = None

    traj_b64, priv_b64 = None, None
    base_m = refine_report['base_metrics'] if refine_report else None
    ref_m = refine_report['refined_metrics'] if refine_report else None

    if refine_report:
        dino_margin = meta.get('dino_margin')
        if dino_margin is not None:
            target_dino = 1.0 - dino_margin
        traj_b64 = _panel_trajectory(refine_report['log'], target_clip, target_dino,
                                     path=os.path.join(output_dir, 'panel_trajectory.png'))

    rows = []
    if base_m is not None:
        rows.append(('CLIP baseline (no refinement)', base_m['clip_sim'], '#9fb3c8'))
        if base_m.get('dino_sim') is not None:
            rows.append(('DINO baseline (no refinement)', base_m['dino_sim'], '#c9bce0'))
    rows.append(('CLIP final (refined)', clip_sim, '#3b6fd4'))
    if dino_sim is not None:
        rows.append(('DINO final (refined)', dino_sim, '#8a5fd4'))
    priv_b64 = _panel_privacy(rows, target_clip, target_dino,
                              path=os.path.join(output_dir, 'panel_privacy.png'))

    msg_b64 = _panel_message(message_bits, decoded_bits, decoded_probs,
                             path=os.path.join(output_dir, 'panel_message.png'))

    # ---- 指标卡片 ----
    cards = _card(f'{bit_accuracy * 100:.1f}%', 'Bit accuracy',
                  _level(bit_accuracy, 0.98, 0.90), f'{correct_bits}/{total_bits} bits correct')
    cards += _card(f'{(1 - clip_sim) * 100:.1f}%', 'CLIP privacy protection',
                   _level(1 - clip_sim, 0.35, 0.20), f'sim {clip_sim:.3f}')
    if dino_sim is not None:
        cards += _card(f'{(1 - dino_sim) * 100:.1f}%', 'DINO privacy protection',
                       _level(1 - dino_sim, 0.15, 0.05), f'sim {dino_sim:.3f}')
    psnr_txt = '∞' if np.isinf(psnr) else f'{psnr:.1f} dB'
    psnr_lvl = 'good' if np.isinf(psnr) else _level(psnr, 38, 35)
    cards += _card(psnr_txt, 'PSNR', psnr_lvl, 'Pixel fidelity')
    cards += _card(f'{ssim:.4f}', 'SSIM', _level(ssim, 0.95, 0.92), 'Structural similarity')
    cards += _card(f'{stego_prob * 100:.1f}%', 'Discriminator stego probability',
                   'info' if stego_prob < 0.5 else 'warn',
                   f'cover side {cover_prob * 100:.1f}%')

    # ---- 元信息表 ----
    meta_rows = ''.join(f'<tr><td><b>{k}</b></td><td>{v}</td></tr>' for k, v in meta.items())

    # ---- 精修前后对照表 ----
    cmp_html = ''
    if refine_report:
        b, r = base_m, ref_m
        def _row(label, fb, fr, fmt):
            return (f'<tr><td>{label}</td><td>{fmt.format(fb)}</td>'
                    f'<td><b>{fmt.format(fr)}</b></td></tr>')
        cmp_rows = _row('CLIP similarity', b['clip_sim'], r['clip_sim'], '{:.4f}')
        if b.get('dino_sim') is not None:
            cmp_rows += _row('DINO similarity', b['dino_sim'], r['dino_sim'], '{:.4f}')
        cmp_rows += _row('Bit accuracy', b['bit_acc'], r['bit_acc'], '{:.2%}')
        cmp_rows += _row('PSNR (dB)', b['psnr'], r['psnr'], '{:.2f}')
        cmp_rows += _row('SSIM', b['ssim'], r['ssim'], '{:.4f}')
        cmp_html = (
            '<section><h2>Before / after refinement</h2><table class="cmp">'
            f'<tr><th>Metric</th><th>Baseline (no refinement)</th><th>Refined</th></tr>{cmp_rows}</table></section>'
        )

    # ---- 组装 HTML ----
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PrivSteg Validation Report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<h1>PrivSteg Validation Report</h1>
<div class="meta"><table>{meta_rows}</table></div>
<div class="cards">{cards}</div>
<section><h2>Image comparison</h2><img class="chart" src="data:image/png;base64,{img_panel}"></section>
{f'<section><h2>Per-image refinement trajectory ({refine_report["steps"]} Adam steps)</h2><img class="chart" src="data:image/png;base64,{traj_b64}"></section>' if traj_b64 else ''}
<section><h2>Privacy escape comparison</h2><img class="chart" src="data:image/png;base64,{priv_b64}"></section>
<section><h2>Message bit verification</h2><img class="chart" src="data:image/png;base64,{msg_b64}"></section>
{cmp_html}
<div class="foot">
Notes: privacy protection = 1 - feature cosine similarity; higher values mean the encoded image is
harder to associate with the cover via CBIR image search. Higher PSNR / SSIM means better visual quality.
The discriminator is this system's internal steganalyzer (monitoring only in the current configuration).
Panel PNGs are saved to the output directory and can be used directly in papers.
</div>
</div>
</body>
</html>"""

    report_path = os.path.join(output_dir, 'report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return report_path
