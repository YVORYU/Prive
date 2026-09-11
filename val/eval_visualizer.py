# 模块名: eval_visualizer.py
# 模块职责: 为 batch_eval.py 与 retrieval_eval.py 生成可视化 HTML 报告。
# 从输出目录的 results.json 读取逐图评估记录，生成自包含 report.html：
#   - 批量评估: 指标卡片 / CLIP·DINO 相似度分布对比 / 逐图改善哑铃图 /
#     质量-隐私权衡散点 / 信道与uint8鲁棒性对比 / 全指标对照表
#   - 检索评估: 指标卡片 / 排名分布与CDF / 逐查询排名散点(log-log) /
#     相似度竞争散点(逃逸机制) / 全指标对照表
# 所有图表以 base64 PNG 内嵌单 HTML 文件（浏览器直接打开），同时保存 PNG 面板。
# 两个评估脚本通过 --report-only 调用本模块时无需加载模型/重跑评估。

import base64
import io
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 图表与报告文本统一使用英文：matplotlib 默认 DejaVu Sans 含完整英文字形，
# 任何平台（Windows/Linux 服务器）无需额外字体文件
plt.rcParams['axes.unicode_minus'] = False

C_BASE = '#9aa7b5'
C_REF = '#3b6fd4'
C_DINO = '#8a5fd4'
C_GOOD = '#2e9e5b'
C_BAD = '#d64545'

_CSS = """
body { font-family: 'Segoe UI', sans-serif; background: #f5f6f8;
       color: #24292f; margin: 0; padding: 26px; }
.wrap { max-width: 1150px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px 0; }
h2 { font-size: 15px; margin: 0 0 10px 0; color: #444; }
.meta { background: #fff; border-radius: 10px; padding: 12px 18px; box-shadow: 0 1px 3px rgba(0,0,0,.08);
        font-size: 13px; color: #57606a; display: inline-block; }
.meta table { border-collapse: collapse; }
.meta td { padding: 2px 16px 2px 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 12px; margin-top: 16px; }
.card { background: #fff; border-radius: 10px; padding: 15px 8px; text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); border-top: 3px solid #9fb3c8; }
.card.good { border-top-color: #2e9e5b; }
.card.warn { border-top-color: #e8a13c; }
.card.bad  { border-top-color: #d64545; }
.card.info { border-top-color: #3b6fd4; }
.card .val { font-size: 24px; font-weight: 700; }
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


def _b64(fig, path=None, dpi=110):
    # 图表输出为 base64 字符串；path 非空时同时保存 PNG 面板文件
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    png = buf.getvalue()
    if path is not None:
        with open(path, 'wb') as f:
            f.write(png)
    return base64.b64encode(png).decode()


def _card(val, name, level='good', sub=''):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ''
    return f'<div class="card {level}"><div class="val">{val}</div><div class="name">{name}</div>{sub_html}</div>'


def _level(v, good, warn):
    return 'good' if v >= good else ('warn' if v >= warn else 'bad')


def _write_html(path, title, meta_rows, cards, sections, foot=''):
    meta = ''.join(f'<tr><td><b>{k}</b></td><td>{v}</td></tr>' for k, v in meta_rows)
    secs = ''.join(f'<section><h2>{h}</h2><img class="chart" src="data:image/png;base64,{b}"></section>'
                  for h, b in sections)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{title}</h1>
<div class="meta"><table>{meta}</table></div>
<div class="cards">{cards}</div>
{secs}
{f'<div class="foot">{foot}</div>' if foot else ''}
</div></body></html>"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


def _load_results(output_dir):
    # 读取 results.json 逐图记录；文件不存在或为空时返回 (None, None)
    rp = os.path.join(output_dir, 'results.json')
    if not os.path.exists(rp):
        return None, rp
    with open(rp, encoding='utf-8') as f:
        results = json.load(f)
    if not results:
        return None, rp
    return list(results.values()), rp


# ======================================================================
# 批量定量评估报告 (batch_eval.py)
# ======================================================================

def _batch_panel_dist(bc, rc, bd, rd, path=None):
    # 相似度分布对比: CLIP 与 DINO 的基线/精修直方图叠加
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    bins = np.linspace(0, 1, 26)
    for ax, b, r, name in [(axes[0], bc, rc, 'CLIP'), (axes[1], bd, rd, 'DINO')]:
        if np.isnan(b).all() and np.isnan(r).all():
            ax.set_title(f'{name} (disabled)')
            continue
        ax.hist(b[~np.isnan(b)], bins=bins, alpha=0.55, color=C_BASE, edgecolor='white',
                label=f'Baseline (mean {np.nanmean(b):.3f})')
        ax.hist(r[~np.isnan(r)], bins=bins, alpha=0.65, color=C_REF, edgecolor='white',
                label=f'Refined (mean {np.nanmean(r):.3f})')
        ax.axvline(0.70, ls='--', color=C_BAD, lw=1)
        ax.set_title(f'{name} similarity distribution')
        ax.set_xlabel('Feature similarity to cover')
        ax.set_ylabel('Number of images')
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _b64(fig, path)


def _batch_panel_dumbbell(bc, rc, path=None):
    # 逐图改善哑铃图: 每张图基线->精修的相似度变化（按基线降序排列）
    order = np.argsort(-bc)
    xs = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    for i, idx in enumerate(order):
        ax.plot([i, i], [bc[idx], rc[idx]], color='#c9d4e0', lw=1, alpha=0.75, zorder=1)
    ax.scatter(xs, bc[order], s=13, color=C_BASE, label='Baseline (no refinement)', zorder=2)
    ax.scatter(xs, rc[order], s=13, color=C_REF, label='Refined', zorder=3)
    ax.axhline(0.70, ls='--', color=C_BAD, lw=1, label='Escape reference 0.70')
    ax.set_xlabel('Images (sorted by baseline similarity, descending)')
    ax.set_ylabel('CLIP similarity')
    ax.set_title('Per-image privacy escape improvement (each line = one image)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _b64(fig, path)


def _batch_panel_pareto(bp, rp_, bc, rc, path=None):
    # 质量-隐私权衡散点: PSNR vs CLIP 相似度, 基线点与精修点连线显示移动方向
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for i in range(len(bc)):
        ax.plot([bp[i], rp_[i]], [bc[i], rc[i]], color='#e1e4e8', lw=0.7, alpha=0.6, zorder=1)
    ax.scatter(bp, bc, s=16, color=C_BASE, label='Baseline', zorder=2)
    ax.scatter(rp_, rc, s=16, color=C_REF, label='Refined', zorder=3)
    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('CLIP similarity')
    ax.set_title('Quality-privacy trade-off (lower-left = trading visual quality for deeper escape)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _b64(fig, path)


def _batch_panel_robustness(b_nm, b_nx, b_rt, r_nm, r_nx, r_rt, path=None):
    # 部署条件鲁棒性: 信道噪声与 uint8 往返的解码率, 基线/精修分组条形图
    labels = ['bit_acc after channel\n(mean of 8)', 'bit_acc after channel\n(worst)', 'bit_acc after uint8\nround-trip']
    base_vals = [np.nanmean(b_nm) * 100, np.nanmean(b_nx) * 100, np.nanmean(b_rt) * 100]
    ref_vals = [np.nanmean(r_nm) * 100, np.nanmean(r_nx) * 100, np.nanmean(r_rt) * 100]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    bars1 = ax.bar(x - w / 2, base_vals, w, color=C_BASE, label='Baseline (no refinement)')
    bars2 = ax.bar(x + w / 2, ref_vals, w, color=C_REF, label='Refined')
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                    f'{b.get_height():.1f}%', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Bit accuracy (%)')
    ax.set_ylim(0, 112)
    ax.set_title('Decoding robustness under deployment conditions')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _b64(fig, path)


def generate_batch_report(output_dir, title='Batch Quantitative Evaluation Report'):
    # 从 results.json 生成批量评估可视化报告, 返回 report.html 路径
    recs, rp = _load_results(output_dir)
    if recs is None:
        print(f"未找到有效的 results.json: {rp}")
        return None

    def get(cond, key, sub=None):
        out = []
        for r in recs:
            src = r[f'{cond}_{sub}'] if sub else r[f'{cond}_metrics']
            v = src.get(key)
            if v is not None:
                out.append(v)
        return np.array(out, dtype=float) if out else np.array([np.nan])

    bc, rc = get('base', 'clip_sim'), get('refined', 'clip_sim')
    bd, rd = get('base', 'dino_sim'), get('refined', 'dino_sim')
    bb, rb = get('base', 'bit_acc'), get('refined', 'bit_acc')
    bp, rp_ = get('base', 'psnr'), get('refined', 'psnr')
    bs, rs = get('base', 'ssim'), get('refined', 'ssim')
    b_nm, r_nm = get('base', 'bit_acc_mean', 'noise'), get('refined', 'bit_acc_mean', 'noise')
    b_nx, r_nx = get('base', 'bit_acc_min', 'noise'), get('refined', 'bit_acc_min', 'noise')
    b_rt, r_rt = get('base', 'bit_acc', 'roundtrip'), get('refined', 'bit_acc', 'roundtrip')

    # ---- 指标卡片 ----
    cards = _card(f'{np.nanmean(rc) * 100:.1f}%', 'CLIP similarity (refined)',
                  _level(1 - np.nanmean(rc), 0.35, 0.20), f'baseline {np.nanmean(bc) * 100:.1f}%')
    cards += _card(f'{(1 - np.nanmean(rc)) * 100:.1f}%', 'CLIP privacy protection',
                   _level(1 - np.nanmean(rc), 0.35, 0.20),
                   f'sim<0.70 ratio {np.mean(rc < 0.70) * 100:.0f}%')
    if not np.isnan(rd).all():
        cards += _card(f'{(1 - np.nanmean(rd)) * 100:.1f}%', 'DINO privacy protection',
                       _level(1 - np.nanmean(rd), 0.15, 0.05), f'baseline {(1 - np.nanmean(bd)) * 100:.1f}%')
    cards += _card(f'{np.nanmean(rb) * 100:.1f}%', 'Bit accuracy (refined)',
                   _level(np.nanmean(rb), 0.98, 0.90),
                   f'perfect-decode ratio {np.mean(rb >= 1.0) * 100:.0f}%')
    cards += _card(f'{np.nanmean(r_nm) * 100:.1f}%', 'bit_acc after channel',
                   _level(np.nanmean(r_nm), 0.95, 0.90),
                   f'worst-case mean {np.nanmean(r_nx) * 100:.1f}%')
    cards += _card(f'{np.nanmean(rp_):.1f} dB', 'PSNR (refined)',
                   _level(np.nanmean(rp_), 38, 35), f'baseline {np.nanmean(bp):.1f} dB')
    cards += _card(f'{np.nanmean(rs):.4f}', 'SSIM (refined)',
                   _level(np.nanmean(rs), 0.95, 0.92), f'baseline {np.nanmean(bs):.4f}')

    # ---- 面板 ----
    img1 = _batch_panel_dist(bc, rc, bd, rd, path=os.path.join(output_dir, 'panel_dist.png'))
    img2 = _batch_panel_dumbbell(bc, rc, path=os.path.join(output_dir, 'panel_dumbbell.png'))
    img3 = _batch_panel_pareto(bp, rp_, bc, rc, path=os.path.join(output_dir, 'panel_pareto.png'))
    img4 = _batch_panel_robustness(b_nm, b_nx, b_rt, r_nm, r_nx, r_rt,
                                   path=os.path.join(output_dir, 'panel_robustness.png'))

    # ---- 对照表 ----
    def row(label, fb, fr, fmt):
        return f'<tr><td>{label}</td><td>{fmt.format(fb)}</td><td><b>{fmt.format(fr)}</b></td></tr>'

    cmp_rows = row('CLIP similarity', np.nanmean(bc), np.nanmean(rc), '{:.4f}')
    if not np.isnan(bd).all():
        cmp_rows += row('DINO similarity', np.nanmean(bd), np.nanmean(rd), '{:.4f}')
    cmp_rows += row('Bit accuracy', np.nanmean(bb), np.nanmean(rb), '{:.2%}')
    cmp_rows += row('PSNR (dB)', np.nanmean(bp), np.nanmean(rp_), '{:.2f}')
    cmp_rows += row('SSIM', np.nanmean(bs), np.nanmean(rs), '{:.4f}')
    cmp_rows += row('bit_acc after channel (mean)', np.nanmean(b_nm), np.nanmean(r_nm), '{:.2%}')
    cmp_rows += row('bit_acc after channel (worst)', np.nanmean(b_nx), np.nanmean(r_nx), '{:.2%}')
    cmp_rows += row('bit_acc after uint8 round-trip', np.nanmean(b_rt), np.nanmean(r_rt), '{:.2%}')
    cmp_rows += row('CLIP sim<0.70 ratio', np.mean(bc < 0.70), np.mean(rc < 0.70), '{:.0%}')

    table_html = (f'<section><h2>All metrics comparison (mean)</h2><table class="cmp">'
                  f'<tr><th>Metric</th><th>Baseline (no refinement)</th><th>Refined</th></tr>{cmp_rows}</table></section>')

    meta_rows = [('Images evaluated', len(recs)),
                 ('Result files', 'results.json / summary.json / details.csv')]
    foot = ('Notes: privacy protection = 1 - feature cosine similarity; higher values make CBIR association '
            'harder. Higher PSNR/SSIM means better visual quality. Channel metrics are averaged over random '
            'samples of JPEG/resize/quant noise; the uint8 round-trip simulates saving and reloading a PNG. '
            'Panel PNGs are saved in this directory and can be used directly in papers.')

    report_path = _write_html(os.path.join(output_dir, 'report.html'), title, meta_rows, cards,
                              [('Similarity distribution comparison', img1), ('Per-image improvement', img2),
                               ('Quality-privacy trade-off', img3), ('Decoding robustness', img4)],
                              foot)
    # 追加对照表 section（保持表格在图片面板之后）
    with open(report_path, 'r', encoding='utf-8') as f:
        html = f.read()
    html = html.replace('</div></body>', table_html + '</div></body>')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return report_path


# ======================================================================
# 检索逃逸评估报告 (retrieval_eval.py)
# ======================================================================

def _retr_panel_rankhist(br, rr, path=None):
    # 排名分布直方图 (对数分箱): 基线 vs 精修
    nmax = max(br.max(), rr.max(), 2)
    bins = np.logspace(0, np.log10(nmax + 1), 22)
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.hist(br, bins=bins, alpha=0.55, color=C_BASE, edgecolor='white',
            label=f'Baseline (median {np.median(br):.0f})')
    ax.hist(rr, bins=bins, alpha=0.65, color=C_REF, edgecolor='white',
            label=f'Refined (median {np.median(rr):.0f})')
    ax.set_xscale('log')
    ax.axvline(1.5, ls='--', color=C_BAD, lw=1, label='top-1 boundary')
    ax.set_xlabel('Rank of the cover image (log scale, 1 = retrieved hit)')
    ax.set_ylabel('Number of queries')
    ax.set_title('Retrieval rank distribution (should shift right after refinement)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _b64(fig, path)


def _retr_panel_rankcdf(br, rr, path=None):
    # 排名累积分布: P(rank <= k), 检查 top-k 命中率
    ks = np.arange(1, max(br.max(), rr.max(), 10) + 1)
    cb = np.array([(br <= k).mean() for k in ks])
    cr = np.array([(rr <= k).mean() for k in ks])
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.plot(ks, cb * 100, '-', color=C_BASE, label='Baseline (no refinement)')
    ax.plot(ks, cr * 100, '-', color=C_REF, label='Refined')
    for k in (1, 5, 10):
        ax.axvline(k, ls=':', color='gray', lw=0.8)
        if k <= ks.max():
            ax.text(k + 0.3, 3, f'top-{k}', fontsize=8, color='gray')
    ax.set_xscale('log')
    ax.set_xlabel('k (log scale)')
    ax.set_ylabel('P(rank \u2264 k) (%)')
    ax.set_title('top-k hit rate cumulative curve (should be lower after refinement)')
    ax.set_ylim(0, 104)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _b64(fig, path)


def _retr_panel_rankscatter(br, rr, path=None):
    # 逐查询排名散点 (log-log): 对角线上方 = 逃逸改善
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    lim = max(br.max(), rr.max(), 2)
    ax.plot([1, lim], [1, lim], '--', color='gray', lw=1, label='No change')
    escaped = rr > br
    ax.scatter(br[~escaped], rr[~escaped], s=26, color=C_BAD, alpha=0.75,
               label=f'No improvement / worse ({(~escaped).sum()})')
    ax.scatter(br[escaped], rr[escaped], s=26, color=C_GOOD, alpha=0.75,
               label=f'Escaped ({escaped.sum()})')
    ax.axhline(1.5, ls=':', color=C_REF, lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Baseline rank (log)')
    ax.set_ylabel('Refined rank (log)')
    ax.set_title('Per-query rank change (above the line = cover pushed out = escape success)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25, which='both')
    fig.tight_layout()
    return _b64(fig, path)


def _retr_panel_sims(bo, ro, bd_, rd_, path=None):
    # 相似度竞争散点: x=与原图相似度, y=与最强干扰相似度; 对角线下方 = 逃逸成功
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Decision boundary (sim_cover = sim_distractor)')
    ax.scatter(bo, bd_, s=26, color=C_BASE, alpha=0.8, label='Baseline (no refinement)')
    ax.scatter(ro, rd_, s=26, color=C_REF, alpha=0.8, label='Refined')
    ax.set_xlabel('CLIP similarity between query and cover')
    ax.set_ylabel('CLIP similarity between query and strongest distractor')
    ax.set_title('Escape mechanism (above the diagonal = cover still nearest neighbor = hit)')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _b64(fig, path)


def generate_retrieval_report(output_dir, title='CBIR Retrieval Escape Evaluation Report'):
    # 从 results.json 生成检索评估可视化报告, 返回 report.html 路径
    recs, rp = _load_results(output_dir)
    if recs is None:
        print(f"未找到有效的 results.json: {rp}")
        return None

    br = np.array([r['base']['rank'] for r in recs], dtype=float)
    rr = np.array([r['refined']['rank'] for r in recs], dtype=float)
    bo = np.array([r['base']['sim_original'] for r in recs])
    ro = np.array([r['refined']['sim_original'] for r in recs])
    bdis = np.array([r['base']['sim_top1_distractor'] for r in recs])
    rdis = np.array([r['refined']['sim_top1_distractor'] for r in recs])
    self_ok = np.array([r.get('self_rank') == 1 for r in recs])

    # 读取评估元信息（图库规模等, 不存在时用排名上界推断）
    gallery_size = None
    meta_path = os.path.join(output_dir, 'eval_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            gallery_size = json.load(f).get('gallery_size')
    if gallery_size is None:
        gallery_size = int(max(br.max(), rr.max()))

    t1b, t1r = float((br <= 1).mean()), float((rr <= 1).mean())
    t5b, t5r = float((br <= 5).mean()), float((rr <= 5).mean())
    t10b, t10r = float((br <= 10).mean()), float((rr <= 10).mean())
    mrr_b, mrr_r = float((1.0 / br).mean()), float((1.0 / rr).mean())

    # ---- 指标卡片 ----
    cards = _card(f'{t1b * 100:.1f}%', 'Baseline top-1 hit rate', 'info' if t1b > 0.5 else 'good',
                  'One-shot retrieval capability of the attacker')
    cards += _card(f'{t1r * 100:.1f}%', 'Refined top-1 hit rate',
                   _level(1 - t1r, 0.5, 0.25), f'baseline {t1b * 100:.1f}%')
    cards += _card(f'{(1 - t1r) * 100:.1f}%', 'Retrieval escape rate',
                   _level(1 - t1r, 0.5, 0.25), f'{int((rr > 1).sum())}/{len(rr)} covers pushed out of top-1')
    cards += _card(f'{np.median(br):.0f} \u2192 {np.median(rr):.0f}', 'Median rank', 'info',
                   f'mean {br.mean():.1f} \u2192 {rr.mean():.1f}')
    cards += _card(f'{mrr_r:.3f}', 'MRR (refined)', _level(1 - mrr_r, 0.5, 0.25),
                   f'baseline {mrr_b:.3f}')
    cards += _card(f'{self_ok.mean() * 100:.1f}%', 'Self-check pass rate',
                   'good' if self_ok.all() else 'warn', 'Cover self-query should be 100% top-1')

    # ---- 面板 ----
    img1 = _retr_panel_rankhist(br, rr, path=os.path.join(output_dir, 'panel_rankhist.png'))
    img2 = _retr_panel_rankcdf(br, rr, path=os.path.join(output_dir, 'panel_rankcdf.png'))
    img3 = _retr_panel_rankscatter(br, rr, path=os.path.join(output_dir, 'panel_rankscatter.png'))
    img4 = _retr_panel_sims(bo, ro, bdis, rdis, path=os.path.join(output_dir, 'panel_sims.png'))

    # ---- 对照表 ----
    def row(label, fb, fr, fmt):
        return f'<tr><td>{label}</td><td>{fmt.format(fb)}</td><td><b>{fmt.format(fr)}</b></td></tr>'

    cmp_rows = row('top-1 hit rate', t1b, t1r, '{:.1%}')
    cmp_rows += row('top-5 hit rate', t5b, t5r, '{:.1%}')
    cmp_rows += row('top-10 hit rate', t10b, t10r, '{:.1%}')
    cmp_rows += row('Mean rank', br.mean(), rr.mean(), '{:.1f}')
    cmp_rows += row('Median rank', np.median(br), np.median(rr), '{:.0f}')
    cmp_rows += row('MRR', mrr_b, mrr_r, '{:.3f}')
    cmp_rows += row('Similarity to cover (mean)', bo.mean(), ro.mean(), '{:.4f}')
    cmp_rows += row('Similarity to strongest distractor (mean)', bdis.mean(), rdis.mean(), '{:.4f}')
    table_html = (f'<section><h2>All metrics comparison</h2><table class="cmp">'
                  f'<tr><th>Metric</th><th>Baseline (no refinement)</th><th>Refined</th></tr>{cmp_rows}</table></section>')

    meta_rows = [('Queries', len(recs)), ('Gallery size', gallery_size),
                 ('Random-guess top-1 hit rate', f'{100.0 / gallery_size:.2f}%'),
                 ('Result files', 'results.json / summary.json')]
    foot = ('Notes: top-1 hit rate = fraction of image searches where the attacker retrieves the cover '
            'as the first result; lower is better protection. Retrieval escape rate = 1 - top-1 hit rate. '
            'MRR = mean reciprocal rank of the cover. Self-check = querying the gallery with the cover '
            'itself should hit top-1 100% of the time (feature pipeline sanity check). '
            'Panel PNGs are saved in this directory and can be used directly in papers.')

    report_path = _write_html(os.path.join(output_dir, 'report.html'), title, meta_rows, cards,
                              [('Retrieval rank distribution', img1), ('top-k hit rate curve', img2),
                               ('Per-query rank change', img3), ('Escape mechanism', img4)],
                              foot)
    # 追加对照表 section（保持表格在图片面板之后）
    with open(report_path, 'r', encoding='utf-8') as f:
        html = f.read()
    html = html.replace('</div></body>', table_html + '</div></body>')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return report_path


# ======================================================================
# 基线评估报告 (batch_baseline.py / retrieval_baseline.py, 单一条件无精修)
# ======================================================================

def _bl_panel_dist(clip, dino, path=None):
    # 单条件相似度分布: CLIP 与 DINO 直方图
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    bins = np.linspace(0, 1, 26)
    for ax, v, name in [(axes[0], clip, 'CLIP'), (axes[1], dino, 'DINO')]:
        if len(v) == 0 or np.isnan(v).all():
            ax.set_title(f'{name} (disabled)')
            continue
        ax.hist(v[~np.isnan(v)], bins=bins, alpha=0.75, color=C_REF, edgecolor='white',
                label=f'Encoder output (mean {np.nanmean(v):.3f})')
        ax.axvline(0.70, ls='--', color=C_BAD, lw=1)
        ax.set_title(f'{name} similarity distribution')
        ax.set_xlabel('Feature similarity to cover')
        ax.set_ylabel('Number of images')
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _b64(fig, path)


def _bl_panel_pareto(psnr, clip, path=None):
    # 单条件质量-隐私散点: PSNR vs CLIP 相似度
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.scatter(psnr, clip, s=16, color=C_REF, alpha=0.8)
    ax.axhline(0.70, ls='--', color=C_BAD, lw=1, label='Escape reference 0.70')
    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('CLIP similarity')
    ax.set_title('Quality-privacy trade-off (encoder single forward, no refinement)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _b64(fig, path)


def _bl_panel_robustness(nm, nx, rt, path=None):
    # 单条件部署鲁棒性: 信道噪声与 uint8 往返解码率
    labels = ['bit_acc after channel\n(mean of 8)', 'bit_acc after channel\n(worst)', 'bit_acc after uint8\nround-trip']
    vals = [np.nanmean(nm) * 100, np.nanmean(nx) * 100, np.nanmean(rt) * 100]
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    bars = ax.bar(np.arange(len(labels)), vals, 0.5, color=C_REF)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f'{b.get_height():.1f}%', ha='center', fontsize=9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Bit accuracy (%)')
    ax.set_ylim(0, 112)
    ax.set_title('Decoding robustness under deployment conditions')
    fig.tight_layout()
    return _b64(fig, path)


def generate_batch_baseline_report(output_dir, title='Batch Baseline Evaluation Report'):
    # 从 results.json 生成批量基线评估报告 (单一条件, 网络直出无精修)
    recs, rp = _load_results(output_dir)
    if recs is None:
        print(f"未找到有效的 results.json: {rp}")
        return None

    def get(key, sub=None):
        out = []
        for r in recs:
            src = r[sub] if sub else r['metrics']
            v = src.get(key)
            if v is not None:
                out.append(v)
        return np.array(out, dtype=float) if out else np.array([np.nan])

    clip, dino = get('clip_sim'), get('dino_sim')
    bit = get('bit_acc')
    psnr, ssim = get('psnr'), get('ssim')
    nm, nx = get('bit_acc_mean', 'noise'), get('bit_acc_min', 'noise')
    rt = get('bit_acc', 'roundtrip')

    # ---- 指标卡片 ----
    cards = _card(f'{np.nanmean(clip) * 100:.1f}%', 'CLIP similarity',
                  _level(1 - np.nanmean(clip), 0.35, 0.20), 'encoder single forward, no refinement')
    cards += _card(f'{(1 - np.nanmean(clip)) * 100:.1f}%', 'CLIP privacy protection',
                   _level(1 - np.nanmean(clip), 0.35, 0.20),
                   f'sim<0.70 ratio {np.mean(clip < 0.70) * 100:.0f}%')
    if not np.isnan(dino).all():
        cards += _card(f'{(1 - np.nanmean(dino)) * 100:.1f}%', 'DINO privacy protection',
                       _level(1 - np.nanmean(dino), 0.15, 0.05))
    cards += _card(f'{np.nanmean(bit) * 100:.1f}%', 'Bit accuracy',
                   _level(np.nanmean(bit), 0.98, 0.90),
                   f'perfect-decode ratio {np.mean(bit >= 1.0) * 100:.0f}%')
    cards += _card(f'{np.nanmean(nm) * 100:.1f}%', 'bit_acc after channel',
                   _level(np.nanmean(nm), 0.95, 0.90),
                   f'worst-case mean {np.nanmean(nx) * 100:.1f}%')
    cards += _card(f'{np.nanmean(psnr):.1f} dB', 'PSNR', _level(np.nanmean(psnr), 38, 35))
    cards += _card(f'{np.nanmean(ssim):.4f}', 'SSIM', _level(np.nanmean(ssim), 0.95, 0.92))

    # ---- 面板 ----
    img1 = _bl_panel_dist(clip, dino, path=os.path.join(output_dir, 'panel_dist.png'))
    img2 = _bl_panel_pareto(psnr, clip, path=os.path.join(output_dir, 'panel_pareto.png'))
    img3 = _bl_panel_robustness(nm, nx, rt, path=os.path.join(output_dir, 'panel_robustness.png'))

    # ---- 指标表 ----
    def row(label, v, fmt):
        return f'<tr><td>{label}</td><td><b>{fmt.format(v)}</b></td></tr>'

    tbl_rows = row('CLIP similarity (mean)', np.nanmean(clip), '{:.4f}')
    if not np.isnan(dino).all():
        tbl_rows += row('DINO similarity (mean)', np.nanmean(dino), '{:.4f}')
    tbl_rows += row('Bit accuracy (mean)', np.nanmean(bit), '{:.2%}')
    tbl_rows += row('Bit accuracy = 100% ratio', np.mean(bit >= 1.0), '{:.0%}')
    tbl_rows += row('PSNR (dB)', np.nanmean(psnr), '{:.2f}')
    tbl_rows += row('SSIM', np.nanmean(ssim), '{:.4f}')
    tbl_rows += row('bit_acc after channel (mean)', np.nanmean(nm), '{:.2%}')
    tbl_rows += row('bit_acc after channel (worst)', np.nanmean(nx), '{:.2%}')
    tbl_rows += row('bit_acc after uint8 round-trip', np.nanmean(rt), '{:.2%}')
    tbl_rows += row('CLIP sim<0.70 ratio', np.mean(clip < 0.70), '{:.0%}')
    table_html = (f'<section><h2>All metrics (mean, encoder single forward)</h2><table class="cmp">'
                  f'<tr><th>Metric</th><th>Value</th></tr>{tbl_rows}</table></section>')

    meta_rows = [('Images evaluated', len(recs)),
                 ('Condition', 'Encoder single forward, no per-image refinement'),
                 ('Result files', 'results.json / summary.json / details.csv')]
    foot = ('Notes: baseline evaluation of the network checkpoint alone (no encoding-time refinement). '
            'Compare against the refined condition from batch_eval.py on the same image sample '
            '(same --seed/--val-dir) to quantify the amortization gap. Privacy protection = 1 - feature '
            'cosine similarity. Channel metrics are averaged over random samples of JPEG/resize/quant '
            'noise; the uint8 round-trip simulates saving and reloading a PNG. '
            'Panel PNGs are saved in this directory and can be used directly in papers.')

    report_path = _write_html(os.path.join(output_dir, 'report.html'), title, meta_rows, cards,
                              [('Similarity distribution', img1),
                               ('Quality-privacy trade-off', img2), ('Decoding robustness', img3)],
                              foot)
    with open(report_path, 'r', encoding='utf-8') as f:
        html = f.read()
    html = html.replace('</div></body>', table_html + '</div></body>')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return report_path


def _bl_retr_panel_rankhist(ranks, path=None):
    # 单条件排名分布直方图 (对数分箱)
    nmax = max(ranks.max(), 2)
    bins = np.logspace(0, np.log10(nmax + 1), 22)
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.hist(ranks, bins=bins, alpha=0.75, color=C_REF, edgecolor='white',
            label=f'Encoder output (median {np.median(ranks):.0f})')
    ax.set_xscale('log')
    ax.axvline(1.5, ls='--', color=C_BAD, lw=1, label='top-1 boundary')
    ax.set_xlabel('Rank of the cover image (log scale, 1 = retrieved hit)')
    ax.set_ylabel('Number of queries')
    ax.set_title('Retrieval rank distribution (encoder single forward)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _b64(fig, path)


def _bl_retr_panel_rankcdf(ranks, path=None):
    # 单条件排名累积分布: P(rank <= k)
    ks = np.arange(1, max(ranks.max(), 10) + 1)
    c = np.array([(ranks <= k).mean() for k in ks])
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.plot(ks, c * 100, '-', color=C_REF, lw=2)
    for k in (1, 5, 10):
        ax.axvline(k, ls=':', color='gray', lw=0.8)
        if k <= ks.max():
            ax.text(k + 0.3, 3, f'top-{k}', fontsize=8, color='gray')
    ax.set_xscale('log')
    ax.set_xlabel('k (log scale)')
    ax.set_ylabel('P(rank \u2264 k) (%)')
    ax.set_title('top-k hit rate cumulative curve')
    ax.set_ylim(0, 104)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _b64(fig, path)


def _bl_retr_panel_sims(so, sd, path=None):
    # 单条件相似度竞争散点: 对角线上方 = 逃逸成功
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Decision boundary (sim_cover = sim_distractor)')
    escaped = sd > so
    ax.scatter(so[~escaped], sd[~escaped], s=26, color=C_BAD, alpha=0.8,
               label=f'Cover still nearest ({(~escaped).sum()})')
    ax.scatter(so[escaped], sd[escaped], s=26, color=C_GOOD, alpha=0.8,
               label=f'Escaped ({escaped.sum()})')
    ax.set_xlabel('CLIP similarity between query and cover')
    ax.set_ylabel('CLIP similarity between query and strongest distractor')
    ax.set_title('Escape mechanism (encoder single forward)')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _b64(fig, path)


def generate_retrieval_baseline_report(output_dir, title='CBIR Retrieval Baseline Report'):
    # 从 results.json 生成检索基线报告 (单一条件, 网络直出无精修)
    recs, rp = _load_results(output_dir)
    if recs is None:
        print(f"未找到有效的 results.json: {rp}")
        return None

    ranks = np.array([r['rank'] for r in recs], dtype=float)
    so = np.array([r['sim_original'] for r in recs])
    sd = np.array([r['sim_top1_distractor'] for r in recs])
    self_ok = np.array([r.get('self_rank') == 1 for r in recs])

    # 读取评估元信息（图库规模等, 不存在时用排名上界推断）
    gallery_size = None
    meta_path = os.path.join(output_dir, 'eval_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            gallery_size = json.load(f).get('gallery_size')
    if gallery_size is None:
        gallery_size = int(ranks.max())

    t1 = float((ranks <= 1).mean())
    t5 = float((ranks <= 5).mean())
    t10 = float((ranks <= 10).mean())
    mrr = float((1.0 / ranks).mean())

    # ---- 指标卡片 ----
    cards = _card(f'{t1 * 100:.1f}%', 'top-1 hit rate', _level(1 - t1, 0.5, 0.25),
                  'One-shot retrieval capability of the attacker')
    cards += _card(f'{(1 - t1) * 100:.1f}%', 'Retrieval escape rate',
                   _level(1 - t1, 0.5, 0.25), f'{int((ranks > 1).sum())}/{len(ranks)} covers pushed out of top-1')
    cards += _card(f'{np.median(ranks):.0f}', 'Median rank', 'info', f'mean {ranks.mean():.1f}')
    cards += _card(f'{mrr:.3f}', 'MRR', _level(1 - mrr, 0.5, 0.25))
    cards += _card(f'{self_ok.mean() * 100:.1f}%', 'Self-check pass rate',
                   'good' if self_ok.all() else 'warn', 'Cover self-query should be 100% top-1')

    # ---- 面板 ----
    img1 = _bl_retr_panel_rankhist(ranks, path=os.path.join(output_dir, 'panel_rankhist.png'))
    img2 = _bl_retr_panel_rankcdf(ranks, path=os.path.join(output_dir, 'panel_rankcdf.png'))
    img3 = _bl_retr_panel_sims(so, sd, path=os.path.join(output_dir, 'panel_sims.png'))

    # ---- 指标表 ----
    def row(label, v, fmt):
        return f'<tr><td>{label}</td><td><b>{fmt.format(v)}</b></td></tr>'

    tbl_rows = row('top-1 hit rate', t1, '{:.1%}')
    tbl_rows += row('top-5 hit rate', t5, '{:.1%}')
    tbl_rows += row('top-10 hit rate', t10, '{:.1%}')
    tbl_rows += row('Mean rank', ranks.mean(), '{:.1f}')
    tbl_rows += row('Median rank', np.median(ranks), '{:.0f}')
    tbl_rows += row('MRR', mrr, '{:.3f}')
    tbl_rows += row('Similarity to cover (mean)', so.mean(), '{:.4f}')
    tbl_rows += row('Similarity to strongest distractor (mean)', sd.mean(), '{:.4f}')
    table_html = (f'<section><h2>All metrics (encoder single forward, no refinement)</h2><table class="cmp">'
                  f'<tr><th>Metric</th><th>Value</th></tr>{tbl_rows}</table></section>')

    meta_rows = [('Queries', len(recs)), ('Gallery size', gallery_size),
                 ('Condition', 'Encoder single forward, no per-image refinement'),
                 ('Random-guess top-1 hit rate', f'{100.0 / gallery_size:.2f}%'),
                 ('Result files', 'results.json / summary.json')]
    foot = ('Notes: baseline retrieval evaluation of the network checkpoint alone (no encoding-time '
            'refinement). top-1 hit rate = fraction of image searches where the attacker retrieves the '
            'cover as the first result; lower is better protection. Retrieval escape rate = 1 - top-1 '
            'hit rate. Self-check = querying the gallery with the cover itself should hit top-1 100% '
            'of the time (feature pipeline sanity check). Compare against the refined condition from '
            'retrieval_eval.py with the same gallery configuration to quantify the amortization gap. '
            'Panel PNGs are saved in this directory and can be used directly in papers.')

    report_path = _write_html(os.path.join(output_dir, 'report.html'), title, meta_rows, cards,
                              [('Retrieval rank distribution', img1), ('top-k hit rate curve', img2),
                               ('Escape mechanism', img3)],
                              foot)
    with open(report_path, 'r', encoding='utf-8') as f:
        html = f.read()
    html = html.replace('</div></body>', table_html + '</div></body>')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return report_path


if __name__ == '__main__':
    # 独立运行: python eval_visualizer.py <输出目录> [batch|retrieval|batch_baseline|retrieval_baseline]
    import sys
    if len(sys.argv) < 2:
        print('用法: python eval_visualizer.py <评估输出目录> '
              '[batch|retrieval|batch_baseline|retrieval_baseline]')
        sys.exit(1)
    out_dir = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'batch'
    fns = {'batch': generate_batch_report, 'retrieval': generate_retrieval_report,
           'batch_baseline': generate_batch_baseline_report,
           'retrieval_baseline': generate_retrieval_baseline_report}
    print(fns[mode](out_dir))
