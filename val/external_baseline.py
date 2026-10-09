# 模块名: external_baseline.py
# 模块职责: 外部水印系统对比评估（论文第 V 节"与最新方法的对比"数据生产脚本）。
#   以外部系统（当前支持 TrustMark, ICCV 2025 / Adobe）为被测对象，复用 PrivSteg
#   验证流水线的全部公共组件——采样逻辑、信道噪声、uint8 往返、CLIP/DINO 特征、
#   CBIR 检索协议——保证与 batch_baseline.py / retrieval_baseline.py / batch_eval.py
#   的产物同口径，同 seed 同 val-dir 时评估同一批图像，可直接对照。
#   三个被测条件:
#     direct       —— 外部编码器直出（无任何后处理）
#     pgd_matched  —— 直出 + 同预算事后对抗逃逸：逐像素 L∞ 上限 = Config.prv_budget
#                     （PrivSteg 隐私残差的最大振幅，对基线只宽不严），
#                     步数 / 停止规则 / 选优规则与 PrivSteg 编码期精修（算法 1）同构
#     pgd_free     —— 直出 + 放开像素预算的事后对抗逃逸（仅图像 clamp），
#                     记录逃逸深度与视觉代价（PSNR/ΔE）的权衡
#   PGD 共用规则（与算法 1 一致，保证对比公平）:
#     Adam 步进、提前终止条件 sim<=target+0.005、
#     选优只在外部解码器完美解码（raw bit_acc==1 且 ECC 校验通过）的已访问状态中
#     挑选 CLIP 相似度最低者，无合格状态则回退直出图（失败安全）。
#   输出产物（与现有验证脚本一致的工件集合）:
#     results.json            批量逐图记录（断点续跑）
#     retrieval_results.json  检索逐查询记录（断点续跑）
#     summary.json            两阶段汇总统计
#     details.csv             批量逐图明细（可直接进论文）
#     eval_meta.json          协议元信息（供报告读取）
#     report.html + panel_*.png  自包含可视化报告（英文图表, base64 内嵌）
# 运行环境: image_vision conda 环境已安装 trustmark（pip install trustmark）。
# 用法示例:
#   /d/Minconda/envs/image_vision/python.exe val/external_baseline.py \
#       --val-dir ./dataset/test --num-images 1000 \
#       --num-queries 100 --gallery-size 1000 -o other/external_trustmark --device cpu
#   仅重出报告:
#   /d/Minconda/envs/image_vision/python.exe val/external_baseline.py --report-only \
#       -o other/external_trustmark

import argparse
import csv
import json
import os
import shutil
import sys
import time

# 本脚本位于 val/ 子目录：定位项目根并加入 sys.path（import models/losses/config 等
# 根模块），同时把工作目录切到项目根，保证 ./models/dinov2_vits14_pretrain.pth 等
# 相对路径与从项目根直接运行时的行为完全一致
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 复用现有验证脚本的公共组件（与 batch_baseline / retrieval_baseline 同源）
from validation_script import ValidationScript, _delta_e_map
from batch_eval import list_images, load_quiet, metric_stats, uint8_roundtrip
from retrieval_eval import build_gallery_features, retrieve

# 被测条件（顺序即 CSV/报告列顺序）
CONDS = ['direct', 'pgd_matched', 'pgd_free']
COND_LABELS = {'direct': '直出', 'pgd_matched': '直出+PGD(同预算)', 'pgd_free': '直出+PGD(放开预算)'}

# 借用 eval_visualizer 的 HTML 构件与配色样式（matplotlib 缺失时报告整体跳过）
try:
    from eval_visualizer import _b64, _card, _write_html
    _MPL_OK = True
except Exception:
    _MPL_OK = False


# ======================================================================
# TrustMark 封装（外部系统以原生协议运行: 原生 100 比特 BCH_5 包, 有效载荷 61 比特,
# 原生解码分辨率 245; 不做任何针对本项目的改动 —— 对比的是"现成可用"的系统）
# ======================================================================

def build_trustmark(model_type, device):
    # 构造 TrustMark 编码/解码器。loadRemover=False 避免下载不使用的水印移除模型。
    from trustmark import TrustMark
    device_str = 'cuda' if 'cuda' in str(device) else 'cpu'
    tm = TrustMark(model_type=model_type, device=device_str, verbose=True, loadRemover=False)
    secret_len = tm.schemaCapacity() if tm.use_ECC else tm.secret_len
    print(f"TrustMark({model_type}) 就绪: 设备 {device_str}, 有效载荷 {secret_len} 比特 (BCH_5 ECC)")
    return tm


def tensor_to_pil(vs, image_tensor):
    # [-1,1] 张量 -> uint8 PIL（vs.save_image 的静默版本，避免逐图打印日志）
    arr = vs.inverse_transform(image_tensor.squeeze(0).detach()).cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray((arr * 255).astype(np.uint8))


@torch.no_grad()
def ext_encode(tm, cover_pil, secret_str):
    # 外部系统编码: 走 TrustMark 原生 encode (PIL 进, PIL 出)
    return tm.encode(cover_pil, secret_str, MODE='binary')


@torch.no_grad()
def ext_decode_raw(tm, stego_pil):
    # 复刻 TrustMark.subimage_decode 的原生解码路径, 但返回未过 ECC 的原始比特,
    # 用于逐比特准确率统计（与 PrivSteg 的 bit_acc 定义同口径）
    img = stego_pil.resize((tm.model_resolution_dec, tm.model_resolution_dec), Image.BILINEAR)
    stego = transforms.ToTensor()(img).unsqueeze(0).to(tm.decoder.device) * 2.0 - 1.0
    return (tm.decoder.decoder(stego) > 0).cpu().numpy()  # (1, packet_len) bool


def ext_bit_metrics(tm, stego_pil, secret_bits, secret_str):
    # 外部解码器原生逐比特准确率 + ECC 完整恢复判据
    raw = ext_decode_raw(tm, stego_pil)
    secret_len = tm.schemaCapacity() if tm.use_ECC else tm.secret_len
    bit_acc = float((raw[0][:secret_len] == secret_bits).mean())
    pred, detected, _ = tm.ecc.decode_bitstream(raw, MODE='binary')[0]
    ecc_ok = bool(detected) and (pred[:secret_len] == secret_str)
    return bit_acc, bool(ecc_ok)


# ======================================================================
# 指标与部署条件测试（与 batch_eval.py 同口径, 仅解码器换成外部系统）
# ======================================================================

def ext_metrics(vs, tm, stego_t, cover, secret_bits, secret_str,
                feat_cover_clip, feat_cover_dino):
    # 单图全指标: CLIP/DINO 相似度 + 外部解码逐比特准确率 + PSNR/SSIM/ΔE
    with torch.no_grad():
        feat_enc = vs.content_recognizer(stego_t)
        clip_sim = F.cosine_similarity(feat_cover_clip, feat_enc, dim=1).item()
        dino_sim = None
        if feat_cover_dino is not None:
            dino_enc = vs._extract_dino_features_grad(stego_t)
            dino_sim = F.cosine_similarity(feat_cover_dino, dino_enc, dim=1).item()
        bit_acc, ecc_ok = ext_bit_metrics(tm, tensor_to_pil(vs, stego_t), secret_bits, secret_str)
        mse = torch.mean((stego_t - cover) ** 2).item()
        psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')
        ssim = 1 - vs.ssim_loss(cover, stego_t).item()
        dE = _delta_e_map(stego_t, cover)
        return {'clip_sim': clip_sim, 'dino_sim': dino_sim, 'bit_acc': bit_acc,
                'psnr': psnr, 'ssim': ssim,
                'dE_mean': dE.mean().item(),
                'dE_gt5_ratio': (dE > 5.0).float().mean().item(),
                'ecc_ok': float(ecc_ok)}


@torch.no_grad()
def ext_noise_robustness(vs, tm, noiser, stego_t, cover, secret_bits, secret_str,
                         feat_cover_clip, k):
    # 信道鲁棒性: k 次随机噪声采样 (JPEG/resize/quant, 与训练/验证同一 Noiser),
    # 统计外部解码器的逐比特解码率 / ECC 完整恢复率, 以及信道后 CLIP 相似度 (逃逸存活)
    accs, eccs, sims = [], [], []
    for _ in range(k):
        noised, _ = noiser((stego_t, cover))
        bit_acc, ecc_ok = ext_bit_metrics(tm, tensor_to_pil(vs, noised), secret_bits, secret_str)
        accs.append(bit_acc)
        eccs.append(float(ecc_ok))
        feat_n = vs.content_recognizer(noised)
        sims.append(F.cosine_similarity(feat_cover_clip, feat_n, dim=1).item())
    return {'bit_acc_mean': float(np.mean(accs)), 'bit_acc_min': float(np.min(accs)),
            'ecc_ok_mean': float(np.mean(eccs)),
            'clip_sim_mean': float(np.mean(sims)), 'clip_sim_max': float(np.max(sims))}


@torch.no_grad()
def ext_roundtrip(vs, tm, stego_t, secret_bits, secret_str, feat_cover_clip):
    # uint8 往返: 内存内模拟 PNG 保存重载后重新解码
    reloaded = uint8_roundtrip(stego_t)
    bit_acc, ecc_ok = ext_bit_metrics(tm, tensor_to_pil(vs, reloaded), secret_bits, secret_str)
    feat_re = vs.content_recognizer(reloaded)
    return {'bit_acc': bit_acc, 'ecc_ok': float(ecc_ok),
            'clip_sim': F.cosine_similarity(feat_cover_clip, feat_re, dim=1).item()}


# ======================================================================
# 事后对抗逃逸 (PGD) —— 对外部系统直出图像直接优化 CLIP 相似度的组合基线。
#   与算法 1 同构的三条规则: Adam 优化全图扰动; 提前终止 sim<=target+tol;
#   选优只接受"外部解码器完美解码"的已访问状态 (失败安全回退直出图)。
# ======================================================================

def pgd_escape(vs, tm, stego_t, cover, secret_bits, secret_str, feat_cover_clip,
               cap=None, steps=150, lr=0.05, target=0.40, tol=0.005, check_every=1):
    # cap 非空 = 同预算档 (逐像素 L∞ 上限); cap=None = 放开预算档。
    # 提前终止与算法 1 同语义: 仅当"完美解码 且 sim<=target+tol"同时成立才停。
    # 返回 (最终图像, 元信息): 选优只接受外部解码器完美解码的已访问状态,
    # 无合格状态则回退直出图 (fallback=1, 失败安全)。meta 另记录不做门控的
    # 攻击者原始终态 (raw_final_*), 用于展示"事后逃逸对解码的破坏"。
    stop = target + tol
    delta = torch.zeros_like(stego_t, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    best, best_sim = None, float('inf')
    used, reached, t = steps, False, 0
    last_sim, last_bit = None, None

    for t in range(1, steps + 1):
        x = (stego_t + delta).clamp(-1, 1)
        sim = F.cosine_similarity(feat_cover_clip, vs.content_recognizer(x), dim=1).mean()
        opt.zero_grad()
        sim.backward()  # 最小化与载体的 CLIP 相似度 = 最大化检索逃逸
        opt.step()
        if cap is not None:
            with torch.no_grad():
                delta.clamp_(-cap, cap)  # L∞ 像素预算投影

        if (t - 1) % check_every == 0 or t == steps:
            with torch.no_grad():
                x_d = (stego_t + delta).clamp(-1, 1)
                sim_v = F.cosine_similarity(feat_cover_clip,
                                            vs.content_recognizer(x_d), dim=1).item()
                bit_acc, ecc_ok = ext_bit_metrics(tm, tensor_to_pil(vs, x_d),
                                                  secret_bits, secret_str)
                last_sim, last_bit = sim_v, bit_acc
                if bit_acc >= 1.0 and ecc_ok and sim_v < best_sim:
                    best_sim, best = sim_v, x_d.detach().clone()
                if bit_acc >= 1.0 and ecc_ok and sim_v <= stop:
                    used, reached = t, True
                    break

    raw_meta = {'raw_final_sim': last_sim, 'raw_final_bit_acc': last_bit}
    if best is None:
        return stego_t.detach().clone(), {'steps_used': used, 'reached_target': 0.0,
                                          'fallback': 1.0, 'final_bit_acc': 1.0, **raw_meta}
    bit_acc, _ = ext_bit_metrics(tm, tensor_to_pil(vs, best), secret_bits, secret_str)
    return best, {'steps_used': used, 'reached_target': float(reached),
                  'fallback': 0.0, 'final_bit_acc': bit_acc, **raw_meta}


# ======================================================================
# 批量评估阶段（batch）—— 产物结构与 batch_baseline.py 一致
# ======================================================================

def summarize_batch(recs):
    # 聚合逐图记录: 每条件一组全指标统计 (口径同 batch_baseline.summarize)
    out = {}
    for cond in CONDS:
        src = [r[cond] for r in recs
               if cond in r and r[cond].get('metrics') is not None]
        s = {'metrics': {}, 'success': {}, 'noise': {}, 'roundtrip': {}, 'pgd': {}}
        if src:
            for key in ['clip_sim', 'dino_sim', 'bit_acc', 'psnr', 'ssim',
                        'dE_mean', 'dE_gt5_ratio', 'ecc_ok']:
                vals = [r['metrics'][key] for r in src if r['metrics'].get(key) is not None]
                if vals:
                    s['metrics'][key] = metric_stats(vals)
            clip = [r['metrics']['clip_sim'] for r in src]
            bit = [r['metrics']['bit_acc'] for r in src]
            s['success'] = {
                'bit_acc_100': float(np.mean([b >= 1.0 for b in bit])),
                'ecc_ok': float(np.mean([r['metrics'].get('ecc_ok', 0.0) for r in src])),
                'clip_below_070': float(np.mean([c < 0.70 for c in clip])),
                'clip_below_060': float(np.mean([c < 0.60 for c in clip])),
            }
            s['noise'] = {
                'bit_acc_mean': metric_stats([r['noise']['bit_acc_mean'] for r in src]),
                'bit_acc_min': metric_stats([r['noise']['bit_acc_min'] for r in src]),
                'ecc_ok_mean': metric_stats([r['noise']['ecc_ok_mean'] for r in src]),
                'clip_sim_mean': metric_stats([r['noise']['clip_sim_mean'] for r in src]),
            }
            s['roundtrip'] = {
                'bit_acc': metric_stats([r['roundtrip']['bit_acc'] for r in src]),
                'ecc_ok': metric_stats([r['roundtrip']['ecc_ok'] for r in src]),
                'clip_sim': metric_stats([r['roundtrip']['clip_sim'] for r in src]),
            }
            if cond != 'direct':
                pg = {
                    'steps_used': metric_stats([r['pgd']['steps_used'] for r in src]),
                    'reached_target': float(np.mean([r['pgd']['reached_target'] for r in src])),
                    'fallback': float(np.mean([r['pgd']['fallback'] for r in src])),
                }
                raw_sims = [r['pgd']['raw_final_sim'] for r in src
                            if r['pgd'].get('raw_final_sim') is not None]
                raw_bits = [r['pgd']['raw_final_bit_acc'] for r in src
                            if r['pgd'].get('raw_final_bit_acc') is not None]
                if raw_sims:
                    pg['raw_final_sim'] = metric_stats(raw_sims)
                if raw_bits:
                    pg['raw_final_bit_acc'] = metric_stats(raw_bits)
                s['pgd'] = pg
        out[cond] = s
    return out


def print_batch_summary(summary, num_images):
    print('\n' + '#' * 96)
    print(f'# 外部系统批量对比评估汇总 ({num_images} 张)')
    print('#' * 96)
    header = f"{'指标':<26}" + ''.join(f"{COND_LABELS[c]:>22}" for c in CONDS)
    print(header)
    print('-' * 96)
    rows = [('CLIP 相似度', 'clip_sim', '{:.4f}'), ('DINO 相似度', 'dino_sim', '{:.4f}'),
            ('比特准确率', 'bit_acc', '{:.2%}'), ('PSNR (dB)', 'psnr', '{:.2f}'),
            ('SSIM', 'ssim', '{:.4f}'), ('感知色差 ΔE (均值)', 'dE_mean', '{:.3f}'),
            ('ΔE>5 像素占比', 'dE_gt5_ratio', '{:.2%}')]
    for label, key, fmt in rows:
        cells = []
        for c in CONDS:
            m = summary[c]['metrics'].get(key)
            cells.append(f"{fmt.format(m['mean'])}±{m['std']:.3f}" if m else '-')
        print(f"{label:<26}" + ''.join(f"{cell:>22}" for cell in cells))
    print('-' * 96)
    succ_rows = [('bit_acc=100% 占比', 'bit_acc_100', '{:.1%}'),
                 ('ECC 完整恢复占比', 'ecc_ok', '{:.1%}'),
                 ('clip_sim<0.70 占比', 'clip_below_070', '{:.1%}'),
                 ('clip_sim<0.60 占比', 'clip_below_060', '{:.1%}')]
    for label, key, fmt in succ_rows:
        cells = []
        for c in CONDS:
            v = summary[c]['success'].get(key)
            cells.append(f"{fmt.format(v)}" if v is not None else '-')
        print(f"{label:<26}" + ''.join(f"{cell:>22}" for cell in cells))
    print('-' * 96)
    for label, key in [('信道后 bit_acc (均值)', 'bit_acc_mean'),
                       ('信道后 bit_acc (最差)', 'bit_acc_min'),
                       ('信道后 ECC 恢复 (均值)', 'ecc_ok_mean'),
                       ('信道后 clip_sim (均值, 逃逸存活)', 'clip_sim_mean')]:
        cells = []
        for c in CONDS:
            n = summary[c]['noise'].get(key)
            v = n['mean'] if n else None
            cells.append(f"{v:.2%}" if isinstance(v, float) and key != 'clip_sim_mean'
                         else (f"{v:.4f}" if v is not None else '-'))
        print(f"{label:<26}" + ''.join(f"{cell:>22}" for cell in cells))
    print('-' * 96)
    for label, key in [('uint8往返 bit_acc', 'bit_acc'), ('uint8往返 clip_sim', 'clip_sim')]:
        cells = []
        for c in CONDS:
            rt = summary[c]['roundtrip'].get(key)
            v = rt['mean'] if rt else None
            cells.append((f"{v:.2%}" if key == 'bit_acc' else f"{v:.4f}") if v is not None else '-')
        print(f"{label:<26}" + ''.join(f"{cell:>22}" for cell in cells))
    for c in ['pgd_matched', 'pgd_free']:
        pg = summary[c].get('pgd') or {}
        if pg:
            raw = pg.get('raw_final_sim')
            raw_line = (f", 裸终态 sim {raw['mean']:.3f}/bit {pg.get('raw_final_bit_acc', {}).get('mean', float('nan')):.0%}"
                        if raw else "")
            print(f"{COND_LABELS[c]}: 步数均值 {pg['steps_used']['mean']:.0f} | "
                  f"达标率 {pg['reached_target']:.1%} | 回退率 {pg['fallback']:.1%}{raw_line}")
    print('#' * 96)


def write_csv(recs, csv_path):
    # 逐图明细 CSV (宽表: 每条件一组列, 可直接导入论文表格)
    fields = ['image']
    for cond in CONDS:
        fields += [f'{cond}_clip_sim', f'{cond}_dino_sim', f'{cond}_bit_acc',
                   f'{cond}_ecc_ok', f'{cond}_psnr', f'{cond}_ssim',
                   f'{cond}_dE_mean', f'{cond}_dE_gt5_ratio',
                   f'{cond}_noise_bit_mean', f'{cond}_noise_ecc', f'{cond}_noise_clip_mean',
                   f'{cond}_roundtrip_bit', f'{cond}_roundtrip_clip']
        if cond in ('pgd_matched', 'pgd_free'):
            fields += [f'{cond}_pgd_steps', f'{cond}_pgd_reached', f'{cond}_pgd_fallback',
                       f'{cond}_pgd_raw_sim', f'{cond}_pgd_raw_bit']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in recs:
            row = {'image': r['image']}
            for cond in CONDS:
                c = r.get(cond, {})
                m, n, rt = c.get('metrics', {}), c.get('noise', {}), c.get('roundtrip', {})
                for key in ['clip_sim', 'dino_sim', 'bit_acc', 'ecc_ok', 'psnr', 'ssim',
                            'dE_mean', 'dE_gt5_ratio']:
                    row[f'{cond}_{key}'] = m.get(key)
                row[f'{cond}_noise_bit_mean'] = n.get('bit_acc_mean')
                row[f'{cond}_noise_ecc'] = n.get('ecc_ok_mean')
                row[f'{cond}_noise_clip_mean'] = n.get('clip_sim_mean')
                row[f'{cond}_roundtrip_bit'] = rt.get('bit_acc')
                row[f'{cond}_roundtrip_clip'] = rt.get('clip_sim')
                if cond in ('pgd_matched', 'pgd_free'):
                    pg = c.get('pgd', {})
                    row[f'{cond}_pgd_steps'] = pg.get('steps_used')
                    row[f'{cond}_pgd_reached'] = pg.get('reached_target')
                    row[f'{cond}_pgd_fallback'] = pg.get('fallback')
                    row[f'{cond}_pgd_raw_sim'] = pg.get('raw_final_sim')
                    row[f'{cond}_pgd_raw_bit'] = pg.get('raw_final_bit_acc')
            writer.writerow(row)


# ======================================================================
# 检索评估阶段（retrieval）—— 产物结构与 retrieval_baseline.py 一致
# ======================================================================

def _cond_stats(recs, cond):
    ranks = np.array([r[cond]['rank'] for r in recs])
    sim_orig = np.array([r[cond]['sim_original'] for r in recs])
    sim_d = np.array([r[cond]['sim_top1_distractor'] for r in recs])
    return {
        'top1_hit_rate': float((ranks <= 1).mean()),
        'top5_hit_rate': float((ranks <= 5).mean()),
        'top10_hit_rate': float((ranks <= 10).mean()),
        'mean_rank': float(ranks.mean()),
        'median_rank': float(np.median(ranks)),
        'mrr': float((1.0 / ranks).mean()),
        'sim_original_mean': float(sim_orig.mean()),
        'sim_top1_distractor_mean': float(sim_d.mean()),
        'escape_rate': float((ranks > 1).mean()),
    }


def summarize_retrieval(recs):
    # 汇总检索统计 (只对实际记录中存在的条件聚合, pgd 条件可被关闭)
    conds = [c for c in CONDS if recs and c in recs[0]]
    stats = {cond: _cond_stats(recs, cond) for cond in conds}
    stats['self_sanity_top1'] = float(np.mean([r['self_rank'] == 1 for r in recs]))
    return stats


def print_retrieval_summary(stats, num_queries, gallery_size):
    print('\n' + '#' * 100)
    print(f'# 外部系统 CBIR 检索对比汇总 (查询 {num_queries} 张, 图库 {gallery_size} 张)')
    print('#' * 100)
    conds = [c for c in CONDS if c in stats]
    header = (f"{'条件':<22} {'top-1命中':>9} {'top-5命中':>9} {'top-10命中':>10} "
              f"{'平均排名':>8} {'MRR':>7} {'sim(原图)':>10} {'sim(最强干扰)':>13}")
    print(header)
    for cond in conds:
        s = stats[cond]
        print(f"{COND_LABELS[cond]:<22} {s['top1_hit_rate'] * 100:>8.1f}% {s['top5_hit_rate'] * 100:>8.1f}% "
              f"{s['top10_hit_rate'] * 100:>9.1f}% {s['mean_rank']:>8.1f} {s['mrr']:>7.3f} "
              f"{s['sim_original_mean']:>10.4f} {s['sim_top1_distractor_mean']:>13.4f}")
    print(f"\n检索逃逸率 (top-1 未命中): " +
          ' | '.join(f"{COND_LABELS[c]} {(1 - stats[c]['top1_hit_rate']) * 100:.1f}%" for c in conds))
    print(f"自检 (原图自查询 top-1 命中率): {stats['self_sanity_top1'] * 100:.1f}% (应为 100%)")
    print('#' * 100)


# ======================================================================
# 批量评估主流程
# ======================================================================

def run_batch(args, vs, tm, noiser):
    # 与 batch_baseline.py 相同的采样逻辑 (同 seed 同目录时评估同一批图像, 便于对照)
    val_paths = list_images(args.val_dir)
    if not val_paths:
        raise FileNotFoundError(f"目录下未找到图像: {args.val_dir}")
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(val_paths))[:min(args.num_images, len(val_paths))]
    eval_paths = [val_paths[i] for i in idx]
    secret_len = tm.schemaCapacity() if tm.use_ECC else tm.secret_len
    print(f"批量评估规模: {len(eval_paths)} 张 (来源: {args.val_dir}, 条件: {', '.join(CONDS)})")

    results_path = os.path.join(args.output_dir, 'results.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"断点续跑: 已有 {len(results)} 条记录")

    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    cap = args.pgd_cap if args.pgd_cap is not None else vs.cfg.prv_budget
    t_start = time.time()
    for i, path in enumerate(eval_paths):
        if path in results:
            continue
        cover = load_quiet(vs, path)
        # 与 batch_baseline.py 同源的消息随机流 (长度改为外部系统的有效载荷)
        msg_rng = np.random.RandomState(args.seed * 100000 + i)
        secret_bits = msg_rng.randint(0, 2, size=secret_len)
        secret_str = ''.join(str(b) for b in secret_bits)

        cover_pil = tensor_to_pil(vs, cover)
        with torch.no_grad():
            feat_cover_clip = vs.content_recognizer(cover)
            feat_cover_dino = vs._extract_dino_features_grad(cover)

        # 条件 1: 直出
        stego_pil = ext_encode(tm, cover_pil, secret_str)
        stego_t = vs.transform(stego_pil).unsqueeze(0).to(vs.device)
        direct = {
            'metrics': ext_metrics(vs, tm, stego_t, cover, secret_bits, secret_str,
                                   feat_cover_clip, feat_cover_dino),
            'noise': ext_noise_robustness(vs, tm, noiser, stego_t, cover, secret_bits,
                                          secret_str, feat_cover_clip, args.noise_eval),
            'roundtrip': ext_roundtrip(vs, tm, stego_t, secret_bits, secret_str,
                                       feat_cover_clip),
        }
        rec = {'image': path, 'direct': direct}

        # 条件 2/3: 事后 PGD 对抗逃逸 (同预算 / 放开预算)
        for cond, pgd_cap in [('pgd_matched', cap), ('pgd_free', None)]:
            if cond not in args.pgd_conds:
                continue
            final_t, pgd_meta = pgd_escape(
                vs, tm, stego_t, cover, secret_bits, secret_str, feat_cover_clip,
                cap=pgd_cap, steps=args.pgd_steps, lr=args.pgd_lr,
                target=args.pgd_target, tol=args.pgd_tol, check_every=args.pgd_check_every)
            rec[cond] = {
                'metrics': ext_metrics(vs, tm, final_t, cover, secret_bits, secret_str,
                                       feat_cover_clip, feat_cover_dino),
                'noise': ext_noise_robustness(vs, tm, noiser, final_t, cover, secret_bits,
                                              secret_str, feat_cover_clip, args.noise_eval),
                'roundtrip': ext_roundtrip(vs, tm, final_t, secret_bits, secret_str,
                                           feat_cover_clip),
                'pgd': pgd_meta,
            }
            if args.save_images:
                name = os.path.splitext(os.path.basename(path))[0]
                tensor_to_pil(vs, final_t).save(
                    os.path.join(args.output_dir, 'images', f'{name}_{cond}.png'))
        if args.save_images and 'direct' in rec:
            name = os.path.splitext(os.path.basename(path))[0]
            tensor_to_pil(vs, stego_t).save(os.path.join(args.output_dir, 'images',
                                                         f'{name}_direct.png'))

        results[path] = rec
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        m0 = direct['metrics']
        line = (f"[{i + 1}/{len(eval_paths)}] {os.path.basename(path)}: "
                f"direct clip {m0['clip_sim']:.3f} bit {m0['bit_acc'] * 100:.0f}%")
        for cond in ['pgd_matched', 'pgd_free']:
            if cond in rec:
                pm = rec[cond]['metrics']
                line += (f" | {cond} clip {pm['clip_sim']:.3f} bit {pm['bit_acc'] * 100:.0f}% "
                         f"({rec[cond]['pgd']['steps_used']}步, ΔE>5 {pm['dE_gt5_ratio'] * 100:.0f}%)")
        print(line + f" | 已用 {time.time() - t_start:.0f}s")

    # 汇总
    recs = [results[p] for p in eval_paths if p in results]
    return recs


# ======================================================================
# 检索评估主流程
# ======================================================================

def run_retrieval(args, vs, tm, noiser):
    val_paths = list_images(args.val_dir)
    if not val_paths:
        raise FileNotFoundError(f"目录下未找到图像: {args.val_dir}")
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(val_paths))

    gallery_size = min(args.gallery_size, len(val_paths))
    if args.num_queries > gallery_size:
        raise ValueError(f"--num-queries ({args.num_queries}) 不能超过 --gallery-size ({gallery_size})")

    # 查询集是图库的子集 (原图必须在图库内, 检索才有"命中"可言) —— 与 retrieval_eval.py 一致
    gallery_paths = [val_paths[i] for i in perm[:gallery_size]]
    query_paths = [val_paths[i] for i in perm[:args.num_queries]]

    if args.extra_gallery_dir and args.extra_gallery_size > 0:
        extra_paths = list_images(args.extra_gallery_dir)
        extra_idx = rng.choice(len(extra_paths),
                               size=min(args.extra_gallery_size, len(extra_paths)), replace=False)
        gallery_paths += [extra_paths[i] for i in extra_idx]
        print(f"图库扩充: 从 {args.extra_gallery_dir} 追加 {len(extra_idx)} 张干扰图")

    path_to_idx = {p: i for i, p in enumerate(gallery_paths)}
    print(f"检索评估规模: 查询 {len(query_paths)} 张, 图库 {len(gallery_paths)} 张 "
          f"(随机 top-1 命中概率 {100.0 / len(gallery_paths):.2f}%)")

    with open(os.path.join(args.output_dir, 'retrieval_eval_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'gallery_size': len(gallery_paths), 'num_queries': len(query_paths),
                   'system': args.system, 'trustmark_model': args.trustmark_model}, f,
                  ensure_ascii=False, indent=2)

    # 图库特征: 原始载体特征, 与 retrieval_eval.py 完全同构 —— 可通过 --gallery-cache
    # 指向既有缓存 (如 other/full_validation_final/retrieval_eval/gallery_feats.pt) 直接命中
    cache_path = os.path.join(args.output_dir, 'gallery_feats.pt')
    if args.gallery_cache and os.path.isfile(args.gallery_cache):
        shutil.copyfile(args.gallery_cache, cache_path)
        print(f"复用既有图库特征缓存: {args.gallery_cache}")
    gallery_feats = build_gallery_features(vs, gallery_paths, cache_path)

    results_path = os.path.join(args.output_dir, 'retrieval_results.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"断点续跑: 已有 {len(results)} 条检索记录")

    secret_len = tm.schemaCapacity() if tm.use_ECC else tm.secret_len
    cap = args.pgd_cap if args.pgd_cap is not None else vs.cfg.prv_budget
    t_start = time.time()
    for qi, qpath in enumerate(query_paths):
        if qpath in results:
            continue
        cover = load_quiet(vs, qpath)
        original_idx = path_to_idx[qpath]
        msg_rng = np.random.RandomState(args.seed * 100000 + qi)
        secret_bits = msg_rng.randint(0, 2, size=secret_len)
        secret_str = ''.join(str(b) for b in secret_bits)

        with torch.no_grad():
            feat_self = vs.content_recognizer(cover)

        stego_pil = ext_encode(tm, tensor_to_pil(vs, cover), secret_str)
        stego_t = vs.transform(stego_pil).unsqueeze(0).to(vs.device)

        rec = {'image': qpath}
        with torch.no_grad():
            feat_direct = vs.content_recognizer(stego_t)
        rank, sim_o, sim_d, top1 = retrieve(feat_direct, gallery_feats, original_idx)
        rec['direct'] = {'rank': rank, 'sim_original': sim_o,
                         'sim_top1_distractor': sim_d, 'top1_path': gallery_paths[top1]}
        for cond, pgd_cap in [('pgd_matched', cap), ('pgd_free', None)]:
            if cond not in args.pgd_conds:
                continue
            final_t, _ = pgd_escape(vs, tm, stego_t, cover, secret_bits, secret_str,
                                    feat_self, cap=pgd_cap, steps=args.pgd_steps,
                                    lr=args.pgd_lr, target=args.pgd_target,
                                    tol=args.pgd_tol, check_every=args.pgd_check_every)
            with torch.no_grad():
                feat_c = vs.content_recognizer(final_t)
            rank, sim_o, sim_d, top1 = retrieve(feat_c, gallery_feats, original_idx)
            rec[cond] = {'rank': rank, 'sim_original': sim_o,
                         'sim_top1_distractor': sim_d, 'top1_path': gallery_paths[top1]}

        self_rank, _, _, _ = retrieve(feat_self, gallery_feats, original_idx)
        rec['self_rank'] = self_rank
        results[qpath] = rec
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        name = os.path.splitext(os.path.basename(qpath))[0]
        ranks_desc = ' | '.join(f"{COND_LABELS[c]} rank {rec[c]['rank']} (sim {rec[c]['sim_original']:.3f})"
                                for c in CONDS if c in rec)
        print(f"[{qi + 1}/{len(query_paths)}] {name}: {ranks_desc} | 已用时 {time.time() - t_start:.0f}s")

    recs = [results[p] for p in query_paths if p in results]
    return recs, len(gallery_paths)


# ======================================================================
# 可视化报告（与 eval_visualizer.py 同风格: 英文图表 + base64 内嵌 report.html）
# ======================================================================

# 报告配色 (与 eval_visualizer 的语言一致: direct=灰, pgd_matched=蓝, pgd_free=紫)
C_DIRECT = '#9aa7b5'
C_PGD_M = '#3b6fd4'
C_PGD_F = '#8a5fd4'

# 图表内一律用英文标签 (matplotlib DejaVu Sans 无 CJK 字形)
COND_LABELS_EN = {'direct': 'direct',
                  'pgd_matched': 'PGD (matched)',
                  'pgd_free': 'PGD (free)'}


def _get_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def _panel_sim_dist(batch_recs, path=None):
    # 相似度分布对比: 三条件 CLIP 直方图叠加 + 逃逸目标线
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    bins = np.linspace(0, 1, 26)
    for cond, color in zip(CONDS, [C_DIRECT, C_PGD_M, C_PGD_F]):
        v = np.array([r[cond]['metrics']['clip_sim'] for r in batch_recs if cond in r], dtype=float)
        if len(v) == 0:
            continue
        ax.hist(v, bins=bins, alpha=0.55, color=color, edgecolor='white',
                label=f"{cond} (mean {np.nanmean(v):.3f})")
    ax.axvline(0.40, ls='--', color='#d64545', lw=1.2, label='PrivSteg refined target (0.40)')
    ax.axvline(0.70, ls=':', color='#e8a13c', lw=1.2, label='Reference 0.70')
    ax.set_xlabel('CLIP similarity to cover')
    ax.set_ylabel('Number of images')
    ax.set_title('Similarity distribution: direct vs post-hoc adversarial escape')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _b64(fig, path)


def _panel_pareto(batch_recs, path=None):
    # 事后逃逸的视觉代价: PSNR vs CLIP 相似度 (每图一点, 三条件同图)
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for cond, color in zip(CONDS, [C_DIRECT, C_PGD_M, C_PGD_F]):
        pairs = [(r[cond]['metrics']['psnr'], r[cond]['metrics']['clip_sim'])
                 for r in batch_recs if cond in r
                 and r[cond]['metrics'].get('psnr') is not None]
        if not pairs:
            continue
        psnr, clip = zip(*pairs)
        ax.scatter(psnr, clip, s=14, alpha=0.75, color=color, label=cond)
    ax.axhline(0.40, ls='--', color='#d64545', lw=1.2, label='PrivSteg refined target (0.40)')
    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('CLIP similarity to cover')
    ax.set_title('Visual cost of post-hoc escape: PSNR vs escape depth')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _b64(fig, path)


def _panel_impercept(summary, path=None):
    # 不可感知性对比: ΔE 均值与 ΔE>5 像素占比 (条形图, 每条件一组)
    plt = _get_mpl()
    conds = [c for c in CONDS if summary.get(c, {}).get('metrics', {}).get('dE_mean')]
    means = [summary[c]['metrics']['dE_mean']['mean'] for c in conds]
    gt5 = [summary[c]['metrics']['dE_gt5_ratio']['mean'] for c in conds]
    labels = [COND_LABELS_EN[c] for c in conds]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, vals, ylab, t in [(axes[0], means, 'mean ΔE (CIE76)', 'Mean ΔE'),
                              (axes[1], [g * 100 for g in gt5], 'pixels with ΔE>5 (%)', 'ΔE>5 pixel ratio')]:
        bars = ax.bar(np.arange(len(vals)), vals, 0.5,
                      color=[C_DIRECT, C_PGD_M, C_PGD_F][:len(vals)])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02,
                    f'{b.get_height():.2f}', ha='center', fontsize=9)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, fontsize=8, rotation=12)
        ax.set_ylabel(ylab)
        ax.set_title(t)
    fig.tight_layout()
    return _b64(fig, path)


def _panel_robustness(summary, path=None):
    # 部署鲁棒性对比: 信道后/uint8 往返解码率 (每条件一组)
    plt = _get_mpl()
    conds = [c for c in CONDS if summary.get(c, {}).get('noise', {}).get('bit_acc_mean')]
    labels = [COND_LABELS_EN[c] for c in conds]
    nm = [summary[c]['noise']['bit_acc_mean']['mean'] * 100 for c in conds]
    nx = [summary[c]['noise']['bit_acc_min']['mean'] * 100 for c in conds]
    rt = [summary[c]['roundtrip']['bit_acc']['mean'] * 100 for c in conds]
    x = np.arange(len(conds))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for off, vals, name, color in [(-w, nm, 'after channel (mean)', C_DIRECT),
                                   (0, nx, 'after channel (worst)', C_PGD_M),
                                   (w, rt, 'after uint8 round-trip', C_PGD_F)]:
        bars = ax.bar(x + off, vals, w, color=color, label=name)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f'{b.get_height():.1f}', ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=12)
    ax.set_ylabel('Bit accuracy (%)')
    ax.set_ylim(0, 112)
    ax.set_title('Decoding robustness under deployment conditions')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _b64(fig, path)


def _panel_sims(retr_recs, path=None):
    # 逃逸机制散点: sim(原图) vs sim(最强干扰), 对角线上方=逃逸
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    lo, hi = 0.2, 1.0
    for cond, color in zip(CONDS, [C_DIRECT, C_PGD_M, C_PGD_F]):
        so = [r[cond]['sim_original'] for r in retr_recs if cond in r]
        sd = [r[cond]['sim_top1_distractor'] for r in retr_recs if cond in r]
        if not so:
            continue
        lo = min(lo, min(so), min(sd))
        ax.scatter(so, sd, s=12, alpha=0.7, color=color, label=cond)
    lims = [lo, hi]
    ax.plot(lims, lims, ls='--', color='#888', lw=1, label='decision boundary (top-1)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Similarity to original cover')
    ax.set_ylabel('Similarity to strongest distractor')
    ax.set_title('Escape mechanism: below diagonal = cover still top-1')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _b64(fig, path)


def _panel_rankcdf(retr_recs, path=None):
    # 载体排名 CDF (对数横轴): 曲线越靠右, 载体排名越深, 逃逸越彻底
    plt = _get_mpl()
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for cond, color in zip(CONDS, [C_DIRECT, C_PGD_M, C_PGD_F]):
        ranks = np.sort(np.array([r[cond]['rank'] for r in retr_recs if cond in r]))
        if len(ranks) == 0:
            continue
        cdf = np.arange(1, len(ranks) + 1) / len(ranks)
        ax.step(ranks, cdf, where='post', color=color, label=cond)
    ax.set_xscale('log')
    ax.set_xlabel('Rank of original cover in gallery (log)')
    ax.set_ylabel('Cumulative fraction of queries')
    ax.set_title('Rank CDF: where the cover lands after embedding')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _b64(fig, path)


def generate_external_report(output_dir, title='External Baseline Comparison Report'):
    # 从 results.json / retrieval_results.json / summary.json 生成对比报告;
    # matplotlib 缺失或失败时跳过, 不影响评估结果本身
    if not _MPL_OK:
        print("eval_visualizer/matplotlib 缺失, 跳过可视化报告")
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False

    rp = os.path.join(output_dir, 'results.json')
    if not os.path.exists(rp):
        print(f"未找到有效的 results.json: {rp}")
        return None
    with open(rp, encoding='utf-8') as f:
        batch_results = json.load(f)
    batch_recs = list(batch_results.values())

    retr_recs = []
    rrp = os.path.join(output_dir, 'retrieval_results.json')
    if os.path.exists(rrp):
        with open(rrp, encoding='utf-8') as f:
            retr_recs = list(json.load(f).values())

    summary = {}
    sp = os.path.join(output_dir, 'summary.json')
    if os.path.exists(sp):
        with open(sp, encoding='utf-8') as f:
            summary = json.load(f)
    batch_summary = summary.get('batch', summarize_batch(batch_recs))

    def bstat(cond, sec, key):
        v = batch_summary.get(cond, {}).get(sec, {}).get(key)
        return v['mean'] if v else None

    # ---- 指标卡片: 每条件 CLIP 相似度 + 逃逸率 (检索阶段存在时) ----
    cards = ''
    for cond, color in zip(CONDS, ['bad', 'info', 'info']):
        m = batch_summary.get(cond, {}).get('metrics', {}).get('clip_sim')
        if m:
            sub = f"bit acc {batch_summary[cond]['metrics'].get('bit_acc', {}).get('mean', float('nan')):.1%}"
            cards += _card(f"{m['mean']:.3f}", f'CLIP sim ({cond})', color, sub)
    if retr_recs:
        for cond in CONDS:
            if retr_recs and cond in retr_recs[0]:
                er = float(np.mean([r[cond]['rank'] > 1 for r in retr_recs]))
                cards += _card(f"{er * 100:.1f}%", f'escape rate ({cond})', 'good')

    # ---- 面板 ----
    sections = []
    img = _panel_sim_dist(batch_recs, path=os.path.join(output_dir, 'panel_dist.png'))
    if img:
        sections.append(('CLIP similarity distribution', img))
    img = _panel_pareto(batch_recs, path=os.path.join(output_dir, 'panel_pareto.png'))
    if img:
        sections.append(('Visual cost of post-hoc escape', img))
    img = _panel_impercept(batch_summary, path=os.path.join(output_dir, 'panel_impercept.png'))
    if img:
        sections.append(('Imperceptibility', img))
    img = _panel_robustness(batch_summary, path=os.path.join(output_dir, 'panel_robustness.png'))
    if img:
        sections.append(('Decoding robustness', img))
    if retr_recs and CONDS[0] in retr_recs[0]:
        img = _panel_sims(retr_recs, path=os.path.join(output_dir, 'panel_sims.png'))
        if img:
            sections.append(('Retrieval escape mechanism', img))
        img = _panel_rankcdf(retr_recs, path=os.path.join(output_dir, 'panel_rankcdf.png'))
        if img:
            sections.append(('Rank CDF', img))

    # ---- 全指标对照表 ----
    def stat_cell(s, key, fmt):
        v = s.get(key)
        return fmt.format(v['mean']) if v else '-'

    metric_rows = [
        ('CLIP similarity (mean, lower = more private)', 'clip_sim', '{:.4f}'),
        ('DINO similarity (mean)', 'dino_sim', '{:.4f}'),
        ('Bit accuracy (mean)', 'bit_acc', '{:.2%}'),
        ('ECC fully-decoded ratio', 'ecc_ok', '{:.2%}'),
        ('PSNR (dB)', 'psnr', '{:.2f}'),
        ('SSIM', 'ssim', '{:.4f}'),
        ('Mean ΔE (CIE76)', 'dE_mean', '{:.3f}'),
        ('Pixels with ΔE>5', 'dE_gt5_ratio', '{:.2%}'),
    ]
    tbl = '<tr><th>Metric</th>' + ''.join(f'<th>{COND_LABELS[c]}</th>' for c in CONDS) + '</tr>'
    for label, key, fmt in metric_rows:
        cells = ''.join(f'<td>{stat_cell(batch_summary.get(c, {}).get("metrics", {}), key, fmt)}</td>'
                        for c in CONDS)
        tbl += f'<tr><td>{label}</td>{cells}</tr>'
    for label, sec, key, fmt in [
            ('bit_acc after channel (mean)', 'noise', 'bit_acc_mean', '{:.2%}'),
            ('bit_acc after channel (worst)', 'noise', 'bit_acc_min', '{:.2%}'),
            ('ECC after channel (mean)', 'noise', 'ecc_ok_mean', '{:.2%}'),
            ('CLIP sim after channel (escape survival)', 'noise', 'clip_sim_mean', '{:.4f}'),
            ('bit_acc after uint8 round-trip', 'roundtrip', 'bit_acc', '{:.2%}'),
            ('CLIP sim after uint8 round-trip', 'roundtrip', 'clip_sim', '{:.4f}')]:
        cells = ''.join(f'<td>{stat_cell(batch_summary.get(c, {}).get(sec, {}), key, fmt)}</td>'
                        for c in CONDS)
        tbl += f'<tr><td>{label}</td>{cells}</tr>'
    for label, key, fmt in [
            ('PGD steps used (mean)', 'steps_used', '{:.0f}'),
            ('PGD raw final CLIP sim (no decode gate)', 'raw_final_sim', '{:.4f}'),
            ('PGD raw final bit accuracy (no decode gate)', 'raw_final_bit_acc', '{:.2%}')]:
        cells = ''.join(f'<td>{stat_cell(batch_summary.get(c, {}).get("pgd", {}), key, fmt)}</td>'
                        for c in CONDS)
        tbl += f'<tr><td>{label}</td>{cells}</tr>'
    if retr_recs and CONDS[0] in retr_recs[0]:
        rstats = summarize_retrieval(retr_recs)
        rconds = [c for c in CONDS if c in rstats]
        retr_rows = [
            ('top-1 hit rate (lower = more private)', 'top1_hit_rate', '{:.1%}'),
            ('top-5 hit rate', 'top5_hit_rate', '{:.1%}'),
            ('escape rate', 'escape_rate', '{:.1%}'),
            ('mean rank', 'mean_rank', '{:.1f}'),
            ('MRR', 'mrr', '{:.3f}'),
            ('sim to cover (mean)', 'sim_original_mean', '{:.4f}'),
            ('sim to strongest distractor (mean)', 'sim_top1_distractor_mean', '{:.4f}'),
        ]
        for label, key, fmt in retr_rows:
            cells = ''.join(f'<td>{fmt.format(rstats[c][key])}</td>' for c in rconds)
            tbl += f'<tr><td>{label}</td>{cells}</tr>'
    table_html = (f'<section><h2>All metrics comparison</h2><table class="cmp">{tbl}</table></section>')

    # ---- 元信息与脚注 ----
    meta_rows = []
    emp = os.path.join(output_dir, 'eval_meta.json')
    if os.path.exists(emp):
        with open(emp, encoding='utf-8') as f:
            meta = json.load(f)
        for k, v in meta.items():
            meta_rows.append((k, v))
    gallery_size = '-'
    remp = os.path.join(output_dir, 'retrieval_eval_meta.json')
    if os.path.exists(remp):
        with open(remp, encoding='utf-8') as f:
            gallery_size = json.load(f)['gallery_size']
    meta_rows.append(('Images (batch) / Queries / Gallery',
                      f"{len(batch_recs)} / {len(retr_recs)} / {gallery_size}"))
    foot = ('Notes: external system evaluated with the SAME protocol as PrivSteg (same image sample, '
            'same seed, same noise channel, same gallery/self-check retrieval protocol). '
            'pgd_matched constrains the perturbation to the same per-pixel budget as PrivSteg '
            '(Config.prv_budget); pgd_free removes the budget and stops at the same similarity target. '
            'Both share the no-degradation selection rule: the published state must decode perfectly '
            'under the external decoder (raw bits + ECC). Panel PNGs are saved in this directory.')
    _write_html(os.path.join(output_dir, 'report.html'), title, meta_rows, cards, sections, foot)
    report_path = os.path.join(output_dir, 'report.html')
    with open(report_path, 'r', encoding='utf-8') as f:
        html = f.read()
    html = html.replace('</div></body>', table_html + '</div></body>')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"可视化报告: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='外部水印系统对比评估 (TrustMark vs PrivSteg 协议)')
    parser.add_argument('--system', type=str, default='trustmark', choices=['trustmark'],
                        help='被测外部系统 (预留扩展)')
    parser.add_argument('--trustmark-model', type=str, default='Q', choices=['C', 'Q', 'B', 'P'],
                        help='TrustMark 变体 (Q: 论文默认, PSNR~43)')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='PrivSteg checkpoint (可选, 仅用于复用特征器; 对比本身不使用其权重)')
    parser.add_argument('--val-dir', type=str, default='./dataset/test',
                        help='图像来源目录 (与 full_validation_final 对照需用 ./dataset/test)')
    parser.add_argument('--num-images', type=int, default=200, help='批量评估图像数')
    parser.add_argument('--noise-eval', type=int, default=8, help='信道鲁棒性噪声采样次数')
    parser.add_argument('--num-queries', type=int, default=100, help='检索查询数')
    parser.add_argument('--gallery-size', type=int, default=1000, help='主图库大小')
    parser.add_argument('--extra-gallery-dir', type=str, default=None, help='可选图库扩充目录')
    parser.add_argument('--extra-gallery-size', type=int, default=0)
    parser.add_argument('--gallery-cache', type=str, default=None,
                        help='可复用的既有 gallery_feats.pt 路径 (与 retrieval_eval 同图库时命中)')
    parser.add_argument('--pgd-steps', type=int, default=150, help='事后 PGD 最大步数 (同精修预算)')
    parser.add_argument('--pgd-lr', type=float, default=0.05)
    parser.add_argument('--pgd-target', type=float, default=0.40,
                        help='PGD 逃逸目标 (与 PrivSteg 精修下限 1-m_c=0.40 对齐)')
    parser.add_argument('--pgd-tol', type=float, default=0.005, help='提前终止容差')
    parser.add_argument('--pgd-cap', type=float, default=None,
                        help='pgd_matched 的逐像素 L∞ 预算 (None=取 Config.prv_budget)')
    parser.add_argument('--pgd-check-every', type=int, default=1, help='候选状态评估间隔步数')
    parser.add_argument('--pgd-conds', type=str, default='pgd_matched,pgd_free',
                        help='启用的事后逃逸条件 (逗号分隔, 可为空字符串关闭)')
    parser.add_argument('--skip-retrieval', action='store_true', help='跳过检索评估阶段')
    parser.add_argument('--save-images', dest='save_images', action='store_true', default=False,
                        help='保存各条件编码图 PNG')
    parser.add_argument('--report-only', action='store_true',
                        help='不跑评估, 从已有 results.json 重新生成可视化报告')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', '-o', type=str, default='./other/external_trustmark')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    args.pgd_conds = [c.strip() for c in args.pgd_conds.split(',') if c.strip()]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else None
    os.makedirs(args.output_dir, exist_ok=True)

    if args.report_only:
        try:
            generate_external_report(args.output_dir)
        except Exception as e:
            print(f"可视化报告生成失败: {e}")
        return

    vs = ValidationScript(checkpoint_path=args.checkpoint, device=device)
    noiser = vs._build_noiser()
    tm = build_trustmark(args.trustmark_model, vs.device)

    cap = args.pgd_cap if args.pgd_cap is not None else vs.cfg.prv_budget
    with open(os.path.join(args.output_dir, 'eval_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'system': args.system, 'trustmark_model': args.trustmark_model,
                   'secret_len': tm.schemaCapacity() if tm.use_ECC else tm.secret_len,
                   'val_dir': args.val_dir, 'num_images': args.num_images,
                   'noise_eval': args.noise_eval, 'pgd_steps': args.pgd_steps,
                   'pgd_lr': args.pgd_lr, 'pgd_target': args.pgd_target,
                   'pgd_tol': args.pgd_tol, 'pgd_cap': cap,
                   'pgd_conds': ','.join(args.pgd_conds), 'seed': args.seed},
                  f, ensure_ascii=False, indent=2)

    # ===== 阶段 1: 批量保真与部署条件 =====
    recs = run_batch(args, vs, tm, noiser)
    batch_summary = summarize_batch(recs)
    summary = {'system': args.system, 'batch': batch_summary}
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(recs, os.path.join(args.output_dir, 'details.csv'))
    print_batch_summary(batch_summary, len(recs))
    print(f"\n批量结果已保存: {args.output_dir} (results.json / summary.json / details.csv)")

    # ===== 阶段 2: CBIR 检索对比 =====
    if not args.skip_retrieval:
        retr_recs, gallery_size = run_retrieval(args, vs, tm, noiser)
        retr_summary = summarize_retrieval(retr_recs)
        summary['retrieval'] = retr_summary
        with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print_retrieval_summary(retr_summary, len(retr_recs), gallery_size)

    # ===== 可视化报告 =====
    try:
        generate_external_report(args.output_dir)
    except Exception as e:
        print(f"可视化报告生成失败: {e}")


if __name__ == '__main__':
    main()
