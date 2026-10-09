# -*- coding: utf-8 -*-
# ==============================================================================
# 信道后检索逃逸评估（channel_retrieval_eval.py）
#
# 目的：论文表 IV 的 89% 逃逸率是在**干净发布文件**上测的，而表 VI 显示信道噪声会把
#       与载体的相似度从 0.4047 抬回均值 0.5189——已经超过最强干扰项均值 0.5000。
#       也就是说 C2（抗信道）与 C4（逃逸）目前是在互不相交的条件下分别成立的。
#       本脚本把两者放进同一个条件：**过一遍传播信道之后再检索**，直接给出
#       "逃逸在信道下还剩多少"。
#
# 这是纯推理测量：不训练、不重新编码、不重新精修。查询图直接用 batch_eval 已保存的
# 精修图，只做信道前向 + 特征提取 + 检索。
#
# 输出：results.json（逐图明细）/ summary.json / 控制台对照表（干净 vs 信道）。
#
# 用法：
#   python val/channel_retrieval_eval.py -o other/channel_retrieval
#   python val/channel_retrieval_eval.py --num-queries 0 -o other/channel_retrieval   # 全部 1000 张
#   python val/channel_retrieval_eval.py --num-samples 8 -o other/channel_retrieval
#   python val/channel_retrieval_eval.py --noise "jpeg+resize(0.8,1.2)+quant" -o other/channel_retrieval
#   python val/channel_retrieval_eval.py -c ./experiments/train/checkpoints/checkpoint.pth \
#       -o other/channel_retrieval          # 额外核对信道后解码比特准确率
# ==============================================================================
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}
DEFAULT_REFINED_DIR = os.path.join('other', 'full_validation', 'batch_eval', 'images')
DEFAULT_GALLERY_CACHE = os.path.join('other', 'full_validation', 'retrieval_eval',
                                     'gallery_feats.pt')


def list_images(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def _norm(p):
    """路径归一化：既有缓存里存的是 './dataset/test/x.jpg'，本地枚举得到 'dataset\\test\\x.jpg'。"""
    return os.path.normpath(p).replace('\\', '/').lstrip('./')


def retrieval_metrics(query_feats, gallery_feats, cover_idx, k_list=(1, 5, 10)):
    """与 val/transfer_eval.py 同口径：载体保留在图库内，返回命中率/排名/MRR/竞争余量。"""
    sims = query_feats @ gallery_feats.t()
    n = sims.shape[0]
    out = {'topk': {}, 'ranks': [], 'sim_to_cover': [], 'sim_to_distractor': [],
           'margin': [], 'mrr': 0.0}
    for q in range(n):
        row = sims[q]
        ci = cover_idx[q]
        sim_cover = float(row[ci])
        masked = row.clone()
        masked[ci] = -2.0
        di = int(torch.argmax(masked))
        sim_dist = float(masked[di])
        rank = int((row > sim_cover).sum().item()) + 1
        out['ranks'].append(rank)
        out['sim_to_cover'].append(sim_cover)
        out['sim_to_distractor'].append(sim_dist)
        out['margin'].append(sim_cover - sim_dist)
        out['mrr'] += 1.0 / rank
    ranks = np.asarray(out['ranks'], float)
    for k in k_list:
        out['topk'][k] = float((ranks <= k).mean())
    out['mrr'] /= max(n, 1)
    for key in ('ranks', 'sim_to_cover', 'sim_to_distractor', 'margin'):
        out[key] = np.asarray(out[key], float)
    return out


def summarize(res, label):
    r = res['ranks']
    return {
        'condition': label,
        'n_queries': int(len(r)),
        'self_check_top1': res.get('self_check_top1'),
        'sim_to_cover_mean': float(res['sim_to_cover'].mean()),
        'sim_to_cover_median': float(np.median(res['sim_to_cover'])),
        'sim_to_cover_p95': float(np.percentile(res['sim_to_cover'], 95)),
        'sim_to_cover_max': float(res['sim_to_cover'].max()),
        'top1_hit_rate': res['topk'][1],
        'top5_hit_rate': res['topk'][5],
        'top10_hit_rate': res['topk'][10],
        'evasion_rate': 1.0 - res['topk'][1],
        'mean_rank': float(r.mean()),
        'median_rank': float(np.median(r)),
        'mrr': res['mrr'],
        'sim_to_distractor_mean': float(res['sim_to_distractor'].mean()),
        'margin_mean': float(res['margin'].mean()),
        'bit_acc_mean': res.get('bit_acc_mean'),
    }


@torch.no_grad()
def clip_feats(vs, tensors, batch_size=32):
    out = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.cat(tensors[i:i + batch_size], 0).to(vs.device)
        out.append(vs.content_recognizer(batch).float().cpu())
    return torch.cat(out, 0)


def main():
    parser = argparse.ArgumentParser(
        description='信道后检索逃逸评估：过一遍传播信道之后再检索，测逃逸还剩多少')
    parser.add_argument('--cover-dir', type=str, default=os.path.join('dataset', 'test'))
    parser.add_argument('--refined-dir', type=str, default=DEFAULT_REFINED_DIR)
    parser.add_argument('--gallery-cache', type=str, default=DEFAULT_GALLERY_CACHE,
                        help='既有图库 CLIP 特征缓存（retrieval_eval 产物），路径一致时直接复用')
    parser.add_argument('--noise', type=str, default=None,
                        help='信道配置串，默认取 Config.noise_config')
    parser.add_argument('--num-samples', type=int, default=8,
                        help='每张图采样的信道次数 k，默认 8（与论文表 VI 同口径）')
    parser.add_argument('--num-queries', type=int, default=100,
                        help='查询数（论文同 seed 同序取前 N 张），0=全部，默认 100')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='可选：提供 checkpoint 后额外核对信道后的解码比特准确率')
    parser.add_argument('-o', '--output-dir', type=str, default='other/channel_retrieval')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    from validation_script import ValidationScript
    vs = ValidationScript(checkpoint_path=args.checkpoint, device=device)
    if args.noise:
        vs.cfg.noise_config = args.noise
    vs._noiser = None
    noiser = vs._build_noiser()
    print(f'信道配置: {vs.cfg.noise_config}')
    print(f'设备: {device}')

    # 复刻 batch_eval / retrieval_eval 的采样顺序（同 seed 同序 → 查询集与论文协议一致）
    val_paths = list_images(args.cover_dir)
    if not val_paths:
        raise SystemExit(f'载体目录为空: {args.cover_dir}')
    perm = np.random.RandomState(args.seed).permutation(len(val_paths))
    gallery_paths = [val_paths[i] for i in perm[:len(val_paths)]]
    n_q = len(gallery_paths) if args.num_queries in (0, None) else \
        min(args.num_queries, len(gallery_paths))
    if args.limit:
        n_q = min(n_q, args.limit)

    pairs = []
    for k in range(n_q):
        p = gallery_paths[k]
        stem = os.path.splitext(os.path.basename(p))[0]
        rp = os.path.join(args.refined_dir, f'{stem}_refined.png')
        if os.path.exists(rp):
            pairs.append((k, p, rp))
    if not pairs:
        raise SystemExit(
            f'在 {args.refined_dir} 找不到任何 <stem>_refined.png。\n'
            f'请先运行 val/batch_eval.py 生成精修图，或用 --refined-dir 指定目录。')
    if len(pairs) < n_q:
        print(f'注意: {n_q - len(pairs)} 张查询缺少精修图，已跳过')

    q_idx = [k for k, _, _ in pairs]
    cover_idx = list(q_idx)          # 载体在图库中的下标 = 采样序号
    query_covers = [p for _, p, _ in pairs]
    query_refined = [r for _, _, r in pairs]
    print(f'图库: {len(gallery_paths)} 张 | 查询: {len(pairs)} 张')

    # ---- 图库 CLIP 特征：优先复用既有缓存 ----
    gallery_feats = None
    if args.gallery_cache and os.path.exists(args.gallery_cache):
        try:
            cached = torch.load(args.gallery_cache, map_location='cpu')
            if [_norm(x) for x in cached.get('paths', [])] == [_norm(p) for p in gallery_paths]:
                gallery_feats = F.normalize(cached['feats'].float(), dim=-1)
                print(f'复用图库特征缓存: {args.gallery_cache}')
            else:
                print('图库缓存路径不一致，重新提取')
        except Exception as exc:
            print(f'图库缓存不可用，重新提取 ({exc})')
    if gallery_feats is None:
        print('提取图库 CLIP 特征 ...')
        tensors = [vs.transform(Image.open(p).convert('RGB')).unsqueeze(0)
                   for p in gallery_paths]
        gallery_feats = clip_feats(vs, tensors, args.batch_size)

    # ---- 查询侧张量（[-1,1]），干净与含噪各一次 ----
    cover_t = [vs.transform(Image.open(p).convert('RGB')).unsqueeze(0) for p in query_covers]
    refined_t = [vs.transform(Image.open(r).convert('RGB')).unsqueeze(0) for r in query_refined]

    # 自检：载体自身作查询必须排第一
    self_sims = gallery_feats[cover_idx] @ gallery_feats.t()
    self_rank1 = float((self_sims.argmax(dim=1)
                        == torch.tensor(cover_idx)).float().mean())
    print(f'自检（载体自查询 top-1 命中）: {self_rank1 * 100:.1f}%')

    # ---- 干净条件 ----
    print('提取干净精修图特征 ...')
    clean_feats = clip_feats(vs, refined_t, args.batch_size)
    clean = retrieval_metrics(clean_feats, gallery_feats, cover_idx)
    clean['self_check_top1'] = self_rank1

    # ---- 信道条件：k 次采样，逐次检索，最后按图汇总 ----
    print(f'信道采样 k={args.num_samples} ...')
    accs, per_sample = [], []
    for s in range(args.num_samples):
        noised_t, bits = [], []
        for t, ct in zip(refined_t, cover_t):
            n_img, _ = noiser((t.to(vs.device), ct.to(vs.device)))
            n_img = n_img.detach().cpu()
            noised_t.append(n_img)
            if args.checkpoint is not None:
                pred = (torch.sigmoid(vs.decoder(n_img.to(vs.device))) > 0.5).float()
                bits.append(pred)
        nf = clip_feats(vs, noised_t, args.batch_size)
        per_sample.append(retrieval_metrics(nf, gallery_feats, cover_idx))
        if bits:
            accs.append(bits)
        print(f'  样本 {s + 1}/{args.num_samples}: '
              f'相似度均值 {per_sample[-1]["sim_to_cover"].mean():.4f}  '
              f'top-1 命中 {per_sample[-1]["topk"][1] * 100:.1f}%')

    # 按图汇总：每张图的信道后相似度取 k 次均值/最差，排名取 k 次中位
    sim_stack = np.stack([p['sim_to_cover'] for p in per_sample])       # (k, Q)
    rank_stack = np.stack([p['ranks'] for p in per_sample])             # (k, Q)
    chan = {
        'sim_to_cover': sim_stack.mean(axis=0),
        'ranks': np.median(rank_stack, axis=0),
        'sim_to_distractor': np.stack([p['sim_to_distractor'] for p in per_sample]).mean(axis=0),
    }
    chan['margin'] = chan['sim_to_cover'] - chan['sim_to_distractor']
    chan['topk'] = {k: float((chan['ranks'] <= k).mean()) for k in (1, 5, 10)}
    chan['mrr'] = float((1.0 / chan['ranks']).mean())
    chan['self_check_top1'] = self_rank1
    if accs:
        chan['bit_acc_mean'] = float(np.mean([b.mean().item() for blk in accs for b in blk]))

    summaries = [summarize(clean, 'clean (published file)'),
                 summarize(chan, f'after channel (k={args.num_samples})')]

    # ---- 逐图明细 ----
    per_image = {}
    for i, stem in enumerate(os.path.splitext(os.path.basename(p))[0] for p in query_covers):
        rec = {
            'clean': {'rank': int(clean['ranks'][i]),
                      'sim_to_cover': float(clean['sim_to_cover'][i]),
                      'margin': float(clean['margin'][i])},
            'channel': {'rank': float(chan['ranks'][i]),
                        'sim_to_cover_mean': float(chan['sim_to_cover'][i]),
                        'sim_to_cover_worst': float(sim_stack[:, i].max()),
                        'sim_to_cover_best': float(sim_stack[:, i].min()),
                        'margin_mean': float(chan['margin'][i])},
        }
        per_image[stem] = rec
    with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(per_image, f, ensure_ascii=False, indent=1)

    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'meta': {'cover_dir': args.cover_dir, 'refined_dir': args.refined_dir,
                            'noise_config': vs.cfg.noise_config, 'seed': args.seed,
                            'num_samples': args.num_samples, 'gallery_size': len(gallery_paths),
                            'num_queries': len(pairs), 'device': str(device),
                            'checkpoint': args.checkpoint},
                   'conditions': summaries}, f, ensure_ascii=False, indent=1)

    # ---- 对照表 ----
    print('\n# 信道后检索逃逸评估汇总（查询 = 精修图，载体保留在图库内）')
    print(f'# 图库 {len(gallery_paths)} 张，查询 {len(pairs)} 张，k={args.num_samples}，seed {args.seed}')
    hdr = (f'{"条件":<26}{"相似度均值":>12}{"相似度p95":>11}{"top-1命中":>11}'
           f'{"逃逸率":>9}{"中位排名":>10}{"MRR":>8}')
    print(hdr)
    print('-' * len(hdr))
    for s in summaries:
        print(f'{s["condition"]:<26}{s["sim_to_cover_mean"]:>12.4f}'
              f'{s["sim_to_cover_p95"]:>11.4f}{s["top1_hit_rate"] * 100:>10.1f}%'
              f'{s["evasion_rate"] * 100:>8.1f}%{s["median_rank"]:>10.0f}{s["mrr"]:>8.3f}')
    print(f'\n信道后相似度: 均值 {summaries[1]["sim_to_cover_mean"]:.4f}  '
          f'最大 {summaries[1]["sim_to_cover_max"]:.4f}  '
          f'(干净: 均值 {summaries[0]["sim_to_cover_mean"]:.4f})')
    print(f'信道后最强干扰项相似度均值: {summaries[1]["sim_to_distractor_mean"]:.4f}  '
          f'→ 竞争余量 {summaries[1]["margin_mean"]:+.4f}')
    if summaries[1]['bit_acc_mean'] is not None:
        print(f'信道后解码比特准确率（逐样本均值）: {summaries[1]["bit_acc_mean"] * 100:.1f}%')
    print(f'\n逐图明细: {os.path.join(args.output_dir, "results.json")}\n'
          f'汇总: {os.path.join(args.output_dir, "summary.json")}')


if __name__ == '__main__':
    main()
