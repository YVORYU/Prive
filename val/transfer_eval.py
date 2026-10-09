# -*- coding: utf-8 -*-
# ==============================================================================
# 跨骨干迁移评估（transfer_eval.py）
#
# 目的：论文的逃逸保证是在白盒设定下针对 CLIP ViT-B/32 与 DINOv2 ViT-S/14 标定的，
#       而攻击者可以换一个检索骨干。本脚本用**未参与标定的第三方骨干**测量逃逸的
#       迁移程度：把已经生成好的精修图当查询，看它在别的特征空间里还能不能逃离载体。
#
# 这是纯推理测量：不训练、不重新编码、不重新精修，只对磁盘上既有的原图与精修图
# 提取特征并做检索。默认骨干为 torchvision 有监督模型（ResNet-50 / ViT-B/16），
# 与 CLIP（对比学习）和 DINOv2（自监督）属于不同架构族与不同训练范式。
#
# 输出：results.json（逐图明细，断点续跑）/ summary.json / 控制台对照表。
#
# 用法：
#   python val/transfer_eval.py -o other/transfer_eval
#   python val/transfer_eval.py --backbones resnet50,vit_b_16,convnext_tiny -o other/transfer_eval
#   python val/transfer_eval.py --include-clip -o other/transfer_eval   # 附带受训空间做对照
#   python val/transfer_eval.py --limit 50 -o other/transfer_smoke      # 冒烟
# ==============================================================================
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# 与 val/ 下其它评估脚本一致：把项目根加入 sys.path 并把工作目录切到项目根
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

# 论文发布配方在 batch_eval 中保存精修图的默认位置
DEFAULT_REFINED_DIR = os.path.join('other', 'full_validation', 'batch_eval', 'images')


def list_images(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def _norm(p):
    return os.path.normpath(p).replace('\\', '/').lstrip('./')


# ------------------------------------------------------------------ 骨干构造
def _strip_head(model, name):
    """把分类头换成 Identity，保留倒数第二层特征。"""
    for attr in ('fc', 'head', 'heads', 'classifier'):
        if hasattr(model, attr):
            head = getattr(model, attr)
            if isinstance(head, nn.Sequential):
                # convnext/efficientnet 的 classifier 是 Sequential，替换最后一层
                setattr(model, attr, nn.Sequential(*list(head.children())[:-1])
                        if len(list(head.children())) > 1 else nn.Identity())
            else:
                setattr(model, attr, nn.Identity())
    return model


def build_torchvision_backbone(name, device):
    import torchvision.models as tvm

    table = {
        'resnet18': (tvm.resnet18, 'ResNet18_Weights'),
        'resnet50': (tvm.resnet50, 'ResNet50_Weights'),
        'resnet101': (tvm.resnet101, 'ResNet101_Weights'),
        'vit_b_16': (tvm.vit_b_16, 'ViT_B_16_Weights'),
        'vit_b_32': (tvm.vit_b_32, 'ViT_B_32_Weights'),
        'vit_l_16': (tvm.vit_l_16, 'ViT_L_16_Weights'),
        'swin_t': (tvm.swin_t, 'Swin_T_Weights'),
        'convnext_tiny': (tvm.convnext_tiny, 'ConvNeXt_Tiny_Weights'),
        'efficientnet_b0': (tvm.efficientnet_b0, 'EfficientNet_B0_Weights'),
    }
    if name not in table:
        raise ValueError(f'未知骨干 {name}；可选: {", ".join(sorted(table))}')
    ctor, wname = table[name]
    weights = getattr(tvm, wname).IMAGENET1K_V1
    model = ctor(weights=weights)
    transform = weights.transforms()
    model = _strip_head(model, name).to(device).eval()
    return model, transform


@torch.no_grad()
def extract_torchvision(model, transform, paths, device, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([
            transform(Image.open(p).convert('RGB')) for p in paths[i:i + batch_size]
        ]).to(device)
        f = model(batch)
        f = f.flatten(1) if f.dim() > 2 else f
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def extract_clip(paths, device, batch_size=32):
    """受训空间对照：复用项目内的 CLIP 封装（输入 [-1,1]，内部自行归一化）。"""
    from validation_script import ValidationScript
    vs = ValidationScript(checkpoint_path=None, device=device)
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([vs.transform(Image.open(p).convert('RGB'))
                             for p in paths[i:i + batch_size]]).to(device)
        feats.append(vs.content_recognizer(batch).float().cpu())
    del vs
    return torch.cat(feats, 0)


# ------------------------------------------------------------------ 检索指标
def retrieval_metrics(query_feats, gallery_feats, cover_idx, k_list=(1, 5, 10)):
    """查询特征对全图库做余弦检索，返回命中率/排名/MRR/竞争余量。

    cover_idx[q] = 查询 q 的载体在图库中的下标（载体本身保留在图库内）。
    """
    sims = query_feats @ gallery_feats.t()                    # (Q, G)
    n = sims.shape[0]
    out = {'topk': {}, 'ranks': [], 'sim_to_cover': [], 'sim_to_distractor': [],
           'margin': [], 'sim_offdiag': [], 'mrr': 0.0}
    for q in range(n):
        row = sims[q]
        ci = cover_idx[q]
        sim_cover = float(row[ci])
        # 最强干扰项：排除载体自身后相似度最高的图库项
        masked = row.clone()
        masked[ci] = -2.0
        di = int(torch.argmax(masked))
        sim_dist = float(masked[di])
        # 载体排名 = 比载体更相似的图库项数 + 1（并列时载体优先）
        rank = int((row > sim_cover).sum().item()) + 1
        out['ranks'].append(rank)
        out['sim_to_cover'].append(sim_cover)
        out['sim_to_distractor'].append(sim_dist)
        out['margin'].append(sim_cover - sim_dist)
        # 与其余图库项的平均相似度：衡量该特征空间的"整体压缩程度"，
        # 用于判断 sim_to_cover 是高还是低（有监督 CNN 特征常整体偏高）
        out['sim_offdiag'].append(float(row[masked > -2.0].mean()))
        out['mrr'] += 1.0 / rank
    ranks = np.asarray(out['ranks'], float)
    for k in k_list:
        out['topk'][k] = float((ranks <= k).mean())
    out['mrr'] /= max(n, 1)
    for key in ('ranks', 'sim_to_cover', 'sim_to_distractor', 'margin', 'sim_offdiag'):
        out[key] = np.asarray(out[key], float)
    return out


def summarize(res, label):
    r = res['ranks']
    return {
        'backbone': label,
        'n_queries': int(len(r)),
        'self_check_top1': res['self_check_top1'],
        'clip_sim_mean': float(res['sim_to_cover'].mean()),
        'clip_sim_median': float(np.median(res['sim_to_cover'])),
        'sim_offdiag_mean': float(res['sim_offdiag'].mean()),
        'top1_hit_rate': res['topk'][1],
        'top5_hit_rate': res['topk'][5],
        'top10_hit_rate': res['topk'][10],
        'evasion_rate': 1.0 - res['topk'][1],
        'mean_rank': float(r.mean()),
        'median_rank': float(np.median(r)),
        'mrr': res['mrr'],
        'margin_mean': float(res['margin'].mean()),
    }


# ------------------------------------------------------------------ 主流程
def main():
    parser = argparse.ArgumentParser(
        description='跨骨干迁移评估：用未参与逃逸标定的第三方骨干测量检索逃逸的迁移程度')
    parser.add_argument('--cover-dir', type=str, default=os.path.join('dataset', 'test'),
                        help='载体图像目录（图库来源），默认 ./dataset/test')
    parser.add_argument('--refined-dir', type=str, default=DEFAULT_REFINED_DIR,
                        help='精修图目录，文件名须为 <载体文件名去扩展名>_refined.png')
    parser.add_argument('--backbones', type=str, default='resnet50,vit_b_16',
                        help='逗号分隔的 torchvision 骨干名，默认 resnet50,vit_b_16')
    parser.add_argument('--include-clip', action='store_true',
                        help='额外测量受训空间 CLIP ViT-B/32 作为对照（需项目内 CLIP）')
    parser.add_argument('--num-queries', type=int, default=100,
                        help='查询数（按论文同 seed 同序取前 N 张），0=全部，默认 100')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0,
                        help='只评估前 N 张（冒烟用），0=不限制')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('-o', '--output-dir', type=str, default='other/transfer_eval')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 复刻 batch_eval / retrieval_eval 的采样顺序，保证查询集与论文协议一致
    val_paths = list_images(args.cover_dir)
    if not val_paths:
        raise SystemExit(f'载体目录为空: {args.cover_dir}')
    perm = np.random.RandomState(args.seed).permutation(len(val_paths))
    gallery_paths = [val_paths[i] for i in perm[:len(val_paths)]]
    n_q = len(gallery_paths) if args.num_queries in (0, None) else \
        min(args.num_queries, len(gallery_paths))
    if args.limit:
        n_q = min(n_q, args.limit)
    query_paths = gallery_paths[:n_q]

    # 把查询路径映射到精修图；缺图的样本直接跳过，保持可复现
    refined_root = args.refined_dir
    pairs = []
    for p in query_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        rp = os.path.join(refined_root, f'{stem}_refined.png')
        if os.path.exists(rp):
            pairs.append((p, rp))
    if not pairs:
        raise SystemExit(
            f'在 {refined_root} 找不到任何 <stem>_refined.png。\n'
            f'请先运行 val/batch_eval.py 生成精修图，或用 --refined-dir 指定目录。')
    if len(pairs) < len(query_paths):
        print(f'注意: {len(query_paths) - len(pairs)} 张查询缺少精修图，已跳过')

    query_covers = [p for p, _ in pairs]
    query_refined = [r for _, r in pairs]
    path_to_gallery = {p: i for i, p in enumerate(gallery_paths)}
    cover_idx = [path_to_gallery[p] for p in query_covers]

    print(f'图库: {len(gallery_paths)} 张 ({args.cover_dir})')
    print(f'查询: {len(pairs)} 张精修图 ({refined_root})')
    print(f'设备: {device}')

    backbones = [b.strip() for b in args.backbones.split(',') if b.strip()]
    results_path = os.path.join(args.output_dir, 'results.json')
    summary_path = os.path.join(args.output_dir, 'summary.json')
    cache_path = os.path.join(args.output_dir, 'transfer_feats.pt')

    # 特征缓存：图库特征与查询特征按骨干名分槽保存，断点续跑
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = torch.load(cache_path, map_location='cpu')
            cached_paths = cache.get('gallery_paths')
            if cached_paths and [_norm(x) for x in cached_paths] != \
                    [_norm(p) for p in gallery_paths]:
                print('特征缓存图库不一致（--cover-dir 或 --seed 变了），重新提取')
                cache = {}
            else:
                print(f'载入特征缓存: {sorted(k for k in cache if k.startswith("feats_"))}')
        except Exception as exc:
            print(f'特征缓存不可用，重新提取 ({exc})')
            cache = {}
    cache['gallery_paths'] = gallery_paths

    summaries = []
    per_image = {}
    todo = backbones + (['clip'] if args.include_clip else [])

    for name in todo:
        key = f'feats_{name}'
        if key not in cache:
            print(f'[{name}] 提取特征 ...')
            if name == 'clip':
                g = extract_clip(gallery_paths, device, args.batch_size)
                q = extract_clip(query_refined, device, args.batch_size)
            else:
                model, transform = build_torchvision_backbone(name, device)
                g = extract_torchvision(model, transform, gallery_paths, device, args.batch_size)
                q = extract_torchvision(model, transform, query_refined, device, args.batch_size)
                del model
            cache[key] = {'gallery': g, 'query': q}
            torch.save(cache, cache_path)
            print(f'[{name}] 特征维度 {g.shape[1]}，已缓存')

        g, q = cache[key]['gallery'], cache[key]['query']
        if q.shape[0] != len(cover_idx):
            # 缓存与当前查询集不一致（改了 --num-queries 或 --limit），重算查询侧
            print(f'[{name}] 查询集变化，重新提取查询特征 ...')
            if name == 'clip':
                q = extract_clip(query_refined, device, args.batch_size)
            else:
                model, transform = build_torchvision_backbone(name, device)
                q = extract_torchvision(model, transform, query_refined, device, args.batch_size)
                del model
            cache[key]['query'] = q
            torch.save(cache, cache_path)

        # 自检：载体自身作查询必须排第一，否则特征/图库构造有问题
        self_sims = g[cover_idx] @ g.t()
        self_rank1 = float((self_sims.argmax(dim=1) == torch.tensor(cover_idx)).float().mean())

        res = retrieval_metrics(q, g, cover_idx)
        res['self_check_top1'] = self_rank1
        s = summarize(res, name)
        summaries.append(s)

        per_image[name] = {
            os.path.basename(p): {
                'rank': int(res['ranks'][i]),
                'sim_to_cover': float(res['sim_to_cover'][i]),
                'sim_to_distractor': float(res['sim_to_distractor'][i]),
                'margin': float(res['margin'][i]),
            } for i, p in enumerate(query_covers)
        }
        print(f'[{name}] 自检 top-1={self_rank1 * 100:.1f}%  '
              f'相似度均值={s["clip_sim_mean"]:.4f}  '
              f'top-1 命中={s["top1_hit_rate"] * 100:.1f}%  '
              f'逃逸率={s["evasion_rate"] * 100:.1f}%')

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(per_image, f, ensure_ascii=False, indent=1)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'meta': {'cover_dir': args.cover_dir, 'refined_dir': refined_root,
                            'gallery_size': len(gallery_paths), 'num_queries': len(pairs),
                            'seed': args.seed, 'device': str(device)},
                   'backbones': summaries}, f, ensure_ascii=False, indent=1)

    # ---- 对照表 ----
    print('\n# 跨骨干迁移评估汇总（查询 = 精修图，载体保留在图库内）')
    print(f'# 图库 {len(gallery_paths)} 张，查询 {len(pairs)} 张，seed {args.seed}')
    hdr = (f'{"骨干":<22}{"自检top1":>10}{"相似度均值":>12}{"库内均值":>10}'
           f'{"top-1命中":>11}{"逃逸率":>9}{"中位排名":>10}{"MRR":>8}')
    print(hdr)
    print('-' * len(hdr))
    for s in summaries:
        mark = ' *' if s['backbone'] == 'clip' else ''
        print(f'{s["backbone"] + mark:<22}{s["self_check_top1"] * 100:>9.1f}%'
              f'{s["clip_sim_mean"]:>12.4f}{s["sim_offdiag_mean"]:>10.4f}'
              f'{s["top1_hit_rate"] * 100:>10.1f}%'
              f'{s["evasion_rate"] * 100:>8.1f}%{s["median_rank"]:>10.0f}{s["mrr"]:>8.3f}')
    if any(s['backbone'] == 'clip' for s in summaries):
        print('* 受训空间（论文标定所用），其余为未参与标定的第三方骨干')
    print('注: "库内均值" = 查询与图库中其余图像的平均相似度，用于判断该空间的整体压缩程度；\n'
          '    "相似度均值" 明显高于 "库内均值" 才说明精修图仍然贴着它的载体。')
    print(f'\n逐图明细: {results_path}\n汇总: {summary_path}')


if __name__ == '__main__':
    main()
