# 模块名: retrieval_eval.py
# 模块职责: CBIR 检索逃逸评估（攻击者视角的实弹测试）。
#   模拟基于 CLIP 特征的以图搜图攻击：
#     1. 图库 (gallery): 原始图像特征库（val 集采样 + 可选 train 集扩充干扰池）
#     2. 查询 (query): 编码图像（基线）与精修图像分别作为查询
#     3. 检索: 查询特征与全图库做余弦相似度排序，检查原图是否命中 top-k
#   报告指标: top-1/top-5/top-10 命中率（越低保护越好）、检索逃逸率 (1-top1)、
#   平均排名、MRR、以及"原图相似度 vs 最强干扰相似度"的竞争余量分析。
#   含自检 (self sanity): 原图自身作查询必须 top-1 命中，用于校验特征提取链路。
#   支持断点续跑: results.json 按图像路径增量保存，重跑自动跳过已完成图像。
# 用法示例:
#   python val/retrieval_eval.py -c other/checkpoint.pth --val-dir ./dataset1/val \
#       --num-queries 100 --gallery-size 1000 --refine-steps 60 \
#       -o other/retrieval_eval --device cpu

import argparse
import json
import os
import sys
import time

# 本脚本位于 val/ 子目录：定位项目根并加入 sys.path（import models/losses/config 等根模块），
# 同时把工作目录切到项目根，保证 ./models/dinov2_vits14_pretrain.pth 等相对路径
# 与从项目根直接运行时的行为完全一致
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import torch
from PIL import Image

from validation_script import ValidationScript

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def list_images(root):
    # 递归收集目录下所有支持的图像文件并排序（保证跨运行一致性）
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def load_quiet(vs, path):
    # 与 vs.load_image 相同的预处理但不打印日志（批量评估时减少输出噪声）
    image = Image.open(path).convert('RGB')
    return vs.transform(image).unsqueeze(0).to(vs.device)


def random_message(vs, rng):
    # 按独立随机源生成单条消息（检索测试不依赖具体消息内容）
    bits = rng.randint(0, 2, size=vs.cfg.message_length)
    return torch.tensor(bits, dtype=torch.float32).unsqueeze(0).to(vs.device)


@torch.no_grad()
def build_gallery_features(vs, paths, cache_path):
    # 提取图库全部图像的 CLIP 特征；结果缓存到磁盘，路径列表变化时自动失效
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        if cached.get('paths') == paths:
            print(f"图库特征缓存命中: {cache_path} ({len(paths)} 张)")
            return cached['feats'].to(vs.device)
        print("图库特征缓存与当前图库不一致，重新提取...")
    feats = []
    t0 = time.time()
    for i, p in enumerate(paths):
        feats.append(vs.content_recognizer(load_quiet(vs, p)))
        if (i + 1) % 200 == 0:
            print(f"  图库特征提取: {i + 1}/{len(paths)} ({time.time() - t0:.0f}s)")
    feats = torch.cat(feats, dim=0).cpu()
    torch.save({'paths': paths, 'feats': feats}, cache_path)
    print(f"图库特征提取完成: {len(paths)} 张, 用时 {time.time() - t0:.0f}s")
    return feats.to(vs.device)


def retrieve(query_feat, gallery_feats, original_idx):
    # 单次检索: 返回原图排名、与原图相似度、最强干扰相似度及其索引
    # 特征均已 L2 归一化，矩阵乘即余弦相似度
    sims = (gallery_feats @ query_feat.view(-1)).cpu().numpy()
    order = np.argsort(-sims)
    rank = int(np.where(order == original_idx)[0][0]) + 1
    distractor_mask = np.ones(len(sims), dtype=bool)
    distractor_mask[original_idx] = False
    d_sims = np.where(distractor_mask, sims, -np.inf)
    top1_d_idx = int(np.argmax(d_sims))
    return rank, float(sims[original_idx]), float(d_sims[top1_d_idx]), top1_d_idx


def summarize(recs):
    # 聚合单图记录为检索统计
    def cond_stats(cond):
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
    stats = {'base': cond_stats('base'), 'refined': cond_stats('refined')}
    stats['self_sanity_top1'] = float(np.mean([r['self_rank'] == 1 for r in recs]))
    return stats


def print_summary(stats, num_queries, gallery_size):
    print('\n' + '#' * 70)
    print(f'# CBIR 检索逃逸评估汇总 (查询 {num_queries} 张, 图库 {gallery_size} 张)')
    print('#' * 70)
    header = f"{'条件':<14} {'top-1命中':>9} {'top-5命中':>9} {'top-10命中':>10} {'平均排名':>8} {'MRR':>7} {'sim(原图)':>10} {'sim(最强干扰)':>13}"
    print(header)
    for cond, label in [('base', '基线(无精修)'), ('refined', '精修后')]:
        s = stats[cond]
        print(f"{label:<14} {s['top1_hit_rate']*100:>8.1f}% {s['top5_hit_rate']*100:>8.1f}% "
              f"{s['top10_hit_rate']*100:>9.1f}% {s['mean_rank']:>8.1f} {s['mrr']:>7.3f} "
              f"{s['sim_original_mean']:>10.4f} {s['sim_top1_distractor_mean']:>13.4f}")
    print(f"\n检索逃逸率 (top-1 未命中): 基线 {(1-stats['base']['top1_hit_rate'])*100:.1f}% "
          f"-> 精修后 {(1-stats['refined']['top1_hit_rate'])*100:.1f}%")
    print(f"自检 (原图自查询 top-1 命中率): {stats['self_sanity_top1']*100:.1f}% (应为 100%)")


def main():
    parser = argparse.ArgumentParser(description='CBIR 检索逃逸评估 (以图搜图攻击测试)')
    parser.add_argument('--checkpoint', '-c', type=str, default='./other/checkpoint.pth')
    parser.add_argument('--val-dir', type=str, default='./dataset1/val',
                        help='查询与图库图像来源目录')
    parser.add_argument('--extra-gallery-dir', type=str, default=None,
                        help='可选的图库扩充目录 (如 train 集, 增大干扰池提升任务难度)')
    parser.add_argument('--extra-gallery-size', type=int, default=0,
                        help='从扩充目录采样的图像数 (0=不扩充)')
    parser.add_argument('--num-queries', type=int, default=100,
                        help='查询图像数 (从图库中采样, 原图保留在图库内)')
    parser.add_argument('--gallery-size', type=int, default=1000,
                        help='主图库大小 (val 目录采样)')
    parser.add_argument('--refine-steps', type=int, default=None,
                        help='查询编码后的逐图精修步数 (0=关闭精修; None=取 Config.refine_steps, 默认 150)')
    parser.add_argument('--refine-lr', type=float, default=0.1)
    parser.add_argument('--refine-budget', type=float, default=None,
                        help='精修隐私残差预算 (None=取 Config.prv_budget, 默认 0.08)')
    parser.add_argument('--refine-margin', type=float, default=None,
                        help='精修 CLIP escape margin (None=取 Config.refine_margin_end, 默认 0.60)')
    parser.add_argument('--refine-chroma-projection', type=int, default=None,
                        help='精修收敛后剔除 prv 低频色度蒙层 (默认取 Config.refine_chroma_projection; '
                             '1 开启 / 0 关闭)。门控: bit_acc 须 100%% 且 clip_sim 达标, 否则回退')
    parser.add_argument('--refine-w-de', type=float, default=None,
                        help='精修感知色差锚权重 (None=取 Config.refine_w_de, 默认 200; 0=关闭)。'
                             '对 ΔE>tau 的像素二次惩罚, 压制 PSNR/SSIM 测不到的色罩/蒙层类失真')
    parser.add_argument('--refine-de-tau', type=float, default=None,
                        help='感知色差锚阈值 de_tau (None=取 Config.refine_de_tau, 默认 4.0)')
    parser.add_argument('--refine-w-message', type=float, default=None,
                        help='精修消息锚权重 (None=取 Config.refine_w_message, 默认 20)')
    parser.add_argument('--refine-w-privacy', type=float, default=None,
                        help='精修隐私(逃逸)锚权重 (None=取 Config.refine_w_privacy, 默认 12)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report-only', action='store_true',
                        help='不跑评估, 直接从已有 results.json 重新生成可视化报告')
    parser.add_argument('--output-dir', '-o', type=str, default='./other/retrieval_eval')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else None
    os.makedirs(args.output_dir, exist_ok=True)

    # 仅重新生成可视化报告（不加载模型、不重跑评估）
    if args.report_only:
        try:
            from eval_visualizer import generate_retrieval_report
            report_path = generate_retrieval_report(args.output_dir)
            if report_path:
                print(f"可视化报告: {report_path}")
        except Exception as e:
            print(f"可视化报告生成失败: {e}")
        return

    # ===== 图库与查询集构造 =====
    val_paths = list_images(args.val_dir)
    if not val_paths:
        raise FileNotFoundError(f"目录下未找到图像: {args.val_dir}")
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(val_paths))

    gallery_size = min(args.gallery_size, len(val_paths))
    if args.num_queries > gallery_size:
        raise ValueError(f"--num-queries ({args.num_queries}) 不能超过 --gallery-size ({gallery_size})")

    # 查询集是图库的子集 (原图必须在图库内, 检索才有"命中"可言)
    gallery_paths = [val_paths[i] for i in perm[:gallery_size]]
    query_paths = [val_paths[i] for i in perm[:args.num_queries]]

    if args.extra_gallery_dir and args.extra_gallery_size > 0:
        extra_paths = list_images(args.extra_gallery_dir)
        extra_idx = rng.choice(len(extra_paths), size=min(args.extra_gallery_size, len(extra_paths)),
                               replace=False)
        gallery_paths += [extra_paths[i] for i in extra_idx]
        print(f"图库扩充: 从 {args.extra_gallery_dir} 追加 {len(extra_idx)} 张干扰图")

    path_to_idx = {p: i for i, p in enumerate(gallery_paths)}
    print(f"评估规模: 查询 {len(query_paths)} 张, 图库 {len(gallery_paths)} 张 "
          f"(随机 top-1 命中概率 {100.0 / len(gallery_paths):.2f}%)")

    # 保存评估元信息（图库规模等, 供可视化报告读取）
    with open(os.path.join(args.output_dir, 'eval_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'gallery_size': len(gallery_paths), 'num_queries': len(query_paths),
                   'refine_steps': args.refine_steps,
                   'refine_w_de': args.refine_w_de, 'refine_de_tau': args.refine_de_tau},
                  f, ensure_ascii=False, indent=2)

    # ===== 模型与图库特征 =====
    vs = ValidationScript(checkpoint_path=args.checkpoint, device=device)
    cache_path = os.path.join(args.output_dir, 'gallery_feats.pt')
    gallery_feats = build_gallery_features(vs, gallery_paths, cache_path)

    # ===== 逐查询处理 (断点续跑) =====
    results_path = os.path.join(args.output_dir, 'results.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"断点续跑: 已有 {len(results)} 条记录")

    t_start = time.time()
    for qi, qpath in enumerate(query_paths):
        if qpath in results:
            continue
        cover = load_quiet(vs, qpath)
        original_idx = path_to_idx[qpath]
        msg = random_message(vs, np.random.RandomState(args.seed * 100000 + qi))

        # 基线编码 (无精修)
        with torch.no_grad():
            encoded_base, _, _, _ = vs.encoder(cover, msg, phase=2, prv_noise_ratio=0.0)

        # 逐图精修
        # refine_steps None = 取 Config.refine_steps (由 refine_encoded_image 内部解析)
        if args.refine_steps is None or args.refine_steps > 0:
            encoded_refined, report = vs.refine_encoded_image(
                cover, msg, steps=args.refine_steps, lr=args.refine_lr,
                margin_end=args.refine_margin, prv_budget=args.refine_budget,
                verbose=False,
                chroma_projection=(None if args.refine_chroma_projection is None
                                   else bool(args.refine_chroma_projection)),
                w_de=args.refine_w_de, de_tau=args.refine_de_tau,
                w_message=args.refine_w_message, w_privacy=args.refine_w_privacy
            )
        else:
            encoded_refined, report = encoded_base, None

        # 三种查询特征: 原图自查询 (自检)、基线编码、精修编码
        with torch.no_grad():
            feat_self = vs.content_recognizer(cover)
            feat_base = vs.content_recognizer(encoded_base)
            feat_ref = vs.content_recognizer(encoded_refined)

        self_rank, _, _, _ = retrieve(feat_self, gallery_feats, original_idx)
        base_rank, base_sim_o, base_sim_d, base_top1 = retrieve(feat_base, gallery_feats, original_idx)
        ref_rank, ref_sim_o, ref_sim_d, ref_top1 = retrieve(feat_ref, gallery_feats, original_idx)

        rec = {
            'image': qpath,
            'base': {'rank': base_rank, 'sim_original': base_sim_o,
                     'sim_top1_distractor': base_sim_d,
                     'top1_path': gallery_paths[base_top1]},
            'refined': {'rank': ref_rank, 'sim_original': ref_sim_o,
                        'sim_top1_distractor': ref_sim_d,
                        'top1_path': gallery_paths[ref_top1]},
            'self_rank': self_rank,
        }
        if report is not None:
            rec['base_metrics'] = report['base_metrics']
            rec['refined_metrics'] = report['refined_metrics']
        results[qpath] = rec

        # 增量保存 (中断不丢进度)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        name = os.path.splitext(os.path.basename(qpath))[0]
        elapsed = time.time() - t_start
        print(f"[{qi + 1}/{len(query_paths)}] {name}: "
              f"基线 rank {base_rank} (sim {base_sim_o:.3f}) -> "
              f"精修 rank {ref_rank} (sim {ref_sim_o:.3f}) | 已用时 {elapsed:.0f}s")

    # ===== 汇总 =====
    recs = [results[p] for p in query_paths if p in results]
    stats = summarize(recs)
    print_summary(stats, len(recs), len(gallery_paths))
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.output_dir}")

    # 生成可视化报告（matplotlib 缺失或失败时跳过，不影响评估结果）
    try:
        from eval_visualizer import generate_retrieval_report
        report_path = generate_retrieval_report(args.output_dir)
        if report_path:
            print(f"可视化报告: {report_path}")
    except Exception as e:
        print(f"可视化报告生成失败: {e}")


if __name__ == '__main__':
    main()
