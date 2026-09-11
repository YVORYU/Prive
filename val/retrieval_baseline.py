# 模块名: retrieval_baseline.py
# 模块职责: CBIR 检索逃逸基线评估（攻击者视角实弹测试, 网络直出无精修）。
#   模拟基于 CLIP 特征的以图搜图攻击:
#     1. 图库 (gallery): 原始图像特征库（val 集采样 + 可选 train 集扩充干扰池）
#     2. 查询 (query): 编码图像（网络单次前向, 不做逐图精修）
#     3. 检索: 查询特征与全图库做余弦相似度排序, 检查原图是否命中 top-k
#   与 retrieval_eval.py 的区别: 查询图不做逐图精修, 记录为单一条件,
#   速度约为含精修版的 20~30 倍——适合大规模快速评估网络原生的检索逃逸能力,
#   或作为论文 amortization gap 分析的基线列。
#   图库特征缓存 (gallery_feats.pt) 与 retrieval_eval.py 共用: 同图库配置时
#   已提取的特征直接命中, 无需重新计算。
# 输出: 逐查询记录 results.json (断点续跑)、汇总 summary.json、可视化报告 report.html。
# 用法示例:
#   python val/retrieval_baseline.py -c other/checkpoint.pth --val-dir ./dataset1/val \
#       --num-queries 100 --gallery-size 1000 -o other/retrieval_baseline --device cpu

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

from validation_script import ValidationScript
from retrieval_eval import (list_images, load_quiet, random_message,
                            build_gallery_features, retrieve)


def summarize(recs):
    # 聚合单查询记录为检索统计（单一条件）
    ranks = np.array([r['rank'] for r in recs])
    sim_orig = np.array([r['sim_original'] for r in recs])
    sim_d = np.array([r['sim_top1_distractor'] for r in recs])
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
        'self_sanity_top1': float(np.mean([r['self_rank'] == 1 for r in recs])),
    }


def print_summary(stats, num_queries, gallery_size):
    print('\n' + '#' * 70)
    print(f'# CBIR 检索逃逸基线评估汇总 (网络直出无精修, 查询 {num_queries} 张, 图库 {gallery_size} 张)')
    print('#' * 70)
    print(f"{'top-1 命中率':<20} {stats['top1_hit_rate']*100:>9.1f}%")
    print(f"{'top-5 命中率':<20} {stats['top5_hit_rate']*100:>9.1f}%")
    print(f"{'top-10 命中率':<20} {stats['top10_hit_rate']*100:>9.1f}%")
    print(f"{'平均排名':<20} {stats['mean_rank']:>10.1f}")
    print(f"{'MRR':<20} {stats['mrr']:>10.3f}")
    print(f"{'sim(原图)':<20} {stats['sim_original_mean']:>10.4f}")
    print(f"{'sim(最强干扰)':<20} {stats['sim_top1_distractor_mean']:>10.4f}")
    print(f"\n检索逃逸率 (top-1 未命中): {(1-stats['top1_hit_rate'])*100:.1f}%")
    print(f"自检 (原图自查询 top-1 命中率): {stats['self_sanity_top1']*100:.1f}% (应为 100%)")


def main():
    parser = argparse.ArgumentParser(description='CBIR 检索逃逸基线评估 (网络直出, 无逐图精修)')
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report-only', action='store_true',
                        help='不跑评估, 直接从已有 results.json 重新生成可视化报告')
    parser.add_argument('--output-dir', '-o', type=str, default='./other/retrieval_baseline')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else None
    os.makedirs(args.output_dir, exist_ok=True)

    # 仅重新生成可视化报告（不加载模型、不重跑评估）
    if args.report_only:
        try:
            from eval_visualizer import generate_retrieval_baseline_report
            report_path = generate_retrieval_baseline_report(args.output_dir)
            if report_path:
                print(f"可视化报告: {report_path}")
        except Exception as e:
            print(f"可视化报告生成失败: {e}")
        return

    # ===== 图库与查询集构造 (与 retrieval_eval.py 相同的采样逻辑) =====
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
          f"(随机 top-1 命中概率 {100.0 / len(gallery_paths):.2f}%, 网络直出无精修)")

    # 保存评估元信息（图库规模等, 供可视化报告读取）
    with open(os.path.join(args.output_dir, 'eval_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'gallery_size': len(gallery_paths), 'num_queries': len(query_paths),
                   'refine_steps': 0}, f, ensure_ascii=False, indent=2)

    # ===== 模型与图库特征 (缓存文件与 retrieval_eval.py 共用, 同图库直接命中) =====
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

        # 网络单次前向编码——不做任何精修
        with torch.no_grad():
            encoded, _, _, _ = vs.encoder(cover, msg, phase=2, prv_noise_ratio=0.0)
            # 两种查询特征: 原图自查询 (自检)、编码图查询 (被测条件)
            feat_self = vs.content_recognizer(cover)
            feat_enc = vs.content_recognizer(encoded)

        self_rank, _, _, _ = retrieve(feat_self, gallery_feats, original_idx)
        enc_rank, enc_sim_o, enc_sim_d, enc_top1 = retrieve(feat_enc, gallery_feats, original_idx)

        rec = {
            'image': qpath,
            'rank': enc_rank,
            'sim_original': enc_sim_o,
            'sim_top1_distractor': enc_sim_d,
            'top1_path': gallery_paths[enc_top1],
            'self_rank': self_rank,
        }
        results[qpath] = rec

        # 增量保存 (中断不丢进度)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        name = os.path.splitext(os.path.basename(qpath))[0]
        elapsed = time.time() - t_start
        print(f"[{qi + 1}/{len(query_paths)}] {name}: "
              f"rank {enc_rank} (sim {enc_sim_o:.3f}) | 已用时 {elapsed:.0f}s")

    # ===== 汇总 =====
    recs = [results[p] for p in query_paths if p in results]
    stats = summarize(recs)
    print_summary(stats, len(recs), len(gallery_paths))
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.output_dir}")

    # 生成可视化报告（matplotlib 缺失或失败时跳过，不影响评估结果）
    try:
        from eval_visualizer import generate_retrieval_baseline_report
        report_path = generate_retrieval_baseline_report(args.output_dir)
        if report_path:
            print(f"可视化报告: {report_path}")
    except Exception as e:
        print(f"可视化报告生成失败: {e}")


if __name__ == '__main__':
    main()
