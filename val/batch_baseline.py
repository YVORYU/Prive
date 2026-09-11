# 模块名: batch_baseline.py
# 模块职责: 批量基线评估（网络单次前向直出，无逐图精修）。
#   对 val 集采样 N 张图像逐一执行: 编码(单次前向) -> 全指标计算,
#   并附加信道鲁棒性 (JPEG/resize/quant 噪声 k 次采样) 与 uint8 往返
#   (模拟 PNG 保存重载的 8bit 量化) 两类部署条件测试。
#   与 batch_eval.py 的区别: 跳过逐图精修, 记录为单一条件 (无 base/refined
#   冗余结构), 速度约为含精修版的 20~30 倍——适合大规模快速评估网络
#   checkpoint 的原生质量, 或作为论文 amortization gap 分析的基线列。
#   复用 batch_eval.py 的采样/噪声/往返等公共函数, 保证与含精修版口径一致。
# 输出: 逐图记录 results.json (断点续跑)、汇总统计 summary.json、
#   CSV 明细表 (可直接进论文)、可视化报告 report.html。
# 用法示例:
#   python val/batch_baseline.py -c other/checkpoint.pth --val-dir ./dataset1/val \
#       --num-images 200 --noise-eval 8 -o other/batch_baseline --device cpu

import argparse
import csv
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
from batch_eval import list_images, load_quiet, noise_robustness, roundtrip_eval, metric_stats


def summarize(recs):
    # 聚合逐图记录: 单一条件全指标统计 + 成功率类指标
    out = {'metrics': {}}
    for key in ['clip_sim', 'dino_sim', 'bit_acc', 'psnr', 'ssim']:
        values = [r['metrics'][key] for r in recs if r['metrics'].get(key) is not None]
        if values:
            out['metrics'][key] = metric_stats(values)
    clip = [r['metrics']['clip_sim'] for r in recs]
    bit = [r['metrics']['bit_acc'] for r in recs]
    out['success'] = {
        'bit_acc_100': float(np.mean([b >= 1.0 for b in bit])),
        'clip_below_070': float(np.mean([c < 0.70 for c in clip])),
        'clip_below_060': float(np.mean([c < 0.60 for c in clip])),
    }
    out['noise'] = {
        'bit_acc_mean': metric_stats([r['noise']['bit_acc_mean'] for r in recs]),
        'bit_acc_min': metric_stats([r['noise']['bit_acc_min'] for r in recs]),
        'clip_sim_mean': metric_stats([r['noise']['clip_sim_mean'] for r in recs]),
    }
    out['roundtrip'] = {
        'bit_acc': metric_stats([r['roundtrip']['bit_acc'] for r in recs]),
        'clip_sim': metric_stats([r['roundtrip']['clip_sim'] for r in recs]),
    }
    return out


def print_summary(summary, num_images):
    print('\n' + '#' * 60)
    print(f'# 批量基线评估汇总 (网络直出, 无精修, {num_images} 张)')
    print('#' * 60)
    m = summary['metrics']
    rows = [
        ('CLIP 相似度', 'clip_sim'), ('DINO 相似度', 'dino_sim'),
        ('比特准确率', 'bit_acc'), ('PSNR (dB)', 'psnr'), ('SSIM', 'ssim'),
    ]
    for label, key in rows:
        if key in m:
            print(f"{label:<22} {m[key]['mean']:>12.4f}±{m[key]['std']:.4f}")
    print('-' * 60)
    s = summary['success']
    print(f"{'bit_acc=100% 占比':<22} {s['bit_acc_100']*100:>11.1f}%")
    print(f"{'clip_sim<0.70 占比':<22} {s['clip_below_070']*100:>11.1f}%")
    print(f"{'clip_sim<0.60 占比':<22} {s['clip_below_060']*100:>11.1f}%")
    print('-' * 60)
    n = summary['noise']
    print(f"{'信道后 bit_acc (均值)':<22} {n['bit_acc_mean']['mean']*100:>11.1f}%")
    print(f"{'信道后 bit_acc (最差)':<22} {n['bit_acc_min']['mean']*100:>11.1f}%")
    print(f"{'信道后 clip_sim (均值)':<22} {n['clip_sim_mean']['mean']:>12.4f}")
    rt = summary['roundtrip']
    print(f"{'uint8往返 bit_acc':<22} {rt['bit_acc']['mean']*100:>11.1f}%")
    print(f"{'uint8往返 clip_sim':<22} {rt['clip_sim']['mean']:>12.4f}")
    print('#' * 60)


def write_csv(recs, csv_path):
    # 逐图明细 CSV (可直接导入论文表格或进一步分析)
    fields = ['image', 'clip_sim', 'dino_sim', 'bit_acc', 'psnr', 'ssim',
              'noise_bit_mean', 'noise_bit_min', 'noise_clip_mean',
              'roundtrip_bit', 'roundtrip_clip']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in recs:
            writer.writerow({
                'image': r['image'],
                'clip_sim': r['metrics']['clip_sim'],
                'dino_sim': r['metrics'].get('dino_sim'),
                'bit_acc': r['metrics']['bit_acc'],
                'psnr': r['metrics']['psnr'],
                'ssim': r['metrics']['ssim'],
                'noise_bit_mean': r['noise']['bit_acc_mean'],
                'noise_bit_min': r['noise']['bit_acc_min'],
                'noise_clip_mean': r['noise']['clip_sim_mean'],
                'roundtrip_bit': r['roundtrip']['bit_acc'],
                'roundtrip_clip': r['roundtrip']['clip_sim'],
            })


def main():
    parser = argparse.ArgumentParser(description='批量基线评估 (网络直出, 无逐图精修)')
    parser.add_argument('--checkpoint', '-c', type=str, default='./other/checkpoint.pth')
    parser.add_argument('--val-dir', type=str, default='./dataset1/val')
    parser.add_argument('--num-images', type=int, default=200,
                        help='评估图像数 (从 val 目录随机采样)')
    parser.add_argument('--noise-eval', type=int, default=8,
                        help='信道鲁棒性噪声采样次数')
    parser.add_argument('--save-images', dest='save_images', action='store_true', default=False,
                        help='保存编码图 (每图一张 PNG)')
    parser.add_argument('--no-save-images', dest='save_images', action='store_false')
    parser.add_argument('--report-only', action='store_true',
                        help='不跑评估, 直接从已有 results.json 重新生成可视化报告')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', '-o', type=str, default='./other/batch_baseline')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else None
    os.makedirs(args.output_dir, exist_ok=True)

    # 仅重新生成可视化报告（不加载模型、不重跑评估）
    if args.report_only:
        try:
            from eval_visualizer import generate_batch_baseline_report
            report_path = generate_batch_baseline_report(args.output_dir)
            if report_path:
                print(f"可视化报告: {report_path}")
        except Exception as e:
            print(f"可视化报告生成失败: {e}")
        return
    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    # 与 batch_eval.py 相同的采样逻辑 (同 seed 同目录时评估同一批图像, 便于对照)
    val_paths = list_images(args.val_dir)
    if not val_paths:
        raise FileNotFoundError(f"目录下未找到图像: {args.val_dir}")
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(val_paths))[:min(args.num_images, len(val_paths))]
    eval_paths = [val_paths[i] for i in idx]
    print(f"评估规模: {len(eval_paths)} 张 (来源: {args.val_dir}, 网络直出无精修)")

    vs = ValidationScript(checkpoint_path=args.checkpoint, device=device)
    noiser = vs._build_noiser()

    results_path = os.path.join(args.output_dir, 'results.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"断点续跑: 已有 {len(results)} 条记录")

    t_start = time.time()
    for i, path in enumerate(eval_paths):
        if path in results:
            continue
        cover = load_quiet(vs, path)
        msg_rng = np.random.RandomState(args.seed * 100000 + i)
        bits = msg_rng.randint(0, 2, size=vs.cfg.message_length)
        message = torch.tensor(bits, dtype=torch.float32).unsqueeze(0).to(vs.device)

        # 网络单次前向编码——不做任何精修
        with torch.no_grad():
            encoded, _, _, _ = vs.encoder(cover, message, phase=2, prv_noise_ratio=0.0)
            feat_cover_clip = vs.content_recognizer(cover)
            feat_cover_dino = vs._extract_dino_features_grad(cover)

        metrics = vs._refine_metrics(encoded, message, cover,
                                     feat_cover_clip, feat_cover_dino)
        noise = noise_robustness(vs, noiser, encoded, cover, message,
                                 feat_cover_clip, args.noise_eval)
        rt = roundtrip_eval(vs, encoded, cover, message, feat_cover_clip)

        rec = {'image': path, 'metrics': metrics, 'noise': noise, 'roundtrip': rt}
        results[path] = rec
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if args.save_images:
            name = os.path.splitext(os.path.basename(path))[0]
            vs.save_image(encoded, os.path.join(args.output_dir, 'images', f'{name}.png'))

        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(eval_paths) - i - 1)
        print(f"[{i + 1}/{len(eval_paths)}] {os.path.basename(path)}: "
              f"clip {metrics['clip_sim']:.3f} | bit {metrics['bit_acc']*100:.0f}% | "
              f"噪声后 bit {noise['bit_acc_mean']*100:.0f}% | "
              f"已用 {elapsed:.0f}s, 预计剩余 {eta:.0f}s")

    # ===== 汇总 =====
    recs = [results[p] for p in eval_paths if p in results]
    summary = summarize(recs)
    print_summary(summary, len(recs))
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(recs, os.path.join(args.output_dir, 'details.csv'))
    print(f"\n结果已保存: {args.output_dir} (results.json / summary.json / details.csv)")

    # 生成可视化报告（matplotlib 缺失或失败时跳过，不影响评估结果）
    try:
        from eval_visualizer import generate_batch_baseline_report
        report_path = generate_batch_baseline_report(args.output_dir)
        if report_path:
            print(f"可视化报告: {report_path}")
    except Exception as e:
        print(f"可视化报告生成失败: {e}")


if __name__ == '__main__':
    main()
