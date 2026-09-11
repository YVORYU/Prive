# 模块名: batch_eval.py
# 模块职责: 批量定量评估（论文核心数据表的生产脚本）。
#   对 val 集采样 N 张图像逐一执行: 编码 -> 逐图精修 -> 全指标计算,
#   并附加信道鲁棒性 (JPEG/resize/quant 噪声 k 次采样) 与 uint8 往返
#   (模拟 PNG 保存重载的 8bit 量化) 两类部署条件测试, 基线与精修同口径对比。
#   输出: 逐图记录 results.json (断点续跑)、汇总统计 summary.json、
#   CSV 明细表 (可直接进论文), 以及精修前后均值±标准差的对比摘要。
# 用法示例:
#   python val/batch_eval.py -c other/checkpoint.pth --val-dir ./dataset1/val \
#       --num-images 200 --refine-steps 60 --noise-eval 8 \
#       -o other/batch_eval --device cpu

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
import torch.nn.functional as F
from PIL import Image

from validation_script import ValidationScript

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def list_images(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def load_quiet(vs, path):
    # 与 vs.load_image 相同的预处理但不打印日志
    image = Image.open(path).convert('RGB')
    return vs.transform(image).unsqueeze(0).to(vs.device)


def uint8_roundtrip(x):
    # 内存内模拟 PNG 保存重载的 8bit 量化 (PNG 无损, 差异仅来自量化)
    q = ((x + 1) / 2).clamp(0, 1)
    q = torch.round(q * 255) / 255
    return q * 2 - 1


@torch.no_grad()
def noise_robustness(vs, noiser, encoded, cover, message, feat_cover_clip, k):
    # 信道鲁棒性: k 次随机噪声实现, 统计解码率与信道后 CLIP 相似度
    accs, sims = [], []
    for _ in range(k):
        noised, _ = noiser((encoded, cover))
        pred = (torch.sigmoid(vs.decoder(noised)) > 0.5).float()
        accs.append((pred == message).float().mean().item())
        feat_n = vs.content_recognizer(noised)
        sims.append(F.cosine_similarity(feat_cover_clip, feat_n, dim=1).item())
    return {'bit_acc_mean': float(np.mean(accs)), 'bit_acc_min': float(np.min(accs)),
            'clip_sim_mean': float(np.mean(sims)), 'clip_sim_max': float(np.max(sims))}


@torch.no_grad()
def roundtrip_eval(vs, encoded, cover, message, feat_cover_clip):
    # uint8 往返: 量化 -> 解码 -> 比特准确率与 CLIP 相似度
    reloaded = uint8_roundtrip(encoded)
    pred = (torch.sigmoid(vs.decoder(reloaded)) > 0.5).float()
    bit_acc = (pred == message).float().mean().item()
    feat_re = vs.content_recognizer(reloaded)
    clip_sim = F.cosine_similarity(feat_cover_clip, feat_re, dim=1).item()
    return {'bit_acc': bit_acc, 'clip_sim': clip_sim}


def metric_stats(values):
    # 单指标统计: 均值/标准差/分位数/极值
    a = np.array(values, dtype=float)
    return {'mean': float(a.mean()), 'std': float(a.std()),
            'p5': float(np.percentile(a, 5)), 'p50': float(np.percentile(a, 50)),
            'p95': float(np.percentile(a, 95)),
            'min': float(a.min()), 'max': float(a.max())}


def summarize(recs):
    # 聚合逐图记录: 基线/精修两组全指标统计 + 成功率类指标
    out = {}
    for cond in ['base', 'refined']:
        out[f'{cond}_metrics'] = {}
        for key in ['clip_sim', 'dino_sim', 'bit_acc', 'psnr', 'ssim']:
            values = [r[f'{cond}_metrics'][key] for r in recs if r[f'{cond}_metrics'][key] is not None]
            if values:
                out[f'{cond}_metrics'][key] = metric_stats(values)
        clip = [r[f'{cond}_metrics']['clip_sim'] for r in recs]
        bit = [r[f'{cond}_metrics']['bit_acc'] for r in recs]
        out[f'{cond}_success'] = {
            'bit_acc_100': float(np.mean([b >= 1.0 for b in bit])),
            'clip_below_070': float(np.mean([c < 0.70 for c in clip])),
            'clip_below_060': float(np.mean([c < 0.60 for c in clip])),
        }
        out[f'{cond}_noise'] = {
            'bit_acc_mean': metric_stats([r[f'{cond}_noise']['bit_acc_mean'] for r in recs]),
            'bit_acc_min': metric_stats([r[f'{cond}_noise']['bit_acc_min'] for r in recs]),
            'clip_sim_mean': metric_stats([r[f'{cond}_noise']['clip_sim_mean'] for r in recs]),
        }
        out[f'{cond}_roundtrip'] = {
            'bit_acc': metric_stats([r[f'{cond}_roundtrip']['bit_acc'] for r in recs]),
            'clip_sim': metric_stats([r[f'{cond}_roundtrip']['clip_sim'] for r in recs]),
        }
    return out


def print_summary(summary, num_images):
    print('\n' + '#' * 78)
    print(f'# 批量定量评估汇总 ({num_images} 张)')
    print('#' * 78)
    print(f"{'指标':<22} {'基线(无精修)':>16} {'精修后':>16}")
    print('-' * 78)
    bm, rm = summary['base_metrics'], summary['refined_metrics']
    rows = [
        ('CLIP 相似度', 'clip_sim'), ('DINO 相似度', 'dino_sim'),
        ('比特准确率', 'bit_acc'), ('PSNR (dB)', 'psnr'), ('SSIM', 'ssim'),
    ]
    for label, key in rows:
        if key in bm and key in rm:
            print(f"{label:<22} {bm[key]['mean']:>10.4f}±{bm[key]['std']:.4f} "
                  f"{rm[key]['mean']:>10.4f}±{rm[key]['std']:.4f}")
    print('-' * 78)
    bs, rs = summary['base_success'], summary['refined_success']
    print(f"{'bit_acc=100% 占比':<20} {bs['bit_acc_100']*100:>15.1f}% {rs['bit_acc_100']*100:>15.1f}%")
    print(f"{'clip_sim<0.70 占比':<20} {bs['clip_below_070']*100:>15.1f}% {rs['clip_below_070']*100:>15.1f}%")
    print(f"{'clip_sim<0.60 占比':<20} {bs['clip_below_060']*100:>15.1f}% {rs['clip_below_060']*100:>15.1f}%")
    print('-' * 78)
    bn, rn = summary['base_noise'], summary['refined_noise']
    print(f"{'信道后 bit_acc (均值)':<18} {bn['bit_acc_mean']['mean']*100:>15.1f}% {rn['bit_acc_mean']['mean']*100:>15.1f}%")
    print(f"{'信道后 bit_acc (最差)':<18} {bn['bit_acc_min']['mean']*100:>15.1f}% {rn['bit_acc_min']['mean']*100:>15.1f}%")
    print(f"{'信道后 clip_sim (均值)':<18} {bn['clip_sim_mean']['mean']:>16.4f} {rn['clip_sim_mean']['mean']:>16.4f}")
    br, rr = summary['base_roundtrip'], summary['refined_roundtrip']
    print(f"{'uint8往返 bit_acc':<19} {br['bit_acc']['mean']*100:>15.1f}% {rr['bit_acc']['mean']*100:>15.1f}%")
    print(f"{'uint8往返 clip_sim':<19} {br['clip_sim']['mean']:>16.4f} {rr['clip_sim']['mean']:>16.4f}")
    print('#' * 78)


def write_csv(recs, csv_path):
    # 逐图明细 CSV (可直接导入论文表格或进一步分析)
    fields = ['image']
    for cond in ['base', 'refined']:
        for key in ['clip_sim', 'dino_sim', 'bit_acc', 'psnr', 'ssim']:
            fields.append(f'{cond}_{key}')
        fields += [f'{cond}_noise_bit_mean', f'{cond}_noise_bit_min', f'{cond}_noise_clip_mean',
                   f'{cond}_roundtrip_bit', f'{cond}_roundtrip_clip']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in recs:
            row = {'image': r['image']}
            for cond in ['base', 'refined']:
                for key in ['clip_sim', 'dino_sim', 'bit_acc', 'psnr', 'ssim']:
                    row[f'{cond}_{key}'] = r[f'{cond}_metrics'][key]
                row[f'{cond}_noise_bit_mean'] = r[f'{cond}_noise']['bit_acc_mean']
                row[f'{cond}_noise_bit_min'] = r[f'{cond}_noise']['bit_acc_min']
                row[f'{cond}_noise_clip_mean'] = r[f'{cond}_noise']['clip_sim_mean']
                row[f'{cond}_roundtrip_bit'] = r[f'{cond}_roundtrip']['bit_acc']
                row[f'{cond}_roundtrip_clip'] = r[f'{cond}_roundtrip']['clip_sim']
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='批量定量评估 (编码+精修+信道鲁棒性+uint8往返)')
    parser.add_argument('--checkpoint', '-c', type=str, default='./other/checkpoint.pth')
    parser.add_argument('--val-dir', type=str, default='./dataset1/val')
    parser.add_argument('--num-images', type=int, default=200,
                        help='评估图像数 (从 val 目录随机采样)')
    parser.add_argument('--refine-steps', type=int, default=30,
                        help='逐图精修步数 (0=关闭精修)')
    parser.add_argument('--refine-lr', type=float, default=0.1)
    parser.add_argument('--refine-budget', type=float, default=0.08,
                        help='精修隐私残差预算 (默认 0.08 深挖配方)')
    parser.add_argument('--refine-margin', type=float, default=0.50,
                        help='精修 CLIP escape margin (默认 0.50 深挖配方)')
    parser.add_argument('--refine-chroma-projection', type=int, default=None,
                        help='精修收敛后剔除 prv 低频色度蒙层 (默认取 Config.refine_chroma_projection; '
                             '1 开启 / 0 关闭)。门控: bit_acc 须 100%% 且 clip_sim 达标, 否则回退')
    parser.add_argument('--noise-eval', type=int, default=8,
                        help='信道鲁棒性噪声采样次数')
    parser.add_argument('--save-images', dest='save_images', action='store_true', default=True,
                        help='保存精修后编码图 (默认开启, 每图一张 PNG)')
    parser.add_argument('--no-save-images', dest='save_images', action='store_false',
                        help='不保存精修后编码图')
    parser.add_argument('--report-only', action='store_true',
                        help='不跑评估, 直接从已有 results.json 重新生成可视化报告')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', '-o', type=str, default='./other/batch_eval')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else None
    os.makedirs(args.output_dir, exist_ok=True)

    # 仅重新生成可视化报告（不加载模型、不重跑评估）
    if args.report_only:
        try:
            from eval_visualizer import generate_batch_report
            report_path = generate_batch_report(args.output_dir)
            if report_path:
                print(f"可视化报告: {report_path}")
        except Exception as e:
            print(f"可视化报告生成失败: {e}")
        return
    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    val_paths = list_images(args.val_dir)
    if not val_paths:
        raise FileNotFoundError(f"目录下未找到图像: {args.val_dir}")
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(val_paths))[:min(args.num_images, len(val_paths))]
    eval_paths = [val_paths[i] for i in idx]
    print(f"评估规模: {len(eval_paths)} 张 (来源: {args.val_dir})")

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

        # 编码 + 精修 (report 内含基线/精修两套指标; 基线张量重新编码一次, 单次前向)
        if args.refine_steps > 0:
            encoded_refined, report = vs.refine_encoded_image(
                cover, message, steps=args.refine_steps, lr=args.refine_lr,
                margin_end=args.refine_margin, prv_budget=args.refine_budget,
                verbose=False,
                chroma_projection=(None if args.refine_chroma_projection is None
                                   else bool(args.refine_chroma_projection))
            )
            with torch.no_grad():
                encoded_base, _, _, _ = vs.encoder(cover, message, phase=2, prv_noise_ratio=0.0)
        else:
            with torch.no_grad():
                encoded_base, _, _, _ = vs.encoder(cover, message, phase=2, prv_noise_ratio=0.0)
            encoded_refined = encoded_base
            report = None

        with torch.no_grad():
            feat_cover_clip = vs.content_recognizer(cover)

        # 信道鲁棒性与 uint8 往返 (基线/精修同口径)
        base_noise = noise_robustness(vs, noiser, encoded_base, cover, message,
                                       feat_cover_clip, args.noise_eval)
        ref_noise = noise_robustness(vs, noiser, encoded_refined, cover, message,
                                     feat_cover_clip, args.noise_eval)
        base_rt = roundtrip_eval(vs, encoded_base, cover, message, feat_cover_clip)
        ref_rt = roundtrip_eval(vs, encoded_refined, cover, message, feat_cover_clip)

        # 无精修模式下手动补齐两套指标 (refine 关闭时两者相同)
        if report is not None:
            base_metrics = report['base_metrics']
            refined_metrics = report['refined_metrics']
        else:
            base_metrics = vs._refine_metrics(encoded_base, message, cover,
                                              feat_cover_clip, None)
            refined_metrics = base_metrics

        rec = {
            'image': path,
            'base_metrics': base_metrics,
            'refined_metrics': refined_metrics,
            'base_noise': base_noise, 'refined_noise': ref_noise,
            'base_roundtrip': base_rt, 'refined_roundtrip': ref_rt,
        }
        results[path] = rec
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if args.save_images:
            name = os.path.splitext(os.path.basename(path))[0]
            vs.save_image(encoded_refined, os.path.join(args.output_dir, 'images',
                                                        f'{name}_refined.png'))

        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(eval_paths) - i - 1)
        print(f"[{i + 1}/{len(eval_paths)}] {os.path.basename(path)}: "
              f"clip {base_metrics['clip_sim']:.3f}->{refined_metrics['clip_sim']:.3f} | "
              f"bit {refined_metrics['bit_acc']*100:.0f}% | "
              f"噪声后 bit {ref_noise['bit_acc_mean']*100:.0f}% | "
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
        from eval_visualizer import generate_batch_report
        report_path = generate_batch_report(args.output_dir)
        if report_path:
            print(f"可视化报告: {report_path}")
    except Exception as e:
        print(f"可视化报告生成失败: {e}")


if __name__ == '__main__':
    main()
