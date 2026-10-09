# 模块名: run_full_validation.py
# 模块职责: 全链路验证编排脚本（论文实验一键入口）。
#   在同一 checkpoint / 同一测试集 / 同一 seed 上按顺序编排 val/ 下的评估脚本:
#     1. smoke              冒烟自检 (3 张图 + 3 查询/20 图库 + 1 张精修, 校验链路后再跑全量)
#     2. single             单图 full 全链路 (编码->精修->解码->判别->相似度 + report.html)
#     3. batch_baseline     批量基线 (网络直出, 快)
#     4. retrieval_baseline 检索基线 (网络直出, 快; 顺带构建图库特征缓存)
#     5. batch_eval         批量精修评估 (慢; 与 3 同 seed 同图像集, 直接对照)
#     6. retrieval_eval     检索精修评估 (慢; 复用 4 的图库特征缓存)
#     7. external           外部系统对比实验 (TrustMark, 不需要 PrivSteg checkpoint):
#                           外部冒烟(2张) -> 批量直出+事后PGD(同预算/放开) -> CBIR检索,
#                           图库特征缓存与 PrivSteg 检索阶段共用 (同图库配置直接命中)
#     8. summary           汇总五个 summary.json -> full_summary.json / full_summary.md
#                          + 跨系统对比段 (PrivSteg 直出/精修 vs TrustMark 直出/PGD)
#                          + 跨系统对比报告 cross_report.html + PNG 面板 (英文图表)
#   设计要点:
#     - 快阶段先行 (失败早暴露), 慢的精修阶段放最后; 各子脚本自带断点续跑
#       (results.json 增量保存), 任一阶段中断后重新执行同一条命令即可续跑。
#     - 结束时输出网络直出 / 精修后的对照表 (控制台中文, full_summary.md 英文可直接进论文),
#       并做两路直出结果的一致性交叉检查 (seed/val-dir 配错时给出警告)。
#     - external 阶段只依赖 trustmark 包与测试集, --stages external 可在无 checkpoint
#       时单独跑对比实验; 与 PrivSteg 阶段同 seed 同 val-dir 时评估同一批图像。
# 用法示例 (云服务器项目根):
#   python val/run_full_validation.py -c ./experiments/full_run/checkpoints/checkpoint_best.pth \
#       --val-dir ./dataset/test --output-root ./other/full_validation
#   只跑外部系统对比 (无需 checkpoint):
#   python val/run_full_validation.py --stages external --val-dir ./dataset/test \
#       --output-root ./other/full_validation

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

# 本脚本位于 val/ 子目录：定位项目根并加入 sys.path，同时把工作目录切到项目根，
# 保证子脚本与相对路径 (checkpoint / 数据集 / 输出目录) 的行为与从项目根运行完全一致
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

_VAL_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}
STAGE_ORDER = ['smoke', 'single', 'batch_baseline', 'retrieval_baseline',
               'batch_eval', 'retrieval_eval', 'external', 'summary']
# PrivSteg 自有评估阶段 (需要 checkpoint); external/summary 阶段不需要
_PRIV_STAGES = {'smoke', 'single', 'batch_baseline', 'retrieval_baseline',
                'batch_eval', 'retrieval_eval'}
BANNER = '=' * 78

_DRY_RUN = False


def find_external_dir(out_root):
    # 定位输出根下的外部系统对比目录 (external_<system>/, 含 summary.json 才有效)
    if not os.path.isdir(out_root):
        return None
    for d in sorted(os.listdir(out_root)):
        if d.startswith('external_') and \
                os.path.isfile(os.path.join(out_root, d, 'summary.json')):
            return os.path.join(out_root, d)
    return None


def list_images(root):
    # 递归收集目录下所有支持的图像文件并排序 (与各评估脚本口径一致)
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def fmt_cmd(cmd):
    # 仅供打印: 含空格/括号等 shell 特殊字符的参数加引号, 方便复制到终端单独重跑
    # (实际执行走 subprocess 参数列表, 不经过 shell, 不受此影响)
    special = ' \t()[]{}$&|;<>"\''
    return ' '.join(f'"{c}"' if any(ch in special for ch in str(c)) else str(c) for c in cmd)


def run_stage(name, cmd):
    print(f"\n{BANNER}\n# 阶段: {name}\n# 命令: {fmt_cmd(cmd)}\n{BANNER}", flush=True)
    if _DRY_RUN:
        print('[dry-run] 跳过执行')
        return 0.0
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    dt = time.time() - t0
    if ret.returncode != 0:
        raise SystemExit(
            f"\n[中止] 阶段 {name} 失败 (退出码 {ret.returncode}), 用时 {dt:.0f}s。\n"
            f"排查修复后重新执行同一条编排命令即可断点续跑 (已完成的图像/查询自动跳过)。")
    print(f"[完成] 阶段 {name} 用时 {dt:.0f}s")
    return dt


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _device_args(a):
    return ['--device', a.device] if a.device else []


def _extra_gallery_args(a):
    if not a.extra_gallery_dir:
        return []
    return ['--extra-gallery-dir', a.extra_gallery_dir,
            '--extra-gallery-size', str(a.extra_gallery_size)]


def _chroma_args(a):
    # 色度蒙层投影开关: None = 取子脚本 Config 默认, 不传参; 0/1 显式覆盖
    if a.refine_chroma_projection is None:
        return []
    return ['--refine-chroma-projection', str(a.refine_chroma_projection)]


def _wde_args(a):
    # 感知色差锚参数转发 (None 安全): None = 取 Config.refine_w_de/refine_de_tau 默认, 不传参
    out = []
    if a.refine_w_de is not None:
        out += ['--refine-w-de', str(a.refine_w_de)]
    if a.refine_de_tau is not None:
        out += ['--refine-de-tau', str(a.refine_de_tau)]
    return out


def _refine_args(a):
    # 精修配方参数转发 (None 安全): None = 取 Config.refine_* 默认, 不传参;
    # 非 None 逐项显式覆盖子脚本默认
    out = []
    if a.refine_steps is not None:
        out += ['--refine-steps', str(a.refine_steps)]
    if a.refine_lr is not None:
        out += ['--refine-lr', str(a.refine_lr)]
    if a.refine_budget is not None:
        out += ['--refine-budget', str(a.refine_budget)]
    if a.refine_margin is not None:
        out += ['--refine-margin', str(a.refine_margin)]
    if a.refine_w_message is not None:
        out += ['--refine-w-message', str(a.refine_w_message)]
    if a.refine_w_privacy is not None:
        out += ['--refine-w-privacy', str(a.refine_w_privacy)]
    return out


def _effective_refine_cfg(a):
    # 显示/汇总用: 把 None 解析为 Config.refine_* 的实际生效值
    # (子脚本对 None 的解析与本处一致, 保证汇总记录的就是真实配方)
    from config import Config
    return {
        'steps': a.refine_steps if a.refine_steps is not None
                 else getattr(Config, 'refine_steps', 30),
        'lr': a.refine_lr if a.refine_lr is not None else 0.1,
        'budget': a.refine_budget if a.refine_budget is not None else Config.prv_budget,
        'margin': a.refine_margin if a.refine_margin is not None
                  else getattr(Config, 'refine_margin_end', Config.privacy_margin_end),
        'w_de': a.refine_w_de if a.refine_w_de is not None
                else getattr(Config, 'refine_w_de', 0.0),
        'de_tau': a.refine_de_tau if a.refine_de_tau is not None
                  else getattr(Config, 'refine_de_tau', 2.0),
        'w_message': a.refine_w_message if a.refine_w_message is not None
                     else getattr(Config, 'refine_w_message', 10.0),
        'w_privacy': a.refine_w_privacy if a.refine_w_privacy is not None
                     else getattr(Config, 'refine_w_privacy', 12.0),
        'chroma_projection': a.refine_chroma_projection,
    }


# ============================ 各阶段 ============================

def _reset_smoke_dir(out_root):
    # 冒烟产物每次全量重跑 (量小): 避免旧 results.json 掩盖新 checkpoint 的问题。
    # 仅清理编排脚本自有的 _smoke 临时目录, 且要求它严格位于输出根之内。
    smoke_root = os.path.join(out_root, '_smoke')
    if os.path.isdir(smoke_root):
        ap, ar = os.path.abspath(smoke_root), os.path.abspath(out_root)
        if ap.startswith(ar + os.sep) and os.path.basename(ap) == '_smoke':
            shutil.rmtree(smoke_root)
    return smoke_root


def run_smoke(a):
    smoke_root = _reset_smoke_dir(a.output_root)
    run_stage('smoke/batch_baseline (3 张, 网络直出)',
              [sys.executable, os.path.join(_VAL_DIR, 'batch_baseline.py'),
               '-c', a.checkpoint, '--val-dir', a.val_dir,
               '--num-images', '3', '--noise-eval', '2', '--seed', str(a.seed),
               '-o', os.path.join(smoke_root, 'batch_baseline')] + _device_args(a))
    run_stage('smoke/retrieval_baseline (3 查询 / 20 图库, 网络直出)',
              [sys.executable, os.path.join(_VAL_DIR, 'retrieval_baseline.py'),
               '-c', a.checkpoint, '--val-dir', a.val_dir,
               '--num-queries', '3', '--gallery-size', '20', '--seed', str(a.seed),
               '-o', os.path.join(smoke_root, 'retrieval_baseline')] + _device_args(a))
    run_stage('smoke/batch_eval (1 张, 含精修)',
              [sys.executable, os.path.join(_VAL_DIR, 'batch_eval.py'),
               '-c', a.checkpoint, '--val-dir', a.val_dir,
               '--num-images', '1', '--noise-eval', '2', '--no-save-images'] \
              + _refine_args(a) + _chroma_args(a) + _wde_args(a) + [
               '--seed', str(a.seed),
               '-o', os.path.join(smoke_root, 'batch_eval')] + _device_args(a))
    if _DRY_RUN:
        return

    rb = read_json(os.path.join(smoke_root, 'retrieval_baseline', 'summary.json'))
    bb = read_json(os.path.join(smoke_root, 'batch_baseline', 'summary.json'))
    be = read_json(os.path.join(smoke_root, 'batch_eval', 'summary.json'))
    if rb is None or bb is None or be is None:
        raise SystemExit('[中止] 冒烟阶段未生成 summary.json, 请检查上方子脚本输出。')

    # 硬性门槛: 特征链路自检必须 100%, 否则全量结果无意义
    sanity = rb.get('self_sanity_top1', 0.0)
    if sanity < 0.999:
        raise SystemExit(
            f"[中止] 检索自检未通过: 原图自查询 top-1 命中率 {sanity * 100:.1f}% (应为 100%)。\n"
            f"特征提取链路异常 (checkpoint / 数据 / 环境), 请先排查再跑全量。")
    print(f"[冒烟通过] 检索自检 100% | 基线编码图查询 top-1 命中 "
          f"{rb['top1_hit_rate'] * 100:.1f}% (信息项, 不设门槛)")

    bit = _dig(bb, 'metrics', 'bit_acc', 'mean')
    if bit is None or bit < 0.5:
        raise SystemExit(f"[中止] 冒烟解码异常: bit_acc 均值 "
                         f"{'-' if bit is None else f'{bit * 100:.1f}%'} (<50%), 请检查 checkpoint 是否正确。")
    if bit < 0.9:
        print(f"[警告] 冒烟 bit_acc 均值 {bit * 100:.1f}% (<90%), 解码质量偏低, 继续执行。")

    b_clip = _dig(be, 'base_metrics', 'clip_sim', 'mean')
    r_clip = _dig(be, 'refined_metrics', 'clip_sim', 'mean')
    r_bit = _dig(be, 'refined_metrics', 'bit_acc', 'mean')
    if r_bit is not None and r_bit < 0.5:
        raise SystemExit(f"[中止] 冒烟精修后 bit_acc {r_bit * 100:.1f}% (<50%), 精修链路异常。")
    print(f"[冒烟通过] 精修链路 clip_sim {b_clip:.4f} -> {r_clip:.4f} | "
          f"精修后 bit_acc {r_bit * 100:.1f}% | 批量基线 bit_acc {bit * 100:.1f}%")


def run_single(a):
    paths = list_images(a.val_dir)
    if not paths:
        raise SystemExit(f"[中止] 测试目录下未找到图像: {a.val_dir}")
    if len(a.message) != 30:
        print(f"[警告] --message 长度 {len(a.message)} != 30 (Config.message_length), "
              f"子脚本可能报错, 请确认消息长度。")
    run_stage('single (单图全链路 full 模式)',
              [sys.executable, os.path.join(_VAL_DIR, 'validation_script.py'),
               '-c', a.checkpoint, '-i', paths[0], '-m', a.message, '--mode', 'full'] \
              + _refine_args(a) + _chroma_args(a) + _wde_args(a) + [
               '-o', os.path.join(a.output_root, 'single')] + _device_args(a))


def run_batch_baseline(a, n_eval):
    cmd = [sys.executable, os.path.join(_VAL_DIR, 'batch_baseline.py'),
           '-c', a.checkpoint, '--val-dir', a.val_dir,
           '--num-images', str(n_eval), '--noise-eval', str(a.noise_eval),
           '--seed', str(a.seed), '-o', os.path.join(a.output_root, 'batch_baseline')]
    run_stage('batch_baseline (批量基线, 网络直出)', cmd + _device_args(a))


def run_retrieval_baseline(a, gallery):
    cmd = [sys.executable, os.path.join(_VAL_DIR, 'retrieval_baseline.py'),
           '-c', a.checkpoint, '--val-dir', a.val_dir,
           '--num-queries', str(a.num_queries), '--gallery-size', str(gallery),
           '--seed', str(a.seed), '-o', os.path.join(a.output_root, 'retrieval_baseline')]
    run_stage('retrieval_baseline (检索基线, 网络直出, 构建图库特征缓存)',
              cmd + _extra_gallery_args(a) + _device_args(a))


def run_batch_eval(a, n_eval):
    cmd = [sys.executable, os.path.join(_VAL_DIR, 'batch_eval.py'),
           '-c', a.checkpoint, '--val-dir', a.val_dir,
           '--num-images', str(n_eval), '--noise-eval', str(a.noise_eval)] \
          + _refine_args(a) + _chroma_args(a) + _wde_args(a) + [
           '--seed', str(a.seed), '-o', os.path.join(a.output_root, 'batch_eval')]
    if a.no_save_images:
        cmd.append('--no-save-images')
    run_stage('batch_eval (批量精修评估, 与 batch_baseline 同一批图像)', cmd + _device_args(a))


def run_retrieval_eval(a, gallery):
    # 图库配置与 retrieval_baseline 完全一致 (同 val-dir/seed/gallery-size/extra),
    # 把基线阶段提取的 gallery_feats.pt 复制过来直接命中, 省一次 1000 张特征提取
    if not _DRY_RUN:
        src = os.path.join(a.output_root, 'retrieval_baseline', 'gallery_feats.pt')
        dst = os.path.join(a.output_root, 'retrieval_eval', 'gallery_feats.pt')
        if os.path.exists(src) and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print('[缓存复用] 已复制基线阶段 gallery_feats.pt -> retrieval_eval (同图库配置直接命中)')
    cmd = [sys.executable, os.path.join(_VAL_DIR, 'retrieval_eval.py'),
           '-c', a.checkpoint, '--val-dir', a.val_dir,
           '--num-queries', str(a.num_queries), '--gallery-size', str(gallery)] \
          + _refine_args(a) + _chroma_args(a) + _wde_args(a) + [
           '--seed', str(a.seed), '-o', os.path.join(a.output_root, 'retrieval_eval')]
    run_stage('retrieval_eval (检索精修评估, 复用图库缓存)', cmd + _extra_gallery_args(a) + _device_args(a))


def _external_pgd_args(a):
    # 外部事后 PGD 参数转发 (None 安全): None = 取 external_baseline.py 默认, 不传参
    out = []
    if a.external_pgd_steps is not None:
        out += ['--pgd-steps', str(a.external_pgd_steps)]
    if a.external_pgd_target is not None:
        out += ['--pgd-target', str(a.external_pgd_target)]
    return out


def run_external(a, ext_n, ext_q, ext_g):
    # 外部系统对比阶段: 冒烟 (2 张, 直出 + 同预算 PGD, 无检索) -> 全量 (批量 + 检索)。
    # 不需要 PrivSteg checkpoint; 同 seed 同 val-dir 时与 PrivSteg 阶段评估同一批图像。
    ext_dir = os.path.join(a.output_root, f'external_{a.external_system}')
    run_stage('external/smoke (2 张, 直出+同预算PGD, 无检索)',
              [sys.executable, os.path.join(_VAL_DIR, 'external_baseline.py'),
               '--system', a.external_system, '--trustmark-model', a.trustmark_model,
               '--val-dir', a.val_dir, '--num-images', '2', '--noise-eval', '1',
               '--skip-retrieval', '--pgd-conds', 'pgd_matched'] \
              + _external_pgd_args(a) + [
               '--seed', str(a.seed), '-o', os.path.join(a.output_root, '_smoke',
                                                         'external_baseline')] \
              + _device_args(a))
    if not _DRY_RUN:
        es = read_json(os.path.join(a.output_root, '_smoke', 'external_baseline', 'summary.json'))
        clip = _dig(es, 'batch', 'direct', 'metrics', 'clip_sim', 'mean')
        bit = _dig(es, 'batch', 'direct', 'metrics', 'bit_acc', 'mean')
        if clip is None:
            raise SystemExit('[中止] 外部系统冒烟失败: 未生成直出指标, '
                             '请检查 trustmark 安装与权重下载 (首次运行需联网自动下载)。')
        warn = '' if (bit is not None and bit >= 0.9) else \
            f' [警告] 外部直出 bit_acc {bit * 100:.1f}% (<90%), 解码质量偏低'
        print(f"[冒烟通过] 外部系统直出: CLIP sim {clip:.4f} | bit_acc {bit * 100:.1f}%{warn}")

        # 图库特征缓存复用: PrivSteg 检索基线已提取且图库配置一致 (同 val-dir/seed/
        # gallery-size/无扩充) 时直接命中, 省一次全图库特征提取
        src = os.path.join(a.output_root, 'retrieval_baseline', 'gallery_feats.pt')
        dst = os.path.join(ext_dir, 'gallery_feats.pt')
        if os.path.exists(src) and not os.path.exists(dst) and ext_g == gallery_of(a):
            os.makedirs(ext_dir, exist_ok=True)
            shutil.copy2(src, dst)
            print('[缓存复用] 已复制 PrivSteg 检索阶段 gallery_feats.pt -> '
                  f'{os.path.basename(ext_dir)} (同图库配置直接命中)')

    run_stage('external/full (批量直出+事后PGD两档+CBIR检索)',
              [sys.executable, os.path.join(_VAL_DIR, 'external_baseline.py'),
               '--system', a.external_system, '--trustmark-model', a.trustmark_model,
               '--val-dir', a.val_dir,
               '--num-images', str(ext_n), '--noise-eval', str(ext_noise_of(a)),
               '--num-queries', str(ext_q), '--gallery-size', str(ext_g)] \
              + _external_pgd_args(a) + _extra_gallery_args(a) + [
               '--seed', str(a.seed), '-o', ext_dir] + _device_args(a))


def gallery_of(a):
    # 编排主流程解析后的实际图库规模 (与 main 中 gallery 同口径)
    return min(a.gallery_size, len(list_images(a.val_dir)))


def ext_noise_of(a):
    return a.external_noise_eval if a.external_noise_eval is not None else a.noise_eval


def external_plan(a, n_img, gallery):
    # 计划打印/执行用: 外部阶段实际规模 (None = 与 PrivSteg 同规模, 均截断到测试集)
    ext_n = min(a.external_num_images if a.external_num_images is not None
                else a.num_images, n_img)
    ext_q = a.external_num_queries if a.external_num_queries is not None else a.num_queries
    ext_g = min(a.external_gallery_size if a.external_gallery_size is not None
                else a.gallery_size, n_img)
    return ext_n, ext_q, ext_g


# ============================ 跨系统对比 ============================

CROSS_COLUMNS = ['PrivSteg direct', 'PrivSteg refined', 'TrustMark direct',
                 'TrustMark + PGD (matched)', 'TrustMark + PGD (free)']
CROSS_ZH_COLUMNS = ['自有直出', '自有精修', 'TM直出', 'TM+PGD同预算', 'TM+PGD放开']
# 行: (中文名, 英文名, 格式, 字段) —— 字段来自 _norm_cond 归一化后的扁平键
CROSS_ROWS = [
    ('CLIP 相似度', 'CLIP similarity (lower = more private)', 'ms4', 'clip_sim'),
    ('DINO 相似度', 'DINO similarity', 'ms4', 'dino_sim'),
    ('比特准确率', 'Bit accuracy', 'msP', 'bit_acc'),
    ('PSNR (dB)', 'PSNR (dB)', 'ms2', 'psnr'),
    ('SSIM', 'SSIM', 'ms4', 'ssim'),
    ('感知色差 ΔE (均值)', 'Perceptual ΔE (mean)', 'ms2', 'dE_mean'),
    ('ΔE>5 像素占比', 'ΔE>5 pixel ratio', 'msP', 'dE_gt5_ratio'),
    ('bit_acc=100% 占比', 'Perfect-decode ratio', 'pct', 'bit_acc_100'),
    ('clip_sim<0.70 占比', 'CLIP sim < 0.70 ratio', 'pct', 'clip_below_070'),
    ('信道后 bit_acc (均值)', 'Bit acc after channel (mean)', 'pct', 'noise_bit_mean'),
    ('信道后 bit_acc (最差)', 'Bit acc after channel (worst)', 'pct', 'noise_bit_min'),
    ('uint8 往返 bit_acc', 'Bit acc after uint8 round-trip', 'pct', 'rt_bit'),
    ('检索 top-1 命中率', 'Retrieval top-1 hit rate', 'pct', 'top1'),
    ('检索逃逸率', 'Retrieval escape rate', 'pct', 'escape'),
    ('检索 MRR', 'Retrieval MRR', 'f3', 'mrr'),
]


def _x_ms(src, sec, key):
    # metric_stats 字段 -> {'mean','std'} (无则 None)
    v = (src or {}).get(sec, {}).get(key)
    return v if isinstance(v, dict) and v.get('mean') is not None else None


def _x_f(src, sec, key):
    # 纯数值字段 -> float (无则 None)
    v = (src or {}).get(sec, {}).get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _x_fm(src, sec, key):
    # metric_stats 字段 -> mean (float, 无则 None)
    v = _x_ms(src, sec, key)
    return v['mean'] if v else None


def _norm_cond(batch_cond, retr_cond):
    # 单条件归一化为跨系统对比扁平字段 (PrivSteg 与外部系统 summary 同构)
    batch_cond = batch_cond or {}
    retr_cond = retr_cond or {}
    return {
        'clip_sim': _x_ms(batch_cond, 'metrics', 'clip_sim'),
        'dino_sim': _x_ms(batch_cond, 'metrics', 'dino_sim'),
        'bit_acc': _x_ms(batch_cond, 'metrics', 'bit_acc'),
        'psnr': _x_ms(batch_cond, 'metrics', 'psnr'),
        'ssim': _x_ms(batch_cond, 'metrics', 'ssim'),
        'dE_mean': _x_ms(batch_cond, 'metrics', 'dE_mean'),
        'dE_gt5_ratio': _x_ms(batch_cond, 'metrics', 'dE_gt5_ratio'),
        'bit_acc_100': _x_f(batch_cond, 'success', 'bit_acc_100'),
        'clip_below_070': _x_f(batch_cond, 'success', 'clip_below_070'),
        'noise_bit_mean': _x_fm(batch_cond, 'noise', 'bit_acc_mean'),
        'noise_bit_min': _x_fm(batch_cond, 'noise', 'bit_acc_min'),
        'rt_bit': _x_fm(batch_cond, 'roundtrip', 'bit_acc'),
        'top1': retr_cond.get('top1_hit_rate'),
        'escape': retr_cond.get('escape_rate'),
        'mrr': retr_cond.get('mrr'),
    }


def cross_columns(bb, be, rb, rv, ext_summary):
    # 汇总五个被测条件: PrivSteg 直出/精修 + 外部系统 直出/PGD 两档
    ext_retr = (ext_summary or {}).get('retrieval') or {}
    ext_batch = (ext_summary or {}).get('batch') or {}

    def be_refined():
        return {'metrics': _dig(be, 'refined_metrics') or {},
                'success': _dig(be, 'refined_success') or {},
                'noise': _dig(be, 'refined_noise') or {},
                'roundtrip': _dig(be, 'refined_roundtrip') or {}}

    cols = [
        (CROSS_COLUMNS[0], _norm_cond(bb, rb)),
        (CROSS_COLUMNS[1], _norm_cond(be_refined(), _dig(rv, 'refined'))),
    ]
    if ext_summary is not None:
        sys_name = ext_summary.get('system', 'external').upper()
        for cond, suffix in [('direct', 'direct'), ('pgd_matched', 'PGD (matched)'),
                             ('pgd_free', 'PGD (free)')]:
            cols.append((f'{sys_name} {suffix}',
                         _norm_cond(ext_batch.get(cond) or {}, ext_retr.get(cond) or {})))
    return cols


def cross_rows(bb, be, rb, rv, ext_summary):
    # 行: (中文名, 英文名, [五列单元格]) ; 列缺失时显示 '-'
    cols = cross_columns(bb, be, rb, rv, ext_summary)
    rows = []
    for zh, en, kind, key in CROSS_ROWS:
        cells = []
        for _, norm in cols:
            v = norm.get(key)
            if v is None:
                cells.append('-')
            elif kind == 'pct':
                cells.append(f'{v * 100:.1f}%')
            elif kind == 'f3':
                cells.append(f'{v:.3f}')
            elif kind == 'msP':
                cells.append(f"{v['mean'] * 100:.1f}±{v['std'] * 100:.1f}%")
            else:
                f = '{:.4f}' if kind == 'ms4' else '{:.2f}'
                cells.append(f"{f.format(v['mean'])}±{f.format(v['std'])}")
        rows.append((zh, en, *cells))
    return rows, cols


def print_cross_table(title, rows):
    print(f'\n[{title}]')
    print(f"  {'指标':<24}" + ''.join(f'{c:>16}' for c in CROSS_ZH_COLUMNS))
    print('  ' + '-' * 104)
    for r in rows:
        zh, en, *cells = r
        print(f'  {zh:<24}' + ''.join(f'{c:>16}' for c in cells))
    print('  ' + '-' * 104)
    print('  注: TM 列载荷为 TrustMark 原生 61 bit (BCH_5), PrivSteg 为 30 bit, '
          '论文表格需脚注说明; 同 seed 同 val-dir 时五列评估同一批图像。')


def _cross_cell(norm, kind, key):
    # 跨系统报告表格单元格 (与 cross_rows 同格式)
    v = (norm or {}).get(key)
    if v is None:
        return '-'
    if kind == 'pct':
        return f'{v * 100:.1f}%'
    if kind == 'f3':
        return f'{v:.3f}'
    if kind == 'msP':
        return f"{v['mean'] * 100:.1f}±{v['std'] * 100:.1f}%"
    f = '{:.4f}' if kind == 'ms4' else '{:.2f}'
    return f"{f.format(v['mean'])}±{f.format(v['std'])}"


def _read_recs(results_path):
    # 逐图记录 -> list (用于分布直方图); 文件不存在返回 []
    data = read_json(results_path)
    return list(data.values()) if data else []


def generate_cross_report(out_root):
    # 跨系统对比报告 (cross_report.html + panel_cross_*.png, 英文图表):
    # PrivSteg 直出/精修 vs 外部系统 直出/PGD 同预算/放开, matplotlib 缺失时跳过
    ext_dir = find_external_dir(out_root)
    bb = read_json(os.path.join(out_root, 'batch_baseline', 'summary.json'))
    be = read_json(os.path.join(out_root, 'batch_eval', 'summary.json'))
    rb = read_json(os.path.join(out_root, 'retrieval_baseline', 'summary.json'))
    rv = read_json(os.path.join(out_root, 'retrieval_eval', 'summary.json'))
    ext_summary = read_json(os.path.join(ext_dir, 'summary.json')) if ext_dir else None
    if bb is None and be is None and ext_summary is None:
        return None
    try:
        from eval_visualizer import _b64, _card, _write_html
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f'[跳过] 跨系统对比报告不可用 (matplotlib/eval_visualizer 缺失): {e}')
        return None

    rows, cols = cross_rows(bb, be, rb, rv, ext_summary)
    short = ['PS direct', 'PS refined', 'TM direct', 'TM+PGD(m)', 'TM+PGD(f)']

    def m(col, key):
        v = col.get(key)
        return v['mean'] if isinstance(v, dict) else v

    # ---- 指标卡片 ----
    cards = ''
    for (label, norm), color in zip(cols, ['#7db8f0', '#f0a0a0', '#c3ccd6', '#93aaf0', '#b39df0']):
        if m(norm, 'clip_sim') is None:
            continue
        sub = f"bit acc {m(norm, 'bit_acc') * 100:.1f}%" if m(norm, 'bit_acc') else ''
        cards += _card(f"{m(norm, 'clip_sim'):.3f}", f'CLIP sim ({label})', 'info', sub)
    for label, norm in cols:
        if norm.get('escape') is not None:
            cards += _card(f"{norm['escape'] * 100:.1f}%", f'escape rate ({label})', 'good')

    # ---- 面板 1: CLIP 相似度分布 (五条件直方图) ----
    series = []
    p1 = _read_recs(os.path.join(out_root, 'batch_baseline', 'results.json'))
    if p1:
        series.append(('PrivSteg direct', np.array([r['metrics']['clip_sim'] for r in p1
                                                    if r.get('metrics') is not None], float),
                       '#1f77b4'))
    p2 = _read_recs(os.path.join(out_root, 'batch_eval', 'results.json'))
    if p2:
        series.append(('PrivSteg refined', np.array([r['refined_metrics']['clip_sim'] for r in p2
                                                     if r.get('refined_metrics') is not None], float),
                       '#d62728'))
    if ext_dir:
        pe = _read_recs(os.path.join(ext_dir, 'results.json'))
        for cond, label, color in [('direct', 'TrustMark direct', '#9aa7b5'),
                                   ('pgd_matched', 'TrustMark + PGD (matched)', '#8a5fd4'),
                                   ('pgd_free', 'TrustMark + PGD (free)', '#b39df0')]:
            v = np.array([r[cond]['metrics']['clip_sim'] for r in pe
                          if cond in r and r[cond].get('metrics') is not None], float)
            if len(v):
                series.append((label, v, color))
    sections = []
    if series:
        fig, ax = plt.subplots(figsize=(10.0, 4.4))
        bins = np.linspace(-0.05, 1.0, 27)
        for label, v, color in series:
            ax.hist(v, bins=bins, alpha=0.5, color=color, edgecolor='white',
                    label=f'{label} (mean {np.nanmean(v):.3f})')
        ax.axvline(0.40, ls='--', color='#d64545', lw=1.2, label='PrivSteg refined target (0.40)')
        ax.axvline(0.70, ls=':', color='#e8a13c', lw=1.2, label='Reference 0.70')
        ax.set_xlabel('CLIP similarity to cover')
        ax.set_ylabel('Number of images')
        ax.set_title('Cross-system CLIP similarity distribution')
        ax.legend(fontsize=8)
        fig.tight_layout()
        img = _b64(fig, path=os.path.join(out_root, 'panel_cross_sim_dist.png'))
        if img:
            sections.append(('CLIP similarity across systems', img))

    # ---- 面板 2: 质量-隐私总览 (CLIP sim / bit_acc 均值条形) ----
    active_idx = [i for i, (label, norm) in enumerate(cols) if m(norm, 'clip_sim') is not None]
    if active_idx:
        active = [cols[i] for i in active_idx]
        labels = [short[i] for i in active_idx]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for ax, key, ylab, t in [(axes[0], 'clip_sim', 'CLIP similarity (mean, lower = more private)',
                                  'Privacy: CLIP similarity to cover'),
                                 (axes[1], 'bit_acc', 'bit accuracy (mean)', 'Function: bit accuracy')]:
            vals = [m(norm, key) for _, norm in active]
            bars = ax.bar(np.arange(len(active)), vals, 0.55,
                          color=['#1f77b4', '#d62728', '#9aa7b5', '#8a5fd4', '#b39df0'][:len(active)])
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                        f'{b.get_height():.3f}', ha='center', fontsize=8)
            ax.set_xticks(np.arange(len(active)))
            ax.set_xticklabels(labels, fontsize=8, rotation=12)
            if key == 'clip_sim':
                ax.axhline(0.40, ls='--', color='#d64545', lw=1.2)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(ylab)
            ax.set_title(t)
        fig.tight_layout()
        img = _b64(fig, path=os.path.join(out_root, 'panel_cross_quality.png'))
        if img:
            sections.append(('Privacy vs function overview', img))

    # ---- 面板 3: 检索逃逸对比 (top-1 命中 / 逃逸率) ----
    retr_active = [(short[i], norm) for i, (label, norm) in enumerate(cols)
                   if norm.get('top1') is not None]
    if retr_active:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        vals = [(lab, norm['top1'] * 100, norm['escape'] * 100) for lab, norm in retr_active]
        for ax, idx, ylab, t in [(axes[0], 1, 'top-1 hit rate (%)',
                                  'Cover still retrievable (lower = better privacy)'),
                                 (axes[1], 2, 'escape rate (%)',
                                  'Retrieval escape rate')]:
            bars = ax.bar(np.arange(len(vals)), [v[idx] for v in vals], 0.55,
                          color=['#1f77b4', '#d62728', '#9aa7b5', '#8a5fd4', '#b39df0'][:len(vals)])
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                        f'{b.get_height():.1f}', ha='center', fontsize=8)
            ax.set_xticks(np.arange(len(vals)))
            ax.set_xticklabels([v[0] for v in vals], fontsize=8, rotation=12)
            ax.set_ylabel(ylab)
            ax.set_title(t)
        fig.tight_layout()
        img = _b64(fig, path=os.path.join(out_root, 'panel_cross_retrieval.png'))
        if img:
            sections.append(('CBIR retrieval attack across systems', img))

    # ---- 跨系统全指标对照表 ----
    tbl = '<tr><th>Metric</th>' + ''.join(f'<th>{c}</th>' for c in CROSS_COLUMNS) + '</tr>'
    for zh, en, *cells in rows:
        tbl += f'<tr><td>{en}</td>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
    table_html = (f'<section><h2>Cross-system all-metrics comparison</h2>'
                  f'<table class="cmp">{tbl}</table></section>')

    meta_rows = []
    ext_meta = read_json(os.path.join(ext_dir, 'eval_meta.json')) if ext_dir else None
    n_priv_direct = len(_read_recs(os.path.join(out_root, 'batch_baseline', 'results.json')))
    n_priv_ref = len(_read_recs(os.path.join(out_root, 'batch_eval', 'results.json')))
    n_ext = len(_read_recs(os.path.join(ext_dir, 'results.json'))) if ext_dir else 0
    meta_rows.append(('Images (PrivSteg direct / refined / external)',
                      f'{n_priv_direct} / {n_priv_ref} / {n_ext}'))
    meta_rows.append(('External system / model',
                      f"{(ext_summary or {}).get('system', '-')}"
                      f"({(ext_meta or {}).get('trustmark_model', '-')})"))
    meta_rows.append(('External payload (bits)', (ext_meta or {}).get('secret_len', '-')))
    foot = ('Protocol parity: both systems are evaluated on the same image sample (same seed and '
            'val-dir), with identical CLIP/DINO features, noise channel, uint8 round-trip and CBIR '
            'retrieval protocol. Payload lengths follow each native design (PrivSteg 30-bit; '
            'TrustMark Q 61-bit BCH_5) and are disclosed in the paper table footnote. '
            'pgd_matched constrains the perturbation to the same per-pixel budget as PrivSteg '
            '(Config.prv_budget); pgd_free removes the budget. Both follow the no-degradation '
            'selection rule: published states must decode perfectly (raw bits + ECC for TrustMark). '
            'Panel PNGs are saved in this directory.')
    _write_html(os.path.join(out_root, 'cross_report.html'),
                'PrivSteg vs External System Comparison', meta_rows, cards, sections, foot)
    report_path = os.path.join(out_root, 'cross_report.html')
    with open(report_path, 'r', encoding='utf-8') as f:
        html = f.read()
    html = html.replace('</div></body>', table_html + '</div></body>')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'跨系统对比报告: {report_path}')
    return report_path


# ============================ 汇总 ============================

def _dig(d, *keys):
    # 安全取嵌套字段: 任一层缺失/为 None 返回 None
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or cur.get(k) is None:
            return None
        cur = cur[k]
    return cur


def batch_rows(bb, be):
    # 行: (中文名, 英文名, 网络直出(独立脚本), 网络直出(精修运行内), 精修后)
    bbm = _dig(bb, 'metrics') or {}
    bem = _dig(be, 'base_metrics') or {}
    brm = _dig(be, 'refined_metrics') or {}

    def ms(src, key, fmt):
        st = src.get(key)
        return '-' if st is None else f"{fmt.format(st['mean'])}±{fmt.format(st['std'])}"

    rows = []
    for zh, en, key, fmt in [
            ('CLIP 相似度', 'CLIP similarity', 'clip_sim', '{:.4f}'),
            ('DINO 相似度', 'DINO similarity', 'dino_sim', '{:.4f}'),
            ('比特准确率', 'Bit accuracy', 'bit_acc', '{:.1%}'),
            ('PSNR (dB)', 'PSNR (dB)', 'psnr', '{:.2f}'),
            ('SSIM', 'SSIM', 'ssim', '{:.4f}'),
            ('感知色差 ΔE (均值)', 'Perceptual ΔE (mean)', 'dE_mean', '{:.2f}'),
            ('ΔE>5 像素占比', 'ΔE>5 pixel ratio', 'dE_gt5_ratio', '{:.1%}')]:
        rows.append((zh, en, ms(bbm, key, fmt), ms(bem, key, fmt), ms(brm, key, fmt)))

    def pct3(v1, v2, v3):
        return tuple('-' if v is None else f'{v * 100:.1f}%' for v in (v1, v2, v3))

    rows.append(('bit_acc=100% 占比', 'Perfect-decode ratio', *pct3(
        _dig(bb, 'success', 'bit_acc_100'), _dig(be, 'base_success', 'bit_acc_100'),
        _dig(be, 'refined_success', 'bit_acc_100'))))
    rows.append(('clip_sim<0.70 占比', 'CLIP sim < 0.70 ratio', *pct3(
        _dig(bb, 'success', 'clip_below_070'), _dig(be, 'base_success', 'clip_below_070'),
        _dig(be, 'refined_success', 'clip_below_070'))))
    rows.append(('信道后 bit_acc (均值)', 'Bit acc after channel (mean)', *pct3(
        _dig(bb, 'noise', 'bit_acc_mean', 'mean'), _dig(be, 'base_noise', 'bit_acc_mean', 'mean'),
        _dig(be, 'refined_noise', 'bit_acc_mean', 'mean'))))
    rows.append(('信道后 bit_acc (最差)', 'Bit acc after channel (worst)', *pct3(
        _dig(bb, 'noise', 'bit_acc_min', 'mean'), _dig(be, 'base_noise', 'bit_acc_min', 'mean'),
        _dig(be, 'refined_noise', 'bit_acc_min', 'mean'))))
    rows.append(('uint8 往返 bit_acc', 'Bit acc after uint8 round-trip', *pct3(
        _dig(bb, 'roundtrip', 'bit_acc', 'mean'), _dig(be, 'base_roundtrip', 'bit_acc', 'mean'),
        _dig(be, 'refined_roundtrip', 'bit_acc', 'mean'))))
    return rows


def retrieval_rows(rb, rv):
    conds = [rb or {}, _dig(rv, 'base') or {}, _dig(rv, 'refined') or {}]
    rows = []
    for zh, en, key, kind in [
            ('top-1 命中率', 'top-1 hit rate', 'top1_hit_rate', 'pct'),
            ('top-5 命中率', 'top-5 hit rate', 'top5_hit_rate', 'pct'),
            ('top-10 命中率', 'top-10 hit rate', 'top10_hit_rate', 'pct'),
            ('检索逃逸率', 'Retrieval escape rate', 'escape_rate', 'pct'),
            ('平均排名', 'Mean rank', 'mean_rank', 'f1'),
            ('中位排名', 'Median rank', 'median_rank', 'f0'),
            ('MRR', 'MRR', 'mrr', 'f3'),
            ('sim(原图)', 'Similarity to cover', 'sim_original_mean', 'f4'),
            ('sim(最强干扰)', 'Similarity to top distractor', 'sim_top1_distractor_mean', 'f4')]:
        cells = []
        for c in conds:
            v = c.get(key)
            if v is None:
                cells.append('-')
            elif kind == 'pct':
                cells.append(f'{v * 100:.1f}%')
            else:
                cells.append({'f1': f'{v:.1f}', 'f0': f'{v:.0f}',
                              'f3': f'{v:.3f}', 'f4': f'{v:.4f}'}[kind])
        rows.append((zh, en, *cells))
    return rows


def print_table(title, rows):
    print(f'\n[{title}]')
    print(f"  {'指标':<26} {'网络直出':>18} {'直出(运行内)':>18} {'精修后':>18}")
    print('  ' + '-' * 84)
    for zh, en, c1, c2, c3 in rows:
        print(f'  {zh:<26} {c1:>18} {c2:>18} {c3:>18}')


def run_summary(out_root, a):
    bb = read_json(os.path.join(out_root, 'batch_baseline', 'summary.json'))
    be = read_json(os.path.join(out_root, 'batch_eval', 'summary.json'))
    rb = read_json(os.path.join(out_root, 'retrieval_baseline', 'summary.json'))
    rv = read_json(os.path.join(out_root, 'retrieval_eval', 'summary.json'))

    missing = [n for n, s in [('batch_baseline', bb), ('batch_eval', be),
                              ('retrieval_baseline', rb), ('retrieval_eval', rv)] if s is None]
    if missing:
        print(f'[警告] 以下阶段的 summary.json 缺失 (未运行?), 对应列显示为 "-": {", ".join(missing)}')

    n_batch = len(read_json(os.path.join(out_root, 'batch_eval', 'results.json')) or {}) or \
        len(read_json(os.path.join(out_root, 'batch_baseline', 'results.json')) or {})
    meta_re = read_json(os.path.join(out_root, 'retrieval_eval', 'eval_meta.json')) or \
        read_json(os.path.join(out_root, 'retrieval_baseline', 'eval_meta.json')) or {}

    b_rows = batch_rows(bb, be)
    r_rows = retrieval_rows(rb, rv)

    # ---- 跨系统对比 (external 阶段产物存在时) ----
    ext_dir = find_external_dir(out_root)
    ext_summary = read_json(os.path.join(ext_dir, 'summary.json')) if ext_dir else None
    if ext_summary is not None:
        c_rows, _ = cross_rows(bb, be, rb, rv, ext_summary)
        print_cross_table(f'跨系统对比 (PrivSteg vs '
                          f'{ext_summary.get("system", "external").upper()})', c_rows)
    else:
        c_rows = None
        print('\n[跳过] 跨系统对比: 未发现 external_*/summary.json (external 阶段未运行)')

    print(f"\n{'#' * 78}")
    print('# 全链路验证汇总')
    print('#' * 78)
    print_table(f'批量定量评估 (N={n_batch} 张)', b_rows)
    print_table(f"CBIR 检索逃逸 (查询 {meta_re.get('num_queries', '-')} / "
                f"图库 {meta_re.get('gallery_size', '-')})", r_rows)

    # ---- 自检与一致性交叉检查 ----
    sanity = _dig(rb, 'self_sanity_top1')
    if sanity is None:
        sanity = _dig(rv, 'self_sanity_top1')
    if sanity is not None:
        flag = '正常' if sanity >= 0.999 else '异常 (应为 100%)'
        print(f'\n自检 (原图自查询 top-1): {sanity * 100:.1f}% [{flag}]')

    d1 = _dig(bb, 'metrics', 'clip_sim', 'mean')
    d2 = _dig(be, 'base_metrics', 'clip_sim', 'mean')
    if d1 is not None and d2 is not None and abs(d1 - d2) > 0.02:
        print(f'[警告] 两次网络直出的 clip_sim 均值差 {abs(d1 - d2):.4f} (>0.02), '
              f'请确认 batch_baseline 与 batch_eval 使用了相同 seed / val-dir')
    t1 = _dig(rb, 'top1_hit_rate')
    t2 = _dig(rv, 'base', 'top1_hit_rate')
    if t1 is not None and t2 is not None and abs(t1 - t2) > 0.05:
        print(f'[警告] 两次网络直出的检索 top-1 差 {abs(t1 - t2) * 100:.1f}% (>5%), '
              f'请确认两个检索脚本的图库配置一致')

    direct = d1 if d1 is not None else d2
    refined = _dig(be, 'refined_metrics', 'clip_sim', 'mean')
    if direct is not None and refined is not None:
        print(f'CLIP 逃逸: 网络直出 {direct:.4f} -> 精修后 {refined:.4f} '
              f'(直出值越低 = 内环蒸馏摊销越成功; 差值 = 逐图精修的额外收益)')

    # ---- 写 full_summary.json / full_summary.md ----
    rc = _effective_refine_cfg(a)

    def to_dicts(rows):
        return [{'metric_zh': zh, 'metric': en, 'direct': c1, 'direct_in_run': c2, 'refined': c3}
                for zh, en, c1, c2, c3 in rows]

    consolidated = {
        'meta': {
            'checkpoint': a.checkpoint, 'val_dir': a.val_dir, 'seed': a.seed,
            'refine': {'steps': rc['steps'], 'lr': rc['lr'],
                       'budget': rc['budget'], 'margin': rc['margin'],
                       'w_de': rc['w_de'], 'de_tau': rc['de_tau'],
                       'w_message': rc['w_message'], 'w_privacy': rc['w_privacy'],
                       'chroma_projection': rc['chroma_projection']},
            'num_images_evaluated': n_batch,
            'num_queries': meta_re.get('num_queries'), 'gallery_size': meta_re.get('gallery_size'),
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'batch': {'batch_baseline': bb, 'batch_eval': be},
        'retrieval': {'retrieval_baseline': rb, 'retrieval_eval': rv},
        'external': {'summary': ext_summary,
                     'dir': os.path.basename(ext_dir) if ext_dir else None},
        'derived': {'batch_table': to_dicts(b_rows), 'retrieval_table': to_dicts(r_rows)},
    }
    if c_rows is not None:
        cross_dicts = []
        for r in c_rows:
            d = {'metric_zh': r[0], 'metric': r[1]}
            for label, cell in zip(CROSS_COLUMNS, r[2:]):
                d[label] = cell
            cross_dicts.append(d)
        consolidated['derived']['comparison_table'] = cross_dicts
    json_path = os.path.join(out_root, 'full_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)

    md = ['# PrivSteg Full Validation Summary', '',
          f"- Checkpoint: `{a.checkpoint}`",
          f"- Test set: `{a.val_dir}`",
          f"- Images evaluated (batch): {n_batch}",
          f"- Retrieval: {meta_re.get('num_queries', '-')} queries / "
          f"{meta_re.get('gallery_size', '-')} gallery",
          f"- Seed: {a.seed} | Refinement: steps={rc['steps']}, lr={rc['lr']}, "
          f"budget={rc['budget']}, margin={rc['margin']}, w_de={rc['w_de']}, "
          f"de_tau={rc['de_tau']}, w_message={rc['w_message']}, w_privacy={rc['w_privacy']}, "
          f"chroma_projection={rc['chroma_projection'] if rc['chroma_projection'] is not None else 'cfg default'}",
          f"- Generated: {consolidated['meta']['generated']}", '',
          '## 1. Batch quantitative evaluation', '',
          '| Metric | Encoder direct | Encoder direct (in-run) | Per-image refined |',
          '|---|---|---|---|']
    md += [f'| {en} | {c1} | {c2} | {c3} |' for _, en, c1, c2, c3 in b_rows]
    md += ['', '## 2. CBIR retrieval attack', '',
           '| Metric | Encoder direct | Encoder direct (in-run) | Per-image refined |',
           '|---|---|---|---|']
    md += [f'| {en} | {c1} | {c2} | {c3} |' for _, en, c1, c2, c3 in r_rows]
    if c_rows is not None:
        sys_name = ext_summary.get('system', 'external').upper()
        md += ['', f'## 3. Cross-system comparison (PrivSteg vs {sys_name})', '',
               'Both systems are evaluated on the same image sample (same seed and val-dir) with '
               'identical CLIP/DINO features, noise channel, uint8 round-trip and CBIR retrieval '
               'protocol. Payload lengths follow each native design (PrivSteg 30-bit; '
               f'{sys_name} payload from its native BCH packet) and should be footnoted '
               'in the paper table.', '',
               '| Metric | ' + ' | '.join(CROSS_COLUMNS) + ' |',
               '|---|' + '---|' * len(CROSS_COLUMNS)]
        md += [f'| {en} | ' + ' | '.join(cells) + ' |' for _, en, *cells in c_rows]
        md += ['', 'PGD (matched) constrains the perturbation to the same per-pixel budget as '
               'PrivSteg (Config.prv_budget); PGD (free) removes the budget. Both follow the '
               'no-degradation selection rule: published states must decode perfectly under the '
               'system\'s own decoder.']
    md += ['']
    if sanity is not None:
        md.append(f"Self-sanity (cover self-query top-1): {sanity * 100:.1f}% (expected 100%).")
    if direct is not None and refined is not None:
        md.append(f"CLIP escape: encoder direct {direct:.4f} -> per-image refined {refined:.4f}. "
                  "A lower direct value means the inner-loop distillation successfully amortized "
                  "the per-instance refinement.")
    artifacts = ['- `single/report.html` - Single-image full pipeline report',
                 '- `batch_baseline/report.html` - Batch baseline report (encoder direct)',
                 '- `retrieval_baseline/report.html` - Retrieval baseline report (encoder direct)',
                 '- `batch_eval/report.html` - Batch refined report (with baseline comparison)',
                 '- `retrieval_eval/report.html` - Retrieval refined report (with baseline comparison)']
    if ext_dir:
        artifacts.append(f'- `{os.path.basename(ext_dir)}/report.html` - '
                         f'{ext_summary.get("system", "external").upper()} external comparison report')
        artifacts.append('- `cross_report.html` - Cross-system comparison report '
                         '(PrivSteg vs external system, English charts)')
    artifacts.append('- `full_summary.json` - Consolidated machine-readable summary')
    md += ['', '## 4. Artifacts', ''] + [a for a in artifacts] + ['']
    md_path = os.path.join(out_root, 'full_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f'\n汇总已保存: {json_path}')
    print(f'论文用英文汇总: {md_path} (表格可直接粘贴进论文)')

    # ---- 跨系统对比报告 (HTML + PNG 面板, matplotlib 缺失时跳过) ----
    try:
        generate_cross_report(out_root)
    except Exception as e:
        print(f'跨系统对比报告生成失败: {e}')

    print('\n产出文件 (输出目录内):')
    for sub_name, desc in [('single', '单图全链路可视化报告'),
                           ('batch_baseline', '批量基线报告 (网络直出)'),
                           ('retrieval_baseline', '检索基线报告 (网络直出)'),
                           ('batch_eval', '批量精修报告 (含基线对照)'),
                           ('retrieval_eval', '检索精修报告 (含基线对照)')]:
        report = os.path.join(out_root, sub_name, 'report.html')
        mark = '' if os.path.exists(report) else ' (未生成: 阶段未运行)'
        print(f'  {sub_name}/report.html - {desc}{mark}')
    if ext_dir:
        ext_report = os.path.join(ext_dir, 'report.html')
        mark = '' if os.path.exists(ext_report) else ' (未生成: 阶段未运行)'
        print(f'  {os.path.basename(ext_dir)}/report.html - '
              f'外部系统对比报告 ({ext_summary.get("system", "external")}){mark}')
    cross_report = os.path.join(out_root, 'cross_report.html')
    mark = '' if os.path.exists(cross_report) else ' (未生成: matplotlib 缺失或无数据)'
    print(f'  cross_report.html - 跨系统对比报告 (PrivSteg vs 外部系统, 英文图表){mark}')
    if be is not None:
        print('\n论文提示: 双配方权衡表可用 --refine-budget 0.04 --refine-margin 0.30 '
              '换一个 --output-root 再跑一遍, 构成深挖/标准配方对照。')


# ============================ 入口 ============================

def main():
    global _DRY_RUN
    parser = argparse.ArgumentParser(
        description='全链路验证编排: 冒烟自检 -> 单图 full -> 批量/检索的基线与精修 -> 汇总对照表')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='PrivSteg checkpoint (PrivSteg 评估阶段必需; '
                             '仅跑 external 阶段时可不传)')
    parser.add_argument('--val-dir', type=str, default='./dataset/test',
                        help='测试集目录 (默认 ./dataset/test)')
    parser.add_argument('--output-root', '-o', type=str, default='./other/full_validation')
    parser.add_argument('--num-images', type=int, default=1000,
                        help='批量评估图像数 (自动截断到测试集规模, 默认 1000)')
    parser.add_argument('--num-queries', type=int, default=100)
    parser.add_argument('--gallery-size', type=int, default=1000,
                        help='检索图库大小 (自动截断到测试集规模)')
    parser.add_argument('--extra-gallery-dir', type=str, default=None,
                        help='可选: 检索图库扩充目录 (如 train 集, 增大干扰池)')
    parser.add_argument('--extra-gallery-size', type=int, default=0)
    parser.add_argument('--external-system', type=str, default='trustmark',
                        help='外部对比系统 (当前支持 trustmark)')
    parser.add_argument('--trustmark-model', type=str, default='Q', choices=['C', 'Q', 'B', 'P'],
                        help='TrustMark 变体 (Q: 论文默认)')
    parser.add_argument('--external-num-images', type=int, default=None,
                        help='外部对比批量图像数 (None=与 --num-images 相同)')
    parser.add_argument('--external-num-queries', type=int, default=None,
                        help='外部对比检索查询数 (None=与 --num-queries 相同)')
    parser.add_argument('--external-gallery-size', type=int, default=None,
                        help='外部对比图库大小 (None=与 --gallery-size 相同)')
    parser.add_argument('--external-noise-eval', type=int, default=None,
                        help='外部对比信道噪声采样次数 (None=与 --noise-eval 相同)')
    parser.add_argument('--external-pgd-steps', type=int, default=None,
                        help='外部事后 PGD 最大步数 (None=取 external_baseline.py 默认 150)')
    parser.add_argument('--external-pgd-target', type=float, default=None,
                        help='外部事后 PGD 逃逸目标 (None=取默认 0.40, 与精修下限对齐)')
    parser.add_argument('--skip-external', action='store_true',
                        help='跳过外部系统对比阶段 (默认运行)')
    parser.add_argument('--refine-steps', type=int, default=None,
                        help='精修步数 (None=取 Config.refine_steps, 默认 150)')
    parser.add_argument('--refine-lr', type=float, default=None,
                        help='精修 Adam 学习率 (None=取子脚本默认 0.1)')
    parser.add_argument('--refine-budget', type=float, default=None,
                        help='精修隐私残差预算 (None=取 Config.prv_budget, 默认 0.08)')
    parser.add_argument('--refine-margin', type=float, default=None,
                        help='精修 CLIP escape margin (None=取 Config.refine_margin_end, 默认 0.60)')
    parser.add_argument('--refine-chroma-projection', type=int, default=None,
                        help='精修收敛后剔除 prv 低频色度蒙层 (默认取 Config.refine_chroma_projection; '
                             '1 开启 / 0 关闭)。门控: bit_acc 须 100%% 且 clip_sim 达标, 否则回退')
    parser.add_argument('--refine-w-de', type=float, default=None,
                        help='精修感知色差锚权重 (None=取 Config.refine_w_de, 默认 200; 0=关闭)。'
                             '对 ΔE>tau 像素二次惩罚, 压制 PSNR/SSIM 测不到的色罩/蒙层类失真')
    parser.add_argument('--refine-de-tau', type=float, default=None,
                        help='感知色差锚阈值 de_tau (None=取 Config.refine_de_tau, 默认 4.0)')
    parser.add_argument('--refine-w-message', type=float, default=None,
                        help='精修消息锚权重 (None=取 Config.refine_w_message, 默认 20)')
    parser.add_argument('--refine-w-privacy', type=float, default=None,
                        help='精修隐私(逃逸)锚权重 (None=取 Config.refine_w_privacy, 默认 12)')
    parser.add_argument('--noise-eval', type=int, default=8)
    parser.add_argument('--message', type=str, default='101000100110100101100011100010',
                        help='单图全链路验证使用的消息 (默认 30 bit)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None,
                        help='默认 auto (子脚本按 CUDA 可用性自动选择)')
    parser.add_argument('--no-save-images', action='store_true',
                        help='批量精修评估不保存精修编码图 (默认保存)')
    parser.add_argument('--stages', type=str, default='all',
                        help=f"逗号分隔的阶段子集: {','.join(STAGE_ORDER)} 或 all")
    parser.add_argument('--report-only', action='store_true',
                        help='不重跑评估: 从已有 results.json 重出各子报告与汇总')
    parser.add_argument('--dry-run', action='store_true', help='只打印将执行的命令, 不实际运行')
    args = parser.parse_args()
    _DRY_RUN = args.dry_run

    out_root = args.output_root
    os.makedirs(out_root, exist_ok=True)

    # ---- 仅重出报告模式 ----
    if args.report_only:
        for sub_name, script in [('batch_baseline', 'batch_baseline.py'),
                                 ('retrieval_baseline', 'retrieval_baseline.py'),
                                 ('batch_eval', 'batch_eval.py'),
                                 ('retrieval_eval', 'retrieval_eval.py')]:
            if os.path.exists(os.path.join(out_root, sub_name, 'results.json')):
                run_stage(f'report-only/{sub_name}',
                          [sys.executable, os.path.join(_VAL_DIR, script), '--report-only',
                           '-o', os.path.join(out_root, sub_name)])
            else:
                print(f'[跳过] {sub_name}: 输出目录无 results.json')
        ext_dir = find_external_dir(out_root)
        if ext_dir:
            run_stage('report-only/external',
                      [sys.executable, os.path.join(_VAL_DIR, 'external_baseline.py'),
                       '--report-only', '-o', ext_dir])
        else:
            print('[跳过] external: 未发现外部对比目录 (external_* 阶段未运行过)')
        run_summary(out_root, args)
        return

    # ---- 阶段解析 ----
    if args.stages == 'all':
        stages = list(STAGE_ORDER)
    else:
        stages = [s.strip() for s in args.stages.split(',') if s.strip()]
        for s in stages:
            if s not in STAGE_ORDER:
                raise SystemExit(f'[中止] 未知阶段: {s} (可选: {", ".join(STAGE_ORDER)})')
    ordered = [s for s in STAGE_ORDER if s in set(stages)]
    if args.skip_external and 'external' in ordered:
        ordered.remove('external')
        print('[跳过] external 阶段已按 --skip-external 移除')

    # ---- 预检 ----
    if not os.path.isdir(args.val_dir):
        raise SystemExit(f'[中止] 测试目录不存在: {args.val_dir}')
    paths = list_images(args.val_dir)
    if not paths:
        raise SystemExit(f'[中止] 测试目录下未找到图像: {args.val_dir}')
    n_img = len(paths)
    n_eval = min(args.num_images, n_img)
    gallery = min(args.gallery_size, n_img)
    if args.num_queries > gallery:
        raise SystemExit(f'[中止] --num-queries ({args.num_queries}) 不能超过图库规模 '
                         f'({gallery}, 由 --gallery-size 与测试集规模取小)')
    needs_ckpt = bool(set(ordered) & _PRIV_STAGES)
    if needs_ckpt:
        if not args.checkpoint:
            raise SystemExit('[中止] 所选阶段包含 PrivSteg 评估, 需要 --checkpoint '
                             '(仅 external 阶段无需 checkpoint)')
        if not os.path.isfile(args.checkpoint):
            raise SystemExit(f'[中止] checkpoint 不存在: {args.checkpoint}')

    # ---- 计划 ----
    rc = _effective_refine_cfg(args)
    # 外部阶段规模提前解析 (计划打印与执行共用同一数值)
    ext_n, ext_q, ext_g = external_plan(args, n_img, gallery)
    print(BANNER)
    print('# PrivSteg 全链路验证')
    print(BANNER)
    print(f'checkpoint : {args.checkpoint or "- (仅 external 阶段)"}')
    print(f'测试集     : {args.val_dir} ({n_img} 张)')
    print(f'评估规模   : 批量 {n_eval} 张 | 检索 查询 {args.num_queries} / 图库 {gallery} 张 '
          f'(随机 top-1 命中概率 {100.0 / gallery:.2f}%)')
    if 'external' in ordered:
        print(f'外部对比   : {args.external_system}({args.trustmark_model}) | '
              f'批量 {ext_n} 张 | 检索 查询 {ext_q} / 图库 {ext_g} | '
              f'PGD steps/target: {args.external_pgd_steps or "默认150"}/'
              f'{args.external_pgd_target or "默认0.40"}')
    print(f'配置       : seed={args.seed} | 精修 steps={rc["steps"]} '
          f'budget={rc["budget"]} margin={rc["margin"]} w_de={rc["w_de"]} '
          f'de_tau={rc["de_tau"]} w_message={rc["w_message"]} w_privacy={rc["w_privacy"]} '
          f'chroma_projection={args.refine_chroma_projection if args.refine_chroma_projection is not None else "cfg默认"} '
          f'| noise-eval={args.noise_eval}')
    print(f'device     : {args.device or "auto (子脚本自动选择 CUDA/CPU)"}')
    print(f'输出目录   : {os.path.abspath(out_root)}')
    print(f'阶段       : {", ".join(ordered)}')
    print('阶段成本   : 冒烟/单图/两个基线为轻量; batch_eval / retrieval_eval 含逐图精修, '
          '为主要耗时 (约为基线版 20~30 倍); external 含事后 PGD 两档, 与精修阶段同量级')
    print('断点续跑   : 任一阶段中断后重新执行本命令即可续跑; 从头重跑需换 --output-root '
          '或删除对应子目录的 results.json')

    t_total = time.time()
    for stage in ordered:
        if stage == 'smoke':
            run_smoke(args)
        elif stage == 'single':
            run_single(args)
        elif stage == 'batch_baseline':
            run_batch_baseline(args, n_eval)
        elif stage == 'retrieval_baseline':
            run_retrieval_baseline(args, gallery)
        elif stage == 'batch_eval':
            run_batch_eval(args, n_eval)
        elif stage == 'retrieval_eval':
            run_retrieval_eval(args, gallery)
        elif stage == 'external':
            run_external(args, ext_n, ext_q, ext_g)
        elif stage == 'summary':
            if not _DRY_RUN:
                run_summary(out_root, args)

    print(f'\n{BANNER}')
    print(f'# 全链路验证{"命令预演 (dry-run)" if _DRY_RUN else "完成"}, '
          f'总用时 {(time.time() - t_total) / 60:.1f} 分钟')
    print(f'# 输出目录: {os.path.abspath(out_root)}')
    print(BANNER)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[中断] 用户中断。已完成的图像/查询均已增量保存, '
              '重新执行同一条命令即可断点续跑。')
        raise SystemExit(130)
