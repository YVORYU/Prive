# 模块名: run_full_validation.py
# 模块职责: 全链路验证编排脚本（论文实验一键入口）。
#   在同一 checkpoint / 同一测试集 / 同一 seed 上按顺序编排 val/ 下的评估脚本:
#     1. smoke              冒烟自检 (3 张图 + 3 查询/20 图库 + 1 张精修, 校验链路后再跑全量)
#     2. single             单图 full 全链路 (编码->精修->解码->判别->相似度 + report.html)
#     3. batch_baseline     批量基线 (网络直出, 快)
#     4. retrieval_baseline 检索基线 (网络直出, 快; 顺带构建图库特征缓存)
#     5. batch_eval         批量精修评估 (慢; 与 3 同 seed 同图像集, 直接对照)
#     6. retrieval_eval     检索精修评估 (慢; 复用 4 的图库特征缓存)
#     7. summary           汇总四个 summary.json -> full_summary.json / full_summary.md
#   设计要点:
#     - 快阶段先行 (失败早暴露), 慢的精修阶段放最后; 各子脚本自带断点续跑
#       (results.json 增量保存), 任一阶段中断后重新执行同一条命令即可续跑。
#     - 结束时输出网络直出 / 精修后的对照表 (控制台中文, full_summary.md 英文可直接进论文),
#       并做两路直出结果的一致性交叉检查 (seed/val-dir 配错时给出警告)。
# 用法示例 (云服务器项目根):
#   python val/run_full_validation.py -c ./experiments/full_run/checkpoints/checkpoint_best.pth \
#       --val-dir ./dataset/test --output-root ./other/full_validation

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
               'batch_eval', 'retrieval_eval', 'summary']
BANNER = '=' * 78

_DRY_RUN = False


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
        'derived': {'batch_table': to_dicts(b_rows), 'retrieval_table': to_dicts(r_rows)},
    }
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
    md += ['']
    if sanity is not None:
        md.append(f"Self-sanity (cover self-query top-1): {sanity * 100:.1f}% (expected 100%).")
    if direct is not None and refined is not None:
        md.append(f"CLIP escape: encoder direct {direct:.4f} -> per-image refined {refined:.4f}. "
                  "A lower direct value means the inner-loop distillation successfully amortized "
                  "the per-instance refinement.")
    md += ['', '## 3. Artifacts', '',
           '- `single/report.html` - Single-image full pipeline report',
           '- `batch_baseline/report.html` - Batch baseline report (encoder direct)',
           '- `retrieval_baseline/report.html` - Retrieval baseline report (encoder direct)',
           '- `batch_eval/report.html` - Batch refined report (with baseline comparison)',
           '- `retrieval_eval/report.html` - Retrieval refined report (with baseline comparison)',
           '- `full_summary.json` - Consolidated machine-readable summary', '']
    md_path = os.path.join(out_root, 'full_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f'\n汇总已保存: {json_path}')
    print(f'论文用英文汇总: {md_path} (表格可直接粘贴进论文)')

    print('\n产出文件 (输出目录内):')
    for sub_name, desc in [('single', '单图全链路可视化报告'),
                           ('batch_baseline', '批量基线报告 (网络直出)'),
                           ('retrieval_baseline', '检索基线报告 (网络直出)'),
                           ('batch_eval', '批量精修报告 (含基线对照)'),
                           ('retrieval_eval', '检索精修报告 (含基线对照)')]:
        report = os.path.join(out_root, sub_name, 'report.html')
        mark = '' if os.path.exists(report) else ' (未生成: 阶段未运行)'
        print(f'  {sub_name}/report.html - {desc}{mark}')
    if be is not None:
        print('\n论文提示: 双配方权衡表可用 --refine-budget 0.04 --refine-margin 0.30 '
              '换一个 --output-root 再跑一遍, 构成深挖/标准配方对照。')


# ============================ 入口 ============================

def main():
    global _DRY_RUN
    parser = argparse.ArgumentParser(
        description='全链路验证编排: 冒烟自检 -> 单图 full -> 批量/检索的基线与精修 -> 汇总对照表')
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
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
        run_summary(out_root, args)
        return

    # ---- 预检 ----
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f'[中止] checkpoint 不存在: {args.checkpoint}')
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

    if args.stages == 'all':
        stages = list(STAGE_ORDER)
    else:
        stages = [s.strip() for s in args.stages.split(',') if s.strip()]
        for s in stages:
            if s not in STAGE_ORDER:
                raise SystemExit(f'[中止] 未知阶段: {s} (可选: {", ".join(STAGE_ORDER)})')
    ordered = [s for s in STAGE_ORDER if s in set(stages)]

    # ---- 计划 ----
    rc = _effective_refine_cfg(args)
    print(BANNER)
    print('# PrivSteg 全链路验证')
    print(BANNER)
    print(f'checkpoint : {args.checkpoint}')
    print(f'测试集     : {args.val_dir} ({n_img} 张)')
    print(f'评估规模   : 批量 {n_eval} 张 | 检索 查询 {args.num_queries} / 图库 {gallery} 张 '
          f'(随机 top-1 命中概率 {100.0 / gallery:.2f}%)')
    print(f'配置       : seed={args.seed} | 精修 steps={rc["steps"]} '
          f'budget={rc["budget"]} margin={rc["margin"]} w_de={rc["w_de"]} '
          f'de_tau={rc["de_tau"]} w_message={rc["w_message"]} w_privacy={rc["w_privacy"]} '
          f'chroma_projection={args.refine_chroma_projection if args.refine_chroma_projection is not None else "cfg默认"} '
          f'| noise-eval={args.noise_eval}')
    print(f'device     : {args.device or "auto (子脚本自动选择 CUDA/CPU)"}')
    print(f'输出目录   : {os.path.abspath(out_root)}')
    print(f'阶段       : {", ".join(ordered)}')
    print('阶段成本   : 冒烟/单图/两个基线为轻量; batch_eval / retrieval_eval 含逐图精修, '
          '为主要耗时 (约为基线版 20~30 倍)')
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
