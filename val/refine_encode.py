# 模块名: refine_encode.py
# 职责: 逐图隐私残差精修（编码期测试时优化，方案 A）。
#   加载训练好的 checkpoint，冻结全部网络参数，仅对单张图像的 96x96 低分辨率
#   隐私残差场做少量 Adam 步优化。目标函数:
#     L = w_privacy * (CLIP/DINO escape) + w_message * (干净+含噪 BCE) + w_visual * (MSE+SSIM)
#   收敛后可选色度蒙层事后投影: 剔除 prv 的 8x8 网格低频色度分量（"发灰/蒙层"主体，
#   梯度惩罚实测无效——逃逸方向与色度方向高度平行，只能结构化剔除；与
#   validation_script.refine_encoded_image 同配方同门控）。
#   约束链与训练完全同构: 96 低通 -> 双三次上采样 -> tanh*prv_budget -> 纹理掩码 -> clamp。
#   消息残差 msg_residual 完全取自冻结网络，不参与优化，bit_acc 由 BCE 项硬保证。
#   不修改任何训练代码与模型权重；产出可直接与 validation_script 相同口径的指标。
# 用法示例:
#   python val/refine_encode.py -c other/checkpoint.pth -i other/test.jpg other/test2.jpg \
#       -m "101000100110100101100011100010" -o other/refine_output --device cpu

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
import torch.nn.functional as F

from validation_script import ValidationScript, _project_out_chroma_lowfreq, _delta_e_map
from losses.advanced_privacy_loss import FeatureRedirectionLoss
from noise_layers.noiser import Noiser


def build_noiser(cfg, device):
    # 与 trainer.py 相同的噪声配置解析逻辑
    from noise_argparser import parse_crop, parse_cropout, parse_dropout, parse_resize
    from noise_layers.jpeg_compression import JpegCompression
    from noise_layers.quantization import Quantization

    layers = []
    if cfg.noise_config != 'identity':
        for command in cfg.noise_config.split('+'):
            command = command.replace(' ', '')
            if command.startswith('cropout'):
                layers.append(parse_cropout(command))
            elif command.startswith('crop'):
                layers.append(parse_crop(command))
            elif command.startswith('dropout'):
                layers.append(parse_dropout(command))
            elif command.startswith('resize'):
                layers.append(parse_resize(command))
            elif command.startswith('jpeg'):
                layers.append(JpegCompression(device))
            elif command.startswith('quant'):
                layers.append(Quantization(device))
    return Noiser(layers, device).to(device)


def extract_dino_features_grad(vs, image):
    # 与 ValidationScript.extract_dino_features 相同的预处理，但保留输入梯度
    if vs.dino_model is None:
        return None
    image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    x = (image_resized + 1) / 2
    x = (x - vs.dino_mean) / vs.dino_std
    feats = vs.dino_model(x)
    return feats / (feats.norm(dim=-1, keepdim=True) + 1e-7)


@torch.no_grad()
def compute_metrics(vs, cover, message, encoded, feat_cover_clip, feat_cover_dino, prv=None):
    # 与 validation_script.py 完全相同口径的指标计算
    feat_enc = vs.content_recognizer(encoded)
    clip_sim = F.cosine_similarity(feat_cover_clip, feat_enc, dim=1).item()
    dino_sim = None
    if feat_cover_dino is not None:
        dino_enc = extract_dino_features_grad(vs, encoded)
        dino_sim = F.cosine_similarity(feat_cover_dino, dino_enc, dim=1).item()
    logits = vs.decoder(encoded)
    pred = (torch.sigmoid(logits) > 0.5).float()
    bit_acc = (pred == message).float().mean().item()
    mse = torch.mean((encoded - cover) ** 2).item()
    psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')
    ssim = 1 - vs.ssim_loss(cover, encoded).item()
    result = {
        'clip_sim': clip_sim, 'dino_sim': dino_sim, 'bit_acc': bit_acc,
        'psnr': psnr, 'ssim': ssim
    }
    if prv is not None:
        result['prv_rms'] = prv.pow(2).mean().sqrt().item()
        # 色度 RMS（蒙层监测指标，与 validation_script._refine_metrics 同口径）
        r, g, b = prv[:, 0:1], prv[:, 1:2], prv[:, 2:3]
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        result['chroma_rms'] = torch.sqrt((cb ** 2 + cr ** 2).mean()).item()
    return result


@torch.no_grad()
def noise_robustness_eval(vs, noiser, cover, message, encoded, feat_cover_clip, k):
    # 信道鲁棒性: 随机采样 k 次噪声实现，统计解码率与信道后 CLIP 相似度
    accs, sims = [], []
    for _ in range(k):
        noised_enc, _ = noiser((encoded, cover))
        pred = (torch.sigmoid(vs.decoder(noised_enc)) > 0.5).float()
        accs.append((pred == message).float().mean().item())
        feat_n = vs.content_recognizer(noised_enc)
        sims.append(F.cosine_similarity(feat_cover_clip, feat_n, dim=1).item())
    return {
        'bit_acc_mean': float(np.mean(accs)), 'bit_acc_min': float(np.min(accs)),
        'clip_sim_mean': float(np.mean(sims)), 'clip_sim_max': float(np.max(sims))
    }


def refine_image(vs, cfg, cover, message, args, noiser):
    device = vs.device
    H, W = cover.shape[-2], cover.shape[-1]
    # 精修配方默认统一取 Config.refine_*（2026-09 视觉-逃逸解耦配方），CLI 显式传参覆盖
    # （热启动比例与新约束链自动一致，无需额外处理）
    steps = args.steps if getattr(args, 'steps', None) is not None else getattr(cfg, 'refine_steps', 60)
    lr = args.lr if getattr(args, 'lr', None) is not None else 0.1
    w_privacy = args.w_privacy if getattr(args, 'w_privacy', None) is not None else getattr(cfg, 'refine_w_privacy', 12.0)
    w_message = args.w_message if getattr(args, 'w_message', None) is not None else getattr(cfg, 'refine_w_message', 10.0)
    w_visual = args.w_visual if getattr(args, 'w_visual', None) is not None else getattr(cfg, 'refine_w_visual', 15.0)
    w_de = args.w_de if getattr(args, 'w_de', None) is not None else getattr(cfg, 'refine_w_de', 0.0)
    de_tau = args.de_tau if getattr(args, 'de_tau', None) is not None else getattr(cfg, 'refine_de_tau', 2.0)
    budget = args.prv_budget if getattr(args, 'prv_budget', None) else cfg.prv_budget

    # ===== 冻结全部网络，准备常量 =====
    for module in (vs.encoder, vs.decoder):
        for p in module.parameters():
            p.requires_grad = False

    with torch.no_grad():
        encoded_base, _, msg_residual, prv_residual = vs.encoder(
            cover, message, phase=2, prv_noise_ratio=0.0
        )
        feat_cover_clip = vs.content_recognizer(cover)
        feat_cover_dino = extract_dino_features_grad(vs, cover)
        texture_mask = vs.encoder._compute_texture_mask(cover)
        base_metrics = compute_metrics(vs, cover, message, encoded_base,
                                       feat_cover_clip, feat_cover_dino, prv=prv_residual)

    # ===== 精修变量: 96x96 低分辨率残差场，从网络已学残差热启动 =====
    with torch.no_grad():
        denom = budget * texture_mask
        ratio = (prv_residual / (denom + 1e-8)).clamp(-0.95, 0.95)
        pre_tanh = torch.atanh(ratio)
        delta0 = F.adaptive_avg_pool2d(pre_tanh, (cfg.prv_lowres_size, cfg.prv_lowres_size))

    delta = torch.nn.Parameter(delta0.clone())
    cur_lr = lr
    opt = torch.optim.Adam([delta], lr=cur_lr)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()
    # escape margin 可 CLI 覆盖：二次 Hinge 在 sim < (1-margin) 后梯度归零，
    # margin 决定逃逸地板——放大 margin 是压穿 0.70 平台的最直接杠杆。
    # None 取 Config.refine_margin_end（与训练期 privacy_margin_end 独立）
    margin_end = (args.margin_end if getattr(args, 'margin_end', None) is not None
                  else getattr(cfg, 'refine_margin_end', cfg.privacy_margin_end))
    # 色度蒙层事后投影：None 取 Config 默认（False = 关闭）
    chroma_projection = (args.chroma_projection
                         if getattr(args, 'chroma_projection', None) is not None
                         else getattr(cfg, 'refine_chroma_projection', False))
    escape = FeatureRedirectionLoss(
        clip_weight=1.0, dino_weight=cfg.dino_weight,
        escape_margin=margin_end,
        clip_redirect_weight=0.0, dino_redirect_weight=0.0
    )
    clean_ratio = cfg.get_msg_clean_ratio(2, 1)

    def build_encoded(d):
        prv_up = F.interpolate(d, size=(H, W), mode='bicubic', align_corners=False)
        prv = budget * torch.tanh(prv_up) * texture_mask
        encoded = torch.clamp(cover + msg_residual + prv, -1, 1)
        return encoded, prv

    log = []
    best = None  # (clip_sim, delta克隆, 指标dict)
    t0 = time.time()
    stop_step = None  # 提前终止发生的步数 (跑满 steps 则为 None)
    guard_fails = 0  # 数值防护触发次数
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        encoded, prv = build_encoded(delta)

        feat_enc = vs.content_recognizer(encoded)
        dino_enc = extract_dino_features_grad(vs, encoded) if feat_cover_dino is not None else None
        L_priv, pinfo = escape(
            feat_cover_clip, feat_enc,
            dino_cover=feat_cover_dino, dino_encoded=dino_enc,
            escape_margin=margin_end,
            dino_escape_margin=cfg.dino_escape_margin_end
        )

        logits_clean = vs.decoder(encoded)
        L_msg = clean_ratio * bce(logits_clean, message)

        # ===== 提前终止: 地板以下 CLIP hinge 梯度归零，触底后继续迭代只剩
        # 消息/视觉项往回拉，纯浪费步数。条件: 干净分支 bit 100% 且
        # clip_sim ≤ 地板+0.005 (sim/bit 均取自本步已算好的前向, 检查零开销,
        # 且放在反向传播之前——触底那步连 backward 都省掉)。
        # 容差 0.005: 全量实测容差 0.02 会把逐图 sim 钉在地板+0.02,
        # 直接损失 ~6 个点检索逃逸率 (90%→84%), 收紧到 0.005 基本无损失
        bit_now = (torch.sigmoid(logits_clean) > 0.5).float().eq(message).float().mean().item()
        if bit_now >= 1.0 and pinfo['clip_sim'] <= (1.0 - margin_end) + 0.005:
            with torch.no_grad():
                enc_stop, _ = build_encoded(delta)
                m_stop = compute_metrics(vs, cover, message, enc_stop,
                                         feat_cover_clip, feat_cover_dino, prv=prv)
            log.append({'step': step, **m_stop})
            if m_stop['bit_acc'] >= 1.0 and (best is None or m_stop['clip_sim'] < best[0]):
                best = (m_stop['clip_sim'], delta.detach().clone(), m_stop)
            print(f"    [提前终止] step {step}: clip {pinfo['clip_sim']:.4f} "
                  f"已达逃逸地板且 bit 100%, 停止优化")
            stop_step = step
            break

        noised_enc, _ = noiser((encoded, cover))
        logits_noised = vs.decoder(noised_enc)
        L_msg = L_msg + (1.0 - clean_ratio) * bce(logits_noised, message)

        L_vis = mse(encoded, cover) + vs.ssim_loss(cover, encoded)

        # 感知色差锚 (可选, w_de>0): 对 ΔE>tau 的像素二次惩罚, 压制色罩/蒙层类失真
        # (逃逸最省力方向是低频色度偏移, PSNR/SSIM 对其失明)
        if w_de > 0:
            L_dE = (torch.relu(_delta_e_map(encoded, cover) - de_tau) ** 2).mean()
        else:
            L_dE = torch.zeros((), device=cover.device)

        L = w_privacy * L_priv + w_message * L_msg + w_visual * L_vis + w_de * L_dE
        if not torch.isfinite(L):
            # 数值防护: 精修发散 (NaN/Inf) 时丢弃该步
            opt.zero_grad(set_to_none=True)
            continue
        prev_delta = delta.detach().clone()
        L.backward()
        opt.step()
        # 钳制 pre-tanh 场值域: tanh 在 |x|>4 已完全饱和, 钳制不改变有效残差,
        # 但阻止 delta 无界增长导致的步长爆炸
        torch.clamp_(delta.data, -8.0, 8.0)
        if not torch.isfinite(delta).all():
            # 数值防护: L 有限但梯度非有限时 Adam 会把优化变量写坏（终局是 NaN 图像）。
            # 回退 + 随机抖动移离病态点 + 重建优化器（非有限梯度下任何学习率都无效,
            # 抖动绕开才是有效手段）。连续失败超过上限则终止本图, 沿用最后有限状态
            delta.data.copy_(prev_delta)
            guard_fails += 1
            if guard_fails > 10:
                print(f"    [数值防护] step {step} 重试次数已用尽, 沿用最后有限状态")
                break
            delta.data.add_(torch.randn_like(delta.data) * (0.02 + 0.01 * guard_fails))
            torch.clamp_(delta.data, -8.0, 8.0)
            opt = torch.optim.Adam([delta], lr=cur_lr)
            opt.zero_grad(set_to_none=True)
            print(f"    [数值防护] step {step} 梯度非有限, 回退+抖动重试 (第 {guard_fails} 次)")
            continue

        if step == 1 or step % args.log_interval == 0:
            with torch.no_grad():
                enc_eval, prv_eval = build_encoded(delta)
                m = compute_metrics(vs, cover, message, enc_eval,
                                    feat_cover_clip, feat_cover_dino, prv=prv_eval)
            log.append({'step': step, **m})
            dino_str = format(m['dino_sim'], '.4f') if m['dino_sim'] is not None else '  --  '
            chroma_str = (format(m['chroma_rms'], '.4f')
                          if m.get('chroma_rms') is not None else '  --  ')
            print(f"    step {step:3d} | clip {m['clip_sim']:.4f} | dino {dino_str} | "
                  f"bit {m['bit_acc']*100:5.1f}% | psnr {m['psnr']:5.2f} | ssim {m['ssim']:.4f} | "
                  f"prv_rms {m['prv_rms']:.4f} | chroma_rms {chroma_str}")
            # 只在消息 100% 正确的状态里挑 clip_sim 最低者
            if m['bit_acc'] >= 1.0 and (best is None or m['clip_sim'] < best[0]):
                best = (m['clip_sim'], delta.detach().clone(), m)

    refine_time = time.time() - t0
    final_delta = best[1] if best is not None else delta.detach().clone()
    if not torch.isfinite(final_delta).all():
        # 兜底: 产出绝不允许 NaN，回退网络自身残差
        final_delta = delta0.clone()
    with torch.no_grad():
        encoded_refined, prv_refined = build_encoded(final_delta)
        # 色度蒙层事后投影: 剔除 prv 的 8x8 网格低频色度（与 validation_script
        # .refine_encoded_image 同配方）。门控: 投影后 bit_acc 必须 100% 且
        # clip_sim 仍 ≤ (1-margin) 逃逸目标，否则回退未投影结果
        projected = False
        if chroma_projection:
            prv_proj = _project_out_chroma_lowfreq(prv_refined)
            enc_proj = torch.clamp(cover + msg_residual + prv_proj, -1, 1)
            m_proj = compute_metrics(vs, cover, message, enc_proj,
                                     feat_cover_clip, feat_cover_dino, prv=prv_proj)
            if m_proj['bit_acc'] >= 1.0 and m_proj['clip_sim'] <= 1.0 - margin_end:
                encoded_refined, prv_refined = enc_proj, prv_proj
                projected = True
        refined_metrics = compute_metrics(vs, cover, message, encoded_refined,
                                          feat_cover_clip, feat_cover_dino, prv=prv_refined)

    return {
        'encoded_base': encoded_base, 'encoded_refined': encoded_refined,
        'base_metrics': base_metrics, 'refined_metrics': refined_metrics,
        'log': log, 'time_sec': refine_time, 'chroma_projection_applied': projected,
        'stop_step': stop_step,
        'feat_cover_clip': feat_cover_clip, 'feat_cover_dino': feat_cover_dino
    }


def main():
    parser = argparse.ArgumentParser(description='逐图隐私残差精修 (per-image refinement)')
    parser.add_argument('--checkpoint', '-c', type=str, default='./other/checkpoint.pth')
    parser.add_argument('--image', '-i', type=str, nargs='+', required=True, help='载体图像路径(可多张)')
    parser.add_argument('--message', '-m', type=str,
                        default='101000100110100101100011100010')
    parser.add_argument('--output-dir', '-o', type=str, default='./other/refine_output')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--steps', type=int, default=None,
                        help='精修步数 (None=取 Config.refine_steps, 默认 150)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Adam 学习率 (None=默认 0.1)')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--w-privacy', type=float, default=None,
                        help='隐私(逃逸)锚权重 (None=取 Config.refine_w_privacy, 默认 12.0)')
    parser.add_argument('--w-message', type=float, default=None,
                        help='消息锚权重 (None=取 Config.refine_w_message, 默认 20.0)')
    parser.add_argument('--w-visual', type=float, default=None,
                        help='视觉锚权重 (None=取 Config.refine_w_visual, 默认 15.0)')
    parser.add_argument('--prv-budget', type=float, default=None,
                        help='覆盖隐私残差预算 (None=取 Config.prv_budget, 默认 0.08 深挖配方)；'
                             '逃逸深度近似随预算平方增长')
    parser.add_argument('--margin-end', type=float, default=None,
                        help='覆盖 CLIP escape margin (None=取 Config.refine_margin_end, '
                             '默认 0.60)。二次 Hinge 梯度在 sim<(1-margin) 后归零，'
                             'margin 即逃逸地板')
    parser.add_argument('--chroma-projection', type=int, default=None,
                        help='收敛后剔除 prv 低频色度蒙层 (默认取 Config.refine_chroma_projection; '
                             '1 开启 / 0 关闭)。门控: bit_acc 须 100%% 且 clip_sim 达标, 否则回退。'
                             '注意: 深逃逸(margin>=0.55)下投影门控基本不会满足, 且投影直接'
                             '削掉逃逸方向, 推荐保持关闭')
    parser.add_argument('--w-de', type=float, default=None,
                        help='感知色差锚权重 (None=取 Config.refine_w_de, 默认 200)。对 ΔE>tau 的像素二次惩罚, '
                             '压制 PSNR/SSIM 测不到的色罩/蒙层类失真')
    parser.add_argument('--de-tau', type=float, default=None,
                        help='感知色差锚阈值 (None=取 Config.refine_de_tau, 默认 4.0)')
    parser.add_argument('--noise-eval', type=int, default=8, help='信道鲁棒性采样次数')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device) if args.device else None

    vs = ValidationScript(checkpoint_path=args.checkpoint, device=device)
    cfg = vs.cfg
    noiser = build_noiser(cfg, vs.device)

    message_tensor, message_str, original_length = vs.process_message(args.message)
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for image_path in args.image:
        name = os.path.splitext(os.path.basename(image_path))[0]
        print('\n' + '#' * 60)
        print(f'# 精修图像: {image_path}')
        print('#' * 60)

        cover = vs.load_image(image_path)
        out = refine_image(vs, cfg, cover, message_tensor, args, noiser)

        # 信道鲁棒性对比（基线 vs 精修）
        noise_base = noise_robustness_eval(vs, noiser, cover, message_tensor,
                                           out['encoded_base'], out['feat_cover_clip'],
                                           args.noise_eval)
        noise_refined = noise_robustness_eval(vs, noiser, cover, message_tensor,
                                              out['encoded_refined'], out['feat_cover_clip'],
                                              args.noise_eval)

        # uint8 往返（PNG 保存->重载->解码），真实部署条件下的最终门禁
        img_dir = os.path.join(args.output_dir, name)
        os.makedirs(img_dir, exist_ok=True)
        base_png = os.path.join(img_dir, 'encoded_baseline.png')
        refined_png = os.path.join(img_dir, 'encoded_refined.png')
        vs.save_image(out['encoded_base'], base_png)
        vs.save_image(out['encoded_refined'], refined_png)
        with torch.no_grad():
            reloaded = vs.load_image(refined_png)
            pred = (torch.sigmoid(vs.decoder(reloaded)) > 0.5).float()
            roundtrip_bit_acc = (pred == message_tensor).float().mean().item()
            feat_re = vs.content_recognizer(reloaded)
            roundtrip_clip = F.cosine_similarity(out['feat_cover_clip'], feat_re, dim=1).item()

        record = {
            'base': out['base_metrics'], 'refined': out['refined_metrics'],
            'noise_base': noise_base, 'noise_refined': noise_refined,
            'roundtrip': {'bit_acc': roundtrip_bit_acc, 'clip_sim': roundtrip_clip},
            'log': out['log'], 'time_sec': out['time_sec'],
            'chroma_projection_applied': out['chroma_projection_applied']
        }
        all_results[name] = record
        with open(os.path.join(img_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        b, r = out['base_metrics'], out['refined_metrics']
        chroma_b = b.get('chroma_rms')
        chroma_r = r.get('chroma_rms')
        chroma_note = (f" | chroma {chroma_b:.4f}->{chroma_r:.4f}"
                       if chroma_b is not None and chroma_r is not None else "")
        proj_note = " | 色度投影已应用" if out['chroma_projection_applied'] else ""
        print(f"  基线:  clip {b['clip_sim']:.4f} | dino {b['dino_sim']:.4f} | bit {b['bit_acc']*100:.2f}% | "
              f"psnr {b['psnr']:.2f} | ssim {b['ssim']:.4f}")
        print(f"  精修:  clip {r['clip_sim']:.4f} | dino {r['dino_sim']:.4f} | bit {r['bit_acc']*100:.2f}% | "
              f"psnr {r['psnr']:.2f} | ssim {r['ssim']:.4f}{chroma_note}{proj_note}")
        print(f"  信道后(基线):  bit_mean {noise_base['bit_acc_mean']*100:.1f}% min {noise_base['bit_acc_min']*100:.1f}% | "
              f"clip_mean {noise_base['clip_sim_mean']:.4f}")
        print(f"  信道后(精修):  bit_mean {noise_refined['bit_acc_mean']*100:.1f}% min {noise_refined['bit_acc_min']*100:.1f}% | "
              f"clip_mean {noise_refined['clip_sim_mean']:.4f}")
        print(f"  uint8往返(精修): bit {roundtrip_bit_acc*100:.2f}% | clip {roundtrip_clip:.4f} | 用时 {out['time_sec']:.0f}s")

    # ===== 汇总表 =====
    print('\n' + '#' * 60)
    print('# 汇总 (基线 -> 精修)')
    print('#' * 60)
    header = f"{'图像':<12} {'clip_sim':<18} {'dino_sim':<18} {'bit_acc':<15} {'PSNR':<14} {'SSIM':<16}"
    print(header)
    for name, rec in all_results.items():
        b, r = rec['base'], rec['refined']
        print(f"{name:<12} {b['clip_sim']:.4f}->{r['clip_sim']:.4f}   "
              f"{b['dino_sim']:.4f}->{r['dino_sim']:.4f}   "
              f"{b['bit_acc']*100:.0f}%->{r['bit_acc']*100:.0f}%   "
              f"{b['psnr']:.1f}->{r['psnr']:.1f}   {b['ssim']:.4f}->{r['ssim']:.4f}")

    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.output_dir}")


if __name__ == '__main__':
    main()
