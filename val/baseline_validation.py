# 模块名: baseline_validation.py
# 模块职责: PrivSteg 项目的单图基线验证脚本（网络直出，无逐图精修）。
# 流程: 加载图像与消息 -> 网络单次前向编码 -> 解码验证 -> 判别器预测 ->
#   CLIP/DINO/SSIM/PSNR 相似度 -> 保存编码图。全程不做逐图精修、不生成
#   report.html，直接输出网络一次前向的编码图像与全部指标。
# 用途: 作为 validation_script.py full 模式（默认含 30 步精修）的基线对照，
#   量化"网络单独"与"网络+编码期精修"的差距（amortization gap 分析），
#   或快速检查某个 checkpoint 的原生质量。
# 用法示例:
#   python val/baseline_validation.py -c other/checkpoint.pth -i other/test.jpg \
#       -m "101000100110100101100011100010" --device cpu -o other/baseline_output

import os
import sys
import argparse

# 本脚本位于 val/ 子目录：定位项目根并加入 sys.path（import models/losses/config 等根模块），
# 同时把工作目录切到项目根，保证 ./models/dinov2_vits14_pretrain.pth 等相对路径
# 与从项目根直接运行时的行为完全一致（与 batch_eval.py 相同的启动约定）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import torch

from validation_script import ValidationScript


# main 函数功能: CLI 入口，加载 checkpoint 后执行网络直出（无精修）的完整验证。
# 输出格式与 validation_script.py full 模式同口径，便于并排对照。
def main():
    parser = argparse.ArgumentParser(description='PrivSteg 单图基线验证（网络直出，不做逐图精修）')

    # --checkpoint/-c: 模型 checkpoint 路径
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='检查点文件路径')
    # --image/-i: 必填，载体图像路径
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='载体图像路径')
    # --message/-m: 待嵌入的二进制消息（字符串或文件路径）
    parser.add_argument('--message', '-m', type=str, default='0' * 30,
                        help='消息 (二进制字符串或消息文件路径)')
    # --output-dir/-o: 输出目录
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='输出目录 (默认在原图旁创建 <name>_baseline_output)')
    # --device: 指定运行设备
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')

    args = parser.parse_args()

    # 未提供 checkpoint 时给出警告（随机权重的验证结果无意义）
    if args.checkpoint is None:
        print('警告: 未提供检查点，将使用随机初始化的权重，验证结果无意义!')
        print('建议使用: -c <checkpoint_path> 指定检查点路径')

    device = torch.device(args.device) if args.device else None
    validator = ValidationScript(checkpoint_path=args.checkpoint, device=device)

    print('\n' + '#' * 60)
    print('# 基线验证流程（网络直出，无逐图精修）')
    print('#' * 60)

    # 1. 加载载体图像与消息
    cover_image = validator.load_image(args.image)
    message_tensor, message_str, original_length = validator.process_message(args.message)

    # 2. 网络单次前向编码——不做任何精修，直接输出
    encoded_image, residual_penalty = validator.encode_image(cover_image, message_tensor)

    # 3. 解码并验证消息比特
    decoded_message, decoded_prob = validator.decode_message(encoded_image)
    bit_accuracy, correct_bits, total_bits = validator.verify_message_extraction(
        message_tensor, decoded_message, original_length
    )

    # 4. 判别器对原图与编码图的隐写检测预测
    cover_prob, _ = validator.discriminator_predict(cover_image, '原始图像')
    stego_prob, _ = validator.discriminator_predict(encoded_image, '隐写图像')

    # 5. CLIP/DINO/SSIM/PSNR 隐私相似度评估
    privacy_results = validator.privacy_similarity(cover_image, encoded_image)

    # 6. 保存原图与编码图
    if args.output_dir is None:
        image_dir = os.path.dirname(os.path.abspath(args.image))
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output_dir = os.path.join(image_dir, f'{image_name}_baseline_output')
    os.makedirs(args.output_dir, exist_ok=True)
    cover_save_path = os.path.join(args.output_dir, 'cover_image.png')
    encoded_save_path = os.path.join(args.output_dir, 'encoded_image.png')
    validator.save_image(cover_image, cover_save_path)
    validator.save_image(encoded_image, encoded_save_path)
    print(f'\n输出目录: {args.output_dir}')

    # 汇总（与 validation_script.py full 模式同口径，便于对照）
    print('\n' + '#' * 60)
    print('# 基线验证结果汇总（网络直出，无精修）')
    print('#' * 60)
    print(f'消息提取准确率: {bit_accuracy * 100:.2f}%')
    print(f'判别器预测 - 原图隐写概率: {cover_prob * 100:.2f}%')
    print(f'判别器预测 - 隐写图隐写概率: {stego_prob * 100:.2f}%')
    print(f'CLIP 隐私相似度: {privacy_results["similarity_probability"] * 100:.2f}%')
    print(f'CLIP 隐私保护程度: {(1 - privacy_results["similarity_probability"]) * 100:.2f}%')
    if privacy_results.get('dino_similarity') is not None:
        print(f'DINO 隐私相似度: {privacy_results["dino_similarity_probability"] * 100:.2f}%')
        print(f'DINO 隐私保护程度: {(1 - privacy_results["dino_similarity_probability"]) * 100:.2f}%')
    print(f'PSNR: {privacy_results["psnr"]:.2f} dB | SSIM: {privacy_results["ssim"]:.6f}')
    print('#' * 60)
    print('对照提示: 本脚本为网络单次前向的基线结果；'
          'validation_script.py full 模式默认含 30 步逐图精修（深挖配方 budget 0.08/margin 0.50）。')


if __name__ == '__main__':
    main()
