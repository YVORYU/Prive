import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import timm

from models import Encoder, Decoder, Steganalyzer, ContentRecognizer
from losses import SSIMLoss
import config


class ValidationScript:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = config.Config()

        self.encoder = Encoder(
            message_length=self.cfg.message_length,
            n_channels=3,
            image_size=self.cfg.image_size,
            msg_budget=self.cfg.msg_budget,
            prv_budget=self.cfg.prv_budget
        ).to(self.device)

        self.decoder = Decoder(
            message_length=self.cfg.message_length,
            n_channels=3
        ).to(self.device)

        self.steganalyzer = Steganalyzer(n_channels=3, use_spectral_norm=True).to(self.device)

        self.content_recognizer = ContentRecognizer(device=self.device).to(self.device)

        self.ssim_loss = SSIMLoss().to(self.device)

        # 加载 DINOv2 模型
        self.dino_model = None
        self.dino_mean = None
        self.dino_std = None
        dino_model_path = getattr(self.cfg, 'dino_model_path', None)
        if getattr(self.cfg, 'dino_weight', 0) > 0:
            self.dino_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=224)
            if dino_model_path and os.path.exists(dino_model_path):
                state_dict = torch.load(dino_model_path, map_location=self.device, weights_only=False)
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']

                model_pos_embed = self.dino_model.pos_embed.data
                ckpt_pos_embed = state_dict.get('pos_embed', None)
                if ckpt_pos_embed is not None and ckpt_pos_embed.shape != model_pos_embed.shape:
                    cls_token = ckpt_pos_embed[:, :1, :]
                    patch_embed = ckpt_pos_embed[:, 1:, :]
                    embed_dim = patch_embed.shape[-1]
                    orig_grid_size = int(patch_embed.shape[1] ** 0.5)
                    new_grid_size = int((model_pos_embed.shape[1] - 1) ** 0.5)
                    patch_embed = patch_embed.reshape(1, orig_grid_size, orig_grid_size, embed_dim).permute(0, 3, 1, 2)
                    patch_embed = F.interpolate(patch_embed, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
                    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, embed_dim)
                    state_dict['pos_embed'] = torch.cat([cls_token, patch_embed], dim=1)

                self.dino_model.load_state_dict(state_dict, strict=False)
                print(f"DINOv2 模型已加载: {dino_model_path}")
            else:
                print(f"警告: DINOv2 模型未找到: {dino_model_path}，使用随机初始化")
            self.dino_model = self.dino_model.to(self.device)
            self.dino_model.eval()
            for param in self.dino_model.parameters():
                param.requires_grad = False
            self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        else:
            print("DINOv2 未启用 (dino_weight=0)")

        self.transform = transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
        ])

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if 'steganalyzer_state_dict' in checkpoint:
            steganalyzer_state_dict = checkpoint['steganalyzer_state_dict'].copy()

            old_keys = ['fc_layers.3.weight', 'fc_layers.3.bias']
            new_keys = ['fc_layers.2.weight', 'fc_layers.2.bias']

            for old_key, new_key in zip(old_keys, new_keys):
                if old_key in steganalyzer_state_dict:
                    steganalyzer_state_dict[new_key] = steganalyzer_state_dict.pop(old_key)
                    print(f"映射键: {old_key} -> {new_key}")

            self.steganalyzer.load_state_dict(steganalyzer_state_dict, strict=False)

        self.encoder.eval()
        self.decoder.eval()
        self.steganalyzer.eval()
        self.content_recognizer.eval()
        print("模型加载完成!")

    def load_image(self, image_path):
        print(f"加载图像: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        print(f"图像形状: {image_tensor.shape}")
        return image_tensor

    def process_message(self, message_input):
        if os.path.isfile(message_input):
            with open(message_input, 'r') as f:
                message_str = f.read().strip()
        else:
            message_str = message_input

        message_str = ''.join(c for c in message_str if c in '01')

        original_length = len(message_str)
        target_length = self.cfg.message_length

        if original_length < target_length:
            message_str = message_str + '0' * (target_length - original_length)
            print(f"消息长度 {original_length} < 目标长度 {target_length}，已自动补0")
        elif original_length > target_length:
            message_str = message_str[:target_length]
            print(f"消息长度 {original_length} > 目标长度 {target_length}，已自动截断")

        print(f"最终消息长度: {len(message_str)}")
        print(f"消息内容: {message_str}")

        message_tensor = torch.tensor([float(c) for c in message_str], dtype=torch.float32).unsqueeze(0).to(self.device)
        return message_tensor, message_str, original_length

    def encode_image(self, cover_image, message):
        print("\n" + "="*60)
        print("开始编码...")

        with torch.no_grad():
            encoded_image, residual_penalty, msg_residual, prv_residual = self.encoder(
                cover_image, message, phase=2, prv_noise_ratio=0.0
            )

        print(f"残差惩罚值: {residual_penalty.item():.6f}")
        return encoded_image, residual_penalty

    def decode_message(self, encoded_image):
        print("\n" + "="*60)
        print("开始解码...")

        with torch.no_grad():
            message_pred = self.decoder(encoded_image)

        message_pred_sigmoid = torch.sigmoid(message_pred)
        message_pred_binary = (message_pred_sigmoid > 0.5).float()

        return message_pred_binary, message_pred_sigmoid

    def verify_message_extraction(self, original_message, decoded_message, original_length=None):
        print("\n" + "="*60)
        print("消息提取验证:")

        original_binary = original_message.squeeze().cpu().numpy()
        decoded_binary = decoded_message.squeeze().cpu().numpy()

        if original_length is not None and original_length < len(original_binary):
            original_binary = original_binary[:original_length]
            decoded_binary = decoded_binary[:original_length]
            print(f"(仅计算原始输入的 {original_length} 位，不包括补位0)")

        correct_bits = np.sum(original_binary == decoded_binary)
        total_bits = len(original_binary)
        bit_accuracy = correct_bits / total_bits

        print(f"原始消息: {''.join(str(int(b)) for b in original_binary)}")
        print(f"解码消息: {''.join(str(int(b)) for b in decoded_binary)}")
        print(f"比特准确率: {bit_accuracy*100:.2f}%")
        print(f"正确比特数: {correct_bits}/{total_bits}")
        print(f"错误比特数: {total_bits - correct_bits}")

        return bit_accuracy, correct_bits, total_bits

    def discriminator_predict(self, image, image_name="图像"):
        print("\n" + "="*60)
        print(f"判别器预测 ({image_name}):")

        with torch.no_grad():
            prediction = self.steganalyzer(image)
            probability = torch.sigmoid(prediction)

        prob_value = probability.item()
        print(f"判别器输出 (logits): {prediction.item():.6f}")
        print(f"隐写概率 (sigmoid): {prob_value*100:.2f}%")
        print(f"判断结果: {'隐写图像' if prob_value > 0.5 else '原始图像'}")

        return prob_value, prediction.item()

    def extract_dino_features(self, image):
        if self.dino_model is None:
            return None
        with torch.no_grad():
            image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            image_normalized = (image_resized + 1) / 2
            image_normalized = (image_normalized - self.dino_mean) / self.dino_std
            features = self.dino_model(image_normalized)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-7)
        return features

    def privacy_similarity(self, cover_image, encoded_image):
        print("\n" + "="*60)
        print("隐私相似度判断:")

        with torch.no_grad():
            feat_cover = self.content_recognizer(cover_image)
            feat_encoded = self.content_recognizer(encoded_image)

        cosine_sim = F.cosine_similarity(feat_cover, feat_encoded, dim=1).item()

        euclidean_dist = torch.norm(feat_encoded - feat_cover, p=2, dim=1).item()

        # CLIP特征已L2归一化，余弦相似度范围即为[0, 1]
        similarity_prob = cosine_sim

        # DINO 相似度
        dino_sim = None
        dino_euclidean_dist = None
        if self.dino_model is not None:
            feat_cover_dino = self.extract_dino_features(cover_image)
            feat_encoded_dino = self.extract_dino_features(encoded_image)
            dino_sim = F.cosine_similarity(feat_cover_dino, feat_encoded_dino, dim=1).item()
            dino_euclidean_dist = torch.norm(feat_encoded_dino - feat_cover_dino, p=2, dim=1).item()

        ssim_value = 1 - self.ssim_loss(cover_image, encoded_image).item()

        mse = torch.mean((encoded_image - cover_image) ** 2).item()
        if mse > 0:
            psnr = 10 * np.log10(4.0 / mse)
        else:
            psnr = float('inf')

        print(f"CLIP 余弦相似度: {cosine_sim:.6f}")
        print(f"CLIP 欧氏距离: {euclidean_dist:.6f}")
        if dino_sim is not None:
            print(f"DINO 余弦相似度: {dino_sim:.6f}")
            print(f"DINO 欧氏距离: {dino_euclidean_dist:.6f}")
        print(f"SSIM: {ssim_value:.6f}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"\n隐私判断结论:")
        print(f"  - CLIP 特征相似度: {similarity_prob*100:.2f}%")
        print(f"  - CLIP 隐私保护程度: {(1-similarity_prob)*100:.2f}%")
        if dino_sim is not None:
            print(f"  - DINO 特征相似度: {dino_sim*100:.2f}%")
            print(f"  - DINO 隐私保护程度: {(1-dino_sim)*100:.2f}%")

        return {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'similarity_probability': similarity_prob,
            'dino_similarity': dino_sim,
            'dino_euclidean_distance': dino_euclidean_dist,
            'ssim': ssim_value,
            'psnr': psnr
        }

    def save_image(self, image_tensor, save_path):
        image_np = self.inverse_transform(image_tensor.squeeze(0)).cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        image.save(save_path)
        print(f"图像已保存: {save_path}")

    def run_full_validation(self, image_path, message, output_dir=None):
        print("\n" + "#"*60)
        print("# 完整验证流程")
        print("#"*60)

        cover_image = self.load_image(image_path)
        message_tensor, message_str, original_length = self.process_message(message)

        encoded_image, residual_penalty = self.encode_image(cover_image, message_tensor)

        decoded_message, decoded_prob = self.decode_message(encoded_image)

        bit_accuracy, correct_bits, total_bits = self.verify_message_extraction(
            message_tensor, decoded_message, original_length
        )

        cover_prob, cover_logit = self.discriminator_predict(cover_image, "原始图像")
        stego_prob, stego_logit = self.discriminator_predict(encoded_image, "隐写图像")

        privacy_results = self.privacy_similarity(cover_image, encoded_image)

        if output_dir is None:
            image_dir = os.path.dirname(os.path.abspath(image_path))
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.join(image_dir, f"{image_name}_stego_output")

        os.makedirs(output_dir, exist_ok=True)
        cover_save_path = os.path.join(output_dir, 'cover_image.png')
        encoded_save_path = os.path.join(output_dir, 'encoded_image.png')
        self.save_image(cover_image, cover_save_path)
        self.save_image(encoded_image, encoded_save_path)
        print(f"\n输出目录: {output_dir}")

        print("\n" + "#"*60)
        print("# 验证结果汇总")
        print("#"*60)
        print(f"消息提取准确率: {bit_accuracy*100:.2f}%")
        print(f"判别器预测 - 原图隐写概率: {cover_prob*100:.2f}%")
        print(f"判别器预测 - 隐写图隐写概率: {stego_prob*100:.2f}%")
        print(f"CLIP 隐私相似度: {privacy_results['similarity_probability']*100:.2f}%")
        print(f"CLIP 隐私保护程度: {(1-privacy_results['similarity_probability'])*100:.2f}%")
        if privacy_results.get('dino_similarity') is not None:
            print(f"DINO 隐私相似度: {privacy_results['dino_similarity']*100:.2f}%")
            print(f"DINO 隐私保护程度: {(1-privacy_results['dino_similarity'])*100:.2f}%")
        print("#"*60)

        return {
            'bit_accuracy': bit_accuracy,
            'cover_discriminator_prob': cover_prob,
            'stego_discriminator_prob': stego_prob,
            'privacy_results': privacy_results
        }

    def run_discriminator_only(self, image_path):
        print("\n" + "#"*60)
        print("# 判别器预测模式")
        print("#"*60)

        image = self.load_image(image_path)
        prob, logit = self.discriminator_predict(image, "输入图像")

        print("\n" + "#"*60)
        print("# 判别器预测结果")
        print("#"*60)
        print(f"隐写概率: {prob*100:.2f}%")
        print(f"判断结果: {'隐写图像' if prob > 0.5 else '原始图像'}")
        print("#"*60)

        return prob, logit

    def run_decode_only(self, image_path):
        print("\n" + "#"*60)
        print("# 解码模式")
        print("#"*60)

        image = self.load_image(image_path)
        decoded_message, decoded_prob = self.decode_message(image)

        decoded_binary = decoded_message.squeeze().cpu().numpy()
        message_str = ''.join(str(int(b)) for b in decoded_binary)

        print("\n" + "#"*60)
        print("# 解码结果")
        print("#"*60)
        print(f"解码消息: {message_str}")
        print(f"消息长度: {len(message_str)} bits")
        print("#"*60)

        return message_str, decoded_prob

    def run_similarity_only(self, cover_image_path, stego_image_path):
        print("\n" + "#"*60)
        print("# 相似度判断模式")
        print("#"*60)

        cover_image = self.load_image(cover_image_path)
        stego_image = self.load_image(stego_image_path)

        privacy_results = self.privacy_similarity(cover_image, stego_image)

        print("\n" + "#"*60)
        print("# 相似度判断结果")
        print("#"*60)
        print(f"CLIP 余弦相似度: {privacy_results['cosine_similarity']:.6f}")
        print(f"CLIP 欧氏距离: {privacy_results['euclidean_distance']:.6f}")
        if privacy_results.get('dino_similarity') is not None:
            print(f"DINO 余弦相似度: {privacy_results['dino_similarity']:.6f}")
            print(f"DINO 欧氏距离: {privacy_results['dino_euclidean_distance']:.6f}")
        print(f"SSIM: {privacy_results['ssim']:.6f}")
        print(f"PSNR: {privacy_results['psnr']:.2f} dB")
        print(f"\n结论:")
        print(f"  - CLIP 相似概率: {privacy_results['similarity_probability']*100:.2f}%")
        print(f"  - CLIP 隐私保护程度: {(1-privacy_results['similarity_probability'])*100:.2f}%")
        if privacy_results.get('dino_similarity') is not None:
            print(f"  - DINO 相似概率: {privacy_results['dino_similarity']*100:.2f}%")
            print(f"  - DINO 隐私保护程度: {(1-privacy_results['dino_similarity'])*100:.2f}%")
        print("#"*60)

        return privacy_results


def main():
    parser = argparse.ArgumentParser(description='PrivSteg 验证脚本')

    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='检查点文件路径 (判别器模式推荐提供)')
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='输入图像路径 (隐写图像)')
    parser.add_argument('--cover', type=str, default=None,
                        help='原图路径 (相似度判断模式必需)')
    parser.add_argument('--message', '-m', type=str, default='0' * 30,
                        help='消息 (二进制字符串或消息文件路径)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')
    parser.add_argument('--mode', type=str, choices=['full', 'discriminator', 'decode', 'similarity'], default='full',
                        help='验证模式: full=完整验证, discriminator=仅判别器预测, decode=仅解码消息, similarity=相似度判断')

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    if args.mode == 'discriminator' and args.checkpoint is None:
        print("警告: 判别器模式未提供检查点，将使用随机初始化的权重，预测结果无意义!")
        print("建议使用: -c <checkpoint_path> 指定检查点路径")

    if args.mode == 'decode' and args.checkpoint is None:
        print("警告: 解码模式未提供检查点，将使用随机初始化的权重，解码结果无意义!")
        print("建议使用: -c <checkpoint_path> 指定检查点路径")

    if args.mode == 'similarity' and args.cover is None:
        print("错误: 相似度判断模式需要指定原图路径!")
        print("建议使用: --cover <cover_image_path> 指定原图路径")
        return

    validator = ValidationScript(checkpoint_path=args.checkpoint, device=device)

    if args.mode == 'full':
        if args.checkpoint is None:
            print("警告: 完整验证模式需要检查点，使用未训练的模型进行验证...")
        results = validator.run_full_validation(
            image_path=args.image,
            message=args.message,
            output_dir=args.output_dir
        )
    elif args.mode == 'discriminator':
        prob, logit = validator.run_discriminator_only(image_path=args.image)
    elif args.mode == 'decode':
        message_str, decoded_prob = validator.run_decode_only(image_path=args.image)
    else:
        privacy_results = validator.run_similarity_only(cover_image_path=args.cover, stego_image_path=args.image)


if __name__ == '__main__':
    main()
