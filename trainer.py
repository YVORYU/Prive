import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from models import Encoder, Decoder, Steganalyzer, ContentRecognizer, RetrievalAttackerWithMemory
from losses import SSIMLoss, PerceptualLoss, PrivacyLoss, RetrievalEscapeLoss
from data import SteganographyDataset
from noise_layers.noiser import Noiser
from noise_argparser import NoiseArgParser
import config


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        #初始化日志记录器
        self.writer = SummaryWriter(cfg.log_dir)
        #定义编码器、解码器、判别器和内容识别器(冻结参数，不进行训练)
        self.encoder = Encoder(message_length=cfg.message_length, n_channels=3, image_size=cfg.image_size).to(self.device)
        self.decoder = Decoder(message_length=cfg.message_length, n_channels=3).to(self.device)
        self.steganalyzer = Steganalyzer(n_channels=3, use_spectral_norm=True).to(self.device)
        self.content_recognizer = ContentRecognizer(device=self.device).to(self.device)
        #定义优化器
        self.optimizer_encoder_decoder = optim.Adam([
            {"params": self.encoder.parameters(), "lr": cfg.lr_encoder},
            {"params": self.decoder.parameters(), "lr": cfg.lr_decoder},
        ])
        self.optimizer_steganalyzer = optim.Adam(self.steganalyzer.parameters(), lr=cfg.lr_steganalyzer)
        #定义损失函数
        self.ssim_loss = SSIMLoss().to(self.device)
        self.use_perceptual_loss = getattr(cfg, 'use_perceptual_loss', False)
        if self.use_perceptual_loss:
            self.perceptual_loss = PerceptualLoss().to(self.device)
        self.privacy_loss = PrivacyLoss(margin=cfg.privacy_margin).to(self.device)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        
        if hasattr(cfg, 'use_advanced_privacy') and cfg.use_advanced_privacy:
            self.use_advanced_privacy = True
            self.dino_weight = getattr(cfg, 'dino_weight', 0.8)
            self.retrieval_escape_loss = RetrievalEscapeLoss(
                clip_weight=getattr(cfg, 'clip_weight', 1.0),
                dino_weight=self.dino_weight,
                triplet_weight=getattr(cfg, 'triplet_weight', 0.5),
                margin=cfg.privacy_margin,
                triplet_margin=getattr(cfg, 'triplet_margin', 0.3),
                attacker_weight=getattr(cfg, 'attacker_weight', 0.5)
            ).to(self.device)
            
            if self.dino_weight > 0:
                dino_model_path = getattr(cfg, 'dino_model_path', None)
                self.dino_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=224)
                if dino_model_path and os.path.exists(dino_model_path):
                    state_dict = torch.load(dino_model_path, map_location=self.device)
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
                        print(f'DINOv2 pos_embed interpolated from {orig_grid_size}x{orig_grid_size} to {new_grid_size}x{new_grid_size}')
                    
                    self.dino_model.load_state_dict(state_dict, strict=False)
                    print(f'DINOv2 model loaded from {dino_model_path}')
                else:
                    print(f'Warning: DINOv2 model not found at {dino_model_path}, using random initialization')
                self.dino_model = self.dino_model.to(self.device)
                self.dino_model.eval()
                for param in self.dino_model.parameters():
                    param.requires_grad = False
                self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            
            if hasattr(cfg, 'use_retrieval_attacker') and cfg.use_retrieval_attacker:
                self.retrieval_attacker = RetrievalAttackerWithMemory(
                    feature_dim=getattr(cfg, 'attacker_feature_dim', 512),
                    memory_size=getattr(cfg, 'attacker_memory_size', 1000),
                    temperature=getattr(cfg, 'attacker_temperature', 0.07)
                ).to(self.device)
                self.optimizer_attacker = optim.Adam(
                    self.retrieval_attacker.parameters(), 
                    lr=cfg.lr_encoder * 0.5
                )
            else:
                self.retrieval_attacker = None
        else:
            self.use_advanced_privacy = False
            self.retrieval_attacker = None
        
        #定义噪声层
        noise_layers = []
        if cfg.noise_config != 'identity':
            from noise_argparser import parse_crop, parse_cropout, parse_dropout, parse_resize
            split_commands = cfg.noise_config.split('+')
            for command in split_commands:
                command = command.replace(' ', '')
                if command.startswith('cropout'):
                    noise_layers.append(parse_cropout(command))
                elif command.startswith('crop'):
                    noise_layers.append(parse_crop(command))
                elif command.startswith('dropout'):
                    noise_layers.append(parse_dropout(command))
                elif command.startswith('resize'):
                    noise_layers.append(parse_resize(command))
                elif command.startswith('jpeg'):
                    from noise_layers.jpeg_compression import JpegCompression
                    noise_layers.append(JpegCompression(self.device))
                elif command.startswith('quant'):
                    from noise_layers.quantization import Quantization
                    noise_layers.append(Quantization(self.device))
        
        self.noiser = Noiser(noise_layers, self.device).to(self.device)
        
        self.best_loss = float('inf')
        
    #获取数据加载器
    def get_dataloaders(self):
        train_dataset = SteganographyDataset(
            root_dir=os.path.join(self.cfg.data_dir, self.cfg.train_dir),
            image_size=self.cfg.image_size,
            message_length=self.cfg.message_length
        )
        
        val_dataset = SteganographyDataset(
            root_dir=os.path.join(self.cfg.data_dir, self.cfg.val_dir),
            image_size=self.cfg.image_size,
            message_length=self.cfg.message_length
        )

        #定义数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )
        #定义验证数据加载器
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def extract_dino_features(self, image):
        if not hasattr(self, 'dino_model') or self.dino_model is None:
            return None
        with torch.no_grad():
            image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            image_normalized = (image_resized + 1) / 2
            image_normalized = (image_normalized - self.dino_mean) / self.dino_std
            features = self.dino_model(image_normalized)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-7)
        return features
    
    #训练batch
    def train_step(self, batch, phase, epoch=1):
        cover_image = batch['cover_image'].to(self.device)
        message = batch['message'].to(self.device)

        weights = self.cfg.get_phase_weights(phase)

        #编码器编码图像
        encoded_image, residual_penalty = self.encoder(cover_image, message)
        
        #添加噪声
        noised_encoded_image, noised_cover_image = self.noiser((encoded_image, cover_image))
        
        #解码器解码图像
        message_pred = self.decoder(noised_encoded_image)

        #计算损失
        L_pixel = self.mse_loss(encoded_image, cover_image)
        L_ssim = self.ssim_loss(encoded_image, cover_image)
        if self.use_perceptual_loss:
            L_perceptual = self.perceptual_loss(encoded_image, cover_image)
            L_visual = (self.cfg.lambda_pixel * L_pixel +
                       self.cfg.lambda_ssim * L_ssim +
                       self.cfg.lambda_perceptual * L_perceptual)
        else:
            L_perceptual = torch.tensor(0.0, device=self.device)
            L_visual = self.cfg.lambda_pixel * L_pixel + self.cfg.lambda_ssim * L_ssim
        #计算噪声图像损失
        L_pixel_noised = self.mse_loss(noised_encoded_image, cover_image)
        L_ssim_noised = self.ssim_loss(noised_encoded_image, cover_image)
        if self.use_perceptual_loss:
            L_perceptual_noised = self.perceptual_loss(noised_encoded_image, cover_image)
            L_visual_noised = (self.cfg.lambda_pixel * L_pixel_noised +
                               self.cfg.lambda_ssim * L_ssim_noised +
                               self.cfg.lambda_perceptual * L_perceptual_noised)
        else:
            L_perceptual_noised = torch.tensor(0.0, device=self.device)
            L_visual_noised = self.cfg.lambda_pixel * L_pixel_noised + self.cfg.lambda_ssim * L_ssim_noised
        #计算消息损失
        L_message = self.bce_with_logits_loss(message_pred, message)
        
        #判别器进行预测
        stego_pred = self.steganalyzer(noised_encoded_image.detach())
        cover_pred = self.steganalyzer(noised_cover_image)

        #判别器损失：cover=1, stego=0
        discr_cover_bce = self.bce_with_logits_loss(cover_pred, torch.ones_like(cover_pred))
        discr_encod_bce = self.bce_with_logits_loss(stego_pred, torch.zeros_like(stego_pred))
        L_D = discr_cover_bce + discr_encod_bce
        
        if weights.get('adv', 0.0) > 0:
            self.optimizer_steganalyzer.zero_grad()
            L_D.backward()
            torch.nn.utils.clip_grad_norm_(self.steganalyzer.parameters(), max_norm=1.0)
            self.optimizer_steganalyzer.step()
        
        stego_pred = self.steganalyzer(noised_encoded_image)
        #对抗损失：生成器希望stego被判别为cover(标签1)
        L_adv = self.bce_with_logits_loss(stego_pred, torch.ones_like(stego_pred))

        feat_cover = self.content_recognizer(cover_image).detach()
        feat_encoded = self.content_recognizer(encoded_image)
        
        if self.use_advanced_privacy:
            feat_negatives = None
            attacker_feat_cover = None
            attacker_feat_encoded = None
            dino_cover = None
            dino_encoded = None
            
            dino_cover = self.extract_dino_features(cover_image)
            dino_encoded = self.extract_dino_features(encoded_image)
            
            if self.retrieval_attacker is not None:
                with torch.no_grad():
                    attacker_feat_cover_for_negatives = self.retrieval_attacker(cover_image)
                
                if hasattr(self.cfg, 'use_hard_negatives') and self.cfg.use_hard_negatives:
                    feat_negatives = self.retrieval_attacker.get_hard_negatives(
                        attacker_feat_cover_for_negatives, k=getattr(self.cfg, 'hard_negative_k', 5)
                    )
                    if feat_negatives is not None:
                        if feat_negatives.dim() == 3:
                            similarities = F.cosine_similarity(
                                attacker_feat_cover_for_negatives.unsqueeze(1),
                                feat_negatives,
                                dim=2
                            )
                            hardest_idx = similarities.argmax(dim=1)
                            feat_negatives = feat_negatives[torch.arange(feat_negatives.size(0)), hardest_idx]
                
                attacker_feat_cover = self.retrieval_attacker(cover_image.detach())
                attacker_feat_encoded = self.retrieval_attacker(encoded_image)
            
            L_privacy, privacy_loss_dict = self.retrieval_escape_loss(
                feat_cover, feat_encoded,
                feat_negatives=feat_negatives,
                dino_cover=dino_cover,
                dino_encoded=dino_encoded,
                attacker_feat_cover=attacker_feat_cover,
                attacker_feat_encoded=attacker_feat_encoded
            )
        else:
            L_privacy = self.privacy_loss(feat_cover, feat_encoded)
            privacy_loss_dict = {'privacy_loss': L_privacy.item()}
        
        residual_penalty_weight = weights.get('residual_penalty', 0.05)
        #计算总损失，按权重加权
        L_total = (weights['visual'] * (L_visual + weights['noised'] * L_visual_noised) +
                  weights['message'] * L_message +
                  weights['adv'] * L_adv +
                  weights['privacy'] * L_privacy +
                  residual_penalty_weight * residual_penalty)
        
        self.optimizer_encoder_decoder.zero_grad()
        L_total.backward()
        
        #梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), max_norm=1.0)
        
        self.optimizer_encoder_decoder.step()
        
        if self.use_advanced_privacy and self.retrieval_attacker is not None and phase >= 2:
            attacker_feat_cover_new = self.retrieval_attacker(cover_image.detach())
            attacker_feat_encoded_new = self.retrieval_attacker(encoded_image.detach())
            
            valid_count = int(self.retrieval_attacker.memory_count[0])
            temperature = getattr(self.cfg, 'attacker_temperature', 0.07)
            
            if valid_count >= self.cfg.batch_size:
                valid_memory = self.retrieval_attacker.memory_bank[:valid_count].detach()
                
                pos_sim = torch.sum(attacker_feat_encoded_new * attacker_feat_cover_new.detach(), dim=1, keepdim=True) / temperature
                neg_sim = torch.mm(attacker_feat_encoded_new, valid_memory.t()) / temperature
                
                logits = torch.cat([pos_sim, neg_sim], dim=1)
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
                L_attacker = F.cross_entropy(logits, labels)
            else:
                L_attacker = F.mse_loss(attacker_feat_encoded_new, attacker_feat_cover_new.detach())
            
            self.optimizer_attacker.zero_grad()
            L_attacker.backward()
            torch.nn.utils.clip_grad_norm_(self.retrieval_attacker.parameters(), max_norm=1.0)
            self.optimizer_attacker.step()
            
            with torch.no_grad():
                self.retrieval_attacker.update_memory(attacker_feat_cover_new)
        
        with torch.no_grad():
            message_pred_binary = (torch.sigmoid(message_pred) > 0.5).float()
            bit_accuracy = (message_pred_binary == message).float().mean().item()
        
        losses = {
            'L_pixel': L_pixel.item(),
            'L_ssim': L_ssim.item(),
            'L_perceptual': L_perceptual.item(),
            'L_visual': L_visual.item(),
            'L_pixel_noised': L_pixel_noised.item(),
            'L_ssim_noised': L_ssim_noised.item(),
            'L_perceptual_noised': L_perceptual_noised.item(),
            'L_visual_noised': L_visual_noised.item(),
            'L_message': L_message.item(),
            'L_adv': L_adv.item(),
            'discr_cover_bce': discr_cover_bce.item(),
            'discr_encod_bce': discr_encod_bce.item(),
            'L_privacy': L_privacy.item(),
            'L_residual_penalty': residual_penalty.item(),
            'L_total': L_total.item(),
            'bit_accuracy': bit_accuracy
        }
        
        if self.use_advanced_privacy:
            for key, value in privacy_loss_dict.items():
                losses[f'privacy_{key}'] = value if isinstance(value, float) else value
        
        return losses

#验证函数
    def validate(self, val_loader, phase, epoch, save_vis=False):
        #切换到评估模式
        self.encoder.eval()
        self.decoder.eval()
        self.steganalyzer.eval()
        
        total_losses = {}
        num_batches = 0
        
        sample_images = []
        max_samples = 8
        #计算验证集的样本数量，如果小于max_samples，就使用所有样本，否则就随机选择max_samples个样本
        all_batches = list(val_loader)
        num_total_samples = sum(batch['cover_image'].size(0) for batch in all_batches)
        if num_total_samples < max_samples:
            selected_indices = list(range(num_total_samples))
        else:
            selected_indices = random.sample(range(num_total_samples), max_samples)
        #将选中的样本索引转换为集合，方便后续判断
        selected_indices_set = set(selected_indices)
        
        with torch.no_grad():
            current_idx = 0
            for batch in all_batches:
                cover_image = batch['cover_image'].to(self.device)
                message = batch['message'].to(self.device)
                
                weights = self.cfg.get_phase_weights(phase)
                
                #编码，加噪声，解码
                encoded_image, residual_penalty = self.encoder(cover_image, message)
                noised_encoded_image, noised_cover_image = self.noiser((encoded_image, cover_image))
                message_pred = self.decoder(noised_encoded_image)
                
                #计算视觉损失
                L_pixel = self.mse_loss(encoded_image, cover_image)
                L_ssim = self.ssim_loss(encoded_image, cover_image)
                if self.use_perceptual_loss:
                    L_perceptual = self.perceptual_loss(encoded_image, cover_image)
                    L_visual = (self.cfg.lambda_pixel * L_pixel + 
                               self.cfg.lambda_ssim * L_ssim + 
                               self.cfg.lambda_perceptual * L_perceptual)
                else:
                    L_perceptual = torch.tensor(0.0, device=self.device)
                    L_visual = self.cfg.lambda_pixel * L_pixel + self.cfg.lambda_ssim * L_ssim
                
                #计算含噪声的视觉损失
                L_pixel_noised = self.mse_loss(noised_encoded_image, cover_image)
                L_ssim_noised = self.ssim_loss(noised_encoded_image, cover_image)
                if self.use_perceptual_loss:
                    L_perceptual_noised = self.perceptual_loss(noised_encoded_image, cover_image)
                    L_visual_noised = (self.cfg.lambda_pixel * L_pixel_noised +
                                       self.cfg.lambda_ssim * L_ssim_noised +
                                       self.cfg.lambda_perceptual * L_perceptual_noised)
                else:
                    L_perceptual_noised = torch.tensor(0.0, device=self.device)
                    L_visual_noised = self.cfg.lambda_pixel * L_pixel_noised + self.cfg.lambda_ssim * L_ssim_noised
                
                #计算消息损失
                L_message = self.bce_with_logits_loss(message_pred, message)
                
                #判别器预测
                stego_pred = self.steganalyzer(noised_encoded_image)
                cover_pred = self.steganalyzer(noised_cover_image)
                L_adv = self.bce_with_logits_loss(stego_pred, torch.ones_like(stego_pred))
                discr_cover_bce = self.bce_with_logits_loss(cover_pred, torch.ones_like(cover_pred))
                discr_encod_bce = self.bce_with_logits_loss(stego_pred, torch.zeros_like(stego_pred))
                
                feat_cover = self.content_recognizer(cover_image)
                feat_encoded = self.content_recognizer(encoded_image)
                
                if self.use_advanced_privacy:
                    feat_negatives = None
                    attacker_feat_cover = None
                    attacker_feat_encoded = None
                    dino_cover = None
                    dino_encoded = None
                    
                    dino_cover = self.extract_dino_features(cover_image)
                    dino_encoded = self.extract_dino_features(encoded_image)
                    
                    if self.retrieval_attacker is not None:
                        attacker_feat_cover = self.retrieval_attacker(cover_image.detach())
                        attacker_feat_encoded = self.retrieval_attacker(encoded_image.detach())
                        
                        if hasattr(self.cfg, 'use_hard_negatives') and self.cfg.use_hard_negatives:
                            feat_negatives = self.retrieval_attacker.get_hard_negatives(
                                attacker_feat_cover, k=getattr(self.cfg, 'hard_negative_k', 5)
                            )
                            if feat_negatives is not None:
                                if feat_negatives.dim() == 3:
                                    sim = F.cosine_similarity(
                                        attacker_feat_cover.unsqueeze(1),
                                        feat_negatives,
                                        dim=2
                                    )
                                    hardest_idx = sim.argmax(dim=1)
                                    feat_negatives = feat_negatives[torch.arange(feat_negatives.size(0)), hardest_idx]
                    
                    L_privacy, _ = self.retrieval_escape_loss(
                        feat_cover, feat_encoded,
                        feat_negatives=feat_negatives,
                        dino_cover=dino_cover,
                        dino_encoded=dino_encoded,
                        attacker_feat_cover=attacker_feat_cover,
                        attacker_feat_encoded=attacker_feat_encoded
                    )
                else:
                    L_privacy = self.privacy_loss(feat_cover, feat_encoded)
                
                residual_penalty_weight = weights.get('residual_penalty', 0.05)
                #计算总损失
                L_total = (weights['visual'] * (L_visual + weights['noised'] * L_visual_noised) + 
                          weights['message'] * L_message + 
                          weights['adv'] * L_adv + 
                          weights['privacy'] * L_privacy +
                          residual_penalty_weight * residual_penalty)
                
                message_pred_binary = (torch.sigmoid(message_pred) > 0.5).float()
                bit_accuracy = (message_pred_binary == message).float().mean()
                
                #计算总损失
                for key, value in [
                    ('L_pixel', L_pixel), ('L_ssim', L_ssim), 
                    ('L_perceptual', L_perceptual), ('L_visual', L_visual),
                    ('L_pixel_noised', L_pixel_noised), ('L_ssim_noised', L_ssim_noised),
                    ('L_perceptual_noised', L_perceptual_noised), ('L_visual_noised', L_visual_noised),
                    ('L_message', L_message), ('L_adv', L_adv),
                    ('discr_cover_bce', discr_cover_bce), ('discr_encod_bce', discr_encod_bce),
                    ('L_privacy', L_privacy), ('L_residual_penalty', residual_penalty),
                    ('L_total', L_total), ('bit_accuracy', bit_accuracy)
                ]:
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value.item()
                
                num_batches += 1
                
                #遍历批次中的所有样本，判断是否在之前选中的样本索引中，如果是，就保存该样本的原始图像和编码后的图像
                for i in range(cover_image.size(0)):
                    if current_idx in selected_indices_set:
                        sample_images.append({
                            'cover': cover_image[i].cpu(),
                            'encoded': encoded_image[i].cpu()
                        })
                    current_idx += 1
        
        #计算平均损失
        for key in total_losses:
            total_losses[key] /= num_batches
        
        if save_vis and sample_images:
            save_samples = sample_images[:5]
            
            cover_images = [img['cover'] for img in save_samples]
            encoded_images = [img['encoded'] for img in save_samples]
            
            cover_grid = make_grid(cover_images, nrow=5, padding=2, normalize=False)
            encoded_grid = make_grid(encoded_images, nrow=5, padding=2, normalize=False)
            combined = torch.cat([cover_grid, encoded_grid], dim=1)
            
            save_dir = os.path.join(self.cfg.checkpoint_dir, 'visualizations')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'phase{phase}_epoch{epoch}.png')
            
            combined_normalized = (combined + 1) / 2
            combined_normalized = torch.clamp(combined_normalized, 0, 1)
            save_image(combined_normalized, save_path)
            print(f' [OK] Visualization saved: {save_path}')
        
        #切换到训练模式
        self.encoder.train()
        self.decoder.train()
        self.steganalyzer.train()
        
        return total_losses

    def warmup_memory_bank(self, train_loader, num_batches=50):
        if self.retrieval_attacker is None:
            return
        
        print(f'Warming up Memory Bank with {num_batches} batches...')
        self.retrieval_attacker.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= num_batches:
                    break
                cover_image = batch['cover_image'].to(self.device)
                features = self.retrieval_attacker(cover_image)
                self.retrieval_attacker.update_memory(features)
        
        self.retrieval_attacker.train()
        print(f'Memory Bank warmed up. Current count: {int(self.retrieval_attacker.memory_count[0])}')

    def save_checkpoint(self, epoch, phase, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'steganalyzer_state_dict': self.steganalyzer.state_dict(),
            'optimizer_encoder_decoder': self.optimizer_encoder_decoder.state_dict(),
            'optimizer_steganalyzer': self.optimizer_steganalyzer.state_dict(),
            'best_loss': self.best_loss
        }
        
        if hasattr(self, 'optimizer_attacker') and self.optimizer_attacker is not None:
            checkpoint['optimizer_attacker'] = self.optimizer_attacker.state_dict()
        
        if hasattr(self, 'retrieval_attacker') and self.retrieval_attacker is not None:
            checkpoint['retrieval_attacker_state_dict'] = self.retrieval_attacker.state_dict()
        #保存当前轮数的检查点
        path = os.path.join(self.cfg.checkpoint_dir, f'checkpoint_phase{phase}_epoch{epoch}.pth')
        torch.save(checkpoint, path)
        
        #更新最新的检查点
        latest_path = os.path.join(self.cfg.checkpoint_dir, 'checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        #如果当前轮数的损失小于最佳损失，就更新最佳损失
        if is_best:
            best_path = os.path.join(self.cfg.checkpoint_dir, f'best_phase{phase}.pth')
            torch.save(checkpoint, best_path)
        
        #删除旧的检查点，只保留最新的5个
        checkpoint_files = [f for f in os.listdir(self.cfg.checkpoint_dir)
                          if f.startswith(f'checkpoint_phase{phase}_epoch') and f.endswith('.pth')]

        if len(checkpoint_files) > 5:
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.cfg.checkpoint_dir, x)))
            oldest_file = checkpoint_files[0]
            os.remove(os.path.join(self.cfg.checkpoint_dir, oldest_file))


"""
pythonw.exe train.py new --data_dir ./dataset1 --name train1 > log.txt 2>&1
"""