import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from models import Encoder, Decoder, Steganalyzer, ContentRecognizer
from losses import SSIMLoss, FeatureRedirectionLoss
from data import SteganographyDataset
from noise_layers.noiser import Noiser
import config


class FeatureMemoryBank:
    def __init__(self, feature_dim, memory_size=1000, device='cuda'):
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.device = device
        self.memory_bank = torch.zeros(memory_size, feature_dim, device=device)
        self.memory_ptr = 0
        self.memory_count = 0

    @torch.no_grad()
    def update(self, features):
        features = F.normalize(features.float(), p=2, dim=1)
        batch_size = features.shape[0]
        ptr = self.memory_ptr

        if ptr + batch_size > self.memory_size:
            remaining = self.memory_size - ptr
            self.memory_bank[ptr:] = features[:remaining]
            self.memory_bank[:batch_size - remaining] = features[remaining:]
            self.memory_ptr = batch_size - remaining
        else:
            self.memory_bank[ptr:ptr + batch_size] = features
            self.memory_ptr = (ptr + batch_size) % self.memory_size

        self.memory_count = min(self.memory_count + batch_size, self.memory_size)

    def get_hard_negatives(self, features, k=5):
        if self.memory_count == 0:
            return None

        features = features.float()
        valid_memory = self.memory_bank[:self.memory_count]
        similarities = torch.mm(features, valid_memory.t())
        actual_k = min(k, self.memory_count)
        _, indices = similarities.topk(actual_k, dim=1)
        hard_negatives = valid_memory[indices]

        return hard_negatives

    def state_dict(self):
        return {
            'memory_bank': self.memory_bank.cpu(),
            'memory_ptr': self.memory_ptr,
            'memory_count': self.memory_count
        }

    def load_state_dict(self, state_dict):
        self.memory_bank = state_dict['memory_bank'].to(self.device)
        self.memory_ptr = state_dict['memory_ptr']
        self.memory_count = state_dict['memory_count']


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.writer = SummaryWriter(cfg.log_dir)

        self.encoder = Encoder(
            message_length=cfg.message_length,
            n_channels=3,
            image_size=cfg.image_size,
            msg_budget=cfg.msg_budget,
            prv_budget=cfg.prv_budget,
            phase1_prv_noise_smooth_kernel=cfg.phase1_prv_noise_smooth_kernel
        ).to(self.device)
        self.decoder = Decoder(message_length=cfg.message_length, n_channels=3).to(self.device)
        self.steganalyzer = Steganalyzer(n_channels=3, use_spectral_norm=True).to(self.device)
        self.content_recognizer = ContentRecognizer(device=self.device).to(self.device)

        self.optimizer_encoder_decoder = optim.Adam([
            {"params": self.encoder.parameters(), "lr": cfg.lr_encoder},
            {"params": self.decoder.parameters(), "lr": cfg.lr_decoder},
        ])
        self.optimizer_steganalyzer = optim.Adam(self.steganalyzer.parameters(), lr=cfg.lr_steganalyzer)

        self.ssim_loss = SSIMLoss().to(self.device)
        self.mse_loss = nn.MSELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()

        self.retrieval_escape_loss = FeatureRedirectionLoss(
            clip_weight=cfg.clip_weight,
            dino_weight=cfg.dino_weight,
            escape_margin=cfg.privacy_margin_start,
            clip_redirect_weight=cfg.clip_redirect_weight,
            dino_redirect_weight=cfg.dino_redirect_weight,
            clip_redirect_margin=cfg.clip_redirect_margin,
            dino_redirect_margin=cfg.dino_redirect_margin,
            clip_max_target_sim=cfg.clip_max_target_sim,
            dino_max_target_sim=cfg.dino_max_target_sim
        ).to(self.device)

        self.use_clip_redirect = cfg.use_clip_redirect
        self.clip_memory_bank = None
        self.dino_memory_bank = None

        if self.use_clip_redirect:
            self.clip_memory_bank = FeatureMemoryBank(
                feature_dim=512,
                memory_size=cfg.clip_memory_size,
                device=self.device
            )
            self.clip_hard_negative_k = cfg.clip_hard_negative_k

        self.dino_weight = cfg.dino_weight
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

            if self.use_clip_redirect:
                self.dino_memory_bank = FeatureMemoryBank(
                    feature_dim=384,
                    memory_size=cfg.clip_memory_size,
                    device=self.device
                )
        else:
            self.dino_model = None

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
        self.has_noise = not (len(self.noiser._layer_names) == 1 and self.noiser._layer_names[0] == 'identity')

        self.best_loss = float('inf')
        self.phase2_msg_head_frozen = False
        self.disc_train_counter = 0

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

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    def extract_dino_features(self, image, allow_grad=False):
        if not hasattr(self, 'dino_model') or self.dino_model is None:
            return None
        with torch.set_grad_enabled(allow_grad):
            image_resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            image_normalized = (image_resized + 1) / 2
            image_normalized = (image_normalized - self.dino_mean) / self.dino_std
            features = self.dino_model(image_normalized)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-7)
        return features

    def freeze_msg_head(self):
        for param in self.encoder.msg_head.parameters():
            param.requires_grad = False
        self.phase2_msg_head_frozen = True

    def unfreeze_msg_head(self):
        for param in self.encoder.msg_head.parameters():
            param.requires_grad = True
        self.phase2_msg_head_frozen = False

    def freeze_prv_head(self):
        for param in self.encoder.prv_head.parameters():
            param.requires_grad = False

    def unfreeze_prv_head(self):
        for param in self.encoder.prv_head.parameters():
            param.requires_grad = True

    def freeze_shared_layers(self):
        for param in self.encoder.conv_layers.parameters():
            param.requires_grad = False
        for param in self.encoder.after_concat_layer.parameters():
            param.requires_grad = False

    def unfreeze_shared_layers(self):
        for param in self.encoder.conv_layers.parameters():
            param.requires_grad = True
        for param in self.encoder.after_concat_layer.parameters():
            param.requires_grad = True

    def _set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _disc_hinge_loss(self, real_logits, fake_logits):
        margin = getattr(self.cfg, 'disc_hinge_margin', 1.0)
        real_loss = F.relu(margin - real_logits).mean()
        fake_loss = F.relu(margin + fake_logits).mean()
        return real_loss + fake_loss

    def _gen_adv_loss(self, fake_logits):
        return F.softplus(-fake_logits).mean()

    def _disc_metrics(self, real_logits, fake_logits):
        real_prob = torch.sigmoid(real_logits)
        fake_prob = torch.sigmoid(fake_logits)
        real_acc = (real_logits > 0).float().mean()
        fake_acc = (fake_logits < 0).float().mean()
        disc_acc = 0.5 * (real_acc + fake_acc)
        return {
            'D_real_logit': real_logits.mean().item(),
            'D_fake_logit': fake_logits.mean().item(),
            'D_real_prob': real_prob.mean().item(),
            'D_fake_prob': fake_prob.mean().item(),
            'D_acc': disc_acc.item()
        }

    def _adversarial_inputs(self, cover_image, encoded_image, noised_cover_image, noised_encoded_image):
        if self.has_noise:
            return noised_cover_image, noised_encoded_image
        return cover_image, encoded_image

    def _adversarial_losses_and_metrics(self, real_logits, fake_logits):
        L_D = self._disc_hinge_loss(real_logits, fake_logits)
        L_adv = self._gen_adv_loss(fake_logits)
        discr_cover_bce = self.bce_with_logits_loss(real_logits, torch.ones_like(real_logits))
        discr_encod_bce = self.bce_with_logits_loss(fake_logits, torch.zeros_like(fake_logits))
        metrics = self._disc_metrics(real_logits, fake_logits)
        return L_D, L_adv, discr_cover_bce, discr_encod_bce, metrics

    def _set_phase_freeze(self, phase, epoch):
        if phase == 1:
            self.unfreeze_msg_head()
            self.freeze_prv_head()
            self.unfreeze_shared_layers()
        else:
            self.freeze_msg_head()
            self.unfreeze_prv_head()
            if epoch <= self.cfg.num_epochs_phase2 // 3:
                self.unfreeze_shared_layers()
            else:
                self.freeze_shared_layers()

    def train_step(self, batch, phase, epoch=1):
        self._set_phase_freeze(phase, epoch)

        cover_image = batch['cover_image'].to(self.device)
        message = batch['message'].to(self.device)

        weights = self.cfg.get_phase_weights(phase, epoch)
        prv_noise_ratio = self.cfg.get_prv_noise_ratio(epoch) if phase == 1 else 0.0

        encoded_image, residual_penalty, msg_residual, prv_residual = self.encoder(
            cover_image, message, phase=phase, prv_noise_ratio=prv_noise_ratio
        )

        noised_encoded_image, noised_cover_image = self.noiser((encoded_image, cover_image))

        message_pred = self.decoder(noised_encoded_image)

        L_pixel = self.mse_loss(encoded_image, cover_image)
        L_ssim = self.ssim_loss(encoded_image, cover_image)
        L_visual = self.cfg.lambda_pixel * L_pixel + self.cfg.lambda_ssim * L_ssim

        visual_target_noised = noised_cover_image if self.has_noise else cover_image
        L_pixel_noised = self.mse_loss(noised_encoded_image, visual_target_noised)
        L_ssim_noised = self.ssim_loss(noised_encoded_image, visual_target_noised)
        L_visual_noised = self.cfg.lambda_pixel * L_pixel_noised + self.cfg.lambda_ssim * L_ssim_noised

        L_message = self.bce_with_logits_loss(message_pred, message)

        disc_real_input, disc_stego_input = self._adversarial_inputs(
            cover_image, encoded_image, noised_cover_image, noised_encoded_image
        )
        fake_logits_for_disc = self.steganalyzer(disc_stego_input.detach())
        real_logits = self.steganalyzer(disc_real_input.detach())
        L_D, _, discr_cover_bce, discr_encod_bce, disc_metrics = self._adversarial_losses_and_metrics(
            real_logits, fake_logits_for_disc
        )

        D_updated = 0.0
        if weights.get('adv', 0.0) > 0:
            self.optimizer_steganalyzer.zero_grad()
            L_D.backward()
            torch.nn.utils.clip_grad_norm_(self.steganalyzer.parameters(), max_norm=1.0)
            self.optimizer_steganalyzer.step()
            D_updated = 1.0

        self._set_requires_grad(self.steganalyzer, False)
        steganalyzer_was_training = self.steganalyzer.training
        self.steganalyzer.eval()
        stego_pred = self.steganalyzer(disc_stego_input)
        L_adv = self._gen_adv_loss(stego_pred)
        if steganalyzer_was_training:
            self.steganalyzer.train()
        self._set_requires_grad(self.steganalyzer, True)

        feat_cover = self.content_recognizer(cover_image).detach()
        feat_encoded = self.content_recognizer(encoded_image)

        dino_cover = self.extract_dino_features(cover_image)
        dino_encoded = self.extract_dino_features(encoded_image, allow_grad=True)

        clip_hard_negatives = None
        dino_hard_negatives = None

        if self.use_clip_redirect and self.clip_memory_bank is not None:
            self.clip_memory_bank.update(feat_cover.detach())
            clip_hard_negatives = self.clip_memory_bank.get_hard_negatives(
                feat_encoded.detach(), k=self.clip_hard_negative_k
            )

        if self.use_clip_redirect and self.dino_memory_bank is not None and dino_cover is not None:
            self.dino_memory_bank.update(dino_cover.detach())
            dino_hard_negatives = self.dino_memory_bank.get_hard_negatives(
                dino_encoded.detach(), k=self.clip_hard_negative_k
            )

        escape_margin = self.cfg.get_escape_margin(phase, epoch)

        L_privacy, privacy_loss_dict = self.retrieval_escape_loss(
            feat_cover, feat_encoded,
            dino_cover=dino_cover,
            dino_encoded=dino_encoded,
            clip_hard_negatives=clip_hard_negatives,
            dino_hard_negatives=dino_hard_negatives,
            escape_margin=escape_margin
        )

        residual_penalty_weight = weights.get('residual_penalty', 0.05)
        L_visual_weighted = weights['visual'] * (L_visual + weights['noised'] * L_visual_noised)
        L_non_visual = (weights['message'] * L_message +
                       weights['adv'] * L_adv +
                       residual_penalty_weight * residual_penalty)
        L_privacy_weighted = weights['privacy'] * L_privacy
        L_total = L_visual_weighted + L_non_visual + L_privacy_weighted

        self.optimizer_encoder_decoder.zero_grad()
        L_total.backward()

        if phase == 2 and self.phase2_msg_head_frozen:
            for param in self.encoder.msg_head.parameters():
                param.grad = None

        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), max_norm=1.0)

        self.optimizer_encoder_decoder.step()

        with torch.no_grad():
            message_pred_binary = (torch.sigmoid(message_pred) > 0.5).float()
            bit_accuracy = (message_pred_binary == message).float().mean().item()

        losses = {
            'L_pixel': L_pixel.item(),
            'L_ssim': L_ssim.item(),
            'L_visual': L_visual.item(),
            'L_pixel_noised': L_pixel_noised.item(),
            'L_ssim_noised': L_ssim_noised.item(),
            'L_visual_noised': L_visual_noised.item(),
            'L_message': L_message.item(),
            'L_adv': L_adv.item(),
            'L_D': L_D.item(),
            'discr_cover_bce': discr_cover_bce.item(),
            'discr_encod_bce': discr_encod_bce.item(),
            'L_privacy': L_privacy.item(),
            'L_residual_penalty': residual_penalty.item(),
            'bit_accuracy': bit_accuracy,
            'escape_margin': escape_margin,
            'prv_noise_ratio': prv_noise_ratio,
            'D_updated': D_updated,
            'adv_weight': weights.get('adv', 0.0)
        }
        losses.update(disc_metrics)

        for key, value in privacy_loss_dict.items():
            losses[f'privacy_{key}'] = value if isinstance(value, float) else value

        return losses

    def recovery_step(self, train_loader, num_batches=5):
        self.encoder.eval()
        self.decoder.train()

        self.freeze_prv_head()

        temp_optimizer = optim.Adam(self.decoder.parameters(), lr=self.cfg.lr_decoder * 0.5)

        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break

            cover_image = batch['cover_image'].to(self.device)
            message = batch['message'].to(self.device)

            with torch.no_grad():
                _, _, msg_residual, _ = self.encoder(
                    cover_image, message, phase=1, prv_noise_ratio=0.0
                )
                clean_encoded = cover_image + msg_residual
                clean_encoded = torch.clamp(clean_encoded, -1, 1)

            message_pred = self.decoder(clean_encoded)
            L_msg = self.bce_with_logits_loss(message_pred, message)

            temp_optimizer.zero_grad()
            L_msg.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
            temp_optimizer.step()

        self.unfreeze_prv_head()
        self.encoder.train()

    def validate(self, val_loader, phase, epoch, save_vis=False):
        self.encoder.eval()
        self.decoder.eval()
        self.steganalyzer.eval()

        total_losses = {}
        num_batches = 0

        sample_images = []
        max_samples = 8
        all_batches = list(val_loader)
        num_total_samples = sum(batch['cover_image'].size(0) for batch in all_batches)
        if num_total_samples < max_samples:
            selected_indices = list(range(num_total_samples))
        else:
            import random
            selected_indices = random.sample(range(num_total_samples), max_samples)
        selected_indices_set = set(selected_indices)

        prv_noise_ratio = self.cfg.get_prv_noise_ratio(epoch) if phase == 1 else 0.0
        escape_margin = self.cfg.get_escape_margin(phase, epoch)

        with torch.no_grad():
            current_idx = 0
            for batch in all_batches:
                cover_image = batch['cover_image'].to(self.device)
                message = batch['message'].to(self.device)

                weights = self.cfg.get_phase_weights(phase, epoch)

                encoded_image, residual_penalty, msg_residual, prv_residual = self.encoder(
                    cover_image, message, phase=phase, prv_noise_ratio=prv_noise_ratio
                )
                noised_encoded_image, noised_cover_image = self.noiser((encoded_image, cover_image))
                message_pred = self.decoder(noised_encoded_image)

                L_pixel = self.mse_loss(encoded_image, cover_image)
                L_ssim = self.ssim_loss(encoded_image, cover_image)
                L_visual = self.cfg.lambda_pixel * L_pixel + self.cfg.lambda_ssim * L_ssim

                visual_target_noised = noised_cover_image if self.has_noise else cover_image
                L_pixel_noised = self.mse_loss(noised_encoded_image, visual_target_noised)
                L_ssim_noised = self.ssim_loss(noised_encoded_image, visual_target_noised)
                L_visual_noised = self.cfg.lambda_pixel * L_pixel_noised + self.cfg.lambda_ssim * L_ssim_noised

                L_message = self.bce_with_logits_loss(message_pred, message)

                disc_real_input, disc_stego_input = self._adversarial_inputs(
                    cover_image, encoded_image, noised_cover_image, noised_encoded_image
                )
                stego_pred = self.steganalyzer(disc_stego_input)
                cover_pred = self.steganalyzer(disc_real_input)
                L_D, L_adv, discr_cover_bce, discr_encod_bce, disc_metrics = self._adversarial_losses_and_metrics(
                    cover_pred, stego_pred
                )

                feat_cover = self.content_recognizer(cover_image)
                feat_encoded = self.content_recognizer(encoded_image)

                dino_cover = self.extract_dino_features(cover_image)
                dino_encoded = self.extract_dino_features(encoded_image)

                L_privacy, _ = self.retrieval_escape_loss(
                    feat_cover, feat_encoded,
                    dino_cover=dino_cover,
                    dino_encoded=dino_encoded,
                    escape_margin=escape_margin
                )

                residual_penalty_weight = weights.get('residual_penalty', 0.05)
                L_total = (weights['visual'] * (L_visual + weights['noised'] * L_visual_noised) +
                          weights['message'] * L_message +
                          weights['adv'] * L_adv +
                          weights['privacy'] * L_privacy +
                          residual_penalty_weight * residual_penalty)

                message_pred_binary = (torch.sigmoid(message_pred) > 0.5).float()
                bit_accuracy = (message_pred_binary == message).float().mean()

                for key, value in [
                    ('L_pixel', L_pixel), ('L_ssim', L_ssim),
                    ('L_visual', L_visual),
                    ('L_pixel_noised', L_pixel_noised), ('L_ssim_noised', L_ssim_noised),
                    ('L_visual_noised', L_visual_noised),
                    ('L_message', L_message), ('L_adv', L_adv), ('L_D', L_D),
                    ('discr_cover_bce', discr_cover_bce), ('discr_encod_bce', discr_encod_bce),
                    ('L_privacy', L_privacy), ('L_residual_penalty', residual_penalty),
                    ('bit_accuracy', bit_accuracy)
                ]:
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value.item()

                for key, value in disc_metrics.items():
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value

                for key, value in [
                    ('D_updated', 0.0), ('adv_weight', weights.get('adv', 0.0)),
                    ('prv_noise_ratio', prv_noise_ratio)
                ]:
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value

                num_batches += 1

                for i in range(cover_image.size(0)):
                    if current_idx in selected_indices_set:
                        sample_images.append({
                            'cover': cover_image[i].cpu(),
                            'encoded': encoded_image[i].cpu()
                        })
                    current_idx += 1

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

        self.encoder.train()
        self.decoder.train()
        self.steganalyzer.train()

        return total_losses

    def warmup_memory_bank(self, train_loader, num_batches=50):
        if self.clip_memory_bank is None and self.dino_memory_bank is None:
            return

        print(f'Warming up Memory Banks with {num_batches} batches...')

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= num_batches:
                    break
                cover_image = batch['cover_image'].to(self.device)

                if self.clip_memory_bank is not None:
                    clip_feat = self.content_recognizer(cover_image)
                    self.clip_memory_bank.update(clip_feat)

                if self.dino_memory_bank is not None:
                    dino_feat = self.extract_dino_features(cover_image)
                    if dino_feat is not None:
                        self.dino_memory_bank.update(dino_feat)

        info_parts = []
        if self.clip_memory_bank is not None:
            info_parts.append(f'CLIP: {self.clip_memory_bank.memory_count}')
        if self.dino_memory_bank is not None:
            info_parts.append(f'DINO: {self.dino_memory_bank.memory_count}')
        print(f'Memory Banks warmed up. [{", ".join(info_parts)}]')

    def save_checkpoint(self, epoch, phase):
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'steganalyzer_state_dict': self.steganalyzer.state_dict(),
            'optimizer_encoder_decoder': self.optimizer_encoder_decoder.state_dict(),
            'optimizer_steganalyzer': self.optimizer_steganalyzer.state_dict(),
            'best_loss': self.best_loss,
            'phase2_msg_head_frozen': self.phase2_msg_head_frozen
        }

        if self.clip_memory_bank is not None:
            checkpoint['clip_memory_bank'] = self.clip_memory_bank.state_dict()

        if self.dino_memory_bank is not None:
            checkpoint['dino_memory_bank'] = self.dino_memory_bank.state_dict()

        path = os.path.join(self.cfg.checkpoint_dir, f'checkpoint_phase{phase}_epoch{epoch}.pth')
        torch.save(checkpoint, path)
