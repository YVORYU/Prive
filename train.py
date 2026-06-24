import argparse
import torch
import random
import numpy as np
import os
import json
from datetime import datetime
from trainer import Trainer
from noise_argparser import NoiseArgParser
import config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_config(config_path, cfg):
    config_dict = {
        'data_dir': cfg.data_dir,
        'train_dir': cfg.train_dir,
        'val_dir': cfg.val_dir,
        'image_size': cfg.image_size,
        'message_length': cfg.message_length,
        'batch_size': cfg.batch_size,
        'num_epochs_phase1': cfg.num_epochs_phase1,
        'num_epochs_phase2': cfg.num_epochs_phase2,
        'discriminator_loss_threshold': cfg.discriminator_loss_threshold,
        'disc_acc_high': cfg.disc_acc_high,
        'disc_hinge_margin': cfg.disc_hinge_margin,
        'lr_encoder': cfg.lr_encoder,
        'lr_decoder': cfg.lr_decoder,
        'lr_steganalyzer': cfg.lr_steganalyzer,
        'lambda_pixel': cfg.lambda_pixel,
        'lambda_ssim': cfg.lambda_ssim,
        'msg_budget': cfg.msg_budget,
        'prv_budget': cfg.prv_budget,
        'phase1_adv_start': cfg.phase1_adv_start,
        'phase1_adv_mid': cfg.phase1_adv_mid,
        'phase1_adv_end': cfg.phase1_adv_end,
        'phase2_adv': cfg.phase2_adv,
        'phase1_prv_noise_start': cfg.phase1_prv_noise_start,
        'phase1_prv_noise_end': cfg.phase1_prv_noise_end,
        'phase1_prv_noise_max_ratio': cfg.phase1_prv_noise_max_ratio,
        'phase1_prv_noise_power': cfg.phase1_prv_noise_power,
        'phase1_prv_noise_smooth_kernel': cfg.phase1_prv_noise_smooth_kernel,
        'privacy_margin_start': cfg.privacy_margin_start,
        'privacy_margin_end': cfg.privacy_margin_end,
        'use_clip_redirect': cfg.use_clip_redirect,
        'clip_memory_size': cfg.clip_memory_size,
        'clip_hard_negative_k': cfg.clip_hard_negative_k,
        'clip_redirect_margin': cfg.clip_redirect_margin,
        'clip_max_target_sim': cfg.clip_max_target_sim,
        'clip_redirect_weight': cfg.clip_redirect_weight,
        'dino_redirect_weight': cfg.dino_redirect_weight,
        'dino_redirect_margin': cfg.dino_redirect_margin,
        'dino_max_target_sim': cfg.dino_max_target_sim,
        'clip_weight': cfg.clip_weight,
        'dino_weight': cfg.dino_weight,
        'dino_model_path': cfg.dino_model_path,
        'bit_accuracy_threshold': cfg.bit_accuracy_threshold,
        'recovery_steps': cfg.recovery_steps,
        'noise_config': cfg.noise_config,
        'num_workers': cfg.num_workers,
        'save_interval': cfg.save_interval,
        'memory_warmup_batches': cfg.memory_warmup_batches
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4)


def setup_logger(log_file, cfg, append=False):
    mode = 'a' if append else 'w'
    with open(log_file, mode, encoding='utf-8') as f:
        if append:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Resuming Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        else:
            f.write("=" * 80 + "\n")
            f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write("-" * 40 + "\n")
        config_dict = {
            'data_dir': cfg.data_dir,
            'msg_budget': cfg.msg_budget,
            'prv_budget': cfg.prv_budget,
            'disc_acc_high': cfg.disc_acc_high,
            'phase1_adv_start': cfg.phase1_adv_start,
            'phase1_adv_mid': cfg.phase1_adv_mid,
            'phase1_adv_end': cfg.phase1_adv_end,
            'phase2_adv': cfg.phase2_adv,
            'phase1_prv_noise_start': cfg.phase1_prv_noise_start,
            'phase1_prv_noise_end': cfg.phase1_prv_noise_end,
            'phase1_prv_noise_max_ratio': cfg.phase1_prv_noise_max_ratio,
            'phase1_prv_noise_power': cfg.phase1_prv_noise_power,
            'phase1_prv_noise_smooth_kernel': cfg.phase1_prv_noise_smooth_kernel,
            'privacy_margin_start': cfg.privacy_margin_start,
            'privacy_margin_end': cfg.privacy_margin_end,
            'use_clip_redirect': cfg.use_clip_redirect,
            'clip_redirect_weight': cfg.clip_redirect_weight,
            'dino_redirect_weight': cfg.dino_redirect_weight,
            'bit_accuracy_threshold': cfg.bit_accuracy_threshold,
            'num_epochs_phase1': cfg.num_epochs_phase1,
            'num_epochs_phase2': cfg.num_epochs_phase2,
        }
        for key, value in config_dict.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nNetwork Architecture:\n")
        f.write("-" * 40 + "\n")
        f.write("  Encoder: Dual-residual (msg_head + prv_head), channels=128, blocks=16\n")
        f.write("  Decoder: CNN with sigmoid output\n")
        f.write("  Steganalyzer: CNN with SpectralNorm\n")
        f.write("  ContentRecognizer: CLIP (frozen)\n")
        f.write("\n")


def log_batch_losses(log_file, batch_idx, losses, phase):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"Phase {phase} - Batch {batch_idx}: ")
        f.write(f"L_visual={losses['L_visual']:.6f}, ")
        f.write(f"L_message={losses['L_message']:.6f}, ")
        f.write(f"L_adv={losses['L_adv']:.6f}, ")
        f.write(f"L_D={losses.get('L_D', 0):.6f}, ")
        f.write(f"D_acc={losses.get('D_acc', 0):.4f}, ")
        f.write(f"D_updated={losses.get('D_updated', 0):.0f}, ")
        f.write(f"D_real_prob={losses.get('D_real_prob', 0):.4f}, ")
        f.write(f"D_fake_prob={losses.get('D_fake_prob', 0):.4f}, ")
        f.write(f"discr_cover_bce={losses['discr_cover_bce']:.6f}, ")
        f.write(f"discr_encod_bce={losses['discr_encod_bce']:.6f}, ")
        f.write(f"L_privacy={losses['L_privacy']:.6f}, ")
        f.write(f"bit_accuracy={losses['bit_accuracy']:.4f}, ")
        f.write(f"prv_noise_ratio={losses.get('prv_noise_ratio', 0):.4f}, ")
        f.write(f"escape_margin={losses.get('escape_margin', 0):.4f}\n")


def log_epoch_losses(log_file, epoch, train_losses, val_losses, phase):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Phase {phase} - Epoch {epoch} Summary\n")
        f.write("=" * 80 + "\n")

        f.write("Training Losses:\n")
        for key, value in train_losses.items():
            f.write(f"  {key}: {value:.6f}\n")

        f.write("\nValidation Losses:\n")
        for key, value in val_losses.items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\n")


def run_training_loop(trainer, train_loader, val_loader, cfg, start_phase, start_epoch, log_file):
    for phase in range(start_phase, 3):
        num_epochs = getattr(cfg, f'num_epochs_phase{phase}')
        if num_epochs <= 0:
            print(f'Skipping Phase {phase} (num_epochs={num_epochs})')
            continue
        phase_start_epoch = start_epoch if phase == start_phase else 1

        print(f'Starting Phase {phase}')

        if phase == 1 and phase_start_epoch == 1:
            trainer.warmup_memory_bank(train_loader, num_batches=getattr(cfg, 'memory_warmup_batches', 50))

        if phase == 2 and not trainer.phase2_msg_head_frozen:
            trainer.freeze_msg_head()
            print('Phase 2: msg_head frozen, training prv_head + decoder only')

        for epoch in range(phase_start_epoch, num_epochs + 1):
            trainer.encoder.train()
            trainer.decoder.train()
            trainer.steganalyzer.train()

            if hasattr(trainer, 'retrieval_escape_loss') and trainer.retrieval_escape_loss is not None:
                trainer.retrieval_escape_loss.set_epoch(epoch)

            epoch_losses = {}
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                losses = trainer.train_step(batch, phase, epoch)
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value

                num_batches += 1
                log_batch_losses(log_file, batch_idx, losses, phase)

            for key in epoch_losses:
                epoch_losses[key] /= num_batches

            save_vis = epoch % cfg.save_interval == 0
            val_losses = trainer.validate(val_loader, phase, epoch, save_vis)
            log_epoch_losses(log_file, epoch, epoch_losses, val_losses, phase)

            print(f'Phase {phase} - Epoch {epoch} - Train Bit Acc: {epoch_losses["bit_accuracy"]:.4f} - Val Bit Acc: {val_losses["bit_accuracy"]:.4f}')

            for key, value in epoch_losses.items():
                trainer.writer.add_scalar(f'Phase{phase}/Train/{key}', value, epoch)
            for key, value in val_losses.items():
                trainer.writer.add_scalar(f'Phase{phase}/Val/{key}', value, epoch)

            is_best = val_losses['L_message'] < trainer.best_loss
            if is_best:
                trainer.best_loss = val_losses['L_message']
            if epoch % cfg.save_interval == 0:
                trainer.save_checkpoint(epoch, phase)

            if phase == 2:
                val_bit_acc = val_losses.get('bit_accuracy', 1.0)
                threshold = getattr(cfg, 'bit_accuracy_threshold', 0.90)
                if val_bit_acc < threshold:
                    print(f'  [RECOVERY] bit_accuracy={val_bit_acc:.4f} < {threshold}, running recovery steps...')
                    recovery_steps = getattr(cfg, 'recovery_steps', 10)
                    trainer.recovery_step(train_loader, num_batches=recovery_steps)
                    print(f'  [RECOVERY] Done, decoder retrained on clean messages')

    print('Training completed!')
    trainer.writer.close()


def main():
    parent_parser = argparse.ArgumentParser(description='Training of GAN Steganography with Privacy Protection')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')

    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    new_run_parser.add_argument('--data-dir', '-d', required=True, type=str,
                                help='The directory where data is stored (required).')
    new_run_parser.add_argument('--train-dir', type=str, default='train',
                                help='Subdirectory name for training data (default: train).')
    new_run_parser.add_argument('--val-dir', type=str, default='val',
                                help='Subdirectory name for validation data (default: val).')
    new_run_parser.add_argument('--batch-size', '-b', type=int, default=config.Config.batch_size,
                                help='The batch size.')
    new_run_parser.add_argument('--name', type=str, default='default',
                                help='The name of experiment.')
    new_run_parser.add_argument('--size', '-s', type=int, default=256,
                                help='The size of images (images are square so this is height and width).')
    new_run_parser.add_argument('--message', '-m', type=int, default=30, help='The length in bits of watermark.')
    new_run_parser.add_argument('--continue-from-folder', '-c', default='', type=str,
                                help='The folder from where to continue a previous run. Leave blank if you are starting a new experiment.')
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')
    new_run_parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
                                help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")
    new_run_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    new_run_parser.add_argument('--epochs-phase1', type=int, default=config.Config.num_epochs_phase1,
                                help='Number of epochs for phase 1 (basic+adversarial training).')
    new_run_parser.add_argument('--epochs-phase2', type=int, default=config.Config.num_epochs_phase2,
                                help='Number of epochs for phase 2 (privacy training).')
    new_run_parser.add_argument('--discriminator-loss-threshold', type=float, default=config.Config.discriminator_loss_threshold,
                                help='Legacy discriminator loss threshold kept for compatibility.')
    new_run_parser.add_argument('--disc-acc-high', type=float, default=config.Config.disc_acc_high,
                                help='Pause discriminator updates when discriminator accuracy reaches this value.')
    new_run_parser.add_argument('--disc-hinge-margin', type=float, default=config.Config.disc_hinge_margin,
                                help='Margin used by discriminator hinge loss.')

    new_run_parser.add_argument('--phase1-adv-start', type=float, default=config.Config.phase1_adv_start,
                                help='Phase 1 adversarial weight during the first third.')
    new_run_parser.add_argument('--phase1-adv-mid', type=float, default=config.Config.phase1_adv_mid,
                                help='Phase 1 adversarial weight during the second third.')
    new_run_parser.add_argument('--phase1-adv-end', type=float, default=config.Config.phase1_adv_end,
                                help='Phase 1 adversarial weight during the final third.')
    new_run_parser.add_argument('--phase2-adv', type=float, default=config.Config.phase2_adv,
                                help='Phase 2 adversarial weight.')

    new_run_parser.add_argument('--phase1-prv-noise-start', type=float, default=config.Config.phase1_prv_noise_start,
                                help='Phase 1 fraction before dummy privacy noise starts ramping.')
    new_run_parser.add_argument('--phase1-prv-noise-end', type=float, default=config.Config.phase1_prv_noise_end,
                                help='Phase 1 fraction where dummy privacy noise reaches its cap.')
    new_run_parser.add_argument('--phase1-prv-noise-max-ratio', type=float, default=config.Config.phase1_prv_noise_max_ratio,
                                help='Maximum Phase 1 dummy privacy noise ratio.')
    new_run_parser.add_argument('--phase1-prv-noise-power', type=float, default=config.Config.phase1_prv_noise_power,
                                help='Power used for the Phase 1 dummy privacy noise ramp.')
    new_run_parser.add_argument('--phase1-prv-noise-smooth-kernel', type=int, default=config.Config.phase1_prv_noise_smooth_kernel,
                                help='Average-pooling kernel for smoothing Phase 1 dummy privacy noise.')

    new_run_parser.add_argument('--lr-encoder', type=float, default=config.Config.lr_encoder,
                                help='Learning rate for encoder')
    new_run_parser.add_argument('--lr-decoder', type=float, default=config.Config.lr_decoder,
                                help='Learning rate for decoder')
    new_run_parser.add_argument('--lr-steganalyzer', type=float, default=config.Config.lr_steganalyzer,
                                help='Learning rate for steganalyzer')

    new_run_parser.add_argument('--lambda-pixel', type=float, default=config.Config.lambda_pixel,
                                help='Weight for pixel loss.')
    new_run_parser.add_argument('--lambda-ssim', type=float, default=config.Config.lambda_ssim,
                                help='Weight for SSIM loss.')
    new_run_parser.add_argument('--num-workers', type=int, default=config.Config.num_workers,
                                help='Number of data loading workers.')
    new_run_parser.add_argument('--save-interval', type=int, default=config.Config.save_interval,
                                help='Save checkpoint every N epochs.')

    continue_parser = subparsers.add_parser('continue', help='continue a previous run')
    continue_parser.add_argument('--folder', '-f', required=True, type=str,
                                  help='The folder from where to continue a previous run.')

    args = parent_parser.parse_args()

    if args.command is None:
        parent_parser.print_help()
        return

    if args.command == 'new':
        set_seed(args.seed)

        cfg = config.Config()
        cfg.data_dir = args.data_dir
        cfg.train_dir = args.train_dir
        cfg.val_dir = args.val_dir
        cfg.batch_size = args.batch_size
        cfg.message_length = args.message
        cfg.image_size = args.size

        cfg.num_epochs_phase1 = args.epochs_phase1
        cfg.num_epochs_phase2 = args.epochs_phase2
        cfg.discriminator_loss_threshold = args.discriminator_loss_threshold
        cfg.disc_acc_high = args.disc_acc_high
        cfg.disc_hinge_margin = args.disc_hinge_margin
        cfg.phase1_adv_start = args.phase1_adv_start
        cfg.phase1_adv_mid = args.phase1_adv_mid
        cfg.phase1_adv_end = args.phase1_adv_end
        cfg.phase2_adv = args.phase2_adv
        cfg.phase1_prv_noise_start = args.phase1_prv_noise_start
        cfg.phase1_prv_noise_end = args.phase1_prv_noise_end
        cfg.phase1_prv_noise_max_ratio = args.phase1_prv_noise_max_ratio
        cfg.phase1_prv_noise_power = args.phase1_prv_noise_power
        cfg.phase1_prv_noise_smooth_kernel = args.phase1_prv_noise_smooth_kernel

        cfg.lr_encoder = args.lr_encoder
        cfg.lr_decoder = args.lr_decoder
        cfg.lr_steganalyzer = args.lr_steganalyzer

        cfg.lambda_pixel = args.lambda_pixel
        cfg.lambda_ssim = args.lambda_ssim

        cfg.num_workers = args.num_workers
        cfg.save_interval = args.save_interval

        if args.noise:
            cfg.noise_config = args.noise

        experiment_name = args.name
        experiment_dir = os.path.join('experiments', experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        cfg.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        cfg.log_dir = os.path.join(experiment_dir, 'tensorboard')
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        log_file = os.path.join(experiment_dir, 'train.log')
        setup_logger(log_file, cfg)
        config_path = os.path.join(experiment_dir, 'config.json')
        save_config(config_path, cfg)

        trainer = Trainer(cfg)

        start_epoch = 1
        start_phase = 1

        if args.continue_from_folder:
            checkpoint_path = os.path.join(args.continue_from_folder, 'checkpoints', 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                trainer.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                trainer.decoder.load_state_dict(checkpoint['decoder_state_dict'])
                trainer.steganalyzer.load_state_dict(checkpoint['steganalyzer_state_dict'])
                trainer.optimizer_encoder_decoder.load_state_dict(checkpoint['optimizer_encoder_decoder'])
                trainer.optimizer_steganalyzer.load_state_dict(checkpoint['optimizer_steganalyzer'])
                if trainer.clip_memory_bank is not None and 'clip_memory_bank' in checkpoint:
                    trainer.clip_memory_bank.load_state_dict(checkpoint['clip_memory_bank'])
                if trainer.dino_memory_bank is not None and 'dino_memory_bank' in checkpoint:
                    trainer.dino_memory_bank.load_state_dict(checkpoint['dino_memory_bank'])
                trainer.best_loss = checkpoint.get('best_loss', float('inf'))
                trainer.phase2_msg_head_frozen = checkpoint.get('phase2_msg_head_frozen', False)
                if trainer.phase2_msg_head_frozen:
                    trainer.freeze_msg_head()
                start_epoch = checkpoint['epoch'] + 1
                start_phase = checkpoint['phase']
                print(f"Resuming from Phase {start_phase}, Epoch {start_epoch}")
            else:
                print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")

        train_loader, val_loader = trainer.get_dataloaders()
        run_training_loop(trainer, train_loader, val_loader, cfg, start_phase, start_epoch, log_file)

    elif args.command == 'continue':
        checkpoint_dir = args.folder
        config_path = os.path.join(checkpoint_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            cfg = config.Config()
            for key, value in config_dict.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        else:
            cfg = config.Config()
        cfg.checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
        cfg.log_dir = os.path.join(checkpoint_dir, 'tensorboard')
        log_file = os.path.join(checkpoint_dir, 'train.log')
        setup_logger(log_file, cfg, append=True)

        trainer = Trainer(cfg)
        checkpoint_path = os.path.join(cfg.checkpoint_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            trainer.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            trainer.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            trainer.steganalyzer.load_state_dict(checkpoint['steganalyzer_state_dict'])
            trainer.optimizer_encoder_decoder.load_state_dict(checkpoint['optimizer_encoder_decoder'])
            trainer.optimizer_steganalyzer.load_state_dict(checkpoint['optimizer_steganalyzer'])
            if trainer.clip_memory_bank is not None and 'clip_memory_bank' in checkpoint:
                trainer.clip_memory_bank.load_state_dict(checkpoint['clip_memory_bank'])
            if trainer.dino_memory_bank is not None and 'dino_memory_bank' in checkpoint:
                trainer.dino_memory_bank.load_state_dict(checkpoint['dino_memory_bank'])
            trainer.best_loss = checkpoint.get('best_loss', float('inf'))
            trainer.phase2_msg_head_frozen = checkpoint.get('phase2_msg_head_frozen', False)
            if trainer.phase2_msg_head_frozen:
                trainer.freeze_msg_head()
            start_epoch = checkpoint['epoch'] + 1
            start_phase = checkpoint['phase']
            print(f"Resuming from Phase {start_phase}, Epoch {start_epoch}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            return

        train_loader, val_loader = trainer.get_dataloaders()
        run_training_loop(trainer, train_loader, val_loader, cfg, start_phase, start_epoch, log_file)


if __name__ == '__main__':
    main()
