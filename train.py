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

#定义设置随机种子的函数，用于确保实验结果的可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#保存配置文件
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
        'discriminator_steps': cfg.discriminator_steps,
        'lr_encoder': cfg.lr_encoder,
        'lr_decoder': cfg.lr_decoder,
        'lr_steganalyzer': cfg.lr_steganalyzer,
        'lambda_pixel': cfg.lambda_pixel,
        'lambda_ssim': cfg.lambda_ssim,
        'lambda_perceptual': cfg.lambda_perceptual,
        'use_perceptual_loss': cfg.use_perceptual_loss,
        'lambda_message': cfg.lambda_message,
        'lambda_adv': cfg.lambda_adv,
        'lambda_privacy': cfg.lambda_privacy,
        'privacy_margin': cfg.privacy_margin,
        'noise_config': cfg.noise_config,
        'num_workers': cfg.num_workers,
        'save_interval': cfg.save_interval,
        'use_advanced_privacy': cfg.use_advanced_privacy,
        'privacy_mode': cfg.privacy_mode,
        'clip_weight': cfg.clip_weight,
        'dino_weight': cfg.dino_weight,
        'dino_model_path': cfg.dino_model_path,
        'triplet_weight': cfg.triplet_weight,
        'triplet_margin': cfg.triplet_margin,
        'use_retrieval_attacker': cfg.use_retrieval_attacker,
        'attacker_feature_dim': cfg.attacker_feature_dim,
        'attacker_memory_size': cfg.attacker_memory_size,
        'attacker_temperature': cfg.attacker_temperature,
        'attacker_weight': cfg.attacker_weight,
        'use_hard_negatives': cfg.use_hard_negatives,
        'hard_negative_k': cfg.hard_negative_k,
        'memory_warmup_batches': cfg.memory_warmup_batches
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4)

#训练刚开始时，写配置信息到日志文件
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
            'train_dir': cfg.train_dir,
            'val_dir': cfg.val_dir,
            'image_size': cfg.image_size,
            'message_length': cfg.message_length,
            'batch_size': cfg.batch_size,
            'num_epochs_phase1': cfg.num_epochs_phase1,
            'num_epochs_phase2': cfg.num_epochs_phase2,
            'discriminator_steps': cfg.discriminator_steps,
            'lr_encoder': cfg.lr_encoder,
            'lr_decoder': cfg.lr_decoder,
            'lr_steganalyzer': cfg.lr_steganalyzer,
            'lambda_pixel': cfg.lambda_pixel,
            'lambda_ssim': cfg.lambda_ssim,
            'lambda_perceptual': cfg.lambda_perceptual,
            'use_perceptual_loss': cfg.use_perceptual_loss,
            'lambda_message': cfg.lambda_message,
            'lambda_adv': cfg.lambda_adv,
            'lambda_privacy': cfg.lambda_privacy,
            'privacy_margin': cfg.privacy_margin,
            'noise_config': cfg.noise_config,
            'num_workers': cfg.num_workers,
            'save_interval': cfg.save_interval,
            'use_advanced_privacy': cfg.use_advanced_privacy,
            'privacy_mode': cfg.privacy_mode,
            'clip_weight': cfg.clip_weight,
            'dino_weight': cfg.dino_weight,
            'dino_model_path': cfg.dino_model_path,
            'triplet_weight': cfg.triplet_weight,
            'triplet_margin': cfg.triplet_margin,
            'use_retrieval_attacker': cfg.use_retrieval_attacker,
            'attacker_feature_dim': cfg.attacker_feature_dim,
            'attacker_memory_size': cfg.attacker_memory_size,
            'attacker_temperature': cfg.attacker_temperature,
            'attacker_weight': cfg.attacker_weight,
            'use_hard_negatives': cfg.use_hard_negatives,
            'hard_negative_k': cfg.hard_negative_k,
            'memory_warmup_batches': cfg.memory_warmup_batches
        }
        for key, value in config_dict.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nNetwork Architecture:\n")
        f.write("-" * 40 + "\n")
        f.write("  Encoder: UNet with message fusion\n")
        f.write("  Decoder: CNN with sigmoid output\n")
        f.write("  Steganalyzer: CNN with SpectralNorm\n")
        f.write("  ContentRecognizer: CLIP (frozen)\n")
        f.write("\n")

#写每个batch的loss到日志文件
def log_batch_losses(log_file, batch_idx, losses, phase):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"Phase {phase} - Batch {batch_idx}: ")
        f.write(f"L_total={losses['L_total']:.6f}, ")
        f.write(f"L_visual={losses['L_visual']:.6f}, ")
        f.write(f"L_message={losses['L_message']:.6f}, ")
        f.write(f"L_adv={losses['L_adv']:.6f}, ")
        f.write(f"discr_cover_bce={losses['discr_cover_bce']:.6f}, ")
        f.write(f"discr_encod_bce={losses['discr_encod_bce']:.6f}, ")
        f.write(f"L_privacy={losses['L_privacy']:.6f}\n")

#写每个epoch的loss到日志文件
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

#定义训练循环的函数
"""
参数:
    trainer: 训练器对象
    train_loader: 训练数据加载器
    val_loader: 验证数据加载器
    cfg: 配置对象
    start_phase: 开始训练的阶段
    start_epoch: 开始训练的epoch
    日志文件路径
"""
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
        
        for epoch in range(phase_start_epoch, num_epochs+1):
            #将所有网络设置为训练模式
            trainer.encoder.train()
            trainer.decoder.train()
            trainer.steganalyzer.train()
            
            epoch_losses = {}
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                #遍历训练数据集，每个batch进行训练，返回当前batch的loss
                losses = trainer.train_step(batch, phase, epoch)
                # 累加当前batch的loss，用于计算epoch的平均loss
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value
                
                num_batches += 1
                # 写入当前batch的loss到日志文件
                log_batch_losses(log_file, batch_idx, losses, phase)
            # 每轮训练完毕，计算epoch的平均loss
            for key in epoch_losses:
                epoch_losses[key] /= num_batches

            #save_vis取值为True时，保存验证集上的生成图像
            save_vis = epoch % cfg.save_interval == 0
            # 验证模型在验证集上的性能
            val_losses = trainer.validate(val_loader, phase, epoch, save_vis)
            # 写入当前epoch的loss到日志文件
            log_epoch_losses(log_file, epoch, epoch_losses, val_losses, phase)
            
            print(f'Phase {phase} - Epoch {epoch} - Train Loss: {epoch_losses["L_total"]:.4f} - Val Loss: {val_losses["L_total"]:.4f}')
            
            #将epoch损失写入tensorboard
            for key, value in epoch_losses.items():
                trainer.writer.add_scalar(f'Phase{phase}/Train/{key}', value, epoch)
            #将epoch损失写入tensorboard
            for key, value in val_losses.items():
                trainer.writer.add_scalar(f'Phase{phase}/Val/{key}', value, epoch)
            
            #比较当前epoch的验证集损失是否优于最佳损失
            is_best = val_losses['L_total'] < trainer.best_loss
            if is_best:
                trainer.best_loss = val_losses['L_total']
            if epoch % cfg.save_interval == 0:
                trainer.save_checkpoint(epoch, phase, is_best)
    
    print('Training completed!')
    trainer.writer.close()

#定义主函数，用于解析命令行参数并启动训练循环
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
    new_run_parser.add_argument('--discriminator-steps', type=int, default=config.Config.discriminator_steps,
                                help='Number of discriminator training steps per batch.')
    
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
    new_run_parser.add_argument('--lambda-perceptual', type=float, default=config.Config.lambda_perceptual,
                                help='Weight for perceptual loss.')
    new_run_parser.add_argument('--use-perceptual-loss', dest='use_perceptual_loss', action='store_true',
                                default=False,
                                help='Enable VGG perceptual loss (disabled by default).')
    new_run_parser.add_argument('--lambda-message', type=float, default=config.Config.lambda_message,
                                help='Weight for message loss.')
    new_run_parser.add_argument('--lambda-adv', type=float, default=config.Config.lambda_adv,
                                help='Weight for adversarial loss.')
    new_run_parser.add_argument('--lambda-privacy', type=float, default=config.Config.lambda_privacy,
                                help='Weight for privacy loss.')
    new_run_parser.add_argument('--privacy-margin', type=float, default=config.Config.privacy_margin,
                                help='Margin for privacy loss.')
    
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
        cfg.discriminator_steps = args.discriminator_steps
        
        cfg.lr_encoder = args.lr_encoder
        cfg.lr_decoder = args.lr_decoder
        cfg.lr_steganalyzer = args.lr_steganalyzer
        
        cfg.lambda_pixel = args.lambda_pixel
        cfg.lambda_ssim = args.lambda_ssim
        cfg.lambda_perceptual = args.lambda_perceptual
        cfg.use_perceptual_loss = args.use_perceptual_loss
        cfg.lambda_message = args.lambda_message
        cfg.lambda_adv = args.lambda_adv
        cfg.lambda_privacy = args.lambda_privacy
        cfg.privacy_margin = args.privacy_margin
        
        cfg.num_workers = args.num_workers
        cfg.save_interval = args.save_interval
        
        if args.noise:
            cfg.noise_config = args.noise
        
        experiment_name = args.name
        #创建实验目录
        experiment_dir = os.path.join('experiments', experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        #创建模型保存和日志保存目录
        cfg.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        cfg.log_dir = os.path.join(experiment_dir, 'tensorboard')
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        
        log_file = os.path.join(experiment_dir, 'train.log')
        setup_logger(log_file, cfg)
        #保存配置文件
        config_path = os.path.join(experiment_dir, 'config.json')
        save_config(config_path, cfg)
        #创建训练器
        trainer = Trainer(cfg)
        
        start_epoch = 1
        start_phase = 1
        
        if args.continue_from_folder:
            #加载最新的检查点
            checkpoint_path = os.path.join(args.continue_from_folder, 'checkpoints', 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                #加载模型状态字典
                trainer.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                trainer.decoder.load_state_dict(checkpoint['decoder_state_dict'])
                trainer.steganalyzer.load_state_dict(checkpoint['steganalyzer_state_dict'])
                trainer.optimizer_encoder_decoder.load_state_dict(checkpoint['optimizer_encoder_decoder'])
                trainer.optimizer_steganalyzer.load_state_dict(checkpoint['optimizer_steganalyzer'])
                if hasattr(trainer, 'optimizer_attacker') and 'optimizer_attacker' in checkpoint:
                    trainer.optimizer_attacker.load_state_dict(checkpoint['optimizer_attacker'])
                if hasattr(trainer, 'retrieval_attacker') and 'retrieval_attacker_state_dict' in checkpoint:
                    trainer.retrieval_attacker.load_state_dict(checkpoint['retrieval_attacker_state_dict'])
                #找到最佳损失
                trainer.best_loss = checkpoint.get('best_loss', float('inf'))
                start_epoch = checkpoint['epoch'] + 1
                start_phase = checkpoint['phase']
                print(f"Resuming from Phase {start_phase}, Epoch {start_epoch}")
            else:
                print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")
        
        #构建数据加载器和验证数据加载器
        train_loader, val_loader = trainer.get_dataloaders()
        #开始训练
        run_training_loop(trainer, train_loader, val_loader, cfg, start_phase, start_epoch, log_file)
    
    #继续训练
    elif args.command == 'continue':
        checkpoint_dir = args.folder
        #加载配置文件
        config_path = os.path.join(checkpoint_dir, 'config.json')
        #如果存在配置文件则加载，否则使用默认配置
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)            
            cfg = config.Config()
            for key, value in config_dict.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        else:
            cfg = config.Config()
        #设置模型保存和日志保存目录
        cfg.checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
        cfg.log_dir = os.path.join(checkpoint_dir, 'tensorboard')
        log_file = os.path.join(checkpoint_dir, 'train.log')
        setup_logger(log_file, cfg, append=True)
        #创建训练器
        trainer = Trainer(cfg)
        #加载最新的检查点
        checkpoint_path = os.path.join(cfg.checkpoint_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            trainer.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            trainer.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            trainer.steganalyzer.load_state_dict(checkpoint['steganalyzer_state_dict'])
            trainer.optimizer_encoder_decoder.load_state_dict(checkpoint['optimizer_encoder_decoder'])
            trainer.optimizer_steganalyzer.load_state_dict(checkpoint['optimizer_steganalyzer'])
            if hasattr(trainer, 'optimizer_attacker') and 'optimizer_attacker' in checkpoint:
                trainer.optimizer_attacker.load_state_dict(checkpoint['optimizer_attacker'])
            if hasattr(trainer, 'retrieval_attacker') and 'retrieval_attacker_state_dict' in checkpoint:
                trainer.retrieval_attacker.load_state_dict(checkpoint['retrieval_attacker_state_dict'])
            #找到最佳损失
            trainer.best_loss = checkpoint.get('best_loss', float('inf'))
            start_epoch = checkpoint['epoch'] + 1
            start_phase = checkpoint['phase']
            print(f"Resuming from Phase {start_phase}, Epoch {start_epoch}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            return
        #构建数据加载器和验证数据加载器
        train_loader, val_loader = trainer.get_dataloaders()
        #开始训练
        run_training_loop(trainer, train_loader, val_loader, cfg, start_phase, start_epoch, log_file)


if __name__ == '__main__':
    main()