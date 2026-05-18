import torch


class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_size = 256
    message_length = 30
    batch_size = 8

    lr_encoder = 1e-3
    lr_decoder = 1e-3
    lr_steganalyzer = 1e-3

    num_epochs_phase1 = 130
    num_epochs_phase2 = 150

    discriminator_loss_threshold = 0.4

    lambda_pixel = 1.0
    lambda_ssim = 2.0
    lambda_message = 1.0
    lambda_adv = 0.1
    lambda_privacy = 0.3

    msg_budget = 0.04
    prv_budget = 0.06

    privacy_margin_start = 0.05
    privacy_margin_end = 0.30

    use_clip_redirect = False
    clip_memory_size = 1000
    clip_hard_negative_k = 5
    clip_redirect_margin = 0.3
    clip_max_target_sim = 0.55
    clip_redirect_weight = 0.3
    dino_redirect_weight = 0.2
    dino_redirect_margin = 0.3
    dino_max_target_sim = 0.55

    clip_weight = 1.0
    dino_weight = 1.0
    dino_model_path = './models/dinov2_vits14_pretrain.pth'

    use_retrieval_attacker = False
    attacker_feature_dim = 512
    attacker_memory_size = 1000

    memory_warmup_batches = 50

    bit_accuracy_threshold = 0.90
    recovery_steps = 10

    data_dir = None
    train_dir = 'train'
    val_dir = 'val'

    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'

    num_workers = 4
    save_interval = 10

    noise_config = 'identity'

    def get_phase_weights(self, phase, epoch=1):
        if phase == 1:
            total = self.num_epochs_phase1
            third = total // 3
            if epoch <= third:
                adv = self.lambda_adv
            elif epoch <= 2 * third:
                adv = self.lambda_adv * 2
            else:
                adv = self.lambda_adv * 3
            return {
                'visual': 10.0,
                'message': 10.0,
                'adv': adv,
                'privacy': 0.0,
                'noised': 1.0,
                'residual_penalty': 0.5
            }
        elif phase == 2:
            total = self.num_epochs_phase2
            third = total // 3
            if epoch <= third:
                privacy = 3.0
                message = 15.0
            elif epoch <= 2 * third:
                privacy = 6.0
                message = 12.0
            else:
                privacy = 10.0
                message = 10.0
            return {
                'visual': 1.0,
                'message': message,
                'adv': 2.0,
                'privacy': privacy,
                'noised': 0.3,
                'residual_penalty': 0.2
            }
        else:
            raise ValueError(f'Invalid phase: {phase}')

    def get_escape_margin(self, phase, epoch):
        if phase == 1:
            return 0.0
        total = self.num_epochs_phase2
        progress = min(1.0, (epoch - 1) / max(1, total - 1))
        return self.privacy_margin_start + (self.privacy_margin_end - self.privacy_margin_start) * progress

    def get_prv_noise_ratio(self, epoch):
        total = self.num_epochs_phase1
        warmup_end = total * 0.3
        ramp_end = total * 0.7
        if epoch <= warmup_end:
            return 0.0
        elif epoch <= ramp_end:
            return (epoch - warmup_end) / (ramp_end - warmup_end)
        else:
            return 1.0
