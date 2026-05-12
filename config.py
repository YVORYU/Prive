import torch

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_size = 256
    message_length = 30
    batch_size = 8

    lr_encoder = 1e-3
    lr_decoder = 1e-3
    lr_steganalyzer = 1e-3

    num_epochs_phase1 = 300
    num_epochs_phase2 = 100

    discriminator_loss_threshold = 0.1

    lambda_pixel = 1.0
    lambda_ssim = 2.0
    lambda_perceptual = 0.3
    use_perceptual_loss = False
    lambda_message = 1.0
    lambda_adv = 0.1
    lambda_privacy = 0.3

    privacy_margin = 0.5

    use_advanced_privacy = True
    privacy_mode = 'retrieval_escape'

    clip_weight = 1.0
    dino_weight = 0.5
    dino_model_path = './models/dinov2_vits14_pretrain.pth'
    triplet_weight = 0.15
    triplet_margin = 0.5

    use_retrieval_attacker = False
    attacker_feature_dim = 512
    attacker_memory_size = 1000
    attacker_temperature = 0.05
    attacker_weight = 0.5

    use_hard_negatives = False
    hard_negative_k = 5
    memory_warmup_batches = 50

    use_clip_redirect = False
    clip_memory_size = 1000
    clip_hard_negative_k = 5
    clip_redirect_margin = 0.45
    clip_max_target_sim = 0.55
    clip_redirect_weight = 0.3
    dino_redirect_weight = 0.2
    dino_redirect_margin = 0.45
    dino_max_target_sim = 0.55

    use_text_escape = False
    text_escape_weight = 0.2
    text_escape_prompts = [
        "a blurry image",
        "an abstract pattern",
        "a textured surface",
        "a low quality photo",
        "a pixelated image",
        "a noisy image",
        "a photo of a mountain landscape",
        "a photo of an ocean sunset",
    ]

    use_ortho_grad = False
    ortho_alpha = 0.3

    data_dir = None
    train_dir = 'train'
    val_dir = 'val'

    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'

    num_workers = 4
    save_interval = 10

    noise_config = 'identity'
    pixel_budget = 0.08

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
                privacy = 1.0
            elif epoch <= 2 * third:
                privacy = 1.5
            else:
                privacy = 2.0
            return {
                'visual': 10.0,
                'message': 20.0,
                'adv': 5.0,
                'privacy': privacy,
                'noised': 0.5,
                'residual_penalty': 0.5
            }
        else:
            raise ValueError(f'Invalid phase: {phase}')
