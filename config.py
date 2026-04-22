import torch

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_size = 256
    message_length = 30
    batch_size = 8

    lr_encoder = 8e-4
    lr_decoder = 8e-4
    lr_steganalyzer = 5e-4

    num_epochs_phase1 = 200
    num_epochs_phase2 = 100
    num_epochs_phase3 = 100

    scheduler_T_max = num_epochs_phase1 + num_epochs_phase2 + num_epochs_phase3

    lambda_pixel = 0.3
    lambda_ssim = 0.3
    lambda_perceptual = 0.3
    lambda_message = 1.0
    lambda_adv = 0.3
    lambda_privacy = 1.0

    privacy_margin = 0.7

    use_advanced_privacy = True
    privacy_mode = 'retrieval_escape'

    clip_weight = 1.0
    dino_weight = 0.5
    dino_model_path = './models/dinov2_vits14_pretrain.pth'
    triplet_weight = 0.15
    triplet_margin = 0.5

    use_retrieval_attacker = True
    attacker_feature_dim = 512
    attacker_memory_size = 1000
    attacker_temperature = 0.05
    attacker_weight = 0.5

    use_hard_negatives = True
    hard_negative_k = 5
    memory_warmup_batches = 50

    data_dir = None
    train_dir = 'train'
    val_dir = 'val'

    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'

    num_workers = 4
    save_interval = 10

    noise_config = 'identity'

    def get_phase_weights(self, phase):
        if phase == 1:
            return {
                'visual': 10.0,
                'message': 10.0,
                'adv': 0.0,
                'privacy': 0.0,
                'noised': 1.0,
                'residual_penalty': 0.5
            }
        elif phase == 2:
            return {
                'visual': 10.0,
                'message': 10.0,
                'adv': 5.0,
                'privacy': 0.0,
                'noised': 1.0,
                'residual_penalty': 0.5
            }
        elif phase == 3:
            return {
                'visual': 10.0,
                'message': 10.0,
                'adv': 3.0,
                'privacy': 10.0,
                'noised': 0.5,
                'residual_penalty': 0.3
            }
        else:
            raise ValueError(f'Invalid phase: {phase}')
