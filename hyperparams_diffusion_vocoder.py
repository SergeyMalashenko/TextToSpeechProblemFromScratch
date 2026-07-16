from hyperparams_base import *


# =============================================================================
# Diffusion vocoder
# =============================================================================

diffusion_vocoder_batch_size = 16
diffusion_vocoder_lr = 2e-4
diffusion_vocoder_epochs = 10000
diffusion_vocoder_segment_size = hop_length * 64
diffusion_vocoder_validate_every_epoch = 1
diffusion_vocoder_save_every_epoch = 1
diffusion_vocoder_sample_every_epoch = 5
diffusion_vocoder_max_checkpoints_to_keep = 0  # 0 keeps all epoch checkpoints

diffusion_vocoder_checkpoint_path = "./outputs/checkpoints/diffusion_vocoder"
diffusion_vocoder_log_dir = "./outputs/logs/diffusion_vocoder"
diffusion_vocoder_sample_path = "./outputs/samples/diffusion_vocoder"

diffusion_vocoder_train_timesteps = 1000
diffusion_vocoder_inference_steps = 50
diffusion_vocoder_beta_start = 1e-4
diffusion_vocoder_beta_end = 0.02

diffusion_vocoder_residual_layers = 30
diffusion_vocoder_residual_channels = 128
diffusion_vocoder_dilation_cycle = 10
diffusion_vocoder_embedding_dim = 128
diffusion_vocoder_conditioner_channels = 256
diffusion_vocoder_conditioner_layers = 3
diffusion_vocoder_upsample_rates = [5, 5, 11]
diffusion_vocoder_upsample_kernel_sizes = [10, 10, 22]
diffusion_vocoder_clip_grad_norm = 1.0
diffusion_vocoder_weight_decay = 1e-6
diffusion_vocoder_val_batches = 0
diffusion_vocoder_stft_weight = 1.0
diffusion_vocoder_mel_loss_weight = 5.0
diffusion_vocoder_stft_resolutions = [
    (512, 128, 512),
    (1024, 256, 1024),
    (2048, hop_length, win_length),
]

resume_diffusion_vocoder_checkpoint = None
restore_diffusion_vocoder_epoch = None
