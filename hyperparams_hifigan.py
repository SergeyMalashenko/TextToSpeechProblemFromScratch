from hyperparams_base import *


# =============================================================================
# HiFi-GAN
# =============================================================================

hifigan_batch_size = 16
hifigan_lr = 2e-4
hifigan_epochs = 250
hifigan_segment_size = hop_length * 64
hifigan_validate_every_epoch = 1
hifigan_save_every_epoch = 1
hifigan_sample_every_epoch = 5
hifigan_max_checkpoints_to_keep = 10  # 0 keeps all epoch checkpoints

hifigan_checkpoint_path = "./outputs/checkpoints/hifigan"
hifigan_log_dir = "./outputs/logs/hifigan"
hifigan_sample_path = "./outputs/samples/hifigan"

hifigan_lambda_fm = 2.0
hifigan_lambda_mel = 45.0

# The product of the upsample rates should match hop_length.
# For sr = 22050 and frame_shift = 0.0125:
# hop_length = 275 = 5 * 5 * 11
hifigan_upsample_rates = [5, 5, 11]
hifigan_upsample_kernel_sizes = [10, 10, 22]
hifigan_upsample_initial_channel = 512
hifigan_resblock_kernel_sizes = [3, 7, 11]
hifigan_resblock_dilation_sizes = [(1, 3, 5), (1, 3, 5), (1, 3, 5)]

resume_hifigan_checkpoint = None
restore_hifigan_epoch = None
