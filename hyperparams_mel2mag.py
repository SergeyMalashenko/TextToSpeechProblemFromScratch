from hyperparams_base import *


# =============================================================================
# Mel2Mag model
# =============================================================================

mel2mag_hidden_dim = 256
mel2mag_kernel_size = 3
mel2mag_dilations = [1, 2, 4, 8, 16, 1, 2, 4]
mel2mag_dropout = 0.1
mel2mag_output_activation = "sigmoid"  # "linear" enables an output-range ablation
mel2mag_mel_consistency_weight = 0.1
mel2mag_temporal_weight = 0.1
mel2mag_frequency_weight = 0.1
mel2mag_spectral_convergence_weight = 0.0  # opt-in: linear amplitude is sensitive to peaks
mel2mag_input_noise_std = 0.0  # opt-in ablation for predicted-mel robustness
mel2mag_warmup_steps = 4000
mel2mag_save_every_epoch = 1  # numbered checkpoints every N completed epochs
mel2mag_sample_every_epoch = 5  # 0 disables Griffin-Lim listening samples
sample_path = "./outputs/samples/mel2mag"

resume_mel2mag_checkpoint = None
restore_mel2mag_epoch = None

checkpoint_path = "./outputs/checkpoints/mel2mag"
mel2mag_checkpoint_path = checkpoint_path
mel2mag_log_dir = "./outputs/logs/mel2mag"
log_dir = "./outputs/logs/mel2mag"
