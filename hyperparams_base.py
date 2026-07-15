from text.symbols import symbols


# =============================================================================
# Audio
# =============================================================================

n_mels = 80
n_fft = 2048
sr = 22050
preemphasis = 0.97

frame_shift = 0.0125
frame_length = 0.05

hop_length = int(sr * frame_shift)
win_length = int(sr * frame_length)

power = 1.2
min_level_db = -100
ref_level_db = 20
max_db = 100
ref_db = 20
n_iter = 60

outputs_per_step = 1

# =============================================================================
# Text
# =============================================================================

cleaners = "english_cleaners"
n_symbols = len(symbols)


# =============================================================================
# Global training defaults
# =============================================================================

seed = 42

epochs = 300
lr = 0.001
batch_size = 32

num_workers = 32
val_ratio = 0.02

max_checkpoints_to_keep = 5
experiment_name = ""


# =============================================================================
# Paths
# =============================================================================

data_path = "./data/LJSpeech-1.1"

# Unified output directory structure
outputs_root = "./outputs"
checkpoint_root = "./outputs/checkpoints"
logs_root = "./outputs/logs"
samples_root = "./outputs/samples"
synthesis_root = "./outputs/synthesis"

# Acoustic model checkpoints
checkpoint_path = "./outputs/checkpoints/rnn"

# Acoustic model logs
rnn_log_dir = "./outputs/logs/rnn"
transformer_log_dir = "./outputs/logs/transformer"
mel2mag_log_dir = "./outputs/logs/mel2mag"

# Training samples and standalone synthesis outputs
sample_path = "./outputs/samples/rnn"
synthesis_path = "./outputs/synthesis/rnn"

# Auxiliary acoustic checkpoints
mel2mag_checkpoint_path = "./outputs/checkpoints/mel2mag"

# Vocoder checkpoints
simple_vocoder_checkpoint_path = "./outputs/checkpoints/vocoder_simple"
hifigan_checkpoint_path = "./outputs/checkpoints/vocoder_hifigan"

# =============================================================================
# Unified synthesis defaults
# =============================================================================

default_acoustic_model = "rnn"
default_vocoder_backend = "griffinlim"

default_wav_gain = 1.0
default_peak_norm = False
default_peak_target = 0.95


# =============================================================================
# Simple neural vocoder
# =============================================================================

vocoder_batch_size = 16
vocoder_lr = 2e-4
vocoder_epochs = 10000
vocoder_segment_size = hop_length * 64
vocoder_save_step = 5000

resume_vocoder_checkpoint = None
restore_simple_vocoder_step = None
restore_vocoder_step = None
simple_vocoder_checkpoint_path = "./outputs/checkpoints/vocoder"


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
hifigan_log_dir = "./outputs/logs/hifigan"
hifigan_sample_path = "./outputs/samples/hifigan"

hifigan_lambda_fm = 2.0
hifigan_lambda_mel = 45.0
hifigan_checkpoint_path = "./outputs/checkpoints/hifigan"

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
