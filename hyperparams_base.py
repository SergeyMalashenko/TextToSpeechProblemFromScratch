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
checkpoint_path = "./checkpoint"
sample_path = "./samples"

logs_root = "./logs"
rnn_log_dir = "./logs/rnn_tacotron"
transformer_log_dir = "./logs/transformer_tacotron"

simple_vocoder_checkpoint_path = "./checkpoint_vocoder"
hifigan_checkpoint_path = "./checkpoint_hifigan"

mel2mag_log_dir = "./logs/mel2mag"

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


# =============================================================================
# HiFi-GAN
# =============================================================================

hifigan_batch_size = 16
hifigan_lr = 2e-4
hifigan_epochs = 10000
hifigan_segment_size = hop_length * 64
hifigan_save_step = 5000
hifigan_val_step = 2000

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
restore_hifigan_step = None
