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

# =============================================================================
# Unified synthesis defaults
# =============================================================================

default_acoustic_model = "rnn"
default_vocoder_backend = "griffinlim"

default_wav_gain = 1.0
default_peak_norm = False
default_peak_target = 0.95
