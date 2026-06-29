from hyperparams_base import *

# =============================================================================
# Training
# =============================================================================

epochs = 300
lr = 1e-3

batch_size = 96
num_workers = 32

# =============================================================================
# Tacotron2 model
# =============================================================================

symbols_embedding_dim = 512

encoder_embedding_dim = 512
encoder_kernel_size = 5
encoder_n_convolutions = 3
encoder_dropout = 0.5

attention_rnn_dim = 1024
decoder_rnn_dim = 1024

attention_dim = 128
attention_location_n_filters = 32
attention_location_kernel_size = 31

prenet_dims = [256, 256]

postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_n_convolutions = 5

max_decoder_steps = 1000
gate_threshold = 0.5

p_attention_dropout = 0.1
p_decoder_dropout = 0.1

# =============================================================================
# Loss
# =============================================================================

guided_attn_weight = 1.5
guided_attn_sigma = 0.4

# =============================================================================
# Attention metrics
# =============================================================================

attention_diag_width = 0.08
attention_peak_threshold = 0.6

# =============================================================================
# Dataset split
# =============================================================================

val_ratio = 0.02

# =============================================================================
# Logging / checkpoints
# =============================================================================

image_step = 500
save_step = 2000
sample_step = 2000

max_checkpoints_to_keep = 5

# =============================================================================
# Resume
# =============================================================================

resume_checkpoint = None

# =============================================================================
# Output paths
# =============================================================================

checkpoint_path = "./outputs/checkpoints/rnn"
rnn_log_dir = "./outputs/logs/rnn"
sample_path = "./outputs/samples/rnn"
synthesis_path = "./outputs/synthesis/rnn"
log_dir = "./outputs/logs/rnn"

# =============================================================================
# TensorBoard
# =============================================================================

log_dir = rnn_log_dir
