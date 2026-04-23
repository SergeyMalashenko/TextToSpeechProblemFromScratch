from hyperparams_base import *

epochs = 300
lr = 0.005
batch_size = 32

num_workers = 32

# =============================================================================
# Tacotron2 (RNN acoustic model)
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
# Loss configuration
# =============================================================================

guided_attn_weight = 1.0
guided_attn_sigma = 0.4
gate_pos_weight = 5.0


# =============================================================================
# Resume / restore
# =============================================================================

resume_checkpoint = None
restore_epoch1 = None
restore_step2 = None


# =============================================================================
# Epoch-based learning-rate schedule
# =============================================================================
# Supported values:
#   - "warmup_invsqrt_by_epoch"
#   - "exponential_by_epoch"
#   - "constant"

lr_schedule_type = "warmup_invsqrt_by_epoch"
lr_warmup_epochs = 20
lr_hold_epochs = 0
lr_min = 1e-5
lr_decay_gamma = 0.98


# =============================================================================
# Epoch-based workflow cadence
# =============================================================================

validate_every_epoch = 1
save_every_epoch = 1
sample_every_epoch = 1
log_alignment_every_epoch = 1


# =============================================================================
# Logging
# =============================================================================

log_dir = rnn_log_dir
