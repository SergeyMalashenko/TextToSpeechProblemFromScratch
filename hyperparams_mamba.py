from hyperparams_base import *

# =============================================================================
# MambaTacotron2 training configuration
# =============================================================================

# =============================================================================
# Training strategy (aligned with RNN baseline)
# =============================================================================

epochs = 300
lr = 4e-4

batch_size = 32
num_workers = 32
pin_memory = True
val_ratio = 0.02
seed = 42

# =============================================================================
# DataLoader
# =============================================================================

use_bucket_sampler = False
bucket_size_multiplier = 20
bucket_drop_last = False

# =============================================================================
# Independent Mamba3 acoustic model
# =============================================================================

mamba_d_model = 384
mamba_encoder_layers = 4
mamba_decoder_layers = 4

mamba_d_state = 32
mamba_d_conv = 4
mamba_expand = 2
mamba_dropout = 0.2

mamba_attention_dim = 128
mamba_attention_location_filters = 32
mamba_attention_location_kernel_size = 31

mamba_prenet_hidden = 256
mamba_prenet_dropout = 0.5
mamba_step_context_feedback = True

# Decoder branch for experiments:
#   "mamba" -> stateful step-by-step Mamba3 decoder
#   "rnn"        -> Tacotron-style step-wise autoregressive RNN decoder
mamba_decoder_type = "mamba"
mamba_rnn_attention_dim = 1024
mamba_rnn_decoder_dim = 1024
mamba_rnn_prenet_dims = [256, 256]
mamba_rnn_prenet_dropout = 0.5
mamba_rnn_attention_dropout = 0.1
mamba_rnn_decoder_dropout = 0.1

postnet_channels = 512
postnet_kernel_size = 5
postnet_layers = 5
postnet_dropout = 0.5

max_decoder_steps = 1000
gate_threshold = 0.5

# =============================================================================
# Loss
# =============================================================================

gate_pos_weight = 1.0
guided_attn_sigma = 0.4

# Keep constant guided-attention pressure, same style as RNN's fixed weight.
guided_attn_weight_start = 1.0
guided_attn_weight_end = 1.0
guided_attn_decay_start_epoch = 0
guided_attn_decay_end_epoch = 0

# =============================================================================
# Optimizer / regularization
# =============================================================================

adam_beta1 = 0.9
adam_beta2 = 0.98
adam_eps = 1e-9
weight_decay = 1.0e-4
clip_grad_norm = 1.0

# =============================================================================
# Epoch-based LR schedule (approximate RNN warmup+decay behavior)
# =============================================================================

lr_warmup_epochs = 40
lr_step_epochs = 1
lr_step_gamma = 0.98
lr_min = 1e-5

# =============================================================================
# Resume / restore
# =============================================================================

#resume_mamba_checkpoint = "outputs/checkpoints/mamba3_step/checkpoint_mamba_tacotron2_epoch_0071.pth.tar"
resume_mamba_checkpoint = None 

# =============================================================================
# Workflow cadence
# =============================================================================

validate_every_epoch = 1
save_every_epoch = 1
sample_every_epoch = 5
log_alignment_every_epoch = 1

val_every_epoch = validate_every_epoch
checkpoint_every_epoch = save_every_epoch
image_every_epoch = log_alignment_every_epoch

# =============================================================================
# Output paths
# =============================================================================

checkpoint_path = "./outputs/checkpoints/mamba3_step"
mel2mag_checkpoint_path = "./outputs/checkpoints/mel2mag"
mamba_log_dir = "./outputs/logs/mamba3_step"
sample_path = "./outputs/samples/mamba3_step"
synthesis_path = "./outputs/synthesis/mamba3_step"
log_dir = "./outputs/logs/mamba3_step"
max_checkpoints_to_keep = 5
