from hyperparams_base import *

# =============================================================================
# MambaTacotron2 training configuration
# =============================================================================
# This file is intentionally independent from hyperparams_rnn.py.
# It keeps the same dataset/audio symbols inherited from hyperparams_base.py,
# but all model-specific parameters below target tts_mamba_model.py.

# =============================================================================
# Training
# =============================================================================

epochs = 400
lr = 5e-4

# Mamba/Transformer-style sequence models are usually more stable with a lower
# LR than the original RNN Tacotron setting. Start here; increase only after the
# attention and mel-energy curves look stable.

batch_size = 48
num_workers = 32
pin_memory = True
val_ratio = 0.02
seed = 42

# =============================================================================
# DataLoader
# =============================================================================

use_bucket_sampler = True
bucket_size_multiplier = 20
bucket_drop_last = False

# =============================================================================
# Independent Mamba acoustic model
# =============================================================================

# Main hidden width used by text embeddings, Mamba encoder, Mamba decoder,
# cross-attention, and mel projection trunk.
mamba_d_model = 384

# Encoder processes token embeddings.
mamba_encoder_layers = 4

# Decoder processes teacher-forced mel frames after the prenet.
mamba_decoder_layers = 4

# mamba-ssm block internals.
mamba_d_state = 16
mamba_d_conv = 4
mamba_expand = 2
mamba_dropout = 0.1

# Cross-attention from mel-time decoder states to encoded text memory.
mamba_attention_heads = 4

# Tacotron-style mel prenet before the Mamba decoder.
mamba_prenet_hidden = 256
mamba_prenet_dropout = 0.5

# Postnet over predicted mel frames.
postnet_channels = 512
postnet_kernel_size = 5
postnet_layers = 5
postnet_dropout = 0.5

# Autoregressive inference controls.
max_decoder_steps = 1000
gate_threshold = 0.5

# =============================================================================
# Loss configuration
# =============================================================================

# The Mamba model still uses Tacotron-style losses:
#   total = mel_loss + gate_loss + guided_attn_weight * attn_loss

gate_pos_weight = 5.0

# Scheduled guided attention.
# First, strong diagonal pressure helps alignment form. Later, the pressure is
# reduced so the acoustic model can refine timing and prosody.
guided_attn_weight_start = 1.0
guided_attn_weight_end = 0.01
guided_attn_decay_start_epoch = 150
guided_attn_decay_end_epoch = 200
guided_attn_sigma = 0.2

# Scheduled teacher forcing.
# During LR warmup the model receives full ground-truth mel input.
# After warmup, the ratio linearly decays to teacher_forcing_end.
teacher_forcing_start = 1.0
teacher_forcing_end = 0.2
teacher_forcing_decay_start_epoch = 200
teacher_forcing_decay_end_epoch = epochs

# =============================================================================
# Optimizer / regularization
# =============================================================================

adam_beta1 = 0.9
adam_beta2 = 0.98
adam_eps = 1e-9
weight_decay = 1e-4
clip_grad_norm = 1.0

# =============================================================================
# Epoch-based learning-rate schedule
# =============================================================================
# Simple strategy:
#   1. linear warmup for lr_warmup_epochs;
#   2. step decay every lr_step_epochs;
#   3. lr is clipped by lr_min.

lr_warmup_epochs = 50
lr_step_epochs = 50
lr_step_gamma = 0.5
lr_min = 1e-5

# =============================================================================
# Resume / restore
# =============================================================================

resume_mamba_checkpoint = None

# =============================================================================
# Epoch-based workflow cadence
# =============================================================================

validate_every_epoch = 5
save_every_epoch = 25
sample_every_epoch = 5
log_alignment_every_epoch = 1

# Transformer-style aliases supported by the trainer.
val_every_epoch = validate_every_epoch
checkpoint_every_epoch = save_every_epoch
image_every_epoch = log_alignment_every_epoch

# =============================================================================
# Output paths
# =============================================================================

mamba_log_dir = "./logs/mamba_tacotron"
log_dir = mamba_log_dir

# If hyperparams_base.py already defines checkpoint_path/sample_path, these
# assignments create Mamba-specific subfolders under the same root.
checkpoint_path = "./checkpoint/mamba_tacotron"
sample_path = "./samples/mamba_tacotron"
max_checkpoints_to_keep = 5
