from hyperparams_base import *


# =============================================================================
# Transformer Tacotron (Transformer acoustic model)
# =============================================================================

symbols_embedding_dim = 512

prenet_dims = [256, 256]

postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_n_convolutions = 5

max_decoder_steps = 1000
gate_threshold = 0.9

transformer_d_model = 256
transformer_nhead = 4
transformer_encoder_layers = 4
transformer_decoder_layers = 4
transformer_ffn_dim = 1024
transformer_dropout = 0.1

max_text_positions = 2048


# =============================================================================
# Epoch-based training schedule and logging
# =============================================================================

# Validation / artifacts are driven by epoch.
val_every_epoch = 1
image_every_epoch = 1
sample_every_epoch = 5
checkpoint_every_epoch = 5

# Guided attention: strong in the beginning, then cosine decay.
guided_attn_weight_start = 2.0
guided_attn_weight_end = 0.1
guided_attn_warmup_epochs = 30
guided_attn_decay_epochs = 60
guided_attn_sigma = 0.2

# Gate learning is intentionally stronger than in the original setup.
gate_loss_weight = 2.0
gate_pos_weight = 10.0


# =============================================================================
# Resume / restore
# =============================================================================

resume_transformer_tacotron_path = None
restore_transformer_step = None


# =============================================================================
# Logging
# =============================================================================

log_dir = transformer_log_dir
