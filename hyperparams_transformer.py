from hyperparams_base import *

# =============================================================================
# Transformer Tacotron: next optimization experiment
# =============================================================================

# Main training budget
# Keep the budget moderate for the diagnostic run. If the dynamics are good,
# continue training from the checkpoint.
epochs = 500

# The previous conservative run used lr=1e-4 and improved very slowly.
# This is still far below the original aggressive lr=1.5e-3.
#lr = 1.5e-4
lr = 3.5e-4
#lr = 4.0e-4

batch_size = 48

# Safer than 32 workers; increase only if the data pipeline is stable.
num_workers = 16


# =============================================================================
# Transformer Tacotron model
# =============================================================================

symbols_embedding_dim = 512

prenet_dims = [256, 256]

postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_n_convolutions = 5

max_decoder_steps = 1000

# More permissive than 0.9. If synthesis stops too early, increase to 0.8.
gate_threshold = 0.7

transformer_d_model = 256
transformer_nhead = 4
transformer_encoder_layers = 4
transformer_decoder_layers = 4
transformer_ffn_dim = 1024
transformer_dropout = 0.1
transformer_ffn_type = "swiglu"

max_text_positions = 2048


# =============================================================================
# Optimizer
# =============================================================================

# Intended optimizer in train file:
# torch.optim.AdamW(..., betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
weight_decay = 1e-4
clip_grad_norm = 1.0


# =============================================================================
# Epoch-based LR schedule
# =============================================================================

# Recommended train-file behavior:
# - update LR once per epoch
# - do not call step-based Noam schedule inside the batch loop
lr_warmup_epochs = 50

#lr_decay_epoch_1 = 200
#lr_decay_epoch_2 = 350

lr_decay_epoch_1 = 250
lr_decay_epoch_2 = 325

lr_decay_factor_1 = 0.5
lr_decay_factor_2 = 0.25


# =============================================================================
# Epoch-based training schedule and logging
# =============================================================================

val_every_epoch = 5
image_every_epoch = 5
sample_every_epoch = 25
checkpoint_every_epoch = 25


# =============================================================================
# Guided attention
# =============================================================================

# Previous logs showed attn_loss ~= 1e-4 and sharpness ~= 0.10-0.11.
# Therefore we keep guided attention pressure high for longer instead of
# decaying it too quickly.
guided_attn_weight_start = 2.0
guided_attn_weight_end = 0.8
guided_attn_warmup_epochs = 50
guided_attn_decay_epochs = 300

# Softer diagonal corridor for Transformer attention.
guided_attn_sigma = 0.45


# =============================================================================
# Gate loss
# =============================================================================

gate_pos_weight = 2.0


# =============================================================================
# Resume / restore
# =============================================================================

resume_transformer_tacotron_path = None
restore_transformer_epoch = None

# =============================================================================
# Logging
# =============================================================================

checkpoint_path = "./outputs/checkpoints/transformer"
transformer_log_dir = "./outputs/logs/transformer"
sample_path = "./outputs/samples/transformer"
synthesis_path = "./outputs/synthesis/transformer"
log_dir = "./outputs/logs/transformer"
