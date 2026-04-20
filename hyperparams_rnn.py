from hyperparams_base import *


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

resume_tacotron_checkpoint = None
restore_step1 = None

log_dir = rnn_log_dir
