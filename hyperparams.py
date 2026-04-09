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
# Model: Tacotron 2 style
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

#postnet_embedding_dim = 512
#postnet_kernel_size = 5
#postnet_n_convolutions = 5

postnet_cbhg_k = 8
postnet_bank_channels = 128
postnet_proj_channels = [256, n_mels]
postnet_highway_units = 128
postnet_highway_layers = 4
postnet_gru_units = 128
postnet_dropout = 0.5

max_decoder_steps = 1000
gate_threshold = 0.5
p_attention_dropout = 0.1
p_decoder_dropout = 0.1

# =============================================================================
# Training
# =============================================================================
epochs = 10000
lr = 0.001
#batch_size = 32
batch_size = 64
save_step = 2000
image_step = 500

# =============================================================================
# Paths
# =============================================================================
data_path = "./data/LJSpeech-1.1"
checkpoint_path = "./checkpoint"
sample_path = "./samples"
