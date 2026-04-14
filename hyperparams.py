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

transformer_d_model = 256
transformer_nhead = 4
transformer_encoder_layers = 4
transformer_decoder_layers = 4
transformer_ffn_dim = 1024
transformer_dropout = 0.1


# =============================================================================
# Mel2Mag model (renamed external PostNet)
# =============================================================================
mel2mag_hidden_dim = 512
mel2mag_kernel_size = 5
mel2mag_n_convolutions = 5
mel2mag_dropout = 0.5
resume_mel2mag_checkpoint = None

# Legacy compatibility aliases
postnet_hidden_dim = mel2mag_hidden_dim
postnet_kernel_size = mel2mag_kernel_size
postnet_n_convolutions = mel2mag_n_convolutions

# =============================================================================
# Simple neural vocoder
# =============================================================================
vocoder_batch_size = 16
vocoder_lr = 2e-4
vocoder_epochs = 10000
vocoder_segment_size = hop_length * 64
vocoder_checkpoint_path = "./checkpoint_vocoder"
vocoder_save_step = 5000

# =============================================================================
# HiFi-GAN
# =============================================================================
hifigan_batch_size = 16
hifigan_lr = 2e-4
hifigan_epochs = 10000
hifigan_segment_size = hop_length * 64
hifigan_checkpoint_path = "./checkpoint_hifigan"
hifigan_save_step = 5000
hifigan_val_step = 2000
hifigan_lambda_fm = 2.0
hifigan_lambda_mel = 45.0

# The product of the upsample rates should match hop_length.
# For sr=22050 and frame_shift=0.0125 we have hop_length = 275 = 5 * 5 * 11.
hifigan_upsample_rates = [5, 5, 11]
hifigan_upsample_kernel_sizes = [10, 10, 22]
hifigan_upsample_initial_channel = 512
hifigan_resblock_kernel_sizes = [3, 7, 11]
hifigan_resblock_dilation_sizes = [(1, 3, 5), (1, 3, 5), (1, 3, 5)]
restore_hifigan_step = None
restore_vocoder_step = None
