from hyperparams_base import *


# =============================================================================
# Mel2Mag model
# =============================================================================

mel2mag_hidden_dim = 512
mel2mag_kernel_size = 5
mel2mag_n_convolutions = 5
mel2mag_dropout = 0.5

resume_mel2mag_checkpoint = None
restore_step2 = None

log_dir = mel2mag_log_dir
