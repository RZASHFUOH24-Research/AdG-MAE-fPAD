"""
Central configuration for AdG-MAE (Adaptive Gated Masked Autoencoder).
Edit the values below directly instead of passing command-line arguments.

Values here follow the implementation details reported in the paper
"AdG-MAE: Adaptive Gated Masked Autoencoder for Domain-Generalised Face
Anti-Spoofing" (Sections IV.B / IV.C and Appendix C, Table "Hyperparameter
settings for AdG-MAE").
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# SHARED / ARCHITECTURE SETTINGS
# =========================================================================
IMG_SIZE = 256          # paper: 256 x 256 input
PATCH_SIZE = 16         # paper: 16 x 16 patches

# TinyViT (Gated) encoder -- shared between pretraining and fine-tuning
ENCODER_EMBED_DIM = 192
ENCODER_DEPTH = 8
ENCODER_HEADS = 3

# MAE decoder (discarded after Stage 1) -- see Appendix "Ablation on Decoder
# Architecture": Depth=8, Width=512 is the configuration used in the paper.
DECODER_EMBED_DIM = 512
DECODER_DEPTH = 8
DECODER_HEADS = 8

# =========================================================================
# STAGE 1: SELF-SUPERVISED PRE-TRAINING (structural grounding)
# =========================================================================
# Root folders (union of ALL source domains) containing "Real" / "Spoof"
# sub-folders, used for unlabeled MAE pre-training.
PRETRAIN_DATA_ROOTS = [
    r"C:\path\to\CASIA_FASD",
    r"C:\path\to\Idiap_Replay_Attack",
    r"C:\path\to\MSU-MFSD",
    r"C:\path\to\OULU_NPU",
]

MASK_RATIO = 0.75          # paper: 75% masking ratio (Section IV.C, Table I)
PRETRAIN_BATCH_SIZE = 16   # paper: batch size 16
PRETRAIN_EPOCHS = 500      # paper: 500 epochs
PRETRAIN_LR = 1.5e-4       # paper: base LR 1.5e-4 (AdamW)
PRETRAIN_WEIGHT_DECAY = 0.05

PRETRAIN_SAVE_DIR = r"C:\path\to\checkpoints\AdG-MAE_pretrain"

# =========================================================================
# STAGE 2: SUPERVISED FINE-TUNING (spectral refinement)
# =========================================================================
# Checkpoint produced by Stage 1 (pretrain.py)
PRETRAINED_CKPT = r"C:\path\to\checkpoints\AdG-MAE_pretrain\latest.pth"

# Manual train / val / test split. Each root is scanned recursively; a file
# is labelled "live" (1) if its path contains "real"/"live"/"true", and
# "spoof" (0) if it contains "spoof"/"attack"/"fake".
FT_TRAIN_ROOTS = [
    r"C:\path\to\dataset\Train",
]
FT_VAL_ROOTS = [
    r"C:\path\to\dataset\Val",
]
FT_TEST_ROOTS = [
    r"C:\path\to\dataset\Test",
]

FT_SAVE_DIR = r"C:\path\to\Final_Fine_Tuning_Results"

FT_BATCH_SIZE = 64      # paper: batch size 64
FT_EPOCHS = 50          # paper: 50 fine-tuning epochs
WARMUP_EPOCHS = 5       # paper: first 5 epochs, encoder frozen (AdamW, head only)

LR_BACKBONE = 4e-6      # paper: backbone LR 4e-6 (post warm-up, SAM)
LR_HEAD = 1e-3          # paper: classification-head LR 1e-3
FT_WEIGHT_DECAY = 0.1   # paper: weight decay 0.1 (Stage 2)

SAM_RHO = 0.1           # paper: SAM neighbourhood size rho = 0.1
DROPOUT_RATE = 0.2

# Spectral Re-weighting Augmentation (SRA), Eq. (3)-(4) in the paper.
# omega_low = 0.75, omega_high = 1.5 is the global-optimum configuration
# found by the grid search in Table "Ablation study on fine-tuning
# components" (Appendix D).
SRA_PROB = 0.5
SRA_OMEGA_LOW = 0.75
SRA_OMEGA_HIGH = 1.5

LABEL_SMOOTHING = 0.1
