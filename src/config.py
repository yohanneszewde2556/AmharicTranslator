import torch

# Configuration hyperparameters for the NMT model

# Tokenization
VOCAB_SIZE = 32000
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

# Transformer Dimensions
D_MODEL = 512
N_HEADS = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1

# Training Constants
BATCH_SIZE = 64
LEARNING_RATE = 1.0       # Used as base for Noam scheduler — do not change
WEIGHT_DECAY = 0.001
WARMUP_STEPS = 12000
NUM_EPOCHS = 40
MAX_LENGTH = 128
