import torch

ARCHITECTURE = "GlobalNet"

DEVICE = "cpu"
PIN_MEMORY = True if DEVICE == "cuda:0" else False

INIT_LR = 1e-6
NUM_EPOCHS = 200
TR_BATCH_SIZE = 6
VAL_BATCH_SIZE = 1
