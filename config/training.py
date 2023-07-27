import torch

ARCHITECTURE = "LocalNet"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda:0" else False

INIT_LR = 1e-3
NUM_EPOCHS = 200
TR_BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
