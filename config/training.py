import torch

ARCHITECTURE = "localnet"

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.20

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda:0" else False

INIT_LR = 1e-3
NUM_EPOCHS = 200
BATCH_SIZE = 2
