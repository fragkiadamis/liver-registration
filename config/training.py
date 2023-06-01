import torch

ARCHITECTURE = "localnet"

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.20

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda:0" else False

INIT_LR = 1e-3
NUM_EPOCHS = 1000
BATCH_SIZE = 1

INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_DEPTH = 192
