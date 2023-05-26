import torch

VAL_SPLIT = 0.15

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda:0" else False

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 1e-5
NUM_EPOCHS = 2000
BATCH_SIZE = 1

# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128
INPUT_IMAGE_DEPTH = 64

# define threshold to filter weak predictions
THRESHOLD = 0.5
