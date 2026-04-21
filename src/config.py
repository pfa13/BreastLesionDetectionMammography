import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data"
ANNOTATIONS_DIR = "data/annotations/fold_0"

NUM_CLASSES = 2  # mass, calcification

IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 3