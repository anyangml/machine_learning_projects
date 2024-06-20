import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

# image
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
IMAGE_CHANNEL = 3


# text
MAX_SEQ_LENGTH = 1024
