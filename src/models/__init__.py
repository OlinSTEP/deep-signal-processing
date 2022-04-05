from .cnn_1d import CNN1D
from .cnn_2d import CNN2D

import torch


MODELS = {
    "1d_cnn": CNN1D,
    "2d_cnn": CNN2D,
}


OPTS = {
    "adam": torch.optim.Adam
}


LOSSES = {
    "cross_entropy": torch.nn.CrossEntropyLoss
}
