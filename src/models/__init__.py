from .cnn_1d import CNN1D
from .cnn_2d import CNN2D
from .cnn_pretrained import (
    ResNet, AlexNet, VGG, SqueezeNet, DenseNet
)

import torch


MODELS = {
    "1d_cnn": CNN1D,
    "2d_cnn": CNN2D,
    "resnet": ResNet,
    "alexnet": AlexNet,
    "vgg": VGG,
    "squeezenet": SqueezeNet,
    "densenet": DenseNet,
}


OPTS = {
    "adam": torch.optim.Adam
}


LOSSES = {
    "cross_entropy": torch.nn.CrossEntropyLoss
}
