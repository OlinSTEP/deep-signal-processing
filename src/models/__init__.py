from typing import Dict, Type, Callable, Any, Union

from .model import Model
from .cnn_1d import CNN1D
from .cnn_2d import CNN2D
from .speechbrain_model import (
    SpeechBrainVoxCelebModel, SpeechBrainGoogleSpeechModel,
    SpeechBrainWav2Vec2Model
)
from .cnn_pretrained import (
    ResNet18, ResNet34, AlexNet, VGG, SqueezeNet, DenseNet
)

import torch


MODELS: Dict[str, Type[Model]] = {
    "1d_cnn": CNN1D,
    "2d_cnn": CNN2D,
    "voxceleb": SpeechBrainVoxCelebModel,
    "google_speech": SpeechBrainGoogleSpeechModel,
    "wav2vec2": SpeechBrainWav2Vec2Model,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "alexnet": AlexNet,
    "vgg": VGG,
    "squeezenet": SqueezeNet,
    "densenet": DenseNet,
}


OPTS: Dict[
    str,
    Union[
        Type[torch.optim.Adam],
        Type[torch.optim.Adadelta]
    ]
] = {
    "adam": torch.optim.Adam,
    "ada_delta": torch.optim.Adadelta
}


LOSSES: Dict[str, Type[Callable[..., Any]]] = {
    "cross_entropy": torch.nn.CrossEntropyLoss
}
