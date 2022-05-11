from typing import Tuple
from torch import Tensor
from argparse import Namespace

from abc import abstractmethod

import torch.nn as nn
import torchvision

from .model import Model


IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


###############################################################################
# FIRST LAYER FUNCTIONS TAKEN FROM FASTAI
# SEE: fastai/vision/learner.py @ https://github.com/fastai/fastai
###############################################################################


def _get_first_layer(m):
    "Access first layer of a model"
    c, p, n = m, None, None  # child, parent, name
    for n in next(m.named_parameters())[0].split('.')[:-1]:
        p, c = c, getattr(c, n)
    return c, p, n


def _load_pretrained_weights(new_layer, previous_layer):
    "Load pretrained weights based on number of input channels"
    n_in = getattr(new_layer, 'in_channels')
    if n_in == 1:
        # we take the sum
        new_layer.weight.data = previous_layer.weight.data.sum(dim=1, keepdim=True)
    elif n_in == 2:
        # we take first 2 channels + 50%
        new_layer.weight.data = previous_layer.weight.data[:, :2] * 1.5
    else:
        # keep 3 channels weights and set others to null
        new_layer.weight.data[:,:3] = previous_layer.weight.data
        new_layer.weight.data[:,3:].zero_()


def _update_first_layer(model, n_in, pretrained):
    "Change first layer based on number of input channels"
    if n_in == 3: return
    first_layer, parent, name = _get_first_layer(model)
    assert isinstance(first_layer, nn.Conv2d), (
        f'Change of input channels only supported with Conv2d,'
        f'found {first_layer.__class__.__name__}'
    )
    assert getattr(first_layer, 'in_channels') == 3, (
        f'Unexpected number of input channels, '
        f'found {getattr(first_layer, "in_channels")} while expecting 3'
    )
    params = {
        attr: getattr(first_layer, attr) for attr in
        [
            'out_channels', 'kernel_size', 'stride', 'padding', 'dilation',
            'groups', 'padding_mode'
        ]
    }
    params['bias'] = getattr(first_layer, 'bias') is not None
    params['in_channels'] = n_in
    new_layer = nn.Conv2d(**params)
    if pretrained:
        _load_pretrained_weights(new_layer, first_layer)
    setattr(parent, name, new_layer)


###############################################################################
# END FASTAI CODE
###############################################################################


class CNNPreTrained(Model):
    """Pre-trained CNN model base class"""

    def __init__(
        self, in_size: Tuple[int, int, int], out_size: int, config: Namespace
    ) -> None:
        """
        :param in_size Tuple[int, int, int]: Input size, expected  to be
        (channels, height, width)
        :param out_size int: Number of output classes
        """
        super().__init__(in_size, out_size, config)

        self.model, input_size = self.build_model(out_size)

        in_channels, *_ = in_size
        _update_first_layer(self.model, in_channels, True)

        # Fit imagenet normalization statistics to our number of channels
        def _pad_concat_norms(stats, pad):
            if len(stats) == in_channels:
                return stats
            elif len(stats) > in_channels:
                return stats[:in_channels]
            else:
                return stats + [pad for _ in range(in_channels - len(stats))]
        norms = []
        norms.append(_pad_concat_norms(IMAGENET_STATS[0], 0))
        norms.append(_pad_concat_norms(IMAGENET_STATS[1], 1))

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.Normalize(*norms)
        ])

    def forward(self, input_data: Tensor) -> Tensor:
        """
        Forward pass of the model

        :param input_data Tensor: Input data of shape [batch, channels, height,
        width] or [channels, height, width]
        """
        transformed_data = self.transforms(input_data)
        out = self.model(transformed_data)
        return out

    @abstractmethod
    def build_model(self, num_classes: int) -> Tuple[torchvision.models.Model, int]:
        """
        Builds the base pretrained model with the correct number of output
        classes. Must be extended by subclasses.

        :param num_classes int: Number of classes to predict
        :rtype Tuple[torchvision.models.Model, int]: Built model and input image
        size to crop to
        """
        pass


class ResNet18(CNNPreTrained):
    def build_model(self, num_classes: int) -> Tuple[torchvision.models.Model, int]:
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        return model, input_size


class ResNet34(CNNPreTrained):
    def build_model(self, num_classes: int) -> Tuple[torchvision.models.Model, int]:
        model = torchvision.models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        return model, input_size


class AlexNet(CNNPreTrained):
    def build_model(self, num_classes: int) -> Tuple[torchvision.models.Model, int]:
        model = torchvision.models.alexnet(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        return model, input_size


class VGG(CNNPreTrained):
    def build_model(self, num_classes: int) -> Tuple[torchvision.models.Model, int]:
        model = torchvision.models.vgg11_bn(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        return model, input_size


class SqueezeNet(CNNPreTrained):
    def build_model(self, num_classes: int) -> Tuple[torchvision.models.Model, int]:
        model = torchvision.models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1,1), stride=(1,1)
        )
        model.num_classes = num_classes
        input_size = 224
        return model, input_size


class DenseNet(CNNPreTrained):
    def build_model(self, num_classes: int) -> Tuple[torchvision.models.Model, int]:
        model = torchvision.models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        return model, input_size


if __name__ == "__main__":
    pass
