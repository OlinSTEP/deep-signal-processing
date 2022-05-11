from typing import List, Tuple, Optional, Union
from argparse import Namespace
from torch import Tensor

import torch
import torch.nn as nn
import numpy as np

from .model import Model


class CNN2D(Model):
    def __init__(
        self, in_size: Tuple[int, int, int], out_size: int, config: Namespace
    ) -> None:
        """
        :param in_size Tuple[int, int, int]: Input size, expected  to be
        (channels, height, width)
        :param out_size int: Number of output classes
        :param config Namespace: Config to use
        """
        super().__init__(in_size, out_size, config)

        in_channels, _, _ = in_size
        self.convs = nn.ModuleList()
        last_size = in_channels

        fcs: List[int] = config.fcs
        convs: List[Tuple[int, int, int]] = config.convs
        pools: Optional[List[Tuple[int, int]]] = config.pools
        adaptive_pool: bool = config.adaptive_pool
        drop_prob: float = config.drop_prob

        # Not setting pools means no pooling used
        if not pools:
            pools = [(0, 0) for _ in convs]

        for conv_params, pool_params in zip(convs, pools):
            kernel_len, kernel_stride, out_channels = conv_params
            pool_len, pool_stride = pool_params
            self.convs.append(nn.Conv2d(
                last_size, out_channels, kernel_len, kernel_stride
            )), _
            self.convs.append(nn.BatchNorm2d(out_channels))
            if pool_len > 1:
                self.convs.append(nn.MaxPool2d(pool_len, pool_stride))
            if drop_prob > 0:
                self.convs.append(nn.Dropout(p=drop_prob))
            last_size = out_channels

        self.flatten = nn.ModuleList()
        if adaptive_pool:
            self.flatten.append(torch.nn.AdaptiveAvgPool2d((1,1)))
        self.flatten.append(torch.nn.Flatten(start_dim=1))

        x = torch.tensor(np.ones(in_size, dtype=np.float32)[None, :])
        for conv in self.convs:
            x = conv(x)
        for flat in self.flatten:
            x = flat(x)
        last_size = x.shape[1]

        self.fcs = nn.ModuleList()
        for fc_size in fcs:
            self.fcs.append(nn.Linear(last_size, fc_size))
            if drop_prob > 0:
                self.fcs.append(nn.Dropout(p=drop_prob))
            last_size = fc_size
        self.out = nn.Linear(last_size, out_size)

        self.activation = nn.ReLU()


    def forward(self, input_data: Tensor, features: bool = False) \
            -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of the model

        :param input_data Tensor: Input data of shape [batch, channels, height,
        width] or [channels, height, width]
        :param features bool: Whether to return intermediate features in
        addition to final outputs
        :rtype Union[Tensor, Tuple[Tensor, Tensor]]: Output data of shape
        [batch, num_targets] or [num_targets]. Optionally returns intermediate
        features of shape [batch, dim_size] or [dim_size] if features = True
        """
        conv_data = input_data
        for conv in self.convs:
            conv_data = conv(conv_data)
            conv_data = self.activation(conv_data)

        flat_data = conv_data
        for flat in self.flatten:
            flat_data = flat(flat_data)

        fc_data = flat_data
        for fc in self.fcs:
            fc_data = fc(fc_data)
            fc_data = self.activation(fc_data)

        out = self.out(fc_data)
        if features:
            return out, flat_data
        return out


if __name__ == "__main__":
    pass
