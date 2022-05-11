from typing import Tuple, List, Optional
from argparse import Namespace
from torch import Tensor

import torch
import torch.nn as nn
import numpy as np

from .model import Model


class CNN1D(Model):
    """A simple custom 1D CNN class with batchnorm and dropout"""

    def __init__(
        self, in_size: Tuple[int, int], out_size: int, config: Namespace
    ) -> None:
        """
        :param in_size Tuple[int, int]: Input size, expected  to be (channels, length)
        :param out_size int: Number of output classes
        :param config Namespace: Config to use
        """
        super().__init__(in_size, out_size, config)

        in_channels, _ = in_size
        self.convs = nn.ModuleList()
        last_size = in_channels

        fcs: List[int] = config.fcs
        convs: List[Tuple[int, int, int]] = config.convs
        pools: Optional[List[Tuple[int, int]]] = config.pools
        drop_prob: float = config.drop_prob

        # Not setting pools means no pooling used
        if not pools:
            pools = [(0, 0) for _ in convs]

        for conv_params, pool_params  in zip(convs, pools):
            kernel_len, kernel_stride, out_channels = conv_params
            pool_len, pool_stride = pool_params
            self.convs.append(nn.Conv1d(
                last_size, out_channels, kernel_len, kernel_stride))
            self.convs.append(nn.BatchNorm1d(out_channels))
            if pool_len > 1:
                self.convs.append(nn.MaxPool1d(pool_len, pool_stride))
            if drop_prob > 0:
                self.convs.append(nn.Dropout(p=drop_prob))
            last_size = out_channels

        x = torch.tensor(np.ones(in_size, dtype=np.float32)[None, :])
        for conv in self.convs:
            x = conv(x)
        x = torch.flatten(x, start_dim=1)
        last_size = x.shape[1]

        self.fcs = nn.ModuleList()
        for fc_size in fcs:
            self.fcs.append(nn.Linear(last_size, fc_size))
            if drop_prob > 0:
                self.fcs.append(nn.Dropout(p=drop_prob))
            last_size = fc_size
        self.out = nn.Linear(last_size, out_size)

        self.activation = nn.ReLU()


    def forward(self, input_data: Tensor) -> Tensor:
        """
        Forward pass of the model

        :param input_data Tensor: Input data of shape [batch, channels, length]
        or [channels, length]
        :rtype Tensor: Output data of shape [batch, num_targets] or [num_targets]
        """
        for conv in self.convs:
            input_data = conv(input_data)
            input_data = self.activation(input_data)

        flat_data = torch.flatten(input_data, start_dim=1)
        for fc in self.fcs:
            flat_data = fc(flat_data)
            flat_data = self.activation(flat_data)

        out = self.out(flat_data)
        return out


if __name__ == "__main__":
    pass
