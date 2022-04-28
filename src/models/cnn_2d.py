import torch
import torch.nn as nn
import numpy as np

from .model import Model


class CNN2D(Model):
    def __init__(self, in_size, out_size, config):
        """
        Parameters:
            in_size: Input size, expected to be (channels, height, width)
            out_size: Number of output classes
        Config Parameters:
            fcs: Iterable containing the amount of neurons per layer
                ex: (1024, 512, 256) would make 3 fully connected layers, with
                1024, 512 and 256 neurons respectively
            convs: Iterable containing the kernel size, stride, output channels
                for each convolutional layer.
                ex: ((3, 1, 16)) would make 1 convolution layer with an 3x3
                kernel, 1 stride, and 16 output channels
            pools: Iterable containing the max pool length and stride for each
                convolutional layer. None indicates no pooling used. Length < 1
                indicates no pooling for the corresponding pooling layer.
                ex: ((2, 2)) would make 1 pooling layer of size 2x2 and 2 stride
            adaptive_pool: Bool indicating whether the final channels should be
                pooled down to single average values or not.
            drop_prob: Probability of dropout. Dropout is not used if < 0,
                otherwise applied between all layers.
        """
        super().__init__()

        in_channels, _, _ = in_size
        self.convs = nn.ModuleList()
        last_size = in_channels

        # Not setting pools means no pooling used
        if not config.pools:
            config.pools = [(0, 0) for _ in config.convs]

        for conv_params, pool_params in zip(config.convs, config.pools):
            kernel_len, kernel_stride, out_channels = conv_params
            pool_len, pool_stride = pool_params
            self.convs.append(nn.Conv2d(
                last_size, out_channels, kernel_len, kernel_stride
            )), _
            self.convs.append(nn.BatchNorm2d(out_channels))
            if pool_len > 1:
                self.convs.append(nn.MaxPool2d(pool_len, pool_stride))
            if config.drop_prob > 0:
                self.convs.append(nn.Dropout(p=config.drop_prob))
            last_size = out_channels

        self.flatten = nn.ModuleList()
        if config.adaptive_pool:
            self.flatten.append(torch.nn.AdaptiveAvgPool2d((1,1)))
        self.flatten.append(torch.nn.Flatten(start_dim=1))

        x = torch.tensor(np.ones(in_size, dtype=np.float32)[None, :])
        for conv in self.convs:
            x = conv(x)
        for flat in self.flatten:
            x = flat(x)
        last_size = x.shape[1]

        self.fcs = nn.ModuleList()
        for fc_size in config.fcs:
            self.fcs.append(nn.Linear(last_size, fc_size))
            if config.drop_prob > 0:
                self.fcs.append(nn.Dropout(p=config.drop_prob))
            last_size = fc_size
        self.out = nn.Linear(last_size, out_size)

        self.activation = nn.ReLU()


    def forward(self, input_data, features=False):
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
