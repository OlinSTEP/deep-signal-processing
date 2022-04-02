import torch
import torch.nn as nn
import numpy as np

from .model import Model


class CNN1D(Model):
    def __init__(self, in_size, out_size, config):
        """
        Parameters:
            in_size: Input size, expected  to be (channels, length)
            n_out: Number of output classes
        Config Parameters:
            fcs: Iterable containing the amount of neurons per layer
                ex: (1024, 512, 256) would make 3 fully connected layers, with
                1024, 512 and 256 neurons respectively
            convs: Iterable containing the kernel size, stride, output channels
                for each convolutional layer.
                ex: ((3, 1, 16)) would make 1 convolution layer with an 3 length
                kernel, 1 stride, and 16 output channels
            pools: Iterable containing the max pool length and stride for each
                convolutional layer. None indicates no pooling used. Length < 1
                indicates no pooling for the corresponding pooling layer.
                ex: ((2, 2)) would make 1 pooling layer with 2 length and 2 stride
            drop_prob: Probability of dropout. Dropout is not used if < 0,
                otherwise applied between all layers.
        """
        super().__init__()

        in_channels, _ = in_size
        self.convs = nn.ModuleList()
        last_size = in_channels

        # Not setting pools means no pooling used
        if not config.pools:
            config.pools = [(0, 0) for _ in config.convs]

        for conv_params, pool_params  in zip(config.convs, config.pools):
            kernel_len, kernel_stride, out_channels = conv_params
            pool_len, pool_stride = pool_params
            self.convs.append(nn.Conv1d(
                last_size, out_channels, kernel_len, kernel_stride))
            self.convs.append(nn.BatchNorm1d(out_channels))
            if pool_len > 1:
                self.convs.append(nn.MaxPool1d(pool_len, pool_stride))
            if config.drop_prob > 0:
                self.convs.append(nn.Dropout(p=config.drop_prob))
            last_size = out_channels

        x = torch.tensor(np.ones(in_size, dtype=np.float32)[None, :])
        for conv in self.convs:
            x = conv(x)
        x = torch.flatten(x, start_dim=1)
        last_size = x.shape[1]

        self.fcs = nn.ModuleList()
        for fc_size in config.fcs:
            self.fcs.append(nn.Linear(last_size, fc_size))
            if config.drop_prob > 0:
                self.fcs.append(nn.Dropout(p=config.drop_prob))
            last_size = fc_size
        self.out = nn.Linear(last_size, out_size)

        self.activation = nn.ReLU()


    def forward(self, input_data):
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
