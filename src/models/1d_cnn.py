import torch
import torch.nn as nn
import numpy as np

from src.models.model import Model


class EMGCNN(Model):
    def __init__(self, in_size, fcs, convs, n_out):
        """
        Constructor.
        Parameters:
            in_size: Input size, expected  to be (channels, length)
            fcs: Iterable containing the amount of neurons per layer
                ex: (1024, 512, 256) would make 3 fully connected layers, with
                1024, 512 and 256 neurons respectively
            convs: Iterable containing the kernel size, stride, output
            channels, max pool size, and max pool stride for each convolutional layer.
                ex: ((3, 1, 16, 2, 2)) would make 1 convolution layer with an 3 length
                kernel, 1 stride, 16 output channels, 2 length max and 2 stride max pooling
            n_out: Number of output classes
        """
        super().__init__()
        in_channels, _ = in_size
        self.convs = nn.ModuleList()
        last_size = in_channels
        for kernel_len, stride, out_size, pool_len, pool_stride in convs:
            self.convs.append(nn.Conv1d(last_size, out_size, kernel_len, stride))
            if pool_len > 1:
                self.convs.append(nn.MaxPool1d(pool_len, pool_stride))
            last_size = out_size

        x = torch.tensor(np.ones(in_size, dtype=np.float32)[None, :])
        for conv in self.convs:
            x = conv(x)
        x = torch.flatten(x, start_dim=1)
        last_size = x.shape[1]

        self.fcs = nn.ModuleList()
        for fc_size in fcs:
            self.fcs.append(nn.Linear(last_size, fc_size))
            last_size = fc_size
        self.out = nn.Linear(last_size, n_out)

        self.activation = nn.ReLU()


    def forward(self, emg_data):
        for conv in self.convs:
            emg_data = conv(emg_data)
            emg_data = self.activation(emg_data)

        flat_data = torch.flatten(emg_data, start_dim=1)
        for fc in self.fcs:
            flat_data = fc(flat_data)
            flat_data = self.activation(flat_data)

        out = self.out(flat_data)
        out = nn.functional.softmax(out, dim=1)
        return out


if __name__ == "__main__":
    model = EMGCNN((5, 150), [128, 64], [(3, 2, 3, 3, 2)], 1)
    print(list(model.named_parameters()))
    print(model(torch.tensor(np.ones((5, 150), dtype=np.float32)[None, :])))
