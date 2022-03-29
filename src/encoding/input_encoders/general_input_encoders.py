from .input_encoder import AbstractInputEncoder

import torch


class PaddedSequenceEncoder(AbstractInputEncoder):
    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_len

    def fit(self, inputs):
        for i in inputs:
            if i.shape[0] > self.max_len:
                raise Exception(
                    f"Sequence of length {i.shape[0]} exceeds max length"
                )
        self._input_dim = (inputs[0].shape[1], self.max_len)

    def transform(self, data, _):
        return data

    def collate_fn(self, batch):
        batch_input = [torch.tensor(d["input"]) for d in batch]
        batch_target = [d["target"] for d in batch]

        # (batch_size, max_batch_len, n_channels)
        packed_input = torch.nn.utils.rnn.pad_sequence(
            batch_input, batch_first=True
        )
        # Add additional padding to reach self.max_len
        # (batch_size, max_len, n_channels)
        size = packed_input.size()
        missing = self.max_len - size[1]
        if missing < 0:
            raise Exception(
                f"Sequence of length {size[1]} exceeds max length"
            )
        padding = torch.zeros(size[0], missing, size[2])
        packed_input = torch.cat((packed_input, padding), dim=1)
        # (batch_size, n_channels, max_len) (CNN friendly format)
        packed_input = packed_input.transpose(1, 2)

        # (batch_size)
        packed_target = torch.tensor(batch_target)

        return {
            "input": packed_input,
            "target": packed_target,
        }
