from abc import ABC, abstractmethod

import torch


class AbstractInputEncoder(ABC):
    def __init__(self):
        self.input_dim = None
        super().__init__()

    @abstractmethod
    def fit(self, inputs):
        pass

    @abstractmethod
    def transform(self, emg_data):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass


class BaseInputEncoder(AbstractInputEncoder):
    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_len

    def fit(self, inputs):
        max_len = max([i.shape[0] for i in inputs])
        if  max_len > self.max_len:
            raise Exception(
                f"Sequence of length {max_len} exceeds max length"
            )
        self.input_dim = (inputs[0].shape[1], self.max_len)

    def transform(self, emg_data):
        return emg_data

    def collate_fn(self, batch):
        batch_emg = [torch.tensor(d["emg"]) for d in batch]
        batch_text = [d["text"] for d in batch]

        # (batch_size, max_batch_len, n_channels)
        packed_emg = torch.nn.utils.rnn.pad_sequence(
            batch_emg, batch_first=True
        )
        # Add additional padding to reach self.max_len
        # (batch_size, max_len, n_channels)
        size = packed_emg.size()
        missing = self.max_len - size[1]
        if  missing < 0:
            raise Exception(
                f"Sequence of length {size[1]} exceeds max length"
            )
        padding = torch.zeros(size[0], missing, size[2])
        packed_emg = torch.cat((packed_emg, padding), dim=1)
        # (batch_size, n_channels, max_len) (CNN friendly format)
        packed_emg = packed_emg.transpose(1, 2)

        # (batch_size)
        packed_text = torch.tensor(batch_text)

        return {
            "emg": packed_emg,
            "text": packed_text,
        }
