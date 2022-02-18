from abc import ABC, abstractmethod

from torch import nn


class AbstractInputEncoder(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def transform(self, emg_data):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass


class BaseInputEncoder(AbstractInputEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.max_len = config.max_len

    def transform(self, emg_data):
        return emg_data

    def collate_fn(self, batch):
        _, packed_batch = nn.utils.rnn.pad_packed_sequence(
            batch, batch_first=True, total_length=self.max_len
        )
        return packed_batch
