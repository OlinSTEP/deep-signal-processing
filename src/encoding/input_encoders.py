from abc import ABC, abstractmethod

from torch import nn


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
        super().__init__(config)
        self.max_len = config.max_len

    def fit(self, inputs):
        # if self.max_len = None:
        #     # Get max sequence from dataset if not specified
        #     self.max_len = max([i.shape[0] for i in inputs])
        self.input_dim = (inputs[0].shape[1], self.max_len)

    def transform(self, emg_data):
        return emg_data

    def collate_fn(self, batch):
        _, packed_batch = nn.utils.rnn.pad_packed_sequence(
            batch, batch_first=True, total_length=self.max_len
        )
        return packed_batch
