
from abc import ABC, abstractmethod


class AbstractInputEncoder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, emg_data):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass


class BaseInputEncoder(AbstractInputEncoder):
    def __init__(self):
        super().__init__()

    def transform(self, emg_data):
        return emg_data

    def collate_fn(self, batch):
        pass
