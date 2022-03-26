from abc import ABC, abstractmethod


class AbstractLoader(ABC):
    def __init__(self, config):
        self.data_path = config.data
        super().__init__()

    @abstractmethod
    def load(self, index):
        """
        Loads a single datapoint from the disk.

        :param index int: Index of datapoint to load
        """
        pass

    @property
    @abstractmethod
    def len(self):
        pass


class AudioLoader(AbstractLoader):
    def __init__(self, config):
        super().__init__(config)

    def load(self, index):
        pass

    @property
    def len(self):
        pass


class GestureLoader(AbstractLoader):
    def __init__(self, config):
        super().__init__(config)

    def load(self, index):
        pass

    @property
    def len(self):
        pass
