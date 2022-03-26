from abc import ABC, abstractmethod


class AbstractLoader(ABC):
    def __init__(self, config, dev=False, test=False):
        self.data_path = config.data
        self.dev = dev
        self.test = test
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
    def __init__(self, config, dev=False, test=False):
        super().__init__(config, dev=dev, test=test)

    def load(self, index):
        pass

    @property
    def len(self):
        pass


class GestureLoader(AbstractLoader):
    def __init__(self, config, dev=False, test=False):
        super().__init__(config, dev=dev, test=test)

    def load(self, index):
        pass

    @property
    def len(self):
        pass
