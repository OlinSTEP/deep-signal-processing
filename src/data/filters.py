from abc import ABC, abstractmethod


class AbstractFilter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def filter(self, data, sample_freq):
        """
        Filters the passed data

        :param data np.array: Numpy array of shape (num_channels, time)
        :param data int: Sample frequency to run filter with
        """
        pass


class MicFilter(AbstractFilter):
    def __init__(self, config):
        super().__init__()

    def filter(self, data):
        pass


class ThroatMicFilter(AbstractFilter):
    def __init__(self, config):
        super().__init__()

    def filter(self, data):
        pass


class BothMicFilter(AbstractFilter):
    def __init__(self, config):
        super().__init__()

    def filter(self, data):
        pass


class GestureFilter(AbstractFilter):
    def __init__(self, config):
        super().__init__()

    def filter(self, data):
        pass
