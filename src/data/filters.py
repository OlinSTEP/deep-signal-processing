from abc import ABC, abstractmethod


class AbstractFilter(ABC):
    """
    Filtering base class

    Filters take in raw data and performs processing on them
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def filter(self, data):
        """
        Filters the passed data

        :param data np.array: Numpy array of single datapoint
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
