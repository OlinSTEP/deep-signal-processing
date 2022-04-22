from abc import ABC, abstractmethod


class AbstractLoader(ABC):
    """
    Loader base class

    Loaders load the raw data of a specific format
    """
    def __init__(self, config):
        self.data_path = config.data
        super().__init__()

    @abstractmethod
    def load(self, index):
        """
        Loads a single datapoint from the disk

        :param index int: Index of datapoint to load
        :returns: Tuple of (input_data, target, is_train).
            input_data is a list of (sample_rate, sequence_data) tuples for
            every input channel
            target is a single value
            is_train is a bool, whether the sample is in the train set or not
        """
        pass

    @abstractmethod
    def build_splits(self):
        """
        Builds train / dev / test split indexs

        :returns: Tuple of (train_idxs, dev_idxs, train_idxs), where each value
            is a list of idxs that can be passed to load()
        """
        pass

    @abstractmethod
    def __len__(self):
        pass
