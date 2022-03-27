from abc import ABC, abstractmethod

import os

import pickle
import numpy as np


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
        Loads a single datapoint from the disk in (time, channels) format

        :param index int: Index of datapoint to load
        """
        pass

    @abstractmethod
    def build_splits(self):
        """
        Returns train / dev / test split indexs
        """
        pass

    @property
    @abstractmethod
    def len(self):
        pass


class AudioLoader(AbstractLoader):
    def __init__(self, config):
        super().__init__(config)

        session_dirs = [
            os.path.join(self.data_path, fn)
            for fn in os.listdir(self.data_path)
        ]

        self.files = []
        for session_dir in session_dirs:
            for file_name in os.listdir(session_dir):
                file_path = os.path.join(session_dir, file_name)
                self.files.append(file_path)

    def load(self, index):
        file_path = self.files[index]
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)
        input_ = np.asarray(list(zip(data_dict["reg"], data_dict["throat"])))
        target = data_dict["target"]
        return input_, target

    def build_splits(self):
        pass

    @property
    def len(self):
        return len(self.files)


class GestureLoader(AbstractLoader):
    def __init__(self, config):
        super().__init__(config)

    def load(self, index):
        pass

    def build_splits(self):
        pass

    @property
    def len(self):
        pass
