from abc import ABC, abstractmethod

import os

import pickle
import numpy as np
from sklearn.model_selection import train_test_split


SEED = 42


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
        :returns: Tuple of (input_data, target). Input data is a numpy array in
            (time, channel) form
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
        datapoints = (self.load(i) for i in range(len(self)))
        targets = [target for _, target in datapoints]

        # 70% / 15% / 15% split
        # Stratified to guarantee reasonable class distributions
        train_idxs, dev_test_idxs, _, dev_test_targets = train_test_split(
            list(range(len(self))),
            targets,
            test_size=0.3,
            random_state=SEED,
            shuffle=True,
            stratify=targets
        )
        dev_idxs, test_idxs = train_test_split(
            dev_test_idxs,
            test_size=0.5,
            random_state=SEED,
            shuffle=True,
            stratify=dev_test_targets,
        )

        return train_idxs, dev_idxs, test_idxs

    def __len__(self):
        return len(self.files)


class GestureLoader(AbstractLoader):
    def __init__(self, config):
        super().__init__(config)

    def load(self, index):
        pass

    def build_splits(self):
        pass

    def __len_(self):
        pass
