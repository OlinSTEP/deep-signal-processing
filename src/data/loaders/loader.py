from typing import Tuple, List
from numpy.typing import NDArray

from abc import ABC, abstractmethod


class AbstractLoader(ABC):
    """
    Loader base class

    Loaders load the raw data of a specific format
    """
    def __init__(self, config) -> None:
        self.data_path: str = config.data
        super().__init__()

    @abstractmethod
    def load(self, index: int) -> Tuple[List[Tuple[int, NDArray]], int, bool]:
        """
        Loads a single datapoint from the disk

        :param index int: Index of datapoint to load
        :rtype Tuple[List[Tuple[int, NDArray]]], int, bool ]: Tuple of
        (input_data, target, is_train). input_data is a list of (sample_rate,
        channel_data) tuples for every input channel, target is the target idx,
        is_train indicates whether the sample is in the train set or not
        """
        pass

    @abstractmethod
    def build_splits(self) -> Tuple[List[int], ...]:
        """
        Builds train / dev / test split indexs

        :rtype Tuple[List[int], ...]: Variable number of lists, where each list
        contains the indicies for items in the respective split. The number of
        split varies on the loader implementation.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
