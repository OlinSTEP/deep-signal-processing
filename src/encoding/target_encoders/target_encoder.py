from abc import ABC, abstractmethod


class AbstractTargetEncoder(ABC):
    """
    Target encoding base class

    Target encoders take in labels and prepare them for use in the training loop
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, targets):
        """
        Fit the target encoder to the train data

        Iterates over all labels to prepare for label processing. Should always
        set self._input_dim for model building.

        :param targets generator->target: Generator that produces targets as
            created by a Loader
        """
        pass

    @abstractmethod
    def transform(self, target, is_train):
        """
        Transform the label into a format the training loop can use

        :param data: A single label, type varies by dataset
        :param is_train bool: Bool indicating if datapoint is in train data
        """
        pass

    @abstractmethod
    def inverse_transform(self, idx):
        """
        Transforms a processed label back to its initial form

        :param idx: A single processed label
        """
        pass

    @property
    def target_dim(self):
        if self._target_dim is None:
            raise NotImplementedError
        return self._target_dim
