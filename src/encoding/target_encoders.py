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

        :param targets list: List of all labels, elements vary by dataset
        """
        pass

    @abstractmethod
    def transform(self, target):
        """
        Transform the label into a format the training loop can use

        :param data: A single label, type varies by dataset
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


class ClassificationEncoder(AbstractTargetEncoder):
    def __init__(self, config):
        super().__init__()
        self.target_to_idx = None
        self.idx_to_target = None

    def fit(self, targets):
        self.target_labels = sorted(list(set(targets)))
        self.target_to_idx = {p: i for i, p in enumerate(self.target_labels)}
        self.idx_to_target = {i: p for i, p in enumerate(self.target_labels)}
        self._target_dim = len(self.target_labels)

    def transform(self, target):
        if self.target_to_idx is None:
            raise Exception("Target encoder must be fit")
        return self.target_to_idx[target]

    def inverse_transform(self, idx):
        if self.idx_to_target is None:
            raise Exception("Target encoder must be fit")
        return self.idx_to_target[idx]
