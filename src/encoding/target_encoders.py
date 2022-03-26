from abc import ABC, abstractmethod


class AbstractTargetEncoder(ABC):
    def __init__(self):
        self.target_labels = None
        super().__init__()

    @abstractmethod
    def fit(self, targets):
        pass

    @abstractmethod
    def transform(self, target):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass

    @property
    def target_dim(self):
        if self.target_labels:
            return len(self.target_labels)
        return None


class ClassificationEncoder(AbstractTargetEncoder):
    def __init__(self, config):
        super().__init__()
        self.target_to_idx = None
        self.idx_to_target = None

    def fit(self, targets):
        self.target_labels = sorted(list(set(targets)))
        self.target_to_idx = {p: i for i, p in enumerate(self.target_labels)}
        self.idx_to_target = {i: p for i, p in enumerate(self.target_labels)}

    def transform(self, target):
        if self.target_to_idx is None:
            raise Exception("Target encoder must be fit")
        return self.target_to_idx[target]

    def inverse_transform(self, idx):
        if self.idx_to_target is None:
            raise Exception("Target encoder must be fit")
        return self.idx_to_target[idx]
