from abc import ABC, abstractmethod


class AbstractTargetEncoder(ABC):
    def __init__(self, config):
        self.config = config
        self.target_labels = None
        self.target_to_idx = None
        self.idx_to_target = None

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


class PhraseTargetEncoder(AbstractTargetEncoder):
    def __init__(self, config):
        super().__init__(config)

    def fit(self, phrases):
        self.target_labels = sorted(list(set(phrases)))
        self.target_to_idx = {p: i for i, p in enumerate(self.target_labels)}
        self.idx_to_target = {i: p for i, p in enumerate(self.target_labels)}

    def transform(self, phrase):
        return self.target_to_idx[phrase]

    def inverse_transform(self, idx):
        return self.idx_to_target[idx]
