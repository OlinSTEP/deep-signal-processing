from .loader import AbstractLoader


class GestureLoader(AbstractLoader):
    def __init__(self, config):
        super().__init__(config)

    def load(self, index):
        pass

    def build_splits(self):
        pass

    def __len_(self):
        pass
