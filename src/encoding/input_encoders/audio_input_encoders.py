from .input_encoder import AbstractInputEncoder


import torch


class AudioInputEncoder(AbstractInputEncoder):
    def __init__(self):
        super().__init__()

    def fit(self, inputs):
        pass

    def transform(self, data):
        pass

    def collate_fn(self, batch):
        pass
