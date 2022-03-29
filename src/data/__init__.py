from .dataset import Dataset
from .loaders import AudioLoader
from src.encoding.input_encoders import (
    AudioInputEncoder, PaddedSequenceEncoder
)
from src.encoding.target_encoders import (
    ClassificationEncoder
)


class MicClassificationDataset(Dataset):
    loader_cls = AudioLoader
    input_encoder_cls = AudioInputEncoder
    target_encoder_cls = ClassificationEncoder


class ThroatMicClassificationDataset(Dataset):
    loader_cls = AudioLoader
    input_encoder_cls = AudioInputEncoder
    target_encoder_cls = ClassificationEncoder


class BothMicClassificationDataset(Dataset):
    loader_cls = AudioLoader
    input_encoder_cls = AudioInputEncoder
    target_encoder_cls = ClassificationEncoder
