from src.data.dataset import Dataset
from src.data.loaders import AudioLoader
from src.data.filters import RegMicFilter, ThroatMicFilter, BothMicFilter
from src.encoding.input_encoders import PaddedSequenceEncoder
from src.encoding.target_encoders import ClassificationEncoder


class MicClassificationDataset(Dataset):
    loader_cls = AudioLoader
    filter_cls = RegMicFilter
    input_encoder_cls = PaddedSequenceEncoder
    target_encoder_cls = ClassificationEncoder


class ThroatMicClassificationDataset(Dataset):
    loader_cls = AudioLoader
    filter_cls = ThroatMicFilter
    input_encoder_cls = PaddedSequenceEncoder
    target_encoder_cls = ClassificationEncoder


class BothMicClassificationDataset(Dataset):
    loader_cls = AudioLoader
    filter_cls = BothMicFilter
    input_encoder_cls = PaddedSequenceEncoder
    target_encoder_cls = ClassificationEncoder
