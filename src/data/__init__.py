from .dataset import Dataset
from .loaders import (
    RegMicAudioLoader, ThroatMicAudioLoader, BothMicAudioLoader
)
from .encoding.input_encoders import (
    AudioInputEncoder
)
from .encoding.target_encoders import (
    ClassificationEncoder
)


class RegMicClassificationDataset(Dataset):
    loader_cls = RegMicAudioLoader
    input_encoder_cls = AudioInputEncoder
    target_encoder_cls = ClassificationEncoder


class ThroatMicClassificationDataset(Dataset):
    loader_cls = ThroatMicAudioLoader
    input_encoder_cls = AudioInputEncoder
    target_encoder_cls = ClassificationEncoder


class BothMicClassificationDataset(Dataset):
    loader_cls = BothMicAudioLoader
    input_encoder_cls = AudioInputEncoder
    target_encoder_cls = ClassificationEncoder


DATASETS = {
    "reg_mic_classif": RegMicClassificationDataset,
    "throat_mic_classif": ThroatMicClassificationDataset,
    "both_mic_classif": BothMicClassificationDataset,
}
