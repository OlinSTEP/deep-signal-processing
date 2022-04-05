from .dataset import Dataset
from .loaders import AudioLoader
from .encoding.input_encoders import (
    RegMicInputEncoder, ThroatMicInputEncoder, BothMicInputEncoder
)
from .encoding.target_encoders import (
    ClassificationEncoder
)


class RegMicClassificationDataset(Dataset):
    loader_cls = AudioLoader
    input_encoder_cls = RegMicInputEncoder
    target_encoder_cls = ClassificationEncoder


class ThroatMicClassificationDataset(Dataset):
    loader_cls = AudioLoader
    input_encoder_cls = ThroatMicInputEncoder
    target_encoder_cls = ClassificationEncoder


class BothMicClassificationDataset(Dataset):
    loader_cls = AudioLoader
    input_encoder_cls = BothMicInputEncoder
    target_encoder_cls = ClassificationEncoder


DATASETS = {
    "reg_mic_classif": RegMicClassificationDataset,
    "throat_mic_classif": ThroatMicClassificationDataset,
    "both_mic_classif": BothMicClassificationDataset,
}
