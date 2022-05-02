from .datasets import (
    Dataset, DomainAdaptionDataset
)
from .loaders import (
    RegMicAudioLoader, ThroatMicAudioLoader, BothMicAudioLoader,
    ThroatMicDomainAdaptionAudioLoader
)
from .encoding.input_encoders import (
    AudioInputEncoder, RawAudioInputEncoder
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


class ThroatMicRawClassificationDataset(Dataset):
    loader_cls = ThroatMicAudioLoader
    input_encoder_cls = RawAudioInputEncoder
    target_encoder_cls = ClassificationEncoder


class ThroatMicClassificationFinetuneDataset(Dataset):
    loader_cls = ThroatMicDomainAdaptionAudioLoader
    input_encoder_cls = AudioInputEncoder
    target_encoder_cls = ClassificationEncoder


class ThroatMicRawClassificationFinetuneDataset(Dataset):
    loader_cls = ThroatMicDomainAdaptionAudioLoader
    input_encoder_cls = RawAudioInputEncoder
    target_encoder_cls = ClassificationEncoder


class ThroatMicClassificationDomainAdaptionDataset(DomainAdaptionDataset):
    loader_cls = ThroatMicDomainAdaptionAudioLoader
    input_encoder_cls = AudioInputEncoder
    target_encoder_cls = ClassificationEncoder


DATASETS = {
    "reg_mic_classif": RegMicClassificationDataset,
    "throat_mic_classif": ThroatMicClassificationDataset,
    "both_mic_classif": BothMicClassificationDataset,
    "speech_brain_throat_mic_classif": ThroatMicRawClassificationDataset,
    "finetune_speech_brain_throat_mic_classif": ThroatMicRawClassificationFinetuneDataset,
    "finetune_throat_mic_classif": ThroatMicClassificationFinetuneDataset,
    "da_throat_mic_classif": ThroatMicClassificationDomainAdaptionDataset,
}
