from src.dataset.emg_dataset import EMGDataset
from src.encoding.input_encoders import BaseInputEncoder
from src.encoding.target_encoders import PhraseTargetEncoder

class SingleFramePhraseDataset(EMGDataset):
    input_encoder_cls = BaseInputEncoder
    target_encoder_cls = PhraseTargetEncoder
