from src.data.dataset import EMGDataset
from src.encoding.input_encoders import PaddedSequenceEncoder
from src.encoding.target_encoders import ClassificationEncoder

class SingleFramePhraseDataset(EMGDataset):
    input_encoder_cls = PaddedSequenceEncoder
    target_encoder_cls = ClassificationEncoder
