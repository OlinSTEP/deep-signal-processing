import os

import torch
import torch.nn as nn
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained.interfaces import foreign_class

from .model import Model


MODEL_DIR = "data/speechbrain_models/"


def make_trainable(model):
    for p in model.parameters():
        p.requires_grad = True


class SpeechBrainModel(Model):
    def __init__(self, in_size, out_size, config):
        """
        Parameters:
            in_size: Input size, expected to be (channels, height, width)
            out_size: Number of output classes
        """

        super().__init__()
        self.model = self.build_model()
        self.embedding_model = self.get_embedding_model(self.model)

        last_size = self.get_embedding_size()
        self.fcs = nn.ModuleList()
        for fc_size in config.fcs:
            self.fcs.append(nn.Linear(last_size, fc_size))
            if config.drop_prob > 0:
                self.fcs.append(nn.Dropout(p=config.drop_prob))
            last_size = fc_size
        self.out = nn.Linear(last_size, out_size)

        self.activation = nn.ReLU()

    def forward(self, batch, features=False):
        batch = self.process_batch(batch)
        embedded_data = self.embedding_model(batch)

        embedded_data = torch.squeeze(embedded_data)

        fc_data = embedded_data
        for fc in self.fcs:
            fc_data = fc(fc_data)
            fc_data = self.activation(fc_data)

        out = self.out(fc_data)
        if features:
            return out, embedded_data
        return out

    def build_model(self):
        raise NotImplementedError

    def get_embedding_model(self, model):
        raise NotImplementedError

    def get_embedding_size(self):
        raise NotImplementedError

    def process_batch(self, batch):
        raise NotImplementedError


class SpeechBrainVoxCelebModel(SpeechBrainModel):
    def __init__(self, in_size, out_size, config):
        assert config.samplerate == 16000
        assert config.channels == 1
        super().__init__(in_size, out_size, config)

    def build_model(self):
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=MODEL_DIR
        )

    def get_embedding_model(self, model):
        embedding_model = model.mods.embedding_model
        make_trainable(embedding_model)
        return embedding_model

    def get_embedding_size(self):
        return 512

    def process_batch(self, batch):
        batch = self.model.mods.compute_features(batch)
        # batch = self.model.mods.mean_var_norm(batch)
        return batch


class SpeechBrainGoogleSpeechModel(SpeechBrainModel):
    def __init__(self, in_size, out_size, config):
        assert config.samplerate == 16000
        assert config.channels == 1
        super().__init__(in_size, out_size, config)

    def build_model(self):
        return EncoderClassifier.from_hparams(
            source="speechbrain/google_speech_command_xvector",
            savedir=MODEL_DIR
        )

    def get_embedding_model(self, model):
        embedding_model = model.mods.embedding_model
        make_trainable(embedding_model)
        return embedding_model

    def get_embedding_size(self):
        return 512

    def process_batch(self, batch):
        batch = self.model.mods.compute_features(batch)
        # batch = self.model.mods.mean_var_norm(batch)
        return batch


class SpeechBrainWav2Vec2Model(SpeechBrainModel):
    def __init__(self, in_size, out_size, config):
        assert config.samplerate == 16000
        assert config.channels == 1
        super().__init__(in_size, out_size, config)

    def build_model(self):
        model = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=os.path.join(MODEL_DIR, "wav2vec2"),
            freeze_params=False,
        )
        return model

    def get_embedding_model(self, model):
        transformer = model.mods.wav2vec2
        make_trainable(transformer)
        transformer.freeze = False
        transformer.model.train()
        pooling = model.mods.avg_pool
        embedding_model = torch.nn.Sequential(transformer, pooling)
        return embedding_model

    def get_embedding_size(self):
        return 768

    def process_batch(self, batch):
        return batch


if __name__ == "__main__":
    pass
