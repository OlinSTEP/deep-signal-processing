import torch
import torch.nn as nn
import numpy as np
from speechbrain.pretrained import EncoderClassifier

from .model import Model


def make_trainable(model):
    for p in model.parameters():
        p.requires_grad = True


class SpeechBrainModel(Model):
    pretrained_path = None
    def __init__(self, in_size, out_size, config):
        """
        Parameters:
            in_size: Input size, expected to be (channels, height, width)
            out_size: Number of output classes
        """
        if self.pretrained_path is None:
            raise NotImplementedError

        super().__init__()
        self.model = EncoderClassifier.from_hparams(self.pretrained_path)
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

        # Unclear why this is needed
        embedded_data = torch.squeeze(embedded_data)

        fc_data = embedded_data
        for fc in self.fcs:
            fc_data = fc(fc_data)
            fc_data = self.activation(fc_data)

        out = self.out(fc_data)
        if features:
            return out, embedded_data
        return out

    def get_embedding_model(self, model):
        raise NotImplementedError

    def get_embedding_size(self):
        raise NotImplementedError

    def process_batch(self, batch):
        raise NotImplementedError


class SpeechBrainVoxCelebModel(SpeechBrainModel):
    pretrained_path = "speechbrain/spkrec-xvect-voxceleb"

    def __init__(self, in_size, out_size, config):
        assert config.samplerate == 16000
        assert config.channels == 1
        super().__init__(in_size, out_size, config)

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
    pretrained_path = "speechbrain/google_speech_command_xvector"

    def __init__(self, in_size, out_size, config):
        assert config.samplerate == 16000
        assert config.channels == 1
        super().__init__(in_size, out_size, config)

    def get_embedding_model(self, model):
        embedding_model = model.mods.embedding_model
        make_trainable(embedding_model)
        return embedding_model

    def get_embedding_size(self):
        return 512

    def process_batch(self, batch):
        batch = self.model.mods.compute_features(batch)
        batch = self.model.mods.mean_var_norm(batch)
        return batch


if __name__ == "__main__":
    pass
