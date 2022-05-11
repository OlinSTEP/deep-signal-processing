from typing import Tuple, List, Any, Union
from argparse import Namespace
from torch import Tensor

import os
import re
from abc import abstractmethod

import torch
import torch.nn as nn
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained.interfaces import foreign_class, Pretrained

from .model import Model


MODEL_DIR = "data/speechbrain_models/"


def make_trainable(model: torch.nn.Module) -> None:
    """
    Makes all model layers require grad

    :param model torch.nn.Module: Model to make trainable
    """
    for p in model.parameters():
        p.requires_grad = True


class SpeechBrainModel(Model):
    """Baseclass for Speechbrain Models"""

    def __init__(self, in_size: Any, out_size: int, config: Namespace) -> None:
        """
        :param in_size Any: Input size, unused
        (channels, height, width)
        :param out_size int: Number of output classes
        :param config Namespace: Config to use
        """
        super().__init__(in_size, out_size, config)
        fcs: List[int] = config.fcs

        self.model = self.build_model()
        self.embedding_model = self.get_embedding_model(self.model)

        last_size = self.get_embedding_size()
        self.fcs = nn.ModuleList()
        for fc_size in fcs:
            self.fcs.append(nn.Linear(last_size, fc_size))
            if config.drop_prob > 0:
                self.fcs.append(nn.Dropout(p=config.drop_prob))
            last_size = fc_size
        self.out = nn.Linear(last_size, out_size)

        self.activation = nn.ReLU()

    def forward(self, input_data: Tensor, features: bool = False) \
            -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of the model

        :param input_data Tensor: Input data of shape [batch, channels, length,
        ] or [channels, length]
        :param features bool: Whether to return intermediate features in
        addition to final outputs
        :rtype Union[Tensor, Tuple[Tensor, Tensor]]: Output data of shape
        [batch, num_targets] or [num_targets]. Optionally returns intermediate
        features of shape [batch, dim_size] or [dim_size] if features = True
        """
        input_data = self.process_batch(input_data)
        embedded_data = self.embedding_model(input_data)

        embedded_data = torch.squeeze(embedded_data)

        fc_data = embedded_data
        for fc in self.fcs:
            fc_data = fc(fc_data)
            fc_data = self.activation(fc_data)

        out = self.out(fc_data)
        if features:
            return out, embedded_data
        return out

    @abstractmethod
    def build_model(self) -> Pretrained:
        """
        Builds the speechbrain model

        :rtype Pretrained: Built speechbrain model
        """
        pass

    @abstractmethod
    def get_embedding_model(self, model: Pretrained) -> torch.nn.Module:
        """
        Builds the model used to generate embeddings

        :param model Pretrained: Model built with build_model()
        :rtype torch.nn.Module: Embedding model
        """
        pass

    @abstractmethod
    def get_embedding_size(self) -> int:
        """
        Returns size of embedded representation

        :rtype int: Size of embeddings
        """
        pass

    @abstractmethod
    def process_batch(self, batch: Tensor) -> Tensor:
        """
        Preprocesses a batch of data

        :param batch Tensor: Input data of shape [batch, channels, length,
        ] or [channels, length]
        :rtype Tensor: Processed batch
        """
        pass


class SpeechBrainVoxCelebModel(SpeechBrainModel):
    def __init__(self, in_size: Any, out_size: int, config: Namespace) -> None:
        assert config.samplerate == 16000
        assert config.channels == 1
        super().__init__(in_size, out_size, config)

    def build_model(self) -> torch.nn.Module:
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=MODEL_DIR
        )

    def get_embedding_model(self, model: Pretrained) -> torch.nn.Module:
        embedding_model = model.mods.embedding_model
        make_trainable(embedding_model)
        return embedding_model

    def get_embedding_size(self) -> int:
        return 512

    def process_batch(self, batch: Tensor) -> Tensor:
        batch = self.model.mods.compute_features(batch)
        # batch = self.model.mods.mean_var_norm(batch)
        return batch


class SpeechBrainGoogleSpeechModel(SpeechBrainModel):
    def __init__(self, in_size: Any, out_size: int, config: Namespace) -> None:
        assert config.samplerate == 16000
        assert config.channels == 1
        super().__init__(in_size, out_size, config)

    def build_model(self) -> Pretrained:
        return EncoderClassifier.from_hparams(
            source="speechbrain/google_speech_command_xvector",
            savedir=MODEL_DIR
        )

    def get_embedding_model(self, model: Pretrained) -> torch.nn.Module:
        embedding_model = model.mods.embedding_model
        make_trainable(embedding_model)
        return embedding_model

    def get_embedding_size(self) -> int:
        return 512

    def process_batch(self, batch: Tensor) -> Tensor:
        batch = self.model.mods.compute_features(batch)
        # batch = self.model.mods.mean_var_norm(batch)
        return batch


class SpeechBrainWav2Vec2Model(SpeechBrainModel):
    def __init__(self, in_size: Any, out_size: int, config: Namespace) -> None:
        self.finetune_layers: int = config.finetune_layers
        assert config.samplerate == 16000
        assert config.channels == 1
        assert self.finetune_layers <= 11
        super().__init__(in_size, out_size, config)

    def build_model(self) -> Pretrained:
        model = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=os.path.join(MODEL_DIR, "wav2vec2"),
            freeze_params=False,
        )
        return model

    def get_embedding_model(self, model: Pretrained) -> torch.nn.Module:
        transformer = model.mods.wav2vec2

        if self.finetune_layers == -1:
            make_trainable(transformer)
        else:
            unfreeze_layers = {11 - i for i in range(self.finetune_layers)}
            regex_str = r"model\.encoder\.layers\.(\d+)\..*$"
            for name, p in transformer.named_parameters():
                match = re.match(regex_str, name)
                if match is None:
                    continue
                layer_num = int(match.group(1))
                if layer_num in unfreeze_layers:
                    p.requires_grad = True

        if self.finetune_layers != 0:
            transformer.freeze = False
            transformer.model.train()

        pooling = model.mods.avg_pool
        embedding_model = torch.nn.Sequential(transformer, pooling)
        return embedding_model

    def get_embedding_size(self) -> int:
        return 768

    def process_batch(self, batch: Tensor) -> Tensor:
        return batch


if __name__ == "__main__":
    pass
