from typing import Tuple
from argparse import Namespace
from torch import Tensor

import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class Model(nn.Module, ABC):
    """
    Base Custom Model Class
    """
    @abstractmethod
    def __init__(
        self, in_size: Tuple[int, ...], out_size: int, config: Namespace
    ) -> None:
        pass

    @abstractmethod
    def forward(self, input_data: Tensor) -> Tensor:
        """
        Forward pass of the model

        :param input_data Tensor: Input data, shape depends on specific model
        :rtype Tensor: Output data of shape [batch, num_targets] or [num_targets]
        """
        pass

    def save(self, name: str, save_dir: str) -> None:
        """
        Saves model weights

        :param name str: Name to save model as
        :param save_dir str: Directory to save model to
        """
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f"{name}.h5")
        torch.save(self.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    def load(self, load_file: str, device: str) -> None:
        """
        Loads model weights

        :param load_file str: File to load weights from
        :param device str: Device string to map model to
        """
        state_dict = torch.load(load_file, map_location=device)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {load_file}")
