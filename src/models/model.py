import os
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Base Custom Model Class
    """
    def __init__(self):
        super().__init__()
        self._save_path = None

    def save(self, name, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f"{name}.h5")
        torch.save(self.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    def load(self, load_file, device):
        state_dict = torch.load(load_file, map_location=device)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {load_file}")
