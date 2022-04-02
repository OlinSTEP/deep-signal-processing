import os
import re

import json
import scipy.io.wavfile
from sklearn.model_selection import train_test_split

from .loader import AbstractLoader, SEED


class AudioLoader(AbstractLoader):
    def __init__(self, config):
        super().__init__(config)

        self.stratify = config.stratify
        self.use_cache = config.cache_raw

        self.train_idxs = {}

        session_dirs = [
            os.path.join(self.data_path, fn)
            for fn in os.listdir(self.data_path)
        ]

        self.files = []
        for session_dir in session_dirs:
            for file_name in os.listdir(session_dir):
                # Folder contains files in the formats `IDX_info.json`,
                # `IDX_reg_audio.wav` and `IDX_throat_audio.wav`
                # We're iterating over just the `IDX_info.json`s
                match = re.match(r'(\d+)_info.json', file_name)
                if match is None:
                    continue

                idx = int(match.group(1))
                json_path = os.path.join(session_dir, file_name)
                reg_path = os.path.join(session_dir, f"{idx}_reg_audio.wav")
                throat_path = os.path.join(session_dir, f"{idx}_throat_audio.wav")
                self.files.append((json_path, reg_path, throat_path))

        if self.use_cache:
            self.use_cache = False
            self.cache = [self.load(i) for i in range(len(self))]
            self.use_cache = True

    def load(self, index):
        """
        Loads a single datapoint from the disk

        :param index int: Index of datapoint to load
        :returns: Tuple of (input_data, target).
            Input data is a list of (sample_rate, sequence_data) tuples for
            every input channel. Corresponds to:
            [reg_audio_0, reg_audio_1, throat_audio_0, throat_audio_1]
            Target is a single value
        """
        if self.use_cache:
            return self.cache[index]

        json_path, reg_path, throat_path = self.files[index]

        with open(json_path, "r") as f:
            data_dict = json.load(f)
        target = data_dict["target"]

        # reg_input.shape: (time, 2), throat_input.shape: (time, 2)
        reg_sample_rate, reg_input = scipy.io.wavfile.read(reg_path)
        throat_sample_rate, throat_input = scipy.io.wavfile.read(throat_path)

        # We seperate the stereo channels into individual channels here
        input_ = [
            (reg_sample_rate, reg_input[:, 0]),
            (reg_sample_rate, reg_input[:, 1]),
            (throat_sample_rate, throat_input[:, 0]),
            (throat_sample_rate, throat_input[:, 1])
        ]

        is_train = index in self.train_idxs

        return input_, target, is_train

    def build_splits(self):
        datapoints = (self.load(i) for i in range(len(self)))
        targets = [target for _, target, _ in datapoints]

        # 70% / 15% / 15% split
        train_idxs, dev_test_idxs, _, dev_test_targets = train_test_split(
            list(range(len(self))),
            targets,
            test_size=0.3,
            random_state=SEED,
            shuffle=True,
            stratify=(targets if self.stratify else None)
        )
        dev_idxs, test_idxs = train_test_split(
            dev_test_idxs,
            test_size=0.5,
            random_state=SEED,
            shuffle=True,
            stratify=(dev_test_targets if self.stratify else None)
        )

        self.train_idxs = set(train_idxs)

        return train_idxs, dev_idxs, test_idxs

    def __len__(self):
        return len(self.files)
