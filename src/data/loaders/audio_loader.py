import os
import re
import random
import json

import scipy.io.wavfile
from sklearn.model_selection import train_test_split

from .loader import AbstractLoader, SEED


class AudioLoader(AbstractLoader):
    def __init__(self, config):
        super().__init__(config)

        self.stratify = config.stratify
        self.train_split, self.dev_split, self.test_split = config.splits
        self.split_sessions = config.split_sessions
        self.use_cache = config.cache_raw

        self.train_idxs = {}

        session_dirs = [
            os.path.join(self.data_path, fn)
            for fn in os.listdir(self.data_path)
        ]

        self.files = []
        self.sessions = []
        for session_dir in session_dirs:
            self.sessions.append([])
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

                self.sessions[-1].append(len(self.files))
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

        # Seperate the stereo channels into individual channels
        input_ = [
            (reg_sample_rate, reg_input[:, 0]),
            (reg_sample_rate, reg_input[:, 1]),
            (throat_sample_rate, throat_input[:, 0]),
            (throat_sample_rate, throat_input[:, 1])
        ]

        is_train = index in self.train_idxs

        return input_, target, is_train

    def build_splits(self):
        if self.split_sessions:
            return self.build_splits_session()
        return self.build_splits_naive()

    def build_splits_naive(self):
        datapoints = (self.load(i) for i in range(len(self)))
        targets = [target for _, target, _ in datapoints]
        all_idxs = list(range(len(self)))

        def _split_helper(idxs, targets, split):
            """
            Handle splits of 0 and 1 manually since sklearn gets mad when we
            pass them to train_test_split()
            """
            if split == 1:
                return [], idxs, targets
            elif split == 0:
                return idxs, [], None
            else:
                a_idxs, b_idxs, _, b_targets = train_test_split(
                    idxs,
                    targets,
                    test_size=split,
                    random_state=SEED,
                    shuffle=True,
                    stratify=(targets if self.stratify else None)
                )
                return a_idxs, b_idxs, b_targets

        dev_test_split = 1 - self.train_split
        train_idxs, dev_test_idxs, dev_test_targets = _split_helper(
            all_idxs, targets, dev_test_split
        )
        self.train_idxs = set(train_idxs)

        if dev_test_split == 0:
            return train_idxs, [], []

        test_split = self.test_split / dev_test_split
        dev_idxs, test_idxs, _ = _split_helper(
            dev_test_idxs, dev_test_targets, test_split
        )

        return train_idxs, dev_idxs, test_idxs

    def build_splits_session(self):
        # 70% / 15% / 15% split
        num_sessions = len(self.sessions)
        train_sessions = int(num_sessions * self.train_split)
        test_sessions = int(num_sessions * self.test_split)
        dev_sessions = num_sessions - train_sessions -  test_sessions

        random.seed(SEED)
        session_idxs = list(range(num_sessions))
        random.shuffle(session_idxs)

        train_idxs = []
        for _ in range(train_sessions):
            idx = session_idxs.pop()
            train_idxs.extend(self.sessions[idx])
        dev_idxs = []
        for _ in range(dev_sessions):
            idx = session_idxs.pop()
            dev_idxs.extend(self.sessions[idx])
        test_idxs = []
        for _ in range(test_sessions):
            idx = session_idxs.pop()
            test_idxs.extend(self.sessions[idx])

        return train_idxs, dev_idxs, test_idxs

    def __len__(self):
        return len(self.files)
