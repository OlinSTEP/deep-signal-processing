import os
import re
import random
import json
from itertools import chain

import scipy.io.wavfile
from sklearn.model_selection import train_test_split

from .loader import AbstractLoader


class AudioLoader(AbstractLoader):
    def __init__(self, config):
        super().__init__(config)

        self.seed = config.seed

        self.stratify = config.stratify
        self.train_split, self.dev_split, self.test_split = config.splits
        self.split_sessions = config.split_sessions
        self.split_subjects = config.split_subjects
        self.use_cache = config.cache_raw

        self.train_idxs = {}

        self.files = []     # Contains file path tuples
        self.sessions = []  # Contains self.files idxs
        self.subjects = []  # Contains self.session idxs

        subject_dirs = sorted([
            os.path.join(self.data_path, fn)
            for fn in os.listdir(self.data_path)
        ])
        for subject_dir in subject_dirs:
            self.subjects.append([])
            session_dirs = sorted([
                os.path.join(subject_dir, fn)
                for fn in os.listdir(subject_dir)
            ])
            for session_dir in session_dirs:
                self.subjects[-1].append(len(self.sessions))
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
        elif self.split_subjects:
            return self.build_splits_subject()
        else:
            return self.build_splits_naive()

    def build_splits_naive(self):
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
                    random_state=self.seed,
                    shuffle=True,
                    stratify=(targets if self.stratify else None)
                )
                return a_idxs, b_idxs, b_targets

        train_idxs, dev_idxs, test_idxs = [], [], []
        for subject_sessions in self.subjects:
            for session_idx in subject_sessions:
                session_files = self.sessions[session_idx]
                targets = [self.load(i)[1] for i in session_files]

                dev_test_split = 1 - self.train_split
                _train_idxs, dev_test_idxs, dev_test_targets = _split_helper(
                    session_files, targets, dev_test_split
                )
                train_idxs.extend(_train_idxs)

                if dev_test_split == 0:
                    continue

                test_split = self.test_split / dev_test_split
                _dev_idxs, _test_idxs, _ = _split_helper(
                    dev_test_idxs, dev_test_targets, test_split
                )
                dev_idxs.extend(_dev_idxs)
                test_idxs.extend(_test_idxs)

        self.train_idxs = set(train_idxs)
        return train_idxs, dev_idxs, test_idxs

    def build_splits_session(self):
        random.seed(self.seed)
        train_idxs, dev_idxs, test_idxs = [], [], []
        for subject_sessions in self.subjects:
            num_sessions = len(subject_sessions)
            train_sessions = int(num_sessions * self.train_split)
            test_sessions = round(num_sessions * self.test_split)
            dev_sessions = num_sessions - train_sessions -  test_sessions

            # Randomizes subject_session not in-place
            session_idxs = random.sample(subject_sessions, num_sessions)

            def _load_session_idxs(dest, n):
                for _ in range(n):
                    session_idx = session_idxs.pop()
                    file_idxs = self.sessions[session_idx]
                    dest.extend(file_idxs)
            _load_session_idxs(train_idxs, train_sessions)
            _load_session_idxs(dev_idxs, dev_sessions)
            _load_session_idxs(test_idxs, test_sessions)

        self.train_idxs = set(train_idxs)
        return train_idxs, dev_idxs, test_idxs

    def build_splits_subject(self):
        num_subjects = len(self.subjects)
        train_subjects = int(num_subjects * self.train_split)
        test_subjects = int(num_subjects * self.test_split)
        dev_subjects = num_subjects - train_subjects -  test_subjects

        random.seed(self.seed)
        subject_idxs = list(range(num_subjects))
        random.shuffle(subject_idxs)

        def _load_subject_idxs(dest, n):
            for _ in range(n):
                subject_idx = subject_idxs.pop()
                session_idxs = self.subjects[subject_idx]
                file_idxs = [self.sessions[i] for i in session_idxs]
                dest.extend(chain(*file_idxs))
        train_idxs, dev_idxs, test_idxs = [], [], []
        _load_subject_idxs(train_idxs, train_subjects)
        _load_subject_idxs(dev_idxs, dev_subjects)
        _load_subject_idxs(test_idxs, test_subjects)

        self.train_idxs = set(train_idxs)
        return train_idxs, dev_idxs, test_idxs

    def __len__(self):
        return len(self.files)
