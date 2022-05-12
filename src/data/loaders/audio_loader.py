from typing import Set, List, Tuple, Optional
from argparse import Namespace
from numpy.typing import NDArray

import os
import re
import random
import json
from abc import abstractmethod
from itertools import chain

import scipy.io.wavfile
import numpy as np
from sklearn.model_selection import train_test_split

from .loader import AbstractLoader


class AudioLoader(AbstractLoader):
    """
    Audio loading base class

    Implements all loader functions, but requires an implementation for loading
    the audio files. Sub-classes can load any number of audio files.
    """

    def __init__(self, config: Namespace) -> None:
        super().__init__(config)

        self.seed: int = config.seed

        self.channels: int = config.channels

        self.train_split: float = config.splits[0]
        self.dev_split: float = config.splits[1]
        self.test_split: float = config.splits[2]

        self.stratify: bool = config.stratify
        self.split_sessions: bool = config.split_sessions
        self.split_subjects: bool = config.split_subjects
        self.use_cache: bool = config.cache_raw

        self.train_idxs: Set[int] = set()

        self.files: List[Tuple[str, Tuple[str, ...]]] = []  # Contains file paths
        self.sessions: List[List[int]] = []  # Contains self.files idxs
        self.subjects: List[List[int]] = []  # Contains self.session idxs

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
                    audio_paths = self._get_audio_paths(session_dir, idx)

                    self.sessions[-1].append(len(self.files))
                    self.files.append((json_path, audio_paths))

        if self.use_cache:
            self.use_cache = False
            self.cache = [self.load(i) for i in range(len(self))]
            self.use_cache = True

    @abstractmethod
    def _get_audio_paths(self, session_dir: str, idx: int) -> Tuple[str, ...]:
        """
        Gets paths of audio files given an idx

        :param session_dir str: Parent directory to load file within
        :param idx int: Index of file
        :rtype Tuple[str, ...]: Tuple of audio files to load
        """
        pass

    def load(self, idx: int) -> Tuple[List[Tuple[int, NDArray]], int, bool]:
        """
        Loads a single datapoint from the disk

        :param index int: Index of datapoint to load
        :rtype Tuple[List[Tuple[int, NDArray]]], int, bool ]: Tuple of
        (input_data, target, is_train). input_data is a list of (sample_rate,
        channel_data) tuples for every input channel, target is the target idx,
        is_train indicates whether the sample is in the train set or not
        """
        if self.use_cache:
            return self.cache[idx]

        json_path, audio_paths = self.files[idx]

        with open(json_path, "r") as f:
            data_dict = json.load(f)
        target = data_dict["target"]

        # Separate data into list of individual channels
        sample_rates, audio_data = [], []
        for audio_path in audio_paths:
            # data.shape: (time, n_channels) if n_channels > 1 else (time,)
            sr, data = scipy.io.wavfile.read(audio_path)

            if len(data.shape) == 1:
                sample_rates.append(sr)
                audio_data.append(data)
            else:
                # Drops extra channels
                data = data[:, :self.channels]
                # Change to (n_channels, time)
                data = np.transpose(data)
                # Duplicate sr for each channel
                srs = [sr for _ in range(data.shape[0])]

                sample_rates.extend(srs)
                audio_data.extend(data)

        input_ = list(zip(sample_rates, audio_data))
        is_train = idx in self.train_idxs

        return input_, target, is_train

    def build_splits(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Builds train / dev / test split indexs

        :rtype Tuple[List[int], List[int], List[int]]: Tuple of train / dev /
        test split idxs, where each list contains the indicies for items in the
        respective split.
        """
        if self.split_sessions:
            return self.build_splits_session()
        elif self.split_subjects:
            return self.build_splits_subject()
        else:
            return self.build_splits_naive()

    def build_splits_naive(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Builds train / dev / test split idxs with no holdout sessions or holdout
        users

        Ensures that an equal portion of each sessions and each subject is
        present in the holdout set.

        :rtype Tuple[List[int], List[int], List[int]]: Tuple of train / dev /
        test split idxs, where each list contains the indicies for items in the
        respective split.
        """
        def _split_helper(idxs: List[int], targets: Optional[List[int]], split: float) \
                -> Tuple[List[int], List[int], Optional[List[int]]]:
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

    def build_splits_session(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Builds train / dev / test split idxs with holdout sessions

        Ensures that an equal portion of each subject is present in the holdout
        set, while ensuring that some sessions only appear in the holdout set.

        :rtype Tuple[List[int], List[int], List[int]]: Tuple of train / dev /
        test split idxs, where each list contains the indicies for items in the
        respective split.
        """
        random.seed(self.seed)
        train_idxs, dev_idxs, test_idxs = [], [], []
        for subject_sessions in self.subjects:
            num_sessions = len(subject_sessions)
            train_sessions = int(num_sessions * self.train_split)
            test_sessions = int(num_sessions * self.test_split)
            dev_sessions = num_sessions - train_sessions -  test_sessions

            # Randomizes subject_session not in-place
            session_idxs = random.sample(subject_sessions, num_sessions)

            def _load_session_idxs(dest: List[int], n: int) -> None:
                for _ in range(n):
                    session_idx = session_idxs.pop()
                    file_idxs = self.sessions[session_idx]
                    dest.extend(file_idxs)
            _load_session_idxs(train_idxs, train_sessions)
            _load_session_idxs(dev_idxs, dev_sessions)
            _load_session_idxs(test_idxs, test_sessions)

        self.train_idxs = set(train_idxs)
        return train_idxs, dev_idxs, test_idxs

    def build_splits_subject(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Builds train / dev / test split idxs with holdout subjejcts

        Ensures that some subjects only appear in the holdout set.

        :rtype Tuple[List[int], List[int], List[int]]: Tuple of train / dev /
        test split idxs, where each list contains the indicies for items in the
        respective split.
        """
        num_subjects = len(self.subjects)
        train_subjects = int(num_subjects * self.train_split)
        test_subjects = int(num_subjects * self.test_split)
        dev_subjects = num_subjects - train_subjects -  test_subjects

        random.seed(self.seed)
        subject_idxs = list(range(num_subjects))
        random.shuffle(subject_idxs)

        def _load_subject_idxs(dest: List[int], n: int) -> None:
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

    def __len__(self) -> int:
        return len(self.files)


class BothMicAudioLoader(AudioLoader):
    """Loads regular and throat mic audio files"""
    def _get_audio_paths(self, session_dir: str, idx: int) -> Tuple[str, str]:
        reg_path = os.path.join(session_dir, f"{idx}_reg_audio.wav")
        throat_path = os.path.join(session_dir, f"{idx}_throat_audio.wav")
        return (reg_path, throat_path)


class RegMicAudioLoader(AudioLoader):
    """Loads regular mic audio files"""
    def _get_audio_paths(self, session_dir: str, idx: int) -> Tuple[str]:
        reg_path = os.path.join(session_dir, f"{idx}_reg_audio.wav")
        return (reg_path,)


class ThroatMicAudioLoader(AudioLoader):
    """Loads throat mic audio files"""
    def _get_audio_paths(self, session_dir: str, idx: int) -> Tuple[str]:
        throat_path = os.path.join(session_dir, f"{idx}_throat_audio.wav")
        return (throat_path,)
