import re
import os
import numpy as np
import random
import json

import torch

from src.dataset.utils import process_emg, get_emg_features


class EMGDirectory(object):
    def __init__(self, session_index, directory, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory


class EMGDataset(torch.utils.data.Dataset):
    input_encoder_cls = None
    target_encoder_cls = None

    def __init__(self, config, dev=False, test=False):
        self.testset = config.testset
        self.data = config.data
        self.remove_channels = config.remove_channels

        self.example_indices = self.build_example_indices(dev=dev, test=test)
        if not dev and not test:
            # dev and test sets should get encoders from the train set
            self.input_encoder = self.build_input_encoder(config)
            self.target_encoder = self.build_target_encoder(config)

    def __len__(self):
        return len(self.example_indices)

    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        emg, text = self.load_utterance(directory_info.directory, idx)
        emg = emg / 10
        session_ids = np.full(
            emg.shape[0], directory_info.session_index, dtype=np.int64
        )
        return {
            # (time, num_channels)
            'emg': self.input_encoder.transform(emg),
            # (1)
            'text': self.target_encoder.transform(text),
            # (time)
            'session_ids': session_ids,
        }

    def build_example_indices(self, test=False, dev=False):
        with open(self.testset) as f:
            testset_json = json.load(f)
            devset = testset_json['dev']
            testset = testset_json['test']

        directories = []
        for sd in self.config.data:
            for session_dir in sorted(os.listdir(sd)):
                directories.append(
                    EMGDirectory(len(directories), os.path.join(sd, session_dir))
                )

        example_indices = []
        for directory_info in directories:
            for fname in os.listdir(directory_info.directory):
                m = re.match(r'(\d+)_info.json', fname)
                if m is None:
                    continue
                idx_str = m.group(1)
                with open(os.path.join(directory_info.directory, fname)) as f:
                    info = json.load(f)
                    if info['sentence_index'] < 0: # boundary clips of silence are marked -1
                        continue
                    location_in_testset = int(idx_str) in testset
                    location_in_devset = int(idx_str) in devset
                    if (test and location_in_testset and not directory_info.exclude_from_testset) \
                        or (dev and location_in_devset and not directory_info.exclude_from_testset) \
                        or (not test and not dev and not location_in_testset and not location_in_devset):
                        example_indices.append((directory_info, int(idx_str)))
        example_indices.sort()
        random.seed(0)
        random.shuffle(example_indices)
        return example_indices

    def load_utterance(self, base_dir, index):
        index = int(index)
        raw_emg = np.load(os.path.join(base_dir, f'{index}_emg.npy'))
        before = os.path.join(base_dir, f'{index - 1}_emg.npy')
        after = os.path.join(base_dir, f'{index + 1}_emg.npy')
        if os.path.exists(before):
            raw_emg_before = np.load(before)
        else:
            raw_emg_before = np.zeros([0, raw_emg.shape[1]])
        if os.path.exists(after):
            raw_emg_after = np.load(after)
        else:
            raw_emg_after = np.zeros([0, raw_emg.shape[1]])

        emg, emg_orig = process_emg(raw_emg_before, raw_emg, raw_emg_after)

        for c in self.remove_channels:
            emg[:, int(c)] = 0
            emg_orig[:, int(c)] = 0

        emg_features = get_emg_features(emg)
        emg = emg[6:6 + 6 * emg_features.shape[0], :]
        emg_orig = emg_orig[8:8 + 8 * emg_features.shape[0], :]
        assert emg.shape[0] == emg_features.shape[0] * 6

        with open(os.path.join(base_dir, f'{index}_info.json')) as f:
            info = json.load(f)

        return emg_orig.astype(np.float32), info['text']

    def build_input_encoder(self, config):
        if self.input_encoder_cls is None:
            raise NotImplementedError()
        datapoints = [self.load_utterance(*loc) for loc in self.example_indices]
        inputs = [emg for emg, _ in datapoints]
        input_encoder = self.input_encoder_cls(config)
        input_encoder.fit(inputs)
        return input_encoder

    def build_target_encoder(self, config):
        if self.target_encoder_cls is None:
            raise NotImplementedError()
        datapoints = [self.load_utterance(*loc) for loc in self.example_indices]
        targets = [target for _, target in datapoints]
        target_encoder = self.target_encoder_cls(config)
        target_encoder.fit(targets)
        return target_encoder

    def set_encoding(self, test_set):
        self.input_encoder = test_set.input_encoder
        self.target_encoder = test_set.target_encoder

    @property
    def collate_fn(self):
        return self.input_encoder.collate_fn

    @property
    def input_dim(self):
        if getattr(self, "input_encoder"):
            return self.input_encoder.input_dim
        return None

    @property
    def target_dim(self):
        if getattr(self, "target_encoder"):
            return self.target_encoder.target_dim
        return None
