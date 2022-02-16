import re
import os
import numpy as np
import random
import scipy
import json
import sys
import pickle

import torch

from data_utils import get_emg_features, FeatureNormalizer

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_list('remove_channels', [], 'channels to remove')
flags.DEFINE_list('silent_data_directories', ['./emg_data/silent_parallel_data'], 'silent data locations')
flags.DEFINE_list('voiced_data_directories', ['./emg_data/voiced_parallel_data','./emg_data/nonparallel_data'], 'voiced data locations')
flags.DEFINE_string('testset_file', 'testset_largedev.json', 'file with testset indices')
flags.DEFINE_string('text_align_directory', 'text_alignments', 'directory with alignment files')

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)

def process_emg(emg_before, emg, emg_after):
    x = np.concatenate([emg_before, emg, emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[emg_before.shape[0]:x.shape[0] - emg_after.shape[0], :]
    emg_orig = apply_to_all(subsample, x, 800, 1000)
    x = apply_to_all(subsample, x, 600, 1000)
    return x, emg_orig

def load_utterance(base_dir, index):
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

    for c in FLAGS.remove_channels:
        emg[:, int(c)] = 0
        emg_orig[:, int(c)] = 0

    emg_features = get_emg_features(emg)
    emg = emg[6:6 + 6 * emg_features.shape[0], :]
    emg_orig = emg_orig[8:8 + 8 * emg_features.shape[0], :]
    assert emg.shape[0] == emg_features.shape[0] * 6

    with open(os.path.join(base_dir, f'{index}_info.json')) as f:
        info = json.load(f)

    return (
        emg_features, emg_orig.astype(np.float32),
        info['text'], (info['book'], info['sentence_index'])
    )


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
    def __init__(self, dev=False, test=False, no_testset=False, no_normalizers=False):
        if no_testset:
            devset = []
            testset = []
        else:
            with open(FLAGS.testset_file) as f:
                testset_json = json.load(f)
                devset = testset_json['dev']
                testset = testset_json['test']

        directories = []
        for sd in FLAGS.silent_data_directories:
            for session_dir in sorted(os.listdir(sd)):
                directories.append(
                    EMGDirectory(len(directories), os.path.join(sd, session_dir))
                )

        self.example_indices = []
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
                        self.example_indices.append((directory_info, int(idx_str)))

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        self.no_normalizers = no_normalizers
        if not self.no_normalizers:
            self.emg_norm = pickle.load(open(FLAGS.normalizers_file,'rb'))

        sample_emg, *_ = load_utterance(
            self.example_indices[0][0].directory, self.example_indices[0][1]
        )
        self.num_features = sample_emg.shape[1]
        self.num_sessions = len(directories)

    def __len__(self):
        return len(self.example_indices)

    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        emg, raw_emg, text, book_location = load_utterance(directory_info.directory, idx)
        raw_emg = raw_emg / 10

        if not self.no_normalizers:
            emg = self.emg_norm.normalize(emg)
            emg = 8*np.tanh(emg/8.)

        session_ids = np.full(emg.shape[0], directory_info.session_index, dtype=np.int64)

        return {
            'emg': emg,
            'raw_emg': raw_emg,
            'text': text,
            'file_label': idx,
            'session_ids': session_ids,
            'book_location': book_location,
        }

    @staticmethod
    def collate_fixed_length(batch):
        pass

def make_normalizers():
    dataset = EMGDataset(no_normalizers=True)
    emg_samples = []
    for d in dataset:
        emg_samples.append(d['emg'])
        if len(emg_samples) > 50:
            break
    emg_norm = FeatureNormalizer(emg_samples, share_scale=False)
    pickle.dump(emg_norm, open(FLAGS.normalizers_file, 'wb'))

if __name__ == '__main__':
    FLAGS(sys.argv)
    d = EMGDataset()
    for i in range(1000):
        d[i]
