import random

import torch
import torchaudio
import numpy as np

from .input_encoder import AbstractInputEncoder
from src.data.filters import (
    filter_audio_channel, resample_channel, loud_norm, normalize_wave
)
from src.utils.norm_image import StatsRecorder


def _get_stereo_sample_rate(channels):
    # Stereo channels should have same input
    sample_rate = channels[0][0]
    assert sample_rate == channels[1][0]
    return sample_rate


class AudioInputEncoder(AbstractInputEncoder):
    """
    Input encoder for audio.
    """
    def __init__(self, config):
        super().__init__(config)

        # Processing
        self.samplerate = config.samplerate
        self.norm_wave = config.norm
        self.norm_spec = config.norm
        self.loudness = config.loudness

        # Padding
        self.max_ms = config.max_ms

        # Augmentations
        self.aug = config.aug
        self.aug_pad = config.aug_pad
        self.aug_shift = config.aug_shift
        self.aug_spec = config.aug_spec
        self.aug_volume = config.aug_volume

        # Mel spectogram
        self.n_fft = config.n_fft
        self.n_mels = config.n_mels
        self.hop_len = config.hop_len

    def fit(self, inputs):
        if self.norm_spec:
            self.spec_stats = StatsRecorder()
            self.norm_spec = 0
            for input_ in inputs:
                spectogram = self.transform(input_, False)
                self.spec_stats.update(spectogram[None, :])
            self.norm_spec = 1
        else:
            spectogram = self.transform(next(inputs), False)
        self._input_dim = spectogram.numpy().shape

    def collate_fn(self, batch):
        batch_input = torch.stack([d["input"] for d in batch])
        batch_target = torch.tensor([d["target"] for d in batch])

        # Normalizing spectograms at batch level for efficiency
        if self.norm_spec:
            batch_input = (
                (batch_input - self.spec_stats.mean) / self.spec_stats.std
            )

        return {
            "input": batch_input,
            "target": batch_target,
        }

    def transform(self, input_, is_train):
        channels = self.process_channels(input_, is_train)
        channels = self.pad_trunc_channels(channels, is_train)
        if self.aug and is_train and self.aug_shift:
            channels = self.aug_channels(channels)

        # (channels, n_mels, time)
        spectogram = self.to_spectogram(channels)
        if self.aug and is_train and self.aug_spec:
            spectogram = self.aug_spectogram(spectogram)

        return spectogram

    def process_channels(self, channels, is_train):
        sample_rates, channels = list(zip(*channels))
        filtered = [
            filter_audio_channel(sr, d)
            for sr, d in zip(sample_rates, channels)
        ]
        processed = [
            resample_channel(d, sr, self.samplerate)
            if sr != self.samplerate else d
            for sr, d in zip(sample_rates, filtered)
        ]

        if self.norm_wave:
            processed = [
                normalize_wave(d)
                for d in channels
            ]
        elif self.loudness:
            processed = [
                loud_norm(sr, d, self.loudness, is_train, shift=self.aug_volume)
                for sr, d in zip(sample_rates, processed)
            ]

        return processed

    def pad_trunc_channels(self, channels, is_train):
        max_len = int(self.samplerate * self.max_ms / 1000)

        resized_channels = []
        for channel in channels:
            l = len(channel)
            if l > max_len:
                channel = channel[:max_len]
            elif l < max_len:
                diff = max_len - l
                if self.aug and is_train and self.aug_pad:
                    start_pad_len = random.randint(0, diff)
                else:
                    start_pad_len = 0
                end_pad_len = diff - start_pad_len

                start_pad = np.zeros(start_pad_len)
                end_pad = np.zeros(end_pad_len)
                channel = np.concatenate((start_pad, channel, end_pad), axis=0)
            resized_channels.append(channel)
        return np.array(resized_channels)

    def aug_channels(self, channels, shift_pct=0.1):
        # channels must be a numpy array
        for channel in channels:
            shift_amt = int(random.random() * shift_pct * len(channels))
            channel[:] = np.roll(channel, shift_amt)
        return channels

    def to_spectogram(self, channels, top_db=80):
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        transform = torchaudio.transforms.MelSpectrogram(
            self.samplerate,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            n_mels=self.n_mels
        )
        spectogram = transform(torch.tensor(channels, dtype=torch.float32))

        # Convert to decibels
        transform = torchaudio.transforms.AmplitudeToDB(top_db=top_db)
        spectogram = transform(spectogram)

        return spectogram

    def aug_spectogram(
        self, spectogram,
        max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1
    ):
        _, n_mels, n_steps = spectogram.shape
        mask_value = spectogram.mean()
        aug_spec = spectogram

        freq_mask_param = int(max_mask_pct * n_mels)
        for _ in range(n_freq_masks):
            transform = torchaudio.transforms.FrequencyMasking(freq_mask_param)
            aug_spec = transform(aug_spec, mask_value)

        time_mask_param = int(max_mask_pct * n_steps)
        for _ in range(n_time_masks):
            transform = torchaudio.transforms.TimeMasking(time_mask_param)
            aug_spec = transform(aug_spec, mask_value)

        return aug_spec
