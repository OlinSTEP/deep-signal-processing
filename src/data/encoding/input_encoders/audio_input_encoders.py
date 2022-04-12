import random

import torch
import torchaudio
import numpy as np

from .input_encoder import AbstractInputEncoder
from src.data.filters import filter_audio_channel, resample_channel, loud_norm


def _get_stereo_sample_rate(channels):
    # Stereo channels should have same input
    sample_rate = channels[0][0]
    assert sample_rate == channels[1][0]
    return sample_rate


class AudioInputEncoder(AbstractInputEncoder):
    """
    Input encoder for audio.

    MUST BE EXTENDED. Does not implement the transform() method which is
    required of Input Encoders. Child classes should call _transform() with
    a selection (or all) input channels, along with a sample rate.
    """
    def __init__(self, config):
        super().__init__(config)

        # Processing
        self.samplerate = config.samplerate
        self.loudness = config.loudness

        # Padding
        self.max_ms = config.max_ms

        # Augmentations
        self.aug = config.aug
        self.aug_pad = config.aug_pad
        self.aug_shift = config.aug_shift
        self.aug_spec = config.aug_spec

        # Mel spectogram
        self.n_fft = config.n_fft
        self.n_mels = config.n_mels
        self.hop_len = config.hop_len

    def fit(self, inputs):
        spectogram = self.transform(next(inputs), False)
        self._input_dim = spectogram.numpy().shape

    def collate_fn(self, batch):
        batch_input = torch.stack([d["input"] for d in batch])
        batch_target = torch.tensor([d["target"] for d in batch])
        return {
            "input": batch_input,
            "target": batch_target,
        }

    def _transform(self, input_, is_train, sample_rate):
        channels = self.process_channels(
            input_, sample_rate
        )
        channels = self.pad_trunc_channels(
            channels, sample_rate, self.max_ms, is_train
        )
        if self.aug and is_train and self.aug_shift:
            self.aug_channels(channels)

        # (channels, n_mels, time)
        spectogram = self.to_spectogram(channels, sample_rate)
        if self.aug and is_train and self.aug_spec:
            spectogram = self.aug_spectogram(spectogram)

        return spectogram

    def process_channels(self, channels, target_sr):
        sample_rates, channels = list(zip(*channels))
        filtered = [
            filter_audio_channel(sr, d)
            for sr, d in zip(sample_rates, channels)
        ]
        resampled = [
            resample_channel(d, sr, target_sr) if sr != target_sr else d
            for sr, d in zip(sample_rates, filtered)
        ]
        if self.loudness:
            loud_normed = [
                loud_norm(sr, d, self.loudness)
                for sr, d in zip(sample_rates, resampled)
            ]
        return loud_normed

    def pad_trunc_channels(self, channels, sample_rate, max_ms, is_train):
        max_len = sample_rate // 1000 * max_ms

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

    def to_spectogram(self, channels, sample_rate, top_db=80):
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate,
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


class RegMicInputEncoder(AudioInputEncoder):
    def transform(self, input_, is_train):
        """
        Expected to receive channels corresponding to:
        [reg_audio_0, reg_audio_1, throat_audio_0, throat_audio_1]
        Only uses channels 0 and 1
        """
        # Only use the reg_audio channels
        sample_rate = _get_stereo_sample_rate(input_[:2])
        return self._transform(
            input_[:2],
            is_train,
            sample_rate
        )


class ThroatMicInputEncoder(AudioInputEncoder):
    def transform(self, input_, is_train):
        """
        Expected to receive channels corresponding to:
        [reg_audio_0, reg_audio_1, throat_audio_0, throat_audio_1]
        Only uses channels 2 and 3
        """
        # Only use the throat_audio channels
        # sample_rate = _get_stereo_sample_rate(input_[2:])
        sample_rate = self.samplerate
        return self._transform(
            input_[2:],
            is_train,
            sample_rate
        )


class BothMicInputEncoder(AudioInputEncoder):
    def transform(self, input_, is_train):
        """
        Expected to receive channels corresponding to:
        [reg_audio_0, reg_audio_1, throat_audio_0, throat_audio_1]
        Uses all channels
        """
        # Get sample rate from reg audio
        sample_rate = _get_stereo_sample_rate(input_[:2])
        # Use all channels
        return self._transform(
            input_,
            is_train,
            sample_rate
        )
