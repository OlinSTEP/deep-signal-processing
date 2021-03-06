import random
import warnings

import pyloudnorm as pln

from .general_filters import smooth, notch_harmonics, remove_drift

# pln throws a meaningless warnings we don't want
warnings.filterwarnings(
    action='ignore',
    module=r'.*pyloudnorm.*'
)

def filter_audio_channel(sample_freq, channel):
    # x = smooth(channel)
    x = channel
    # x = notch_harmonics(x, 60, sample_freq)
    x = remove_drift(x, sample_freq)
    return x


def loud_norm(sample_freq, channel, target, is_train, shift=0):
    meter = pln.Meter(sample_freq)
    loudness = meter.integrated_loudness(channel)
    if loudness < 0:
        return channel
    if shift > 0 and is_train:
        target += random.randint(-target, target)
    normed = pln.normalize.loudness(channel, loudness, target)
    return normed

def normalize_wave(channel):
    return (channel - channel.mean()) / channel.std()
