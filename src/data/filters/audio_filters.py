import pyloudnorm as pln

from .general_filters import smooth, notch_harmonics, remove_drift


def filter_audio_channel(sample_freq, channel):
    # x = smooth(channel)
    x = channel
    # x = notch_harmonics(x, 60, sample_freq)
    x = remove_drift(x, sample_freq)
    return x


def loud_norm(sample_freq, channel, target):
    meter = pln.Meter(sample_freq)
    loudness = meter.integrated_loudness(channel)
    normed = pln.normalize.loudness(channel, loudness, target)
    return normed
