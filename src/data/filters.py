import numpy as np

from scipy import signal
import torchaudio.transforms


def smooth(data):
    """
    Smoothes data by taking average of every two datapoints in array
    Returns a shorter np.array
    """
    #TODO: make the number of datapoints to be averaged variable
    return ((data + np.roll(data, 1))/2.0)[1::2]


def remove_drift(data, sample_frequency):
    """
    sample_frequency: nominal srate
    """
    # [lowcut, highcut] = [40, 280] are currently arbitrary values
    # typical adult male will have a fundamental frequency from 85 to 155
    # Hz, and that of a typical adult female from 165 to 255 Hz
    b, a = signal.butter(3, [40, 280], btype='bandpass', fs=sample_frequency)
    return signal.filtfilt(b, a, data)


def notch(data, freq, sample_freq):
    # quality factor (Q): (f2 - f1)/f_center
    b, a = signal.iirnotch(freq, 2, sample_freq)
    return signal.filtfilt(b, a, data)


def notch_harmonics(data, freq, sample_freq):
    for f in range(freq, sample_freq//2, freq):
        signal = notch(data, f, sample_freq)
    return signal


def filter_audio_channel(sample_freq, channel):
    # x = smooth(channel)
    x = channel
    x = notch_harmonics(x, 60, sample_freq)
    x = remove_drift(x, sample_freq)
    return x


def resample_channel(data, sample_freq, target_freq):
    transform = torchaudio.transforms.Resample(sample_freq, target_freq)
    tensor = transform(data[None, :])
    return tensor.numpy()
