from abc import ABC, abstractmethod

import numpy as np

from scipy import signal


class AbstractFilter(ABC):
    """
    Filtering base class

    Filters take in raw data and performs processing on them
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def filter(self, data):
        """
        Filters the passed data

        :param data list: List of (sample_rate, sequence_data) per input channel
        """
        pass


class AudioFilter(AbstractFilter):
    def smooth(self, data):
        """
        Smoothes data by taking average of every two datapoints in array
        Returns a shorter np.array
        """
        #TODO: make the number of datapoints to be averaged variable
        return ((data + np.roll(data, 1))/2.0)[1::2]

    def remove_drift(self, data, fs):
        """
        fs: sampling frequency; nominal srate
        """
        # [lowcut, highcut] = [40, 280] are currently arbitrary values
        # typical adult male will have a fundamental frequency from 85 to 155
        # Hz, and that of a typical adult female from 165 to 255 Hz
        b, a = signal.butter(3, [40, 280], btype='bandpass', fs=fs)
        return signal.filtfilt(b, a, data)

    def notch(self, freq, sample_freq):
        # quality factor (Q): (f2 - f1)/f_center
        b, a = signal.iirnotch(freq, 2, sample_freq)
        return signal.filtfilt(b, a, signal)

    def notch_harmonics(self, data, freq, sample_freq):
        for f in range(freq, sample_freq//2, freq):
            signal = self.notch(data, f, sample_freq)
        return signal

    def filter_channel(self, sample_freq, channel):
        # x = self.smooth(channel)
        x = channel
        x = self.notch_harmonics(x, 60, sample_freq)
        x = self.remove_drift(x, sample_freq)
        return x


class RegMicFilter(AudioFilter):
    def __init__(self, config):
        super().__init__()

    def filter(self, data):
        """
        Filters the passed data

        Expected to receive channels corresponding to:
        [reg_audio_0, reg_audio_1, throat_audio_0, throat_audio_1]
        as specified in the AudioLoader.  Only channels 0 and 1 are used

        :param data list: List of (sample_rate, sequence_data) per input channel
            sample_rate is an integer, sequence_data is a 1-D array

        :returns: List of np arrays, where each element is a filtered channel
        """
        return [
            self.filter_channel(sr, d)
            for sr, d in data[:2]  # Use only regular mic channels
        ]


class ThroatMicFilter(AudioFilter):
    def __init__(self, config):
        super().__init__()

    def filter(self, data, fs):
        """
        Filters the passed data

        :param data list: List of (sample_rate, sequence_data) per input channel
            Expected to receive channels corresponding to:
            [reg_audio_0, reg_audio_1, throat_audio_0, throat_audio_1]
            as specified in the AudioLoader. Only channels 2 and 3 are used
        """
        return [
            self.filter_channel(sr, d)
            for sr, d in data[2:]  # Use only throat mic channels
        ]


class BothMicFilter(AudioFilter):
    def __init__(self, config):
        super().__init__()
        self.reg_filter = RegMicFilter(config)
        self.throat_filter = ThroatMicFilter(config)

    def filter(self, data):
        """
        Filters the passed data

        :param data list: List of (sample_rate, sequence_data) per input channel
            Expected to receive channels corresponding to:
            [reg_audio_0, reg_audio_1, throat_audio_0, throat_audio_1]
            as specified in the AudioLoader. All channels are used
        """
        # All channels processed and concat together
        return self.reg_filter.filter(data) + self.throat_filter.filter(data)


class GestureFilter(AbstractFilter):
    def __init__(self, config):
        super().__init__()

    def filter(self, data):
        pass
