from abc import ABC, abstractmethod
from scipy import signal
import numpy as np

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

        :param data np.array: Numpy array of single datapoint
        """
        pass


class MicFilter(AbstractFilter):
    def __init__(self, config):
        super().__init__()

    def smooth(self, data):
        """
        Smoothes data by taking average of every two datapoints in array
        Returns a shorter np.array

        TODO: make the number of datapoints to be averaged variable
        """
        return ((data + np.roll(data, 1))/2.0)[1::2]

    def remove_drift(self, data, fs):
        """
        fs: sampling frequency; nominal srate
        """
        # [lowcut, highcut] = [40, 280] are currently arbitrary values 
        # typical adult male will have a fundamental frequency from 85 to 155 Hz, and that of a typical adult female from 165 to 255 Hz
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

    def filter(self, data, fs):
        """ data is 2d?: time, channels """
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            x = data[:,i]
            x = self.smooth(data)
            x = self.notch_harmonics(x, 60, fs)
            x = self.remove_drift(x, fs)
            result[:,i] = x
        return result


class ThroatMicFilter(AbstractFilter):
    def __init__(self, config):
        super().__init__()

    def smooth(self, data):
        """
        Smoothes data by taking average of every two datapoints in array
        Returns a shorter np.array

        TODO: make the number of datapoints to be averaged variable
        """
        return ((data + np.roll(data, 1))/2.0)[1::2]

    def remove_drift(self, data, fs):
        """
        fs: sampling frequency; nominal srate
        """
        # [lowcut, highcut] = [40, 280] are currently arbitrary values 
        # typical adult male will have a fundamental frequency from 85 to 155 Hz, and that of a typical adult female from 165 to 255 Hz
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

    def filter(self, data, fs):
        """ data is 2d?: time, channels """
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            x = data[:,i]
            x = self.smooth(data)
            x = self.notch_harmonics(x, 60, fs)
            x = self.remove_drift(x, fs)
            result[:,i] = x
        return result


class BothMicFilter(AbstractFilter):
    def __init__(self, config):
        super().__init__()

    def filter(self, data):
        pass


class GestureFilter(AbstractFilter):
    def __init__(self, config):
        super().__init__()

    def filter(self, data):
        pass
