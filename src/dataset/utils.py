import numpy as np
import librosa
import scipy

import matplotlib.pyplot as plt

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('normalizers_file', 'normalizers.pkl', 'file with pickled feature normalizers')


def double_average(x):
    assert len(x.shape) == 1
    f = np.ones(9)/9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w


def get_emg_features(emg_data, debug=False):
    xs = emg_data - emg_data.mean(axis=0, keepdims=True)
    frame_features = []
    for i in range(emg_data.shape[1]):
        x = xs[:,i]
        w = double_average(x)
        p = x - w
        r = np.abs(p)

        w_h = librosa.util.frame(w, frame_length=16, hop_length=6).mean(axis=0)
        p_w = librosa.feature.rms(w, frame_length=16, hop_length=6, center=False)
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(r, frame_length=16, hop_length=6, center=False)
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(p, frame_length=16, hop_length=6, center=False)
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(r, frame_length=16, hop_length=6).mean(axis=0)

        s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=16, hop_length=6, center=False))
        # s has feature dimension first and time second

        if debug:
            plt.subplot(7,1,1)
            plt.plot(x)
            plt.subplot(7,1,2)
            plt.plot(w_h)
            plt.subplot(7,1,3)
            plt.plot(p_w)
            plt.subplot(7,1,4)
            plt.plot(p_r)
            plt.subplot(7,1,5)
            plt.plot(z_p)
            plt.subplot(7,1,6)
            plt.plot(r_h)

            plt.subplot(7,1,7)
            plt.imshow(s, origin='lower', aspect='auto', interpolation='nearest')

            plt.show()

        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        frame_features.append(s.T)

    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)


class FeatureNormalizer(object):
    def __init__(self, feature_samples, share_scale=False):
        """ features_samples should be list of 2d matrices with dimension (time, feature) """
        feature_samples = np.concatenate(feature_samples, axis=0)
        self.feature_means = feature_samples.mean(axis=0, keepdims=True)
        if share_scale:
            self.feature_stddevs = feature_samples.std()
        else:
            self.feature_stddevs = feature_samples.std(axis=0, keepdims=True)

    def normalize(self, sample):
        sample -= self.feature_means
        sample /= self.feature_stddevs
        return sample

    def inverse(self, sample):
        sample = sample * self.feature_stddevs
        sample = sample + self.feature_means
        return sample


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
