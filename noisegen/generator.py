import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


class NoiseGenerator:
    def __init__(self, n_frequencies, f_interval):
        self.f_interval = f_interval
        self.t_end = 1 / self.f_interval
        self.n_frequencies = n_frequencies
        self.n_fft_frequencies = 2 * self.n_frequencies - 1
        self.n_times = self.n_fft_frequencies
        self.t_interval = self.t_end / self.n_times
        self.nyquist_frequency = 0.5 * self.n_fft_frequencies * self.f_interval
        self.positive_frequencies = np.arange(n_frequencies) * f_interval
        self.sample_times = np.linspace(0, self.t_end, self.n_times, endpoint=False)
        self.samples = None
        self.fft_coeffs = None
        self.fft_frequencies = np.fft.fftfreq(self.n_fft_frequencies, self.t_interval)
        self.measured_psd = None
        self.mean_square_fft_coeffs = None
        self.autocorrelation = None
        self.psd = None
        self.fft_power_filter = None
        self.fft_amplitude_filter = None

    def specify_psd(self, psd='white', f_ir=None, normalization=None):
        if psd is 'white':
            self.psd = np.ones(self.n_frequencies)
        elif psd is 'pink':
            assert f_ir is not None
            cutoff_idx = np.sum(self.positive_frequencies < f_ir)
            self.psd = np.zeros(self.n_frequencies)
            self.psd[cutoff_idx:] = 1 / self.positive_frequencies[cutoff_idx:]
            self.psd[:cutoff_idx] = self.psd[cutoff_idx]
        else:
            self.psd = psd

        self.psd = np.hstack([self.psd, np.flip(self.psd)[:-1]])
        if normalization is not None:
            self.psd *= normalization / (np.sum(self.psd) * self.f_interval)

        self.fft_power_filter = self.psd * self.f_interval * self.n_fft_frequencies ** 2

        self.fft_amplitude_filter = np.sqrt(self.fft_power_filter)

        self.psd = pd.Series(self.psd, index=self.fft_frequencies)
        self.psd.sort_index(inplace=True)
        self.fft_amplitude_filter = pd.Series(self.fft_amplitude_filter, index=self.fft_frequencies)
        self.fft_power_filter = pd.Series(self.fft_power_filter, index=self.fft_frequencies)
        self.fft_power_filter.sort_index(inplace=True)

    def generate_trace(self, seed=None, n_traces=1):

        np.random.seed(seed)

        fft_coeffs_array = np.zeros([self.n_fft_frequencies, n_traces], dtype=complex)
        signal_array = np.zeros([self.n_times, n_traces], dtype=float)

        for i in tqdm(range(n_traces)):

            fft_coeffs = np.random.randn(self.n_frequencies - 1) + 1j * np.random.randn(self.n_frequencies - 1)
            fft_coeffs /= np.sqrt(2)
            fft_coeffs = np.hstack([fft_coeffs, np.conjugate(np.flip(fft_coeffs))])
            fft_coeffs = np.hstack([np.random.randn(), fft_coeffs])
            fft_coeffs *= self.fft_amplitude_filter.values

            fft_coeffs_array[:, i] = fft_coeffs

            signal = np.fft.ifft(fft_coeffs).real
            signal_array[:, i] = signal

        self.fft_coeffs = pd.DataFrame(fft_coeffs_array, index=self.fft_frequencies, columns=np.arange(n_traces))
        self.fft_coeffs.index.name = 'frequency'
        self.fft_coeffs.columns.name = 'sample'

        self.samples = pd.DataFrame(signal_array, index=self.sample_times, columns=np.arange(n_traces))
        self.samples.index.name = 'time'
        self.samples.columns.name = 'sample'

    def calc_autocorrelation(self, t_idx=0):
        self.autocorrelation = self.samples.copy()
        self.autocorrelation *= self.autocorrelation.iloc[t_idx, :]
        self.autocorrelation = self.autocorrelation.mean(axis=1)
        self.autocorrelation = pd.Series(self.autocorrelation, index=self.sample_times)

    def measure_psd(self):
        self.mean_square_fft_coeffs = (self.fft_coeffs.abs() ** 2).mean(axis=1).values
        self.measured_psd = np.copy(self.mean_square_fft_coeffs)
        self.measured_psd /= (self.n_fft_frequencies ** 2)*self.f_interval
        self.measured_psd = pd.Series(self.measured_psd, index=self.fft_frequencies)
        self.measured_psd.sort_index(inplace=True)
        self.mean_square_fft_coeffs = pd.Series(self.mean_square_fft_coeffs, index=self.fft_frequencies)
        self.mean_square_fft_coeffs.sort_index(inplace=True)

    def plot_psd(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        if self.measured_psd is None:
            self.measure_psd()
        self.measured_psd.plot(ax=ax, **kwargs)
        self.psd.plot(ax=ax, **kwargs)
        return ax

    def plot_fft_filter(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        if self.mean_square_fft_coeffs is None:
            self.measure_psd()
        combined_data = pd.DataFrame({'Power filter': self.fft_power_filter,
                                      'Mean square Fourier coefficients': self.mean_square_fft_coeffs})
        combined_data.plot.bar(ax=ax, **kwargs)
        return ax

    def plot_autocorrelation(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        t_idx = (self.n_times - 1) // 2
        self.calc_autocorrelation(t_idx=t_idx)
        self.autocorrelation.index = self.autocorrelation.index - self.t_end / 2
        self.autocorrelation.plot(ax=ax, **kwargs)
        return ax
