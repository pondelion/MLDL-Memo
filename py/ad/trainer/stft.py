from scipy.signal import stft as sp_stft
import numpy as np


def stft(
    signal: np.ndarray,
    window_size: int=0.5,
    fs: int=44100,
    window_stride: float=0.25
) -> np.ndarray:
    n_sample = int(window_size * fs)
    n_overlap = int((window_size - window_stride) * fs)
    print(n_sample)
    print(n_overlap)
    f, t, sp = sp_stft(signal, fs, nperseg=n_sample, noverlap=n_overlap)
    return sp[:, 1]
