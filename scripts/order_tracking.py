import numpy as np
import librosa

def extract_instantaneous_frequency(y, sr):
    stft = librosa.stft(y)
    spectrogram = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr)
    peak_indices = np.argmax(spectrogram, axis=0)
    inst_freq = freqs[peak_indices]
    return inst_freq

def resample_order_domain(y, sr, inst_freq):
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    even_phase = np.linspace(phase[0], phase[-1], len(y))
    resampled_y = np.interp(even_phase, phase, y)
    return resampled_y
