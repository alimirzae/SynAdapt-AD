import numpy as np
import librosa

def extract_7d_timbral_features(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    sharpness = np.mean(cent) / (sr / 2.0)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    stft = np.abs(librosa.stft(y))
    roughness = np.mean(np.diff(stft, axis=0)**2)
    rms = librosa.feature.rms(y=y)
    shimmer = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-8)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)
    boominess = np.mean(mfcc)
    
    vec = np.array([sharpness, roughness, shimmer, boominess, flatness, rolloff, zcr], dtype=np.float32)
    return vec
