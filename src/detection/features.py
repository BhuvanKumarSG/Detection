import numpy as np
import librosa
from scipy.signal import hilbert


def extract_features(path, sr=22050, n_mfcc=13, feature_set="base"):
    """Load an audio file and compute a compact feature vector.

    feature_set: "base" (fast, small) or "bio" (richer, "bio-inspired" auditory features)

    Returns a 1D numpy array.
    """
    # librosa.load supports many formats via soundfile/audioread backends
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        # conservative default size (base): n_mfcc*3 + 3
        if feature_set == "base":
            return np.zeros((n_mfcc * 3 + 3,), dtype=float)
        else:
            # estimate for bio set (mfcc + delta + std + contrast + chroma + tonnetz + 3 spectral)
            return np.zeros((n_mfcc * 3 + 12 + 3,), dtype=float)

    # Base features: MFCC mean/std/delta, plus centroid/rolloff/zcr
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

    spec_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))

    base_feat = np.concatenate([
        mfcc_mean,
        mfcc_std,
        mfcc_delta_mean,
        [spec_centroid_mean, spec_rolloff_mean, zcr_mean]
    ])

    if feature_set == "base":
        feat = base_feat
    else:
        # Bio-inspired feature set (chosen algorithm): cochleagram / mel-filterbank envelopes
        # + low-frequency modulation spectrum features (speech-relevant modulation bands)
        # Compute mel spectrogram (power)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=sr // 2)
        # Convert to amplitude
        S_amp = np.sqrt(S)

        # For each mel band compute the temporal envelope via Hilbert transform
        # and compute modulation spectrum energy in a few bands (e.g. 0.5-2Hz, 2-8Hz, 8-16Hz)
        # Parameters
        n_mels = S_amp.shape[0]
        # Reconstruct band signals by inverse mel (approx) not necessary; instead use per-band frames
        # S_amp is (n_mels, t_frames)
        env = np.abs(hilbert(S_amp, axis=1))

        # Compute modulation spectrum (FFT across time for each band)
        # Use real FFT
        mod_spec = np.fft.rfft(env, axis=1)
        mod_power = np.abs(mod_spec) ** 2

        # Build modulation frequency axis
        hop_length = 512
        sr_frames = sr / hop_length
        freqs = np.fft.rfftfreq(env.shape[1], d=1.0 / sr_frames)

        # Define modulation bands (Hz)
        mod_bands = [(0.5, 2.0), (2.0, 8.0), (8.0, 16.0)]
        mod_feats = []
        for lo, hi in mod_bands:
            # find indices
            idx = (freqs >= lo) & (freqs < hi)
            if not np.any(idx):
                # fallback zero
                band_energy = np.zeros((n_mels,))
            else:
                band_energy = np.mean(mod_power[:, idx], axis=1)
            # summarize across mel bands (mean and std)
            mod_feats.append(np.mean(band_energy))
            mod_feats.append(np.std(band_energy))

        # spectral contrast and chroma and tonnetz remain useful complements
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_contrast_mean = np.mean(spec_contrast, axis=1)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Tonnetz requires harmonic component
        y_harmonic = librosa.effects.harmonic(y)
        try:
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
        except Exception:
            tonnetz_mean = np.zeros((6,))

        # concatenate into an extended feature vector
        feat = np.concatenate([
            base_feat,
            spec_contrast_mean,
            chroma_mean,
            tonnetz_mean,
            np.array(mod_feats)
        ])

    # Ensure finite
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat
