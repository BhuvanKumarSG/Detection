"""Generate a tiny synthetic dataset for quick training/testing.
Creates `dataset/small/real` and `dataset/small/fake` with short WAV files.
"""
import os
import numpy as np
import soundfile as sf

OUT = os.path.join(os.path.dirname(__file__), "..", "dataset", "small")
REAL = os.path.join(OUT, "real")
FAKE = os.path.join(OUT, "fake")
os.makedirs(REAL, exist_ok=True)
os.makedirs(FAKE, exist_ok=True)

sr = 22050
n = 8
length = 1.0  # seconds

for i in range(n):
    t = np.linspace(0, length, int(sr * length), endpoint=False)
    # Real: vowel-like sum of harmonics with slow amplitude modulation
    a = 0.6 * (1.0 + 0.3 * np.sin(2 * np.pi * 2.0 * t))
    signal = a * (0.5 * np.sin(2 * np.pi * 220 * t) + 0.25 * np.sin(2 * np.pi * 440 * t))
    path = os.path.join(REAL, f"real_{i}.wav")
    sf.write(path, signal, sr)

for i in range(n):
    t = np.linspace(0, length, int(sr * length), endpoint=False)
    # Fake: synthetic with faster modulation + added noise and slight clipping
    a = 0.3 * (1.0 + 0.6 * np.sign(np.sin(2 * np.pi * 10.0 * t)))
    signal = a * (0.6 * np.sin(2 * np.pi * 220 * t + 0.5) + 0.3 * np.sin(2 * np.pi * 660 * t))
    # add high-frequency noise
    signal += 0.02 * np.random.randn(signal.shape[0])
    # soft clip
    signal = np.tanh(signal * 2.0)
    path = os.path.join(FAKE, f"fake_{i}.wav")
    sf.write(path, signal, sr)

print(f"Generated synthetic dataset at: {OUT}")
