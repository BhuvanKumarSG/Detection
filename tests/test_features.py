import os
import numpy as np
import soundfile as sf
from detection.features import extract_features

def test_extract_features(tmp_path):
    # generate a short sine wave
    sr = 16000
    t = np.linspace(0, 0.5, int(sr*0.5), endpoint=False)
    x = 0.1 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "sine.wav"
    sf.write(str(path), x, sr)

    feat = extract_features(str(path), sr=sr)
    assert feat is not None
    assert feat.size > 0
    assert not np.any(np.isnan(feat))
