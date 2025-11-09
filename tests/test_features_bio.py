import tempfile
import os
import numpy as np
import soundfile as sf

from detection.features import extract_features


def test_extract_features_bio_sane():
    # generate a short 0.5s sine wave at 22050 Hz
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        fname = tf.name
    try:
        sf.write(fname, y, sr)
        feat = extract_features(fname, sr=sr, feature_set="bio")
        assert isinstance(feat, np.ndarray)
        assert feat.ndim == 1
        assert feat.size > 0
        assert np.isfinite(feat).all()
    finally:
        try:
            os.remove(fname)
        except Exception:
            pass
