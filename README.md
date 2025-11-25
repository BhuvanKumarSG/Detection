Deepfake Voice Detector (starter)

This repository is a starter Python project for detecting deepfake (synthetic) voices using simple audio features and a classical ML classifier. It's intended as a baseline you can extend with larger datasets and modern deep-learning anti-spoofing models.

Project layout
- src/detection: library code (features, model, CLI)
- examples: example usage
- tests: unit tests

Quick start
1. Create a virtual environment and install dependencies:

   # Create venv and install dependencies (Windows PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt

   # OR, activate and use the venv Python directly without changing shell state
   .venv\Scripts\python.exe -m pip install -r requirements.txt

2. Prepare a dataset with the structure:

   dataset/
     real/   (wav files of real speech)
     fake/   (wav files of synthetic/manipulated speech)

3. Train a model:

   # Using the activated venv in PowerShell
   python -m detection.cli train --dataset dataset --out-model model.ckpt

   # Or run with the venv Python directly (works from any shell):
   .venv\Scripts\python.exe -m detection.cli train --dataset dataset --out-model model.ckpt

   # Quick example: train & compare baseline and bio models on the supplied tiny sample dataset
   .venv\Scripts\python.exe -m detection.cli train-compare --dataset dataset\small --out-base base_model.ckpt --out-bio bio_model.ckpt

Commands I ran (PowerShell, project root)
-------------------------------------------------
Below are the exact commands used during this session for reproducibility. They assume you're in the project root (where `pyproject.toml` and `requirements.txt` live) on Windows PowerShell.

# 1) Create and activate the virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies into the venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3) Run the unit test used during development
.venv\Scripts\python.exe -m pytest -vv tests\test_features.py -s

# 4) Train and compare the baseline and bio models on the tiny sample dataset
.venv\Scripts\python.exe -m detection.cli train-compare --dataset dataset\small --out-base base_model.ckpt --out-bio bio_model.ckpt

# 5) Quick evaluation on a small ASVspoof subset (balanced sampling from protocol)
.venv\Scripts\python.exe scripts\predict_external.py "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_dev\flac" 20 base_model.ckpt bio_model.ckpt "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt" --balanced

# 5b) Compare two saved models on the ASVspoof dev set (balanced, 200 samples)
# This runs both models, prints per-file confidences and predictions, reports Accuracy/AUC/EER, and declares the winner.
.venv\Scripts\python.exe scripts\predict_external.py "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_dev\flac" 200 base_model_full.ckpt bio_model_full.ckpt "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt" --balanced

# 6) Created and trained on a 100/100 ASVspoof training subset (100 real + 100 fake)
.venv\Scripts\python.exe scripts\make_asvspoof_sample.py "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_train\flac" "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt" dataset\asvspoof_100_100 --n-per-class 100 --seed 123
.venv\Scripts\python.exe -m detection.cli train-compare --dataset dataset\asvspoof_100_100 --out-base base_model_100.ckpt --out-bio bio_model_100.ckpt

# 7) (Earlier) created and trained on a 500/500 ASVspoof subset used for larger runs
.venv\Scripts\python.exe scripts\make_asvspoof_sample.py "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_train\flac" "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt" dataset\asvspoof_500_500 --n-per-class 500 --seed 123
.venv\Scripts\python.exe -m detection.cli train-compare --dataset dataset\asvspoof_500_500 --out-base base_model_500.ckpt --out-bio bio_model_500.ckpt

# 8) Create and train on the full ASVspoof training set (created during this session)
.venv\Scripts\python.exe scripts\make_asvspoof_sample.py "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_train\flac" "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt" dataset\asvspoof_full --n-per-class 999999 --seed 123
.venv\Scripts\python.exe -m detection.cli train-compare --dataset dataset\asvspoof_full --out-base base_model_full.ckpt --out-bio bio_model_full.ckpt

-----------------------------------------------------------------------------------

4. Predict a file:

   # If you saved the model with the default .ckpt extension
   python -m detection.cli predict --model model.ckpt --file path/to/audio.wav

5. Run tests

   # Run the project's unit tests with the venv Python (PowerShell)
   .venv\Scripts\python.exe -m pytest -q

   # Or, if you've activated the venv in PowerShell already:
   pytest -q

Notes & next steps
- This is a feature-based baseline (MFCCs, spectral features) with a RandomForest classifier. For production or research-grade detection, consider using specialized anti-spoofing networks and large curated datasets (ASVspoof, etc.).
- Add data augmentation, balanced sampling, cross-validation, and stronger models (XGBoost, CNNs on spectrograms) for better performance.

Models & feature algorithms
---------------------------
Below is a concise summary of the two model variants used in this project and the high-level feature pipelines they use.

- base model (`base_model_100.ckpt`, `base_model_500.ckpt`)
   - Classifier: scikit-learn RandomForestClassifier (ensemble of decision trees).
   - Features: classic short-time spectral summaries (keeps the baseline small and fast):
      - MFCCs (mean, std, delta mean)
      - Spectral centroid and spectral rolloff
      - Zero-crossing rate
      - Small complementary spectral stats used in experiments (e.g. contrast)
   - Notes: quick to compute and a good baseline; tuning is done with AUC/EER in mind rather than raw accuracy.

- bio model (`bio_model_100.ckpt`, `bio_model_500.ckpt`)
   - Classifier: same RandomForest backend as the base model to keep comparisons fair.
   - Features: a "bio-inspired" cochleagram + modulation summary pipeline that emphasizes per-band envelopes and slow temporal modulations:
      1. Compute a mel-spectrogram with increased spectral resolution (e.g. `n_mels=64`).
      2. Convert power->amplitude and apply mild compressive nonlinearity (sqrt then `log1p`) to approximate cochlear compression.
      3. Extract per-mel-band temporal envelopes by computing the analytic signal (Hilbert transform) and taking amplitude.
      4. Per-band normalization (zero mean / unit std) to reduce dominance of loud bands.
      5. Compute modulation spectra (FFT across time per mel band) and summarize energy in multiple modulation bands (e.g. 0.5–2 Hz, 2–8 Hz, 8–16 Hz, 16–32 Hz) plus modulation centroids.
      6. Add dynamic cues (envelope temporal-delta RMS) and complementary spectral features (spectral contrast, chroma, tonnetz, mel-band mean/std).
   - Why it helps: spoofing artifacts often change fine spectral structure and amplitude-modulation patterns; explicit per-band envelope and modulation features can expose those differences even when MFCCs look similar.

Evaluation notes
----------------
- We report accuracy, ROC-AUC and EER. For spoof-detection tasks, AUC and EER are more informative than raw accuracy because they capture score separation and threshold robustness.
- Model selection in scripts currently combines the metrics (equal weight across 1−EER, AUC and accuracy) to pick a winner; you can change the weighting if your priority differs.

Recent changes in this branch
---------------------------
- The `bio` feature extractor's cochleagram pipeline was improved (more mel bands, modulation features, per-band normalization). The Hilbert transform is now applied per-band and feature extraction is robust for short signals.
- To reduce noisy warnings from `librosa` ("n_fft is too large for input signal of length=..."), the feature code now auto-adjusts `n_fft` and `hop_length` based on the input signal length so short audio files won't trigger that warning.
- Added small training subsets and models committed in this session:
   - `dataset/asvspoof_100_100/` — sampled 100 real + 100 fake from ASVspoof train.
   - `base_model_100.ckpt`, `bio_model_100.ckpt` — models trained on the 100/100 subset.
   - `dataset/asvspoof_500_500/` — sampled 500 real + 500 fake (used earlier).
   - `base_model_500.ckpt`, `bio_model_500.ckpt` — models trained on the 500/500 subset.

Repro tips
----------
- If you prefer to suppress the librosa warning globally, you can set the Python warnings filter in your session (not recommended globally):

```powershell
.venv\Scripts\python.exe - <<'PY'
import warnings
warnings.filterwarnings("ignore", message="n_fft=.*is too large for input signal")
PY
```

However, the code now adapts `n_fft` automatically and this global suppression is not necessary in most cases.

Downloading ASVspoof & creating the full dataset (`dataset/asvspoof_full`)
---------------------------------------------------------------
This project does not include the ASVspoof audio (large, redistributed under ASVspoof terms). To reproduce the experiments and create `dataset/asvspoof_full` locally, follow these steps:

1. Obtain the ASVspoof2019 LA archive
   - Register and download the ASVspoof2019 LA dataset from the official ASVspoof website or your institutional mirror. You need at least:
     - `ASVspoof2019_LA_train/flac` (training audio files)
     - `ASVspoof2019_LA_dev/flac` (development audio files)
     - `ASVspoof2019_LA_cm_protocols/` (protocol text files, e.g. `ASVspoof2019.LA.cm.train.trn.txt` and `ASVspoof2019.LA.cm.dev.trl.txt`)
   - Place or extract the downloaded archive somewhere accessible on your machine (we used `D:\Danush\detection\archive (1)` in examples).

2. Create `dataset/asvspoof_full` using the provided script
   - From the project root (PowerShell), run the `make_asvspoof_sample.py` helper to copy all protocol-matched files into `dataset/asvspoof_full`:

```powershell
$ASV_ROOT = "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_train"
.venv\Scripts\python.exe scripts\make_asvspoof_sample.py "$ASV_ROOT\flac" "$ASV_ROOT\..\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt" dataset\asvspoof_full --n-per-class 999999 --seed 123
```

Notes:
 - The script will copy the available protocol-matched files per class (it will not exceed what's available in the archive). If you want a balanced N-per-class sample, set `--n-per-class` to the number you need.
 - Replace `$ASV_ROOT` with the path where you extracted the ASVspoof data.

3. Train the models on the full dataset
   - After `dataset/asvspoof_full` is created, train both models and save the checkpoints:

```powershell
.venv\Scripts\python.exe -m detection.cli train-compare --dataset dataset\asvspoof_full --out-base base_model_full.ckpt --out-bio bio_model_full.ckpt
```

4. Storage & sharing recommendations
   - Do NOT add large audio folders to the Git repository directly. Instead consider one of the following:
     - Use Git LFS to track audio files (`git lfs track "*.flac"`) and push the data via LFS (check your remote's LFS quota first).
     - Upload a compressed archive (zip/tar) to cloud storage (S3, Google Drive, institutional server) and add a small download manifest to this repo.
     - Keep the dataset outside the repo and document the local archive path in your environment or CI scripts.

5. Quick verification
   - Example dev evaluation (balanced sampling) comparing the two saved models and declaring a winner:

```powershell
.venv\Scripts\python.exe scripts\predict_external.py "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_dev\flac" 200 base_model_full.ckpt bio_model_full.ckpt "D:\Danush\detection\archive (1)\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt" --balanced
```

If you want, I can add a small helper script to download/upload archives to a release or cloud provider — tell me which hosting you prefer and I can prepare it.
