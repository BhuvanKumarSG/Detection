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

Notes on git
---------
I will create a separate branch for this README change and commit it there so the previous code on `main` remains unchanged. If you don't have push access to the remote repository, push will fail with permission denied (HTTP 403). In that case you can either:
- Add the account you're using as a collaborator on the remote repository, or
- Push to your fork and open a PR against the original repository.


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
