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

4. Predict a file:

   python -m detection.cli predict --model model.joblib --file path/to/audio.wav

5. Run tests

   # Run the project's unit tests with the venv Python (PowerShell)
   .venv\Scripts\python.exe -m pytest -q

   # Or, if you've activated the venv in PowerShell already:
   pytest -q

Notes & next steps
- This is a feature-based baseline (MFCCs, spectral features) with a RandomForest classifier. For production or research-grade detection, consider using specialized anti-spoofing networks and large curated datasets (ASVspoof, etc.).
- Add data augmentation, balanced sampling, cross-validation, and stronger models (XGBoost, CNNs on spectrograms) for better performance.
