"""Simple PyTorch CNN trainer for mel-spectrograms with optional GPU support.

This module is optional — it requires torch and will raise a helpful error if not installed.
"""
from typing import List, Tuple
import os
import math
import joblib
import librosa
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    torch = None


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # use adaptive pooling so the flattened feature size is fixed regardless of input time-frames
        self.adapt_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(64 * 8 * 8, 128)
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, C=1, H, W)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        # adaptive pool to fixed 8x8 feature maps
        x = self.adapt_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        x = self.out(x)
        return x


def _collect_files(dataset_dir: str) -> List[Tuple[str, int]]:
    files = []
    for label in ("real", "fake"):
        lab_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(lab_dir):
            continue
        for fname in os.listdir(lab_dir):
            if fname.lower().endswith((".wav", ".flac", ".mp3")):
                files.append((os.path.join(lab_dir, fname), 0 if label == "real" else 1))
    return files


class MelDataset(Dataset):
    def __init__(self, file_label_list: List[Tuple[str, int]], sr=22050, n_mels=64, duration=2.0):
        self.items = file_label_list
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.frames = int(math.ceil((self.duration * sr) / 512))

    def __len__(self):
        return len(self.items)

    def _load_mel(self, path):
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        # pad/trim
        target_len = int(self.duration * self.sr)
        if y.shape[0] < target_len:
            y = np.pad(y, (0, target_len - y.shape[0]))
        else:
            y = y[:target_len]
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=512)
        S_db = librosa.power_to_db(S, ref=np.max)
        # normalize
        S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
        return S_db.astype(np.float32)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        mel = self._load_mel(path)
        # shape (n_mels, t) -> add channel
        mel = np.expand_dims(mel, axis=0)
        return mel, label, path


def train_deep(dataset_dir: str, out_model_path: str, epochs=10, batch_size=8, lr=1e-3, device: str = "cuda"):
    if torch is None:
        raise RuntimeError("PyTorch not installed. Install torch to use deep training: pip install torch")

    files = _collect_files(dataset_dir)
    if len(files) == 0:
        raise RuntimeError("No audio files found in dataset")

    # simple split
    np.random.shuffle(files)
    split = int(0.8 * len(files))
    train_files = files[:split]
    test_files = files[split:]

    train_ds = MelDataset(train_files)
    test_ds = MelDataset(test_files)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = SimpleCNN(in_channels=1, n_classes=2).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0
        for X, y, _ in train_loader:
            # X is a batch of numpy arrays: stack into ndarray then convert
            X = torch.from_numpy(np.stack(X)).float().to(dev)
            y = torch.from_numpy(np.array(y)).long().to(dev)
            outputs = model(X)
            loss = criterion(outputs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            preds = outputs.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} train_acc={acc:.3f}")

    # quick eval
    model.eval()
    total = 0
    correct = 0
    all_probs = []
    with torch.no_grad():
        for X, y, _ in test_loader:
            X = torch.from_numpy(np.stack(X)).float().to(dev)
            y = torch.from_numpy(np.array(y)).long().to(dev)
            outputs = model(X)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            all_probs.extend(probs.tolist())
    acc = correct / total if total > 0 else 0
    print(f"Test accuracy={acc:.3f}")

    # save checkpoint
    chk = {"model_state": model.state_dict(), "args": {"n_mels": train_ds.n_mels}}
    torch.save(chk, out_model_path)
    print(f"Saved deep model to {out_model_path}")
    return out_model_path


def load_deep(path: str, device: str = "cpu"):
    if torch is None:
        raise RuntimeError("PyTorch not installed. Install torch to use deep models: pip install torch")
    chk = torch.load(path, map_location=device)
    model = SimpleCNN(in_channels=1, n_classes=2)
    model.load_state_dict(chk["model_state"])
    model.to(device)
    model.eval()
    return model


def predict_deep(model_or_path, audio_path: str, device: str = "cpu"):
    if torch is None:
        raise RuntimeError("PyTorch not installed. Install torch to use deep models: pip install torch")
    if isinstance(model_or_path, str):
        model = load_deep(model_or_path, device=device)
    else:
        model = model_or_path
    mel = MelDataset([(audio_path, 0)])._load_mel(audio_path)
    X = torch.from_numpy(np.expand_dims(np.expand_dims(mel, 0), 0)).float().to(device)
    with torch.no_grad():
        out = model(X)
        prob = F.softmax(out, dim=1)[:, 1].item()
    return prob
