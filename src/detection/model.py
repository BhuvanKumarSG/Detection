import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from .features import extract_features


def _ensure_ckpt(path: str) -> str:
    """Ensure the model filename uses a .ckpt extension.

    If the provided path has a different extension (for example .joblib), replace it
    with .ckpt. If it has no extension, append .ckpt.
    """
    base, ext = os.path.splitext(path)
    if ext.lower() == ".ckpt":
        return path
    # If there is an extension (like .joblib), replace it; otherwise append .ckpt
    return base + ".ckpt"


def _iter_files_in_dir(dataset_dir):
    # Expect dataset_dir/{real,fake}/*.wav
    for label in ("real", "fake"):
        lab_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(lab_dir):
            continue
        for fname in os.listdir(lab_dir):
            if fname.lower().endswith((".wav", ".flac", ".mp3")):
                yield os.path.join(lab_dir, fname), (0 if label == "real" else 1)


def train_model(dataset_dir, out_model_path, feature_set="base", test_size=0.2, random_state=42):
    X = []
    y = []
    for path, label in _iter_files_in_dir(dataset_dir):
        try:
            feat = extract_features(path, feature_set=feature_set)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    X = np.vstack(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["real", "fake"]))

    # Normalize to .ckpt extension and overwrite any existing model file
    out_model_path = _ensure_ckpt(out_model_path)
    try:
        if os.path.exists(out_model_path):
            os.remove(out_model_path)
    except Exception:
        pass
    joblib.dump(clf, out_model_path)
    print(f"Saved model to {out_model_path}")
    return out_model_path


def train_compare(dataset_dir, out_model_base, out_model_bio, test_size=0.2, random_state=42):
    """Train baseline (base features) and bio (bio features) models and compare metrics on the same holdout set.

    Saves both models to the provided paths and prints side-by-side classification reports and AUC.
    """
    # Collect features, labels and paths once, for reproducibility
    X_base = []
    X_bio = []
    y = []
    paths = []
    for path, label in _iter_files_in_dir(dataset_dir):
        try:
            fbase = extract_features(path, feature_set="base")
            fbio = extract_features(path, feature_set="bio")
            X_base.append(fbase)
            X_bio.append(fbio)
            y.append(label)
            paths.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if len(y) == 0:
        raise RuntimeError("No audio files found in dataset directory")

    X_base = np.vstack(X_base)
    X_bio = np.vstack(X_bio)
    y = np.array(y)
    paths = np.array(paths)

    # split indices so we can map back to file paths for per-sample reporting
    indices = np.arange(len(y))
    train_idx, test_idx, y_train, y_test = train_test_split(indices, y, test_size=test_size, random_state=random_state, stratify=y)

    Xb_train = X_base[train_idx]
    Xb_test = X_base[test_idx]
    Xbio_train = X_bio[train_idx]
    Xbio_test = X_bio[test_idx]

    clf_base = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    clf_bio = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)

    clf_base.fit(Xb_train, y_train)
    clf_bio.fit(Xbio_train, y_train)

    yb_pred = clf_base.predict(Xb_test)
    ybio_pred = clf_bio.predict(Xbio_test)

    # Print classification reports
    print("Baseline model (base features):")
    print(classification_report(y_test, yb_pred, target_names=["real", "fake"]))
    print("Bio model (bio features):")
    print(classification_report(y_test, ybio_pred, target_names=["real", "fake"]))

    # Per-sample confidences and final classification
    print("Per-sample confidences and final classification:")
    try:
        yb_proba = clf_base.predict_proba(Xb_test)[:, 1]
        ybio_proba = clf_bio.predict_proba(Xbio_test)[:, 1]
    except Exception:
        # if predict_proba not supported, fallback to decision_function or predictions
        yb_proba = clf_base.predict(Xb_test)
        ybio_proba = clf_bio.predict(Xbio_test)

    # iterate over test samples in original order
    for i, idx in enumerate(test_idx):
        fp = paths[idx]
        true = int(y_test[i])
        base_conf = float(yb_proba[i])
        bio_conf = float(ybio_proba[i])
        base_pred = int(yb_pred[i])
        bio_pred = int(ybio_pred[i])

        # Determine final classification: if both agree, use that; otherwise pick the model with
        # higher confidence for its predicted class (use the class's probability)
        final = None
        if base_pred == bio_pred:
            final = base_pred
            reason = "agreement"
        else:
            # compare confidence in their own predicted class
            # for our binary case, probs represent P(fake). For base_pred==1 use base_conf else use 1-base_conf
            base_score = base_conf if base_pred == 1 else 1.0 - base_conf
            bio_score = bio_conf if bio_pred == 1 else 1.0 - bio_conf
            if base_score >= bio_score:
                final = base_pred
                reason = "base_higher_conf"
            else:
                final = bio_pred
                reason = "bio_higher_conf"

        print(f"{os.path.basename(fp)} | true={ 'fake' if true==1 else 'real' } | base_conf={base_conf:.3f} base_pred={'fake' if base_pred==1 else 'real'} | bio_conf={bio_conf:.3f} bio_pred={'fake' if bio_pred==1 else 'real'} | final={'fake' if final==1 else 'real'} ({reason})")

    # Compute aggregate metrics to decide winner
    acc_base = accuracy_score(y_test, yb_pred)
    acc_bio = accuracy_score(y_test, ybio_pred)
    auc_base = auc_bio = None
    try:
        auc_base = roc_auc_score(y_test, yb_proba)
        auc_bio = roc_auc_score(y_test, ybio_proba)
    except Exception:
        pass

    print(f"Aggregate: accuracy base={acc_base:.4f}, bio={acc_bio:.4f}")
    if auc_base is not None and auc_bio is not None:
        print(f"Aggregate: AUC base={auc_base:.4f}, bio={auc_bio:.4f}")

    # Decide winner: higher accuracy, tie-breaker higher AUC, else tie
    winner = None
    if acc_base > acc_bio:
        winner = 'baseline'
    elif acc_bio > acc_base:
        winner = 'bio'
    else:
        if auc_base is not None and auc_bio is not None:
            if auc_base > auc_bio:
                winner = 'baseline'
            elif auc_bio > auc_base:
                winner = 'bio'
            else:
                winner = 'tie'
        else:
            winner = 'tie'

    if winner == 'tie':
        print('Result: tie between baseline and bio models')
    else:
        print(f"Winner: {winner} model")

    # Overwrite existing files if present
    # Ensure .ckpt extension for both output paths and overwrite if present
    out_model_base = _ensure_ckpt(out_model_base)
    out_model_bio = _ensure_ckpt(out_model_bio)
    try:
        if os.path.exists(out_model_base):
            os.remove(out_model_base)
    except Exception:
        pass
    try:
        if os.path.exists(out_model_bio):
            os.remove(out_model_bio)
    except Exception:
        pass
    joblib.dump(clf_base, out_model_base)
    joblib.dump(clf_bio, out_model_bio)
    print(f"Saved baseline -> {out_model_base}")
    print(f"Saved bio -> {out_model_bio}")
    return out_model_base, out_model_bio


def load_model(path):
    return joblib.load(path)


def predict_file(model, audio_path):
    if isinstance(model, str):
        model = load_model(model)
    feat = extract_features(audio_path)
    return int(model.predict([feat])[0])
