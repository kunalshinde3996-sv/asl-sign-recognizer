"""Shared paths, data splits and evaluation helpers for the Phase 2 scripts.

Image pools (all from data/raw/asl_alphabet_train, 3000 images per class):

    landmark pool  the 1000/class images sampled by extract_landmarks.py
                   (seed 42). The MLP train/val/test split comes from these.
                   The CNN baseline samples its 300/class from here too.
    held-out pool  the remaining ~2000/class. Neither model ever trains on
                   these. compare.py and lighting_test.py evaluate on them,
                   end to end, starting from raw pixels.

The landmark pool is rebuilt by replaying the extraction sampling, so it is only
correct if landmarks.csv was produced with the default --max-per-class 1000 and
--seed 42.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info spam

import json
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from utils import DATASET_PAD_RATIO, NO_HAND_LABEL, PROJECT_ROOT, detect_hand_features

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "asl_alphabet_train"
LANDMARKS_CSV = PROJECT_ROOT / "data" / "processed" / "landmarks.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLP_MODEL_PATH = MODELS_DIR / "asl_landmark_model.keras"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
CNN_MODEL_PATH = MODELS_DIR / "baseline_cnn.keras"

SEED = 42
EXTRACTION_MAX_PER_CLASS = 1000  # must match the extract_landmarks.py run
CNN_IMG_SIZE = 64

# Chart colors (validated categorical palette, slots 1 and 2) and ink.
MLP_COLOR = "#2a78d6"
CNN_COLOR = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"


# ---------------------------------------------------------------- data splits
def split_80_10_10(
    x: np.ndarray, y: np.ndarray, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 80/10/10 split. Deterministic for a fixed seed and input order."""
    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size=0.2, stratify=y, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return x_train, x_val, x_test, y_train, y_val, y_test


def load_landmarks() -> tuple[np.ndarray, np.ndarray]:
    """Load landmarks.csv as (features float32 [N, 63], string labels [N]).

    "nothing" rows are dropped if present: that class is decided by "no hand
    detected", never by the classifier.
    """
    df = pd.read_csv(LANDMARKS_CSV)
    df = df[df["label"] != NO_HAND_LABEL]
    return df.drop(columns="label").to_numpy(np.float32), df["label"].to_numpy()


# ---------------------------------------------------------------- image pools
def landmark_pool() -> dict[str, set[Path]]:
    """Replay the extract_landmarks.py sampling to recover which images it used."""
    from extract_landmarks import list_images  # local import: only needed here

    rng = random.Random(SEED)
    class_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    return {d.name: set(list_images(d, EXTRACTION_MAX_PER_CLASS, rng)) for d in class_dirs}


def held_out_images(classes: list[str], per_class: int, seed: int = SEED) -> list[tuple[Path, str]]:
    """Sample images that were NOT used for landmark extraction (unseen by both models)."""
    used = landmark_pool()
    rng = random.Random(seed)
    samples: list[tuple[Path, str]] = []
    for label in classes:
        candidates = sorted(p for p in (RAW_DIR / label).iterdir() if p.suffix.lower() == ".jpg" and p not in used[label])
        samples += [(p, label) for p in rng.sample(candidates, min(per_class, len(candidates)))]
    return samples


def read_rgb(path: Path) -> np.ndarray:
    """Read an image file as an RGB uint8 array."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise OSError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- predictions
def cnn_input(rgb: np.ndarray) -> np.ndarray:
    """Resize an RGB image to the CNN input size. Scaling to [0, 1] happens inside the model."""
    return cv2.resize(rgb, (CNN_IMG_SIZE, CNN_IMG_SIZE), interpolation=cv2.INTER_AREA)


def predict_cnn(model, rgb: np.ndarray) -> int:
    """Class index predicted by the CNN for one RGB image."""
    probs = model(cnn_input(rgb)[None].astype(np.float32), training=False).numpy()[0]
    return int(np.argmax(probs))


def detect_features(landmarker, rgb: np.ndarray, pad_ratio: float = DATASET_PAD_RATIO) -> np.ndarray | None:
    """MediaPipe on one Kaggle-style RGB image; same padding as extraction by default."""
    return detect_hand_features(landmarker, rgb, pad_ratio)


def predict_mlp_pipeline(landmarker, model, rgb: np.ndarray, pad_ratio: float = DATASET_PAD_RATIO) -> int | None:
    """Full landmark pipeline on one RGB image: MediaPipe, then normalization, then MLP.

    Returns None when no hand is detected. The live demo would show "nothing" in
    that case, which is wrong for every one of the 28 hand classes.
    """
    features = detect_features(landmarker, rgb, pad_ratio)
    if features is None:
        return None
    return int(np.argmax(model(features[None], training=False).numpy()[0]))


# ---------------------------------------------------------------- reporting
def save_classification_outputs(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], tag: str, title: str
) -> dict:
    """Write the classification report, confusion matrix heatmap and metrics JSON for one model.

    Files: reports/classification_report_<tag>.txt, reports/confusion_matrix[_<tag>].png
    (the MLP's matrix is saved as plain confusion_matrix.png), reports/metrics_<tag>.json.
    Returns the metrics dict.
    """
    REPORTS_DIR.mkdir(exist_ok=True)
    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=3, zero_division=0)
    (REPORTS_DIR / f"classification_report_{tag}.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    annot = np.where(cm > 0, cm.astype(str), "")  # blank out zeros so errors stand out
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar_kws={"label": "Test samples"},
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 7}, ax=ax)
    ax.set_title(title, color=INK, fontsize=13)
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("True", color=MUTED)
    fig.tight_layout()
    cm_name = "confusion_matrix.png" if tag == "mlp" else f"confusion_matrix_{tag}.png"
    fig.savefig(REPORTS_DIR / cm_name, dpi=150)
    plt.close(fig)

    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "test_samples": int(len(y_true)),
    }
    (REPORTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def plot_training_curves(history: dict, path: Path, title: str) -> None:
    """Two panels: accuracy and loss, train vs validation."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, label in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        ax.plot(epochs, history[key], color=MLP_COLOR, linewidth=2, label="Train")
        ax.plot(epochs, history[f"val_{key}"], color=CNN_COLOR, linewidth=2, label="Validation")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.set_title(label, color=INK)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
