"""Robustness to lighting: accuracy of both models on darkened, brightened and low-contrast images.

Uses held-out raw images that neither model trained on. For the landmark MLP
the whole pipeline runs on the modified image (MediaPipe + MLP), and a missed
detection counts as an error. The detection rate is reported separately so you
can tell detection failures apart from classification errors.

Outputs:
    reports/lighting_results.csv
    reports/lighting_robustness.png
    reports/lighting_examples.png   (what each condition looks like)

Usage:
    python src/lighting_test.py --per-class 40
"""

from __future__ import annotations

import common  # noqa: F401  (sets TF log level before TensorFlow is imported)

import argparse
from collections.abc import Callable

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from common import (
    CNN_COLOR, CNN_MODEL_PATH, ENCODER_PATH, INK, MLP_COLOR, MLP_MODEL_PATH, MUTED, REPORTS_DIR,
    cnn_input, held_out_images, predict_mlp_pipeline, read_rgb,
)
from utils import create_hand_landmarker


def adjust(alpha: float, beta: float) -> Callable[[np.ndarray], np.ndarray]:
    """Pixel transform out = clip(alpha * img + beta)."""
    return lambda img: cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# alpha scales contrast and brightness, beta shifts brightness.
CONDITIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "original": lambda img: img,
    "dark": adjust(0.35, 0),            # ~1/3 of the light, like a dim room
    "bright": adjust(1.0, 100),         # washed out; highlights clip to white
    "low_contrast": adjust(0.35, 85),   # squashed toward mid-grey, like haze or a cheap webcam
}


def evaluate_condition(transform, images, y_true, landmarker, mlp, cnn) -> dict:
    """Accuracy of both models on one transformed copy of the image set."""
    modified = [transform(im) for im in images]
    mlp_pred = np.array([
        -1 if (p := predict_mlp_pipeline(landmarker, mlp, im)) is None else p
        for im in tqdm(modified, desc="  MLP pipeline", unit="img", leave=False)
    ])
    cnn_pred = np.argmax(cnn.predict(np.stack([cnn_input(im) for im in modified]), verbose=0), axis=1)
    return {
        "mlp_accuracy": float(np.mean(mlp_pred == y_true)),
        "mlp_detection_rate": float(np.mean(mlp_pred >= 0)),
        "cnn_accuracy": float(np.mean(cnn_pred == y_true)),
    }


def plot_results(df: pd.DataFrame) -> None:
    """Grouped bars: MLP vs CNN accuracy per lighting condition."""
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, col, color, label in ((-width / 2, "mlp_accuracy", MLP_COLOR, "Landmark MLP (MediaPipe + MLP)"),
                                      (width / 2, "cnn_accuracy", CNN_COLOR, "Baseline CNN")):
        bars = ax.bar(x + offset, df[col] * 100, width - 0.02, color=color, label=label)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, color=INK, fontsize=9)
    ax.set_xticks(x, [c.replace("_", " ") for c in df["condition"]])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy on held-out images (%)", color=MUTED)
    ax.set_title("Lighting robustness (missed hand detection counts as an error)", color=INK)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "lighting_robustness.png", dpi=150)
    plt.close(fig)


def plot_examples(image: np.ndarray) -> None:
    """One sample image under each condition, side by side."""
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(3 * len(CONDITIONS), 3.4))
    for ax, (name, transform) in zip(axes, CONDITIONS.items()):
        ax.imshow(transform(image))
        ax.set_title(name.replace("_", " "), color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "lighting_examples.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lighting robustness test for both models.")
    parser.add_argument("--per-class", type=int, default=40, help="Held-out images per class.")
    args = parser.parse_args()

    encoder = joblib.load(ENCODER_PATH)
    class_names = list(encoder.classes_)
    mlp = tf.keras.models.load_model(MLP_MODEL_PATH)
    cnn = tf.keras.models.load_model(CNN_MODEL_PATH)

    samples = held_out_images(class_names, args.per_class, seed=7)  # different seed from compare.py
    images = [read_rgb(p) for p, _ in tqdm(samples, desc="Loading held-out images", unit="img")]
    y_true = encoder.transform([label for _, label in samples])
    plot_examples(images[0])

    rows = []
    with create_hand_landmarker() as landmarker:
        for name, transform in CONDITIONS.items():
            print(f"Condition: {name}")
            rows.append({"condition": name, **evaluate_condition(transform, images, y_true, landmarker, mlp, cnn)})

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "lighting_results.csv", index=False)
    plot_results(df)

    print(f"\nHeld-out images: {len(samples)} ({args.per_class}/class)")
    print(df.to_string(index=False, formatters={c: "{:.1%}".format for c in df.columns if c != "condition"}))
    print(f"Saved {REPORTS_DIR / 'lighting_results.csv'} and lighting_robustness.png")


if __name__ == "__main__":
    main()
