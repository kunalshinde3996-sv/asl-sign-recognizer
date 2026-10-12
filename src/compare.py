"""Compare the landmark MLP with the baseline CNN: accuracy, model size, inference time.

Two kinds of accuracy are reported, because they answer different questions:
  * Test-split accuracy (from evaluate.py / baseline_cnn.py). For the MLP this
    only covers images where MediaPipe found a hand.
  * End-to-end accuracy on held-out raw images that neither model saw. For the
    MLP a missed detection counts as an error. This is what the live demo
    actually experiences.

Output: reports/comparison.md

Usage:
    python src/compare.py
"""

from __future__ import annotations

import common  # noqa: F401  (sets TF log level before TensorFlow is imported)

import argparse
import json
import time

import joblib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from common import (
    CNN_MODEL_PATH, ENCODER_PATH, MLP_MODEL_PATH, REPORTS_DIR,
    cnn_input, detect_features, held_out_images, load_landmarks, predict_cnn, predict_mlp_pipeline, read_rgb,
)
from utils import HAND_MODEL_PATH, create_hand_landmarker


def median_ms(fn, inputs: list, warmup: int = 10) -> float:
    """Median wall-clock latency of fn(x) in milliseconds, one input at a time (like a live frame)."""
    for x in inputs[:warmup]:
        fn(x)
    times = []
    for x in inputs:
        start = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def size_mb(*paths) -> float:
    return sum(p.stat().st_size for p in paths) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare landmark MLP vs baseline CNN.")
    parser.add_argument("--per-class", type=int, default=50, help="Held-out images per class.")
    parser.add_argument("--timing-samples", type=int, default=200)
    args = parser.parse_args()

    encoder = joblib.load(ENCODER_PATH)
    class_names = list(encoder.classes_)
    mlp = tf.keras.models.load_model(MLP_MODEL_PATH)
    cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
    split_metrics = {tag: json.loads((REPORTS_DIR / f"metrics_{tag}.json").read_text()) for tag in ("mlp", "cnn")}

    samples = held_out_images(class_names, args.per_class)
    images = [read_rgb(p) for p, _ in tqdm(samples, desc="Loading held-out images", unit="img")]
    y_true = encoder.transform([label for _, label in samples])

    with create_hand_landmarker() as landmarker:
        mlp_pred = [predict_mlp_pipeline(landmarker, mlp, im) for im in tqdm(images, desc="MLP pipeline", unit="img")]
        detected = np.array([p is not None for p in mlp_pred])
        mlp_pred_arr = np.array([-1 if p is None else p for p in mlp_pred])
        cnn_pred = np.argmax(cnn.predict(np.stack([cnn_input(im) for im in images]), verbose=0), axis=1)

        mlp_e2e = float(np.mean(mlp_pred_arr == y_true))
        mlp_detected_only = float(np.mean(mlp_pred_arr[detected] == y_true[detected])) if detected.any() else 0.0
        cnn_e2e = float(np.mean(cnn_pred == y_true))

        # Timing: one image at a time, as in the live demo.
        timing_imgs = images[: args.timing_samples]
        features = [f for im in timing_imgs[:50] if (f := detect_features(landmarker, im)) is not None]
        t_mlp_pipeline = median_ms(lambda im: predict_mlp_pipeline(landmarker, mlp, im), timing_imgs)
        t_mlp_only = median_ms(lambda f: mlp(f[None], training=False), features) if features else float("nan")
        t_cnn = median_ms(lambda im: predict_cnn(cnn, im), timing_imgs)

    rows = [
        ("Test-split accuracy", f"{split_metrics['mlp']['test_accuracy']:.1%} *", f"{split_metrics['cnn']['test_accuracy']:.1%}"),
        ("Test-split macro F1", f"{split_metrics['mlp']['macro_f1']:.3f} *", f"{split_metrics['cnn']['macro_f1']:.3f}"),
        (f"End-to-end accuracy, held-out raw images (n={len(samples)})", f"{mlp_e2e:.1%}", f"{cnn_e2e:.1%}"),
        ("Accuracy when a hand is detected (held-out)", f"{mlp_detected_only:.1%}", "n/a"),
        ("Hand detection rate (held-out)", f"{detected.mean():.1%}", "n/a"),
        ("Trainable parameters", f"{mlp.count_params():,}", f"{cnn.count_params():,}"),
        ("Classifier file size", f"{size_mb(MLP_MODEL_PATH):.2f} MB", f"{size_mb(CNN_MODEL_PATH):.2f} MB"),
        ("Total pipeline size", f"{size_mb(MLP_MODEL_PATH, HAND_MODEL_PATH):.2f} MB (incl. MediaPipe)", f"{size_mb(CNN_MODEL_PATH):.2f} MB"),
        ("Inference, classifier only (median, CPU)", f"{t_mlp_only:.2f} ms", f"{t_cnn:.2f} ms"),
        ("Inference, full pipeline per image (median, CPU)", f"{t_mlp_pipeline:.2f} ms (MediaPipe + MLP)", f"{t_cnn:.2f} ms"),
    ]

    md = [
        "# Landmark MLP vs baseline CNN",
        "",
        "| Metric | Landmark MLP | Baseline CNN |",
        "|---|---|---|",
        *[f"| {m} | {a} | {b} |" for m, a, b in rows],
        "",
        "\\* MLP test-split numbers only include images where MediaPipe detected a hand. "
        "The end-to-end row counts missed detections as errors.",
        "",
        "## How to read this",
        "",
        "- **Both test splits are optimistic.** Kaggle ASL Alphabet images are consecutive video frames of one signer "
        "against one background, so test images are near-duplicates of training images. That holds for the "
        "held-out pool too. Real generalization (new people, backgrounds, cameras) has to be checked on the webcam.",
        "- **The landmark MLP never sees the background or the lighting.** It only sees 63 normalized coordinates. "
        "The CNN sees every pixel, so it can learn background or lighting shortcuts that don't transfer. "
        "See reports/lighting_robustness.png.",
        "- **The MLP's weak point is detection.** If MediaPipe misses the hand, the classifier never runs. "
        "The detection rate row is the ceiling on end-to-end accuracy.",
        f"- Data: MLP {len(load_landmarks()[1]):,} landmark rows (80/10/10 split); CNN 300 images per class (80/10/10 split).",
        "",
    ]
    out = REPORTS_DIR / "comparison.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
