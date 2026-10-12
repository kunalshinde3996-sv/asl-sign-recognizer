"""Baseline: a small CNN trained directly on raw pixels, for comparison with the landmark MLP.

Setup, chosen to make the comparison fair:
  * Same 28 classes and the same label encoder as the MLP ("nothing" excluded).
  * 300 images/class, sampled from the same landmark pool the MLP trained on,
    so the held-out pool in compare.py/lighting_test.py stays unseen by both.
  * 64x64 RGB, stratified 80/10/10 split, same seed.
  * No photometric augmentation. It's a plain baseline; lighting_test.py shows
    what that costs.

Outputs:
    models/baseline_cnn.keras
    reports/classification_report_cnn.txt, reports/confusion_matrix_cnn.png,
    reports/metrics_cnn.json, reports/training_curves_cnn.png

Usage:
    python src/baseline_cnn.py
"""

from __future__ import annotations

import common  # noqa: F401  (sets TF log level before TensorFlow is imported)

import argparse
import random

import joblib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from common import (
    CNN_IMG_SIZE, CNN_MODEL_PATH, ENCODER_PATH, REPORTS_DIR, SEED,
    cnn_input, landmark_pool, plot_training_curves, read_rgb, save_classification_outputs, split_80_10_10,
)


def build_cnn(num_classes: int) -> tf.keras.Model:
    """Three conv blocks (32, 64, 128 filters), global average pooling, softmax."""
    inputs = tf.keras.Input(shape=(CNN_IMG_SIZE, CNN_IMG_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="baseline_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_images(class_names: list[str], per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample per_class images per class from the landmark pool and resize them to 64x64."""
    pool = landmark_pool()
    rng = random.Random(SEED)
    paths: list[tuple] = []
    for idx, label in enumerate(class_names):
        chosen = rng.sample(sorted(pool[label]), min(per_class, len(pool[label])))
        paths += [(p, idx) for p in chosen]
    x = np.stack([cnn_input(read_rgb(p)) for p, _ in tqdm(paths, desc="Loading images", unit="img")])
    y = np.array([idx for _, idx in paths])
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline CNN on raw images.")
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    encoder = joblib.load(ENCODER_PATH)  # shared with the MLP so class indices match
    class_names = list(encoder.classes_)

    x, y = load_images(class_names, args.per_class)
    x_train, x_val, x_test, y_train, y_val, y_test = split_80_10_10(x, y)
    print(f"Classes: {len(class_names)} | train {len(x_train)} | val {len(x_val)} | test {len(x_test)}")

    model = build_cnn(len(class_names))
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(CNN_MODEL_PATH, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
                        batch_size=64, callbacks=callbacks, verbose=2)
    plot_training_curves(history.history, REPORTS_DIR / "training_curves_cnn.png", "Baseline CNN training")

    best = tf.keras.models.load_model(CNN_MODEL_PATH)
    y_pred = np.argmax(best.predict(x_test, verbose=0), axis=1)
    metrics = save_classification_outputs(y_test, y_pred, class_names, "cnn", "Baseline CNN: confusion matrix (test split)")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f} | macro F1: {metrics['macro_f1']:.4f} | n={metrics['test_samples']}")
    print((REPORTS_DIR / "classification_report_cnn.txt").read_text())


if __name__ == "__main__":
    main()
