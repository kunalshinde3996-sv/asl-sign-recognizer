"""Train the landmark MLP classifier (28 classes: A-Z, del, space).

"nothing" is not a classifier output. The live demo predicts it whenever
MediaPipe finds no hand (see extract_landmarks.py).

Outputs:
    models/asl_landmark_model.keras   best model by validation loss
    models/label_encoder.joblib       maps class index <-> label
    reports/training_curves.png, reports/training_history.csv

Usage:
    python src/train.py
"""

from __future__ import annotations

import common  # noqa: F401  (sets TF log level before TensorFlow is imported)

import argparse

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from common import ENCODER_PATH, MLP_MODEL_PATH, MODELS_DIR, REPORTS_DIR, SEED, load_landmarks, plot_training_curves, split_80_10_10
from utils import NUM_FEATURES


def build_model(num_classes: int) -> tf.keras.Model:
    """Small MLP: 63 -> 256 -> 128 -> 64 -> softmax.

    BatchNorm stabilizes training; Dropout fights overfitting. Kaggle frames
    within a class are very similar, so the model can memorize them easily.
    """
    inputs = tf.keras.Input(shape=(NUM_FEATURES,), name="landmarks")
    x = inputs
    for units, drop in ((256, 0.3), (128, 0.3), (64, 0.2)):
        x = tf.keras.layers.Dense(units, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(drop)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="asl_landmark_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the landmark MLP.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    x, labels = load_landmarks()
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    x_train, x_val, x_test, y_train, y_val, y_test = split_80_10_10(x, y)
    print(f"Classes: {len(encoder.classes_)} | train {len(x_train)} | val {len(x_val)} | test {len(x_test)}")

    # "balanced" weight = n_samples / (n_classes * class_count). Classes with
    # fewer detected hands (M, N, space) count more per sample, so the loss
    # doesn't simply favor the larger classes.
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(weights))
    top = sorted(zip(encoder.classes_, weights), key=lambda t: -t[1])[:5]
    print("Highest class weights:", ", ".join(f"{c}={w:.2f}" for c, w in top))

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    model = build_model(len(encoder.classes_))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MLP_MODEL_PATH, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    joblib.dump(encoder, ENCODER_PATH)
    pd.DataFrame(history.history).to_csv(REPORTS_DIR / "training_history.csv", index_label="epoch")
    plot_training_curves(history.history, REPORTS_DIR / "training_curves.png", "Landmark MLP training")

    best = tf.keras.models.load_model(MLP_MODEL_PATH)
    _, val_acc = best.evaluate(x_val, y_val, verbose=0)
    print(f"Best model val accuracy: {val_acc:.4f}  (test accuracy: run evaluate.py)")
    print(f"Saved {MLP_MODEL_PATH} and {ENCODER_PATH}")


if __name__ == "__main__":
    main()
