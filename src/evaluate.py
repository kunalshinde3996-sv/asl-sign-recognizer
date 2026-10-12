"""Evaluate the landmark MLP on its held-out test split.

Outputs:
    reports/classification_report_mlp.txt, reports/confusion_matrix.png,
    reports/metrics_mlp.json, reports/top_confusions_mlp.txt

Note: these numbers only cover images where MediaPipe found a hand (that is
all landmarks.csv contains). compare.py measures end-to-end accuracy from raw
images, where detection failures count as errors.

Usage:
    python src/evaluate.py
"""

from __future__ import annotations

import common  # noqa: F401  (sets TF log level before TensorFlow is imported)

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from common import ENCODER_PATH, MLP_MODEL_PATH, REPORTS_DIR, load_landmarks, save_classification_outputs, split_80_10_10

# Why pairs of static ASL letters get confused, keyed by unordered pair.
CONFUSION_REASONS: dict[frozenset[str], str] = {
    frozenset("MN"): "Both are fists with the thumb under the fingers (3 vs 2 fingers over it); the thumb is mostly hidden, so landmarks are noisy.",
    frozenset("MS"): "Closed fists; they differ only in where the thumb sits, which is partly hidden.",
    frozenset("NS"): "Closed fists; they differ only in where the thumb sits, which is partly hidden.",
    frozenset("MT"): "Fists with the thumb tucked between fingers; small, hidden thumb differences.",
    frozenset("NT"): "Fists with the thumb tucked between fingers; small, hidden thumb differences.",
    frozenset("ST"): "Both are fists; T tucks the thumb under the index finger, S wraps it in front.",
    frozenset("AS"): "Both are closed fists; A keeps the thumb at the side, S across the front. Small thumb change.",
    frozenset("AE"): "Closed hand shapes; E curls the fingertips down over the thumb.",
    frozenset("ES"): "Closed hand shapes; the difference is fingertip curl and thumb placement.",
    frozenset("AT"): "Closed fists; the thumb position (side vs tucked) is the only cue.",
    frozenset("UV"): "Same two raised fingers; U keeps them together, V spreads them. Only the spread angle differs.",
    frozenset("RU"): "Two raised fingers; R crosses them. The crossing is small in 2D landmarks.",
    frozenset("RV"): "Two raised fingers; crossed (R) vs spread (V).",
    frozenset("KV"): "Two raised fingers in a V; K adds the thumb between them.",
    frozenset("KP"): "Same hand shape; P is K pointed downward. Orientation is the only cue.",
    frozenset("GH"): "Sideways hands; G extends one finger, H two.",
    frozenset("GQ"): "Same shape; Q is G pointed downward.",
    frozenset("HU"): "Same two fingers together; H is sideways, U points up.",
    frozenset("IJ"): "J is I plus a motion. A single static frame of J looks like I.",
    frozenset("DZ"): "Z is the index finger tracing a Z. A static frame looks like D (index up).",
    frozenset("DX"): "Index finger raised (D) vs hooked (X); the hook is a small change in z.",
    frozenset("CO"): "Curved hand; O closes the fingertips onto the thumb, C leaves a gap.",
    frozenset("FW"): "Three raised fingers; F touches thumb to index, W touches thumb to pinky.",
    frozenset("BW"): "Raised fingers; B keeps four together, W spreads three.",
    frozenset("HJ"): "J ends with the hand turned sideways and the pinky out; a static frame can look like the sideways H.",
    frozenset(("C", "space")): "In this dataset 'space' is an open, curved, sideways hand; with the fingers bent it looks like C.",
    frozenset(("B", "space")): "In this dataset 'space' is an open, sideways hand; a flat B differs mainly in orientation.",
    frozenset(("del", "space")): "Dataset-specific gestures that are both open hands; they differ mainly in orientation.",
}
DEFAULT_REASON = "Similar static hand shapes; the landmark differences are small. Check per-class recall."


def top_confused_pairs(cm: np.ndarray, class_names: list[str], k: int = 5) -> list[tuple[str, str, int, int]]:
    """Return the k unordered class pairs with the most mutual errors.

    Each item is (class_a, class_b, a_predicted_as_b, b_predicted_as_a).
    """
    pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(i + 1, n):
            if cm[i, j] + cm[j, i] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j]), int(cm[j, i])))
    return sorted(pairs, key=lambda p: -(p[2] + p[3]))[:k]


def main() -> None:
    x, labels = load_landmarks()
    encoder = joblib.load(ENCODER_PATH)
    y = encoder.transform(labels)
    _, _, x_test, _, _, y_test = split_80_10_10(x, y)

    model = tf.keras.models.load_model(MLP_MODEL_PATH)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    class_names = list(encoder.classes_)

    metrics = save_classification_outputs(y_test, y_pred, class_names, "mlp", "Landmark MLP: confusion matrix (test split)")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f} | macro F1: {metrics['macro_f1']:.4f} | n={metrics['test_samples']}")
    print((REPORTS_DIR / "classification_report_mlp.txt").read_text())

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    lines = ["Top 5 most-confused class pairs (test split):", ""]
    pairs = top_confused_pairs(cm, class_names)
    if not pairs:
        lines.append("No misclassifications on the test split.")
    for rank, (a, b, ab, ba) in enumerate(pairs, 1):
        reason = CONFUSION_REASONS.get(frozenset((a, b)), DEFAULT_REASON)
        lines.append(f"{rank}. {a} <-> {b}: {ab + ba} errors ({a}->{b}: {ab}, {b}->{a}: {ba})")
        lines.append(f"   Likely reason: {reason}")
    text = "\n".join(lines)
    (REPORTS_DIR / "top_confusions_mlp.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
