"""Live ASL fingerspelling demo: webcam -> MediaPipe landmarks -> MLP -> letters -> text.

Per frame:
    1. MediaPipe HandLandmarker (VIDEO mode) on the camera frame. No hand -> "nothing".
    2. The same 63-feature normalization as training (utils.detect_hand).
    3. MLP prediction. Confidence below --threshold -> "?".
    4. Majority vote over the last --window frames (removes single-frame flicker).
    5. Word builder: a letter that stays the voted result for --hold seconds is
       typed once. "space" types a space, "del" deletes the last character. To
       type the same letter twice, change the sign (or drop your hand) in between.

The classifier sees the UNMIRRORED frame, like the Kaggle training photos. The
window shows a mirrored view only because that feels natural (--no-mirror to
turn it off). If you sign with the other hand than the dataset signer and get
poor predictions, try --flip-input.

Keys: q quit | c clear text | s screenshot to reports/

Usage:
    python src/live_demo.py
    python src/live_demo.py --camera 1 --min-detection-confidence 0.3
"""

from __future__ import annotations

import common  # noqa: F401  (sets TF log level before TensorFlow is imported)

import argparse
import sys
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import joblib
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python import vision

from common import ENCODER_PATH, MLP_MODEL_PATH, REPORTS_DIR
from utils import LIVE_MIN_SIDE, LIVE_PAD_RATIO, NO_HAND_LABEL, create_hand_landmarker, detect_hand

UNSURE_LABEL = "?"
NOT_TYPED = {UNSURE_LABEL, NO_HAND_LABEL}  # voted labels that never go into the text
HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
MAX_TEXT_CHARS = 40  # characters shown in the text bar (the full text is kept)

# BGR colors
GREEN = (80, 200, 60)
ORANGE = (40, 140, 240)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5,
                        help="MediaPipe hand detection/presence confidence (default 0.5).")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Classifier confidence below which the frame counts as '?' (default 0.7).")
    parser.add_argument("--window", type=int, default=10, help="Frames in the majority vote (default 10).")
    parser.add_argument("--hold", type=float, default=1.0,
                        help="Seconds a letter must stay stable before it is typed (default 1.0).")
    parser.add_argument("--no-mirror", action="store_true", help="Show the camera image unmirrored.")
    parser.add_argument("--flip-input", action="store_true",
                        help="Mirror the frame BEFORE detection (try this if you sign with the other hand).")
    return parser.parse_args()


def open_camera(index: int) -> cv2.VideoCapture:
    """Open the webcam and read one test frame, or exit with a readable message."""
    # DirectShow opens much faster than the default MSMF backend on most Windows webcams.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)  # fall back to the default backend
    if not cap.isOpened() or not cap.read()[0]:
        cap.release()
        sys.exit(
            f"ERROR: could not open camera {index}.\n"
            "  - Is a webcam connected and not in use by another app (Zoom, Teams, Camera)?\n"
            "  - Windows: Settings > Privacy & security > Camera > allow desktop apps.\n"
            "  - Try another index: --camera 1"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def load_models(min_detection_confidence: float):
    """Load the MLP, label encoder and VIDEO-mode landmarker, or exit with a readable message."""
    for path, hint in ((MLP_MODEL_PATH, "python src/train.py"), (ENCODER_PATH, "python src/train.py")):
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Create it with: {hint}")
    try:
        landmarker = create_hand_landmarker(vision.RunningMode.VIDEO, min_detection_confidence)
    except FileNotFoundError as err:
        sys.exit(f"ERROR: {err}")
    model = tf.keras.models.load_model(MLP_MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return landmarker, model, list(encoder.classes_)


class WordBuilder:
    """Types a voted label once it has been stable for hold_s seconds."""

    def __init__(self, hold_s: float) -> None:
        self.hold_s = hold_s
        self.text = ""
        self.current: str | None = None  # label being held
        self.since = 0.0                 # when the current label started
        self.typed = False               # current hold already produced a character

    def update(self, label: str, now: float) -> None:
        if label != self.current:
            self.current, self.since, self.typed = label, now, False
            return
        if self.typed or label in NOT_TYPED or now - self.since < self.hold_s:
            return
        if label == "space":
            self.text += " "
        elif label == "del":
            self.text = self.text[:-1]
        else:
            self.text += label
        self.typed = True

    def progress(self, now: float) -> float:
        """0..1 hold progress of the current label (0 if it will not be typed)."""
        if self.current is None or self.current in NOT_TYPED or self.typed:
            return 0.0
        return min(1.0, (now - self.since) / self.hold_s)

    def clear(self) -> None:
        self.text = ""


def classify(model, class_names: list[str], features: np.ndarray | None, threshold: float) -> tuple[str, float]:
    """Per-frame label and confidence: 'nothing' without a hand, '?' when unsure."""
    if features is None:
        return NO_HAND_LABEL, 1.0
    probs = model(features[None], training=False).numpy()[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return (class_names[idx] if conf >= threshold else UNSURE_LABEL), conf


def majority_vote(history: deque) -> tuple[str, float]:
    """Most common label in the window and the mean confidence of the frames that voted for it."""
    label, _ = Counter(lbl for lbl, _ in history).most_common(1)[0]
    confs = [c for lbl, c in history if lbl == label]
    return label, float(np.mean(confs))


def draw_text(img, text, org, scale, color, thickness=2) -> None:
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, FONT, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_hand(img, points: np.ndarray, color=GREEN) -> None:
    pts = [tuple(int(v) for v in p) for p in points]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], WHITE, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 4, color, -1, cv2.LINE_AA)


def draw_overlay(img, label: str, conf: float, frame_label: str, frame_conf: float,
                 words: WordBuilder, fps: float, now: float, threshold: float) -> None:
    h, w = img.shape[:2]

    # Top-left panel: smoothed prediction, confidence and hold progress.
    cv2.rectangle(img, (0, 0), (330, 150), BLACK, -1)
    color = GREY if label in NOT_TYPED else GREEN
    draw_text(img, label, (15, 85), 2.4 if len(label) == 1 else 1.6, color, 4)
    if label != NO_HAND_LABEL:
        draw_text(img, f"conf {conf:.2f}", (15, 120), 0.7, color)
    draw_text(img, f"frame: {frame_label} {frame_conf:.2f}", (15, 142), 0.5, GREY, 1)
    progress = words.progress(now)
    if progress > 0:
        cv2.rectangle(img, (180, 100), (180 + int(130 * progress), 115), ORANGE, -1)
        cv2.rectangle(img, (180, 100), (310, 115), WHITE, 1)

    # Top-right: FPS and threshold.
    draw_text(img, f"FPS {fps:4.1f}", (w - 150, 30), 0.7, WHITE)
    draw_text(img, f"thr {threshold:.2f}", (w - 150, 58), 0.55, GREY, 1)

    # Bottom: text bar and key help.
    cv2.rectangle(img, (0, h - 80), (w, h), BLACK, -1)
    shown = words.text[-MAX_TEXT_CHARS:]
    cursor = "|" if int(now * 2) % 2 == 0 else " "
    draw_text(img, shown + cursor, (15, h - 35), 1.2, WHITE)
    draw_text(img, "q quit   c clear   s screenshot", (15, h - 10), 0.5, GREY, 1)


def save_screenshot(img) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(str(path), img)
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    landmarker, model, class_names = load_models(args.min_detection_confidence)
    cap = open_camera(args.camera)
    print(f"Running. {len(class_names)} classes, threshold {args.threshold}, "
          f"detection confidence {args.min_detection_confidence}. Press q to quit.")

    history: deque = deque(maxlen=args.window)
    words = WordBuilder(args.hold)
    fps, last_time, last_ts = 0.0, time.perf_counter(), -1
    window = "ASL fingerspelling demo"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: camera stopped delivering frames (disconnected?).", file=sys.stderr)
                break
            # VIDEO mode needs strictly increasing timestamps.
            now = time.perf_counter()
            ts = max(int(now * 1000), last_ts + 1)
            last_ts = ts
            detect_frame = cv2.flip(frame, 1) if args.flip_input else frame
            rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            features, points = detect_hand(landmarker, rgb, LIVE_PAD_RATIO, LIVE_MIN_SIDE, timestamp_ms=ts)

            frame_label, frame_conf = classify(model, class_names, features, args.threshold)
            history.append((frame_label, frame_conf))
            label, conf = majority_vote(history)
            words.update(label, now)

            mirror = not args.no_mirror
            display = cv2.flip(frame, 1) if mirror else frame
            if points is not None:
                if mirror != args.flip_input:  # detection and display images are mirror images
                    points = points.copy()
                    points[:, 0] = display.shape[1] - 1 - points[:, 0]
                draw_hand(display, points)
            dt = now - last_time
            last_time = now
            fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / max(dt, 1e-6)
            draw_overlay(display, label, conf, frame_label, frame_conf, words, fps, now, args.threshold)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("c"):
                words.clear()
            elif key == ord("s"):
                save_screenshot(display)
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
    if words.text:
        print(f"Final text: {words.text}")


if __name__ == "__main__":
    main()
