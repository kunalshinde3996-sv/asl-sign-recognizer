"""Shared hand-landmark utilities.

Everything that turns a MediaPipe detection into a feature vector lives here so
that dataset extraction and the live webcam demo use *identical* preprocessing.
If training and inference preprocessing ever differ, the classifier sees a
different input distribution live than it saw in training, and accuracy drops
without any error message.

MediaPipe API note: mediapipe >= 0.10.30 removed the legacy ``mp.solutions.hands``
API. Only the Tasks ``HandLandmarker`` is available, and it needs the
``hand_landmarker.task`` model bundle (see MODEL_URL).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3  # 63
FEATURE_COLUMNS: list[str] = [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]

# Optional white border added around an image before hand detection, as a
# fraction of the image's larger side, added on every side. OFF by default.
#
# Tested on the Kaggle images (50/class, see reports/detection_experiments.md):
#   pad 0%  -> 76.0% hands detected
#   pad 10% -> 76.2%  (no real change, 1 false hand on "nothing")
#   pad 30% -> 68.9%  (worse in 23 of 28 classes)
# Padding shrinks the hand in the image MediaPipe sees, and its landmark
# estimates depend on that size. Padded vs unpadded features of the SAME image
# differed more than features of two different images of the same letter.
# So the padding must be identical for training data and live frames. Mixing
# them causes a train/test mismatch.
#
#   DATASET_PAD_RATIO: extraction and the evaluation scripts (Kaggle images).
#   LIVE_PAD_RATIO:    webcam frames. Must equal DATASET_PAD_RATIO unless the
#                      model is retrained. Only change both after testing that
#                      padding helps on the webcam.
DATASET_PAD_RATIO = 0.0
LIVE_PAD_RATIO = DATASET_PAD_RATIO

# Optional upscaling before detection: images whose shorter side is below this
# many pixels are enlarged (aspect ratio kept) until it reaches it. 0 = off.
# The same rules as padding apply: training and live must use the same value.
DATASET_MIN_SIDE = 0
LIVE_MIN_SIDE = DATASET_MIN_SIDE

# Raw-dataset classes that contain no hand. A live frame with no detected hand
# is labelled with this class by rule, not by the classifier.
NO_HAND_LABEL = "nothing"


def create_hand_landmarker(
    running_mode: vision.RunningMode = vision.RunningMode.IMAGE,
    min_detection_confidence: float = 0.5,
    model_path: Path = HAND_MODEL_PATH,
) -> vision.HandLandmarker:
    """Build a single-hand MediaPipe HandLandmarker.

    Use RunningMode.IMAGE for independent images (dataset extraction) and
    RunningMode.VIDEO for webcam frames (enables tracking between frames).
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found at {model_path}.\n"
            f"Download it from:\n  {MODEL_URL}"
        )
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_array(
    hand_landmarks: list, image_width: int, image_height: int
) -> np.ndarray:
    """Convert one hand's MediaPipe landmarks into a (21, 3) array in pixel units.

    MediaPipe returns x in [0, 1] relative to image *width* and y relative to
    image *height*. The Kaggle images are square (200x200) but a webcam frame is
    usually 4:3 or 16:9. Left in normalized units, the same hand shape would look
    horizontally stretched on the webcam compared to training. Multiplying back
    by width/height restores the true geometry. MediaPipe's z uses roughly the
    same scale as x, so it is multiplied by the width.
    """
    return np.array(
        [[lm.x * image_width, lm.y * image_height, lm.z * image_width] for lm in hand_landmarks],
        dtype=np.float32,
    )


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Make landmarks translation- and scale-invariant, then flatten to 63 features.

    Why each step:
      1. Subtract the wrist (landmark 0). The wrist becomes the origin, so the
         features describe the hand *shape* and not where the hand is in the
         frame. A sign at the top-left corner and the same sign at the center
         give the same vector (translation invariance).
      2. Divide by the largest wrist-to-landmark distance. A hand close to the
         camera looks bigger than one far away, but the sign is the same. After
         scaling, the farthest landmark is at distance 1 and every coordinate
         lies in [-1, 1] (scale invariance). Bounded inputs of similar magnitude
         also help a neural network train well.

    Rotation is deliberately NOT normalized: some letters differ mainly by hand
    orientation (e.g. H vs U, G vs Q), so orientation is a useful signal.

    Args:
        landmarks: array of shape (21, 3), ideally from ``landmarks_to_array``.

    Returns:
        float32 array of shape (63,), ordered x0, y0, z0, x1, y1, z1, ...
    """
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"Expected shape ({NUM_LANDMARKS}, 3), got {landmarks.shape}")

    centered = landmarks - landmarks[0]
    max_dist = float(np.max(np.linalg.norm(centered, axis=1)))
    if max_dist > 1e-6:  # guard against a degenerate detection where all points coincide
        centered = centered / max_dist
    return centered.astype(np.float32).flatten()


def pad_pixels(shape: tuple[int, ...], pad_ratio: float) -> int:
    """Border width in pixels that pad_image adds on every side of an image of this shape."""
    return int(round(pad_ratio * max(shape[:2]))) if pad_ratio > 0 else 0


def pad_image(rgb: np.ndarray, pad_ratio: float) -> np.ndarray:
    """Add a white border of pad_ratio * max(height, width) pixels on every side.

    The same pixel amount on every side keeps the hand's geometry unchanged.
    """
    pad = pad_pixels(rgb.shape, pad_ratio)
    if pad == 0:
        return rgb
    return cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def upscale_image(rgb: np.ndarray, min_side: int) -> np.ndarray:
    """Enlarge the image (keeping aspect ratio) so its shorter side is at least min_side pixels."""
    short = min(rgb.shape[:2])
    if min_side <= 0 or short >= min_side:
        return rgb
    scale = min_side / short
    return cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def detect_hand(
    landmarker: vision.HandLandmarker,
    rgb: np.ndarray,
    pad_ratio: float = 0.0,
    min_side: int = 0,
    timestamp_ms: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Optionally upscale and pad, run MediaPipe, return (features, points) or (None, None).

    features: the 63 normalized features (the classifier input).
    points:   (21, 2) landmark pixel positions in the ORIGINAL rgb image, for drawing.

    Pass timestamp_ms for a landmarker in RunningMode.VIDEO (must increase on
    every call); leave it None for RunningMode.IMAGE. Either way the features
    come from the same code path, so training and live preprocessing match.

    Landmarks are converted to pixels using the size of the image MediaPipe
    actually saw, so the geometry stays correct.
    """
    upscaled = upscale_image(rgb, min_side)
    pad = pad_pixels(upscaled.shape, pad_ratio)
    image = np.ascontiguousarray(pad_image(upscaled, pad_ratio))
    height, width = image.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    if timestamp_ms is None:
        result = landmarker.detect(mp_image)
    else:
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None, None

    raw = landmarks_to_array(result.hand_landmarks[0], width, height)
    scale = upscaled.shape[1] / rgb.shape[1]
    points = (raw[:, :2] - pad) / scale
    return normalize_landmarks(raw), points


def detect_hand_features(
    landmarker: vision.HandLandmarker,
    rgb: np.ndarray,
    pad_ratio: float = 0.0,
    min_side: int = 0,
    timestamp_ms: int | None = None,
) -> np.ndarray | None:
    """The 63 normalized features of the first detected hand, or None. See detect_hand."""
    return detect_hand(landmarker, rgb, pad_ratio, min_side, timestamp_ms)[0]
