"""Extract normalized MediaPipe hand landmarks from the Kaggle ASL Alphabet dataset.

Input:  data/raw/asl_alphabet_train/<class>/*.jpg   (A..Z, space, del, nothing)
Output: data/processed/landmarks.csv          63 normalized features + label
        data/processed/extraction_stats.csv   per-class detection statistics

Handling the "nothing" class:
    "nothing" images contain no hand, so there are no landmarks to extract. A
    row of zeros would be a fake feature vector that never occurs at inference
    time, because the live demo cannot run the classifier without landmarks.
    The pipeline therefore treats "nothing" as a RULE and not a learned class:

        no hand detected  ->  predict "nothing"   (live demo, no classifier call)
        hand detected     ->  classifier picks among the 28 hand classes

    So "nothing" images are processed but contribute no rows to landmarks.csv.
    They are not silently dropped: their counts go into extraction_stats.csv,
    and any images where MediaPipe wrongly finds a hand are reported as false
    positives. A high false-positive rate means the "no hand => nothing" rule
    is unreliable, and you should raise --min-detection-confidence.

Usage:
    python src/extract_landmarks.py --max-per-class 1000
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from utils import DATASET_MIN_SIDE, DATASET_PAD_RATIO, FEATURE_COLUMNS, NO_HAND_LABEL, PROJECT_ROOT, create_hand_landmarker, detect_hand_features

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

logger = logging.getLogger("extract_landmarks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "asl_alphabet_train",
        help="Folder containing one sub-folder per class.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Where landmarks.csv and extraction_stats.csv are written.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1000,
        help="Max images sampled per class (0 = use all). Default: 1000.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe hand detection threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=DATASET_PAD_RATIO,
        help=f"White border per side, as a fraction of image size, added before detection "
             f"(0 = off). Default: {DATASET_PAD_RATIO}. See utils.py.",
    )
    parser.add_argument(
        "--min-side",
        type=int,
        default=DATASET_MIN_SIDE,
        help=f"Upscale images so the shorter side is at least this many pixels before detection "
             f"(0 = off). Default: {DATASET_MIN_SIDE}. See utils.py.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


def list_images(class_dir: Path, max_per_class: int, rng: random.Random) -> list[Path]:
    """Return a reproducible random sample of image paths from one class folder.

    Random sampling matters here: Kaggle images are numbered in recording order,
    so consecutive files are near-duplicate video frames. Taking the first N
    would cover far less variation than a random N.
    """
    images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if 0 < max_per_class < len(images):
        images = rng.sample(images, max_per_class)
    return images


def process_class(
    landmarker, label: str, images: list[Path], pad_ratio: float = 0.0, min_side: int = 0
) -> tuple[list[list], dict]:
    """Run hand detection on every image of one class.

    Returns:
        rows: feature rows (63 floats + label) for images with a detected hand.
        stats: per-class detection counts.
    """
    rows: list[list] = []
    detected = unreadable = 0

    for path in tqdm(images, desc=f"{label:>8}", unit="img", leave=False):
        bgr = cv2.imread(str(path))
        if bgr is None:
            unreadable += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        features = detect_hand_features(landmarker, rgb, pad_ratio, min_side)
        if features is None:
            continue
        detected += 1
        if label != NO_HAND_LABEL:  # see module docstring
            rows.append([*features.tolist(), label])

    processed = len(images) - unreadable
    stats = {
        "label": label,
        "total": len(images),
        "unreadable": unreadable,
        "detected": detected,
        "not_detected": processed - detected,
        # For "nothing", a detection is a false positive, so its "failure" is a detection.
        "failure_rate": (
            detected / processed if label == NO_HAND_LABEL else (processed - detected) / processed
        ) if processed else 0.0,
        "rows_saved": len(rows),
    }
    return rows, stats


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.data_dir.is_dir():
        logger.error("Dataset folder not found: %s", args.data_dir)
        sys.exit(1)

    class_dirs = sorted(d for d in args.data_dir.iterdir() if d.is_dir())
    if not class_dirs:
        logger.error("No class folders found in %s", args.data_dir)
        sys.exit(1)
    logger.info("Found %d classes in %s (pad ratio %.2f, min side %d, min confidence %.2f)",
                len(class_dirs), args.data_dir, args.pad_ratio, args.min_side, args.min_detection_confidence)

    rng = random.Random(args.seed)
    all_rows: list[list] = []
    all_stats: list[dict] = []

    with create_hand_landmarker(min_detection_confidence=args.min_detection_confidence) as landmarker:
        for class_dir in tqdm(class_dirs, desc="Classes", unit="class"):
            images = list_images(class_dir, args.max_per_class, rng)
            rows, stats = process_class(landmarker, class_dir.name, images, args.pad_ratio, args.min_side)
            all_rows.extend(rows)
            all_stats.append(stats)

            if class_dir.name == NO_HAND_LABEL:
                tqdm.write(
                    f"{class_dir.name:>8}: {stats['detected']}/{stats['total']} false-positive hands "
                    f"({stats['failure_rate']:.1%}), handled by the no-hand rule, 0 rows saved"
                )
            else:
                tqdm.write(
                    f"{class_dir.name:>8}: {stats['detected']}/{stats['total']} detected, "
                    f"failure rate {stats['failure_rate']:.1%}"
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    landmarks_path = args.out_dir / "landmarks.csv"
    stats_path = args.out_dir / "extraction_stats.csv"

    pd.DataFrame(all_rows, columns=[*FEATURE_COLUMNS, "label"]).to_csv(landmarks_path, index=False)
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(stats_path, index=False)

    hand_stats = stats_df[stats_df["label"] != NO_HAND_LABEL]
    overall = hand_stats["not_detected"].sum() / max(hand_stats["total"].sum() - hand_stats["unreadable"].sum(), 1)
    logger.info("Saved %d rows -> %s", len(all_rows), landmarks_path)
    logger.info("Saved stats  -> %s", stats_path)
    logger.info("Overall detection failure rate (hand classes): %.1f%%", overall * 100)

    worst = hand_stats.nlargest(5, "failure_rate")[["label", "failure_rate"]]
    logger.info("Worst classes:\n%s", worst.to_string(index=False))


if __name__ == "__main__":
    main()
