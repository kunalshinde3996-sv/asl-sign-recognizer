"""Download the Kaggle "ASL Alphabet" dataset and place it in data/raw/.

kagglehub downloads into its own cache (~/.cache/kagglehub). The archive is
nested: asl_alphabet_train/asl_alphabet_train/A..Z. This script finds the folder
that actually contains the class folders and copies it to:

    data/raw/asl_alphabet_train/<class>/*.jpg

If the target already exists, the copy is skipped.

Usage:
    python src/download_data.py
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import truststore

# Verify HTTPS against the Windows certificate store instead of Python's bundled
# certifi list. Antivirus HTTPS scanning (e.g. Avast Web Shield) re-signs traffic
# with a root certificate that only Windows trusts. Verification stays enabled.
# Must run before kagglehub/requests create any SSL context.
truststore.inject_into_ssl()

import kagglehub  # noqa: E402

# Not imported from utils: that would load mediapipe just to get a path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_HANDLE = "grassknoted/asl-alphabet"
TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "asl_alphabet_train"
# Marker folders that identify the real training root.
REQUIRED_CLASSES = ("A", "Z", "nothing")

logger = logging.getLogger("download_data")


def is_class_root(path: Path) -> bool:
    """True if ``path`` directly contains the class folders (A, Z, nothing)."""
    return all((path / name).is_dir() for name in REQUIRED_CLASSES)


def find_train_root(download_dir: Path) -> Path:
    """Find the ``asl_alphabet_train`` folder that directly holds the class folders.

    Searches the whole download, so single or double nesting both work. The
    dataset also contains ``asl_alphabet_test`` (flat files, no class folders),
    which ``is_class_root`` rules out.
    """
    candidates = [download_dir, *download_dir.rglob("asl_alphabet_train")]
    for candidate in candidates:
        if candidate.is_dir() and is_class_root(candidate):
            return candidate
    raise FileNotFoundError(f"No folder with class sub-folders {REQUIRED_CLASSES} found under {download_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if TARGET_DIR.is_dir() and is_class_root(TARGET_DIR):
        logger.info("Dataset already present at %s, skipping download/copy.", TARGET_DIR)
        return

    logger.info("Downloading %s via kagglehub (cached after the first run)...", DATASET_HANDLE)
    download_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    logger.info("Downloaded to %s", download_dir)

    source = find_train_root(download_dir)
    logger.info("Found class folders in %s", source)

    # Copy to a temporary name, then rename, so an interrupted copy is never
    # mistaken for a complete dataset on the next run.
    tmp_dir = TARGET_DIR.with_name(TARGET_DIR.name + ".partial")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if TARGET_DIR.exists():
        logger.error("%s exists but is incomplete. Delete it and re-run.", TARGET_DIR)
        sys.exit(1)

    logger.info("Copying to %s (this can take a few minutes)...", TARGET_DIR)
    shutil.copytree(source, tmp_dir)
    tmp_dir.rename(TARGET_DIR)

    classes = sorted(d.name for d in TARGET_DIR.iterdir() if d.is_dir())
    logger.info("Done: %d classes -> %s", len(classes), ", ".join(classes))


if __name__ == "__main__":
    main()
