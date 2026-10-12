"""Count false hand detections on an empty background at confidence 0.5 vs 0.3.

The live demo shows "nothing" whenever MediaPipe finds no hand, so a "hand"
found on an empty background is a false prediction. On the Kaggle "nothing"
images neither 0.5 nor 0.3 produced false hands (reports/detection_experiments.md),
but a live room has more clutter than those photos. This test checks it on YOUR
webcam and background before deciding whether a lower threshold is safe live.

Both landmarkers see exactly the same frames, so the comparison is fair.
Point the camera at the background you will demo in, start the script and move
out of the frame during the countdown.

Outputs:
    reports/false_hand_test.md    summary table (overwritten each run)
    reports/false_hand_test.csv   one row per run (appended)
    reports/false_hands/*.png     up to --max-images frames with a false detection,
                                  landmarks drawn (green = conf 0.5, orange = 0.3).
                                  Unmirrored camera view. Cleared at the start of each run.

Usage:
    python src/false_hand_test.py --seconds 20
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime

import cv2
from mediapipe.tasks.python import vision

from live_demo import GREEN, ORANGE, draw_hand, open_camera
from utils import LIVE_MIN_SIDE, LIVE_PAD_RATIO, PROJECT_ROOT, create_hand_landmarker, detect_hand

REPORTS_DIR = PROJECT_ROOT / "reports"
FALSE_HANDS_DIR = REPORTS_DIR / "false_hands"
CONFIDENCES = (0.5, 0.3)
HAND_COLORS = {0.5: GREEN, 0.3: ORANGE}


def save_false_hand(frame, found: dict, frame_no: int, elapsed: float) -> None:
    """Save the frame with every threshold's detected landmarks drawn and labelled."""
    img = frame.copy()
    # Draw 0.3 first so the 0.5 hand (usually the same spot) ends up on top.
    for conf in sorted(found):
        if found[conf] is not None:
            draw_hand(img, found[conf], HAND_COLORS[conf])
    lines = [(f"frame {frame_no}  t={elapsed:.1f}s", (255, 255, 255))]
    lines += [(f"conf {c}: {'HAND' if found[c] is not None else '-'}", HAND_COLORS[c]) for c in CONFIDENCES]
    for i, (line, color) in enumerate(lines):
        cv2.putText(img, line, (15, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, line, (15, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(FALSE_HANDS_DIR / f"false_hand_{frame_no:05d}.png"), img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    parser.add_argument("--seconds", type=float, default=20.0, help="Recording length (default 20).")
    parser.add_argument("--countdown", type=int, default=5, help="Seconds to leave the frame before recording.")
    parser.add_argument("--no-preview", action="store_true", help="Do not show the camera window.")
    parser.add_argument("--max-images", type=int, default=20,
                        help="Max false-detection frames saved to reports/false_hands/ (default 20).")
    return parser.parse_args()


def show(frame, lines: list[str], preview: bool) -> bool:
    """Draw status lines on a mirrored preview. Returns False if the user pressed q."""
    if not preview:
        return True
    img = cv2.flip(frame, 1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (15, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, line, (15, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("False hand test", img)
    return cv2.waitKey(1) & 0xFF != ord("q")


def main() -> None:
    args = parse_args()
    preview = not args.no_preview
    landmarkers = {c: create_hand_landmarker(vision.RunningMode.VIDEO, c) for c in CONFIDENCES}
    cap = open_camera(args.camera)

    # Countdown: time to step out of the frame.
    end = time.perf_counter() + args.countdown
    while (left := end - time.perf_counter()) > 0:
        ok, frame = cap.read()
        if not ok or not show(frame, [f"Leave the frame! Starting in {left:.0f} s"], preview):
            cap.release()
            sys.exit("Aborted.")
        if not preview:
            time.sleep(0.05)

    # Fresh folder per run so images from different runs never mix.
    FALSE_HANDS_DIR.mkdir(parents=True, exist_ok=True)
    for old in FALSE_HANDS_DIR.glob("false_hand_*.png"):
        old.unlink()
    # Spread saves over the run: consecutive false frames are near-identical,
    # so the first 20 would all show the same moment.
    save_gap = args.seconds / max(args.max_images, 1)
    saved, last_save = 0, -save_gap

    frames = 0
    hand_frames = dict.fromkeys(CONFIDENCES, 0)
    episodes = dict.fromkeys(CONFIDENCES, 0)  # runs of consecutive false-hand frames
    in_episode = dict.fromkeys(CONFIDENCES, False)
    start = time.perf_counter()
    last_ts = -1
    try:
        while (elapsed := time.perf_counter() - start) < args.seconds:
            ok, frame = cap.read()
            if not ok:
                sys.exit("ERROR: camera stopped delivering frames.")
            ts = max(int(time.perf_counter() * 1000), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames += 1
            points = {}
            for conf, landmarker in landmarkers.items():
                points[conf] = detect_hand(landmarker, rgb, LIVE_PAD_RATIO, LIVE_MIN_SIDE, timestamp_ms=ts)[1]
                found = points[conf] is not None
                hand_frames[conf] += found
                episodes[conf] += found and not in_episode[conf]
                in_episode[conf] = found
            if any(p is not None for p in points.values()) and saved < args.max_images \
                    and elapsed - last_save >= save_gap:
                save_false_hand(frame, points, frames, elapsed)
                saved, last_save = saved + 1, elapsed
            lines = [f"Recording empty background: {args.seconds - elapsed:4.1f} s left"]
            lines += [f"conf {c}: {hand_frames[c]} false-hand frames" for c in CONFIDENCES]
            if not show(frame, lines, preview):
                sys.exit("Aborted.")
    finally:
        cap.release()
        for landmarker in landmarkers.values():
            landmarker.close()
        cv2.destroyAllWindows()

    duration = time.perf_counter() - start
    stamp = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    rows = [
        {"timestamp": stamp, "confidence": c, "frames": frames, "seconds": round(duration, 1),
         "false_hand_frames": hand_frames[c], "false_hand_rate": round(hand_frames[c] / max(frames, 1), 4),
         "false_hand_episodes": episodes[c]}
        for c in CONFIDENCES
    ]

    REPORTS_DIR.mkdir(exist_ok=True)
    csv_path = REPORTS_DIR / "false_hand_test.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)

    table = "\n".join(
        f"| {r['confidence']} | {r['false_hand_frames']} / {r['frames']} | {100 * r['false_hand_rate']:.2f}% "
        f"| {r['false_hand_episodes']} |"
        for r in rows
    )
    md = (
        "# False hand detections on an empty background (live webcam)\n\n"
        f"Run: {stamp}, {duration:.1f} s, {frames} frames ({frames / duration:.1f} FPS), camera {args.camera}. "
        "Both thresholds saw the same frames.\n\n"
        "| Detection confidence | False-hand frames | Rate | Episodes (consecutive runs) |\n"
        "|---|---|---|---|\n"
        f"{table}\n\n"
        "A false hand makes the demo show a letter instead of \"nothing\". Majority voting over ~10 frames "
        "hides isolated single frames, so episodes longer than ~5 frames are the ones that matter.\n"
        f"\n{saved} example frames (green = conf 0.5, orange = 0.3): false_hands/\n"
        "All runs: false_hand_test.csv\n"
    )
    (REPORTS_DIR / "false_hand_test.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved {REPORTS_DIR / 'false_hand_test.md'} and {csv_path}")


if __name__ == "__main__":
    main()
