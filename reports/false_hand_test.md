# False hand detections on an empty background (live webcam)

Run: 2026-10-12 23:48:24, 20.9 s, 200 frames (9.5 FPS), camera 0. Both thresholds saw the same frames.

| Detection confidence | False-hand frames | Rate | Episodes (consecutive runs) |
|---|---|---|---|
| 0.5 | 121 / 200 | 60.50% | 3 |
| 0.3 | 122 / 200 | 61.00% | 3 |

A false hand makes the demo show a letter instead of "nothing". Majority voting over ~10 frames hides isolated single frames, so episodes longer than ~5 frames are the ones that matter.
All runs: false_hand_test.csv
