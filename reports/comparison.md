# Landmark MLP vs baseline CNN

| Metric | Landmark MLP | Baseline CNN |
|---|---|---|
| Test-split accuracy | 98.5% * | 91.1% |
| Test-split macro F1 | 0.984 * | 0.909 |
| End-to-end accuracy, held-out raw images (n=1400) | 74.4% | 89.1% |
| Accuracy when a hand is detected (held-out) | 99.0% | n/a |
| Hand detection rate (held-out) | 75.1% | n/a |
| Trainable parameters | 60,700 | 97,532 |
| Classifier file size | 0.78 MB | 1.24 MB |
| Total pipeline size | 8.60 MB (incl. MediaPipe) | 1.24 MB |
| Inference, classifier only (median, CPU) | 5.93 ms | 17.78 ms |
| Inference, full pipeline per image (median, CPU) | 34.32 ms (MediaPipe + MLP) | 17.78 ms |

\* MLP test-split numbers only include images where MediaPipe detected a hand. The end-to-end row counts missed detections as errors.

## How to read this

- **Both test splits are optimistic.** Kaggle ASL Alphabet images are consecutive video frames of one signer against one background, so test images are near-duplicates of training images. That holds for the held-out pool too. Real generalization (new people, backgrounds, cameras) has to be checked on the webcam.
- **The landmark MLP never sees the background or the lighting.** It only sees 63 normalized coordinates. The CNN sees every pixel, so it can learn background or lighting shortcuts that don't transfer. See reports/lighting_robustness.png.
- **The MLP's weak point is detection.** If MediaPipe misses the hand, the classifier never runs. The detection rate row is the ceiling on end-to-end accuracy.
- Data: MLP 21,266 landmark rows (80/10/10 split); CNN 300 images per class (80/10/10 split).
