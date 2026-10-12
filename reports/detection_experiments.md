# Detection experiments: getting MediaPipe to find more hands

Same 50 images per class (seed 42) for every setting. Rule for adopting a change: more than +5 points detection
with zero false hands on 'nothing'.

| Setting | Hands detected (28 classes) | Detection rate | Change | False hands on 'nothing' (of 50) |
|---|---|---|---|---|
| baseline (no pad, conf 0.5) | 1064 / 1400 | 76.0% | +0.0 pts | 0 |
| pad 10% | 1067 / 1400 | 76.2% | +0.2 pts | 1 |
| pad 30% | 965 / 1400 | 68.9% | -7.1 pts | 0 |
| upscale to 400 | 1035 / 1400 | 73.9% | -2.1 pts | 0 |
| conf 0.3 | 1119 / 1400 | 79.9% | +3.9 pts | 0 |
| upscale 400 + conf 0.3 | 1081 / 1400 | 77.2% | +1.2 pts | 0 |

**Decision:** keep the baseline (no padding, no upscaling, confidence 0.5). No setting cleared +5 points.

- **Padding** shrinks the hand in what MediaPipe sees; 30% hurts in 23 of 28 classes.
- **Upscaling** adds no information (MediaPipe resizes its input internally anyway); interpolation slightly hurts.
- **Confidence 0.3** is the only real gain (+3.9 pts), mostly on the weakest classes: M 21->29, space 22->30,
  N 22->26 (of 50), with no false hands. It's below the bar, and a lower threshold also makes false hands more likely live.
- Padded vs unpadded features of the same image differed by a median max of 0.10 ([-1, 1] scale), more than
  two different images of the same letter (0.04). So any preprocessing change must be identical in training and live.

Per-class detections: detection_experiments.csv
