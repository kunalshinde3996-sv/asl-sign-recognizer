# Jury prep: likely questions and short answers

Numbers come from this folder's reports. Answer in 2–4 sentences, give the number, then stop.

---

## Results and model choice

**Q: What accuracy do you get?**
98.5% on the test split (macro F1 0.984), counting only images where MediaPipe found a hand. End to end on 1400 held-out raw images it's 74.4%: when the hand is found it's 99.0% correct, but the hand is found in only 75% of images. The bottleneck is detection, not classification.

**Q: The CNN gets 89.1% end to end and your pipeline 74.4%. Why choose landmarks?**
Because 89.1% is measured on near-duplicates of the CNN's own training data: same signer, same background, same lighting, consecutive video frames. When I changed only the lighting, the CNN fell to 8–13%, which is near chance. The landmark pipeline stayed at 94–96% whenever it found the hand. A live webcam is a new person, room and light, and that's exactly the situation the lighting test simulates. The landmark model is also smaller (0.78 MB vs 1.24 MB), only processes coordinates (privacy), and its errors are explainable: I can tell "didn't find the hand" apart from "wrong letter".

**Q: Isn't the 98.5% inflated?**
Yes, and I say so. Kaggle frames are consecutive video frames of one person, so test images are near-duplicates of training images. It's an upper bound. The honest numbers are the end-to-end and lighting tests, and the real test is the live demo with a new person.

**Q: Why not just make the CNN robust with augmentation?**
That's possible, and it's a fair next step. But augmentation only covers the variations you think of. Landmarks remove background and lighting from the classifier's input by design. The CNN baseline was deliberately plain, so it shows what a pixel model learns from this dataset.

---

## Errors

**Q: Which signs get confused, and why?**
- **M ↔ N** (the top pair): both are fists with the thumb tucked under the fingers, three fingers for M and two for N. The thumb is hidden, so its landmarks are guessed and noisy. M and N are also the hardest to *detect* (46% and 57% missed).
- **R ↔ U**: two raised fingers; R crosses them. The crossing is a few pixels in 2D.
- **G ↔ H**: both sideways, one finger vs two.
- **C ↔ space**: in this dataset "space" is an open, curved, sideways hand, close to C.
- **H ↔ J**: J is a motion sign. Its end pose is a sideways hand with the pinky out, which looks like H in a single frame.

All of these are 2–3 errors out of 2127 test samples. The bigger loss is missed detections.

**Q: Why does detection fail on 24% of images?**
The Kaggle images are small (200×200), often dark, and closed-fist letters hide fingers. MediaPipe is trained on everyday hands, not tight fists. Worst classes: N 57%, space 48%, M 46%, del 40% missed. On a live webcam the hand is larger and better lit, and VIDEO mode tracks it between frames, so I expect detection to be better there. That's an expectation, not a measured number.

---

## Method

**Q: Why normalize the landmarks?**
Two steps. (1) Subtract the wrist, so the vector describes hand *shape*, not where the hand is in the frame. (2) Divide by the largest wrist-to-point distance, so a hand close to the camera and one far away give the same vector, with every value in [-1, 1], which also helps training. I convert to pixel units first, because a 16:9 webcam would otherwise stretch the hand compared to the square training images. I deliberately do *not* normalize rotation: H vs U and G vs Q differ mainly by orientation.

**Q: Why is "nothing" not a class?**
A frame without a hand has no landmarks, so there is nothing to classify. Training on rows of zeros would teach the model an input it never sees live. So "no hand detected → nothing" is a rule. On the Kaggle "nothing" images MediaPipe found 0 false hands at confidence 0.5. I also measure false hands live on an empty background (false_hand_test.md).

**Q: How do training and live preprocessing stay identical?**
There is one function, `detect_hand` in `src/utils.py`, used by both landmark extraction and the live demo. A test confirmed that re-extracting dataset images through it gives exactly the stored training features (difference 0.0).

**Q: What did you try to improve detection? (padding / upscaling)**
Six settings on the same 50 images per class. Rule: adopt only if it gains more than +5 points with zero false hands on "nothing".
- 10% white padding: +0.2 points, and 1 false hand. Not worth it.
- 30% padding: −7.1 points, worse in 23 of 28 classes, because padding shrinks the hand in what MediaPipe sees.
- Upscaling to 400 px: −2.1 points. It adds no information, since MediaPipe resizes internally anyway.
- Confidence 0.3: +3.9 points (M 21→29, space 22→30 of 50). Best gain, but below the bar.

Nothing passed, so I kept the baseline. Something else I found: padding changed the features of the *same* image more (0.10) than two different images of the same letter differ (0.04). So any preprocessing change must be applied identically in training and live, or the model breaks without any error.

**Q: How does the live demo avoid flickering?**
Three layers. (1) A confidence threshold of 0.7: below it, the frame shows "?". (2) A majority vote over the last 10 frames. (3) The word builder only types a letter after it has been the vote winner for 1 second, and types it once per hold.

**Q: How does lighting affect the system?**

| | MLP end-to-end | MLP when detected | CNN |
|---|---|---|---|
| Original | 76.4% | 99.1% | 89.7% |
| Dark | 53.5% | 94.3% | 12.9% |
| Bright | 71.1% | 95.7% | 12.8% |
| Low contrast | 59.0% | 95.4% | 7.7% |

Darkness and low contrast mostly hurt *detection*. The classifier itself stays at 94–96%. The CNN collapses to near chance under any change. Practical advice: light the hand from the front.

---

## Use and impact

**Q: Who is this for? Accessibility use?**
Fingerspelling is used for names, places and technical words. Possible uses: a learning tool that gives beginners instant feedback; a bridge for short exchanges with someone who doesn't know ASL (names, a room number); a building block for a fuller sign-language interface. It runs on a laptop CPU with an ordinary webcam and no cloud, so it's cheap and private.

**Q: Is this a sign-language translator?**
No, and I wouldn't claim it is. ASL is a full language with its own grammar, two-handed signs, motion and facial expressions. This recognizes the static fingerspelling alphabet of one hand. It should be built and tested *with* Deaf users, not just for them.

---

## Limitations and fixes

| Limitation | Fix |
|---|---|
| J and Z are motion signs; single frames only see the end pose | Sequence model (LSTM/Transformer) over landmark sequences |
| One hand only | `num_hands=2` and a two-hand feature vector |
| One signer, one background; test split near-duplicates | Collect multi-signer webcam data; leave-one-signer-out evaluation |
| ~24% detection misses on the dataset | Better lighting, VIDEO-mode tracking, a hand detector fine-tuned on hard images; confidence 0.3 if the false-hand test is clean |
| M/N, R/U confusion | Engineered features (finger angles, fingertip distances), more data for those pairs |
| Handedness | Mirror augmentation; `--flip-input` for now |
| Can't type double letters without changing the sign | Release gesture or a timed repeat |

**Q: If you had one more month?**
Collect webcam data from 10+ signers with both hands and different rooms, retrain, and report per-signer accuracy. That answers the biggest open question: does it generalize beyond one person?
