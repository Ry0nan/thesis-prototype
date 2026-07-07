# Model Documentation

This document describes how each of the three ResNet18 classification models in this project was built, and how the YOLOv8n detector that feeds them was trained. It is the per-model build report for the thesis:

> _A Data-Efficient Semi-Supervised Learning Framework for Automated Kamias (Averrhoa bilimbi) Surface Defect Detection Using YOLOv8 and ResNet18._
> Group 16, Mapúa University — Ferrer, Lopez, Palma.

The pipeline has two stages: **YOLOv8n** locates and crops the fruit, then **ResNet18** classifies each crop as `defective` or `healthy`. This document covers the detector first, then the three classifier models that are the core of the study.

---

## The Shared Classifier Recipe

All three ResNet18 models are trained with an **identical recipe**. Only the _training data_ differs between them. Stating the recipe once here means each model card below only has to describe what makes it different.

| Component    | Setting                                                                             |
| ------------ | ----------------------------------------------------------------------------------- |
| Architecture | ResNet18, ImageNet-pretrained (`weights="DEFAULT"`)                                 |
| Final layer  | Original 1000-class FC layer replaced with a 2-class layer (`defective`, `healthy`) |
| Input size   | 224 × 224, ImageNet mean/std normalization                                          |
| Loss         | Cross-entropy, class-weighted by inverse class frequency                            |
| Optimizer    | Adam, initial LR 0.001                                                              |
| LR schedule  | Cosine annealing over 15 epochs                                                     |
| Batch size   | 8                                                                                   |
| Epochs       | 15                                                                                  |
| Augmentation | Random horizontal flip, vertical flip, ±20° rotation, color jitter (0.3)            |
| Checkpoint   | Best-validation-accuracy checkpoint saved (not the last epoch)                      |
| Seed         | 42                                                                                  |

**Why these choices, in plain terms:**

- **ImageNet-pretrained, fine-tuned (not trained from scratch):** the labeled set is small, so we start from features ResNet already learned on millions of images and adapt them, rather than learning everything from a few hundred kamias photos.
- **Class-weighted loss:** there are more defective than healthy images, so the loss penalizes mistakes on the smaller class more heavily to stop the model favoring the majority class.
- **Best-validation checkpoint:** we keep the epoch that performed best on the held-out validation set, not whatever the model looked like at the final epoch, which guards against overfitting late in training.
- **Same recipe across all three models:** this is deliberate. Because architecture, optimizer, schedule, and augmentation are held constant, any difference in test performance between the three models is attributable to the **training data alone**, not to a training-setup difference. That makes the comparison clean.

Two classes are ordered alphabetically by `ImageFolder`, so `defective = 0`, `healthy = 1`. This ordering matters when reading confusion matrices.

---

## Stage 1 — YOLOv8n Detector

**What it is and why.** Before classification, the fruit has to be located and cropped out of the photo so the classifier sees the fruit and not the background. YOLOv8n (the smallest, "nano" YOLOv8 variant) does this. It is trained as a **single-class detector** — it only learns to find _kamias_, not to judge quality. The healthy/defective decision is left entirely to the downstream ResNet18. Splitting the problem this way lets each model specialize: the detector localizes, the classifier classifies.

**How it was trained.**

- Images annotated in Roboflow with single-class (`kamias`) bounding boxes, exported in YOLO format.
- Fine-tuned from the pretrained `yolov8n.pt` checkpoint.
- 640 × 640 input, batch size 16, up to 100 epochs with early-stopping patience of 20.
- Best weights selected automatically and copied to `models/yolov8_kamias.pt` for the downstream cropping step.

**Result.** On the validation set the detector reached precision 1.000, recall 1.000, and mAP@0.5 of 0.995, with best weights at epoch 37. In the controlled imaging setup (plain background, single fruit per frame), localization is effectively solved, which is the precondition the classification stage relies on.

---

## Stage 2 — The Three ResNet18 Classifiers

The three models exist to answer one question: **how much can semi-supervised pseudo-labeling recover of the performance you would get from fully labeling the data, when you only actually label a quarter of it?**

- The **Baseline** is the floor: what you get from 252 real labels alone.
- The **Full Supervised** model is the reference ceiling: what you get if _all_ 1,014 training images are labeled by hand.
- The **Semi-Supervised (SSL)** model is the method under test: it starts from the same 252 real labels as the baseline, then grows its own training set with pseudo-labels — no additional human labeling.

All three are evaluated on the **same held-out test set of 222 cropped images** (121 defective, 101 healthy).

---

### Model 1 — Supervised Baseline

**Role** The baseline establishes the floor. It answers "what can the model do with only the 252 real labels a small team could realistically produce?" Every other result is measured against it.

**Data it saw.** 252 labeled crops — 25% of each class in the training set, stratified (114 healthy, 138 defective). Nothing else.

**How it was trained.** The shared recipe above, run once on those 252 images.

**What makes it different from the other two.** It sees the _fewest_ images and only _real_ human labels. It is the "small labeled set, no tricks" condition.

**Result.**

| Metric                           | Value            |
| -------------------------------- | ---------------- |
| Test accuracy                    | 89.19% (198/222) |
| Macro F1                         | 0.8915           |
| Defective F1                     | 0.898            |
| Healthy F1                       | 0.885            |
| False negatives (missed defects) | 15               |
| Total errors                     | 24               |

The baseline is also well-calibrated: on the test set its correct predictions averaged 88.5% confidence versus 67.7% for incorrect ones. That gap is what makes the pseudo-labeling threshold work in Model 2 — the model's confidence is a meaningful signal, so a high-confidence cutoff filters out most of its mistakes.

---

### Model 2 — Semi-Supervised (SSL) via Iterative Pseudo-Labeling

**Role (say this out loud).** This is the method the thesis is testing. It starts from the _exact same_ 252 real labels as the baseline, then teaches itself from the unlabeled pool: it predicts labels for unlabeled images, keeps only the ones it is very confident about, adds those to its training set, and retrains. It repeats this a few times. Crucially, **no human labels a single additional image.**

**Data it saw.** The 252 real labels, plus pseudo-labels it generated itself. Across 5 iterations the labeled pool grew **252 → 1,010**.

**How it was trained — the loop.** Starting from the baseline model:

1. **Predict** on every remaining unlabeled image.
2. **Filter** — keep only predictions with confidence ≥ 0.90.
3. **Accept** those as pseudo-labels and move them into the labeled pool under the predicted class.
4. **Retrain from scratch** (from ImageNet initialization — _not_ continuing the previous iteration's weights) on the enlarged pool, using the shared recipe.
5. **Repeat** until fewer than 5 new labels are accepted, or 5 iterations are reached.

**Why retrain from ImageNet each round rather than continue training?** So each iteration is interpretable in isolation and no representational bias compounds across rounds. Every model in the loop is a clean fine-tune of ImageNet on whatever the pool currently contains.

**How we measured pseudo-label quality (the honesty check).** Each unlabeled image's _true_ label was encoded into its filename (e.g. `__true_healthy__…`). The model never sees the filename — it only sees pixels — but after each iteration we compare its prediction to the encoded truth. This gives an honest per-iteration accuracy for the pseudo-labels the model accepted.

**Pseudo-labeling dynamics (this run):**

| Iteration | Pool at start   | Pseudo-labels added | Verifiable | Correct | Accuracy  |
| --------- | --------------- | ------------------- | ---------- | ------- | --------- |
| 1         | 252             | 455                 | 422        | 410     | 97.2%     |
| 2         | 707             | 197                 | 172        | 159     | 92.4%     |
| 3         | 904             | 58                  | 54         | 44      | 81.5%     |
| 4         | 962             | 27                  | 23         | 17      | 73.9%     |
| 5         | 989             | 21                  | 15         | 10      | 66.7%     |
| **Total** | 252 → **1,010** | **758**             | **686**    | **640** | **93.3%** |

_(Verifiable = pseudo-labels placed on images with an encoded ground-truth label. The remaining 72 were placed on originally-ambiguous images with no reference label and are excluded from the accuracy figure.)_

**What this table shows — the decay story.** The easy images are absorbed first at very high accuracy (97.2% in iteration 1). As those run out, the pool that remains is progressively harder — borderline fruit and originally-ambiguous images — so accuracy falls each round, down to 66.7% by iteration 5. This is the classic **confirmation-bias** failure mode of iterative pseudo-labeling documented by Arazo et al. (2020): the model stays confident enough to pass its own threshold even as it starts being wrong. It matters little here because the damage is small and late — the first three iterations added 613 of 640 correct labels at **94.6%** accuracy, and the last two iterations added only 27 correct against 11 wrong, a small amount of noise in a pool of over a thousand.

**What makes it different from the other two.** Same _real_ labels as the baseline (252), but a much larger _effective_ training set built by the model itself. It sits between the baseline (252 real) and the full model (1,014 real) — with the key property that the extra data cost **zero additional human labeling.**

**Result.**

| Metric                           | Value            |
| -------------------------------- | ---------------- |
| Test accuracy                    | 90.09% (200/222) |
| Macro F1                         | 0.9004           |
| Defective F1                     | 0.9076           |
| Healthy F1                       | 0.8932           |
| False negatives (missed defects) | 13               |
| Total errors                     | 22               |

**Versus the baseline (the headline comparison).** Same 252 real labels, but: accuracy 89.19% → 90.09%, macro F1 0.8915 → 0.9004, and missed defects (false negatives) down 15 → 13. The defective-class recall rose from 0.876 to 0.893. The gain is modest but real, and it came at no additional labeling cost — which is the entire point of the data-efficiency argument.

---

### Model 3 — Full Supervised (Upper Bound)

**Role (say this out loud).** This is the reference ceiling. It answers "if we had paid to hand-label the _entire_ training set, how good would the model get?" It tells us how much performance the SSL model recovers without that labeling effort.

**Data it saw.** All 1,014 real labeled crops in the training set. No pseudo-labels, no ambiguous images.

**How it was trained.** The shared recipe, run once on the full 1,014-image training set. It differs from the baseline in exactly one way — quantity of real labels (1,014 vs 252) — which is what makes it a clean upper bound.

**What makes it different from the other two.** The most _real_ labels of any model, and no pseudo-labels at all. It isolates the effect of label quantity with everything else held constant.

**Result.**

| Metric                           | Value            |
| -------------------------------- | ---------------- |
| Test accuracy                    | 90.54% (201/222) |
| Macro F1                         | 0.9051           |
| Defective F1                     | 0.9106           |
| Healthy F1                       | 0.8995           |
| False negatives (missed defects) | 14               |
| Total errors                     | 21               |

---

## Comparative Summary

| Model                         | Real labels | Pool size | Test acc. | Macro F1 | Def. F1 | Healthy F1 | False neg. | Errors |
| ----------------------------- | ----------- | --------- | --------- | -------- | ------- | ---------- | ---------- | ------ |
| Supervised baseline           | 252         | 252       | 89.19%    | 0.8915   | 0.898   | 0.885      | 15         | 24     |
| Semi-supervised               | 252         | 1,010     | 90.09%    | 0.9004   | 0.9076  | 0.8932     | 13         | 22     |
| Full supervised (upper bound) | 1,014       | 1,014     | 90.54%    | 0.9051   | 0.9106  | 0.8995     | 14         | 21     |

**How to read this in one breath.** Using the same 252 real labels as the baseline, the SSL model improves on every metric and closes most of the gap to the fully-labeled upper bound — despite the upper bound using roughly **4× as many human labels**. The SSL model lands just below the full model (0.9004 vs 0.9051 macro F1), which is the expected and sensible result: pseudo-labels are helpful but imperfect, so the method recovers most, not all, of the benefit of full labeling — at none of the labeling cost.

---

## Reproducibility Note

Iterative pseudo-labeling on a GPU is sensitive to run-to-run nondeterminism: CUDA does not guarantee bitwise-identical training across runs even with a fixed seed, so the exact pseudo-label counts and final pool size can vary slightly between runs. The figures in this document correspond to the model checkpoints currently in `models/` and the logs in `outputs/` (`ssl_log.csv`, `pseudo_label_accuracy.csv`). The **pattern** is stable across runs — high early pseudo-label accuracy, decay after the third iteration, and SSL improving on the baseline — even when individual counts shift. Determinism controls (`cudnn.deterministic`, `use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`) have been added to the training scripts so that future runs are reproducible.

## How to Reproduce

From the `scripts/` directory, with the virtual environment active:

```
# 1. Train the detector (once)
python train_yolo.py

# 2. Crop the dataset to fruit regions
python crop_for_classifier.py

# 3. Train the supervised baseline (set MODE="baseline" in train_resnet.py)
python train_resnet.py

# 4. Run the semi-supervised pseudo-labeling loop
python pseudo_label.py

# 5. Train the full-supervised upper bound (set MODE="full" in train_resnet.py)
python train_resnet.py

# 6. Evaluate any model against the test set
python evaluate.py
```
