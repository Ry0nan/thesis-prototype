# Kamias Surface Defect Detection

A data-efficient semi-supervised learning framework for automated detection
of surface defects in Kamias (*Averrhoa bilimbi*), built with YOLOv8 and ResNet18.

Undergraduate thesis — Mapua Institute of Technology, BS Information Technology.

## Overview

The system uses a two-stage pipeline. A YOLOv8n detector locates the Kamias
fruit in an image; the detected region is cropped and passed to a ResNet18
classifier that labels it as **healthy** or **defective**. The project compares
a fully-supervised baseline against a semi-supervised approach that expands a
small labeled set using iterative pseudo-labeling.

## Pipeline

1. **Detection** — YOLOv8n locates the Kamias fruit (single-class detection).
2. **Cropping** — the detected bounding box is cropped to isolate the fruit.
3. **Classification** — ResNet18 classifies each crop as healthy or defective.
4. **Semi-supervised learning** — starting from a 25% labeled subset, the model
   iteratively pseudo-labels high-confidence predictions (threshold 0.90) from
   the unlabeled pool and retrains, up to 5 iterations.

## Dataset

- 1,449 labeled images (657 healthy, 792 defective), single fruit per image.
- 81 images held as ambiguous, used only in the unlabeled pool.
- Stratified 70/15/15 train/validation/test split by class and camera source.
- Two-class annotation protocol validated by inter-rater agreement
  (Cohen's kappa = 0.65, substantial agreement).

## Environment

- Python with PyTorch (CUDA 12.8 build), torchvision, ultralytics, OpenCV,
  scikit-learn.
- Developed on Windows with an NVIDIA RTX 5060 GPU.
- A virtual environment (`venv/`) is used; activate before running scripts.

## Repository structure

- `scripts/` — all pipeline scripts (detection, cropping, splitting,
  training, evaluation, semi-supervised learning, reliability testing).
- `dataset/` — raw images, cropped images, and generated splits (not tracked).
- `models/` — trained model weights (not tracked).
- `outputs/` — logs and metrics from semi-supervised runs (not tracked).
- `runs/` — YOLO training outputs and plots (not tracked).

## Scripts

| Script | Purpose |
|--------|---------|
| `count_dataset.py` | Inventory class distribution and camera balance |
| `crop_for_classifier.py` | Crop fruit regions from raw images using trained YOLO |
| `split_dataset.py` | Stratified train/validation/test split |
| `prepare_ssl_split.py` | Carve training data into labeled/unlabeled subsets |
| `train_yolo.py` | Fine-tune YOLOv8 for Kamias detection |
| `train_resnet.py` | Train ResNet18 (baseline / SSL / full modes) |
| `pseudo_label.py` | Iterative semi-supervised pseudo-labeling |
| `evaluate.py` | Evaluate a model on the test set |
| `classifier.py` | Run classification on individual crops |
| `main.py` | End-to-end detection and cropping |
| `sample_for_kappa.py` | Sample images for inter-rater reliability testing |
| `compute_kappa.py` | Compute Cohen's kappa between annotators |
| `check_pseudo.py` | Report pseudo-label accuracy per iteration |

## Status

Core experiments complete. Both the supervised baseline and the
semi-supervised model have been trained and evaluated.