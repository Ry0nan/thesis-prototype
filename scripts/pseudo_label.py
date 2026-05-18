"""
Semi-Supervised Learning via iterative pseudo-labeling.

Algorithm:
1. Load the supervised baseline model trained on labeled 25%.
2. Predict on the unlabeled pool with confidence scores.
3. For each prediction above CONFIDENCE_THRESHOLD, copy the image into the
   labeled pool under the predicted class (this is the "pseudo-label").
4. Retrain ResNet18 on the now-expanded labeled pool.
5. Repeat from step 2 with the new model.
6. Stop when no new pseudo-labels are added OR MAX_ITERATIONS reached.

The "__true_X__" prefix in unlabeled filenames lets us measure pseudo-label
accuracy (did we recover the original class?) without the model ever seeing
the true label.

Outputs:
- models/resnet_ssl.pth                — final SSL model
- outputs/ssl_log.csv                  — per-iteration stats for Chapter 4
- outputs/pseudo_label_accuracy.csv    — which pseudo-labels matched true class
"""

import os
import re
import csv
import shutil
import random
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# =========================
# REPRODUCIBILITY
# =========================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# CONFIG
# =========================
LABELED_DIR = "../dataset/train_labeled"   # gets EXPANDED across iterations
UNLABELED_DIR = "../dataset/unlabeled"     # gets SHRUNK as pseudo-labels move out
VAL_DIR = "../dataset/val"

BASELINE_MODEL_PATH = "../models/resnet_baseline.pth"
SSL_MODEL_PATH = "../models/resnet_ssl.pth"

LOG_PATH = "../outputs/ssl_log.csv"
ACCURACY_LOG_PATH = "../outputs/pseudo_label_accuracy.csv"

# SSL hyperparameters
CONFIDENCE_THRESHOLD = 0.90  # Only accept pseudo-labels above this confidence
MAX_ITERATIONS = 5
MIN_NEW_LABELS_PER_ITER = 5  # Stop if fewer than this many new pseudo-labels added

# Training hyperparameters (per-iteration retraining)
BATCH_SIZE = 8
EPOCHS_PER_ITER = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 3
IMG_SIZE = 224

CLASS_NAMES = ['healthy', 'major', 'minor']  # alphabetical, matches ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORMS (match train_resnet.py exactly)
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# =========================
# HELPERS
# =========================
def extract_true_label(filename):
    """
    Extract original class from filename like '__true_healthy__IMG_xyz.jpg'.
    Returns the class string or 'ambiguous' if encoded as such, or None if pattern not found.
    """
    match = re.match(r'__true_([a-z_]+)__', filename)
    if match:
        return match.group(1)
    return None

def load_model(checkpoint_path=None):
    """Build ResNet18 with 3-class head; optionally load pretrained or fine-tuned weights."""
    model = models.resnet18(weights="DEFAULT" if checkpoint_path is None else None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"  Loaded weights from {checkpoint_path}")
    model = model.to(device)
    return model

def predict_unlabeled_pool(model, unlabeled_dir):
    """
    Run inference on every image in the unlabeled folder.
    Returns list of (filename, predicted_class, confidence, all_probs).
    """
    model.eval()
    results = []

    files = [f for f in os.listdir(unlabeled_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    with torch.no_grad():
        for filename in files:
            path = os.path.join(unlabeled_dir, filename)
            try:
                image = Image.open(path).convert("RGB")
                tensor = val_transform(image).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, predicted = torch.max(probs, 1)
                pred_class = CLASS_NAMES[predicted.item()]
                results.append((filename, pred_class, conf.item(),
                                probs[0].cpu().numpy().tolist()))
            except Exception as e:
                print(f"  ERROR predicting {filename}: {e}")

    return results

def train_one_iteration(labeled_dir, val_dir, save_path):
    """
    Train ResNet18 on the current labeled pool. Returns best val accuracy.
    Replicates train_resnet.py logic but as a function we can call repeatedly.
    """
    train_data = datasets.ImageFolder(labeled_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Class weights for current labeled pool
    train_counts = Counter([label for _, label in train_data.samples])
    total = sum(train_counts.values())
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * train_counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float
    ).to(device)

    model = load_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_ITER)

    best_val_acc = 0.0
    for epoch in range(EPOCHS_PER_ITER):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        correct = total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_acc = 100 * correct / total_val
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(f"    Epoch {epoch+1:2d}/{EPOCHS_PER_ITER} | Val Acc: {val_acc:.2f}%")

    return best_val_acc

# =========================
# MAIN SSL LOOP
# =========================
def main():
    print("=" * 70)
    print("SEMI-SUPERVISED LEARNING — ITERATIVE PSEUDO-LABELING")
    print("=" * 70)
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Max iterations:       {MAX_ITERATIONS}")
    print(f"Min new labels/iter:  {MIN_NEW_LABELS_PER_ITER}")
    print(f"Epochs per iter:      {EPOCHS_PER_ITER}")
    print(f"Device:               {device}\n")

    # Sanity checks
    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"ERROR: Baseline model not found at {BASELINE_MODEL_PATH}")
        print("Run train_resnet.py with MODE='baseline' first.")
        return

    if not os.path.exists(LABELED_DIR):
        print(f"ERROR: Labeled folder not found at {LABELED_DIR}")
        print("Run prepare_ssl_split.py first.")
        return

    # Set up output logs
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_file = open(LOG_PATH, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        'iteration', 'labeled_size_before', 'predictions_made',
        'predictions_above_threshold', 'new_labels_added',
        'new_healthy', 'new_minor', 'new_major',
        'val_acc', 'unlabeled_remaining'
    ])

    accuracy_log = open(ACCURACY_LOG_PATH, 'w', newline='', encoding='utf-8')
    accuracy_writer = csv.writer(accuracy_log)
    accuracy_writer.writerow([
        'iteration', 'filename', 'true_class', 'predicted_class', 'confidence', 'correct'
    ])

    # Start by copying the baseline as the initial SSL model
    shutil.copy(BASELINE_MODEL_PATH, SSL_MODEL_PATH)
    print(f"Initialized SSL model from baseline: {SSL_MODEL_PATH}\n")

    # =========================
    # ITERATION LOOP
    # =========================
    for iteration in range(1, MAX_ITERATIONS + 1):
        print("=" * 70)
        print(f"ITERATION {iteration}/{MAX_ITERATIONS}")
        print("=" * 70)

        # ----- STEP 1: Count current state -----
        labeled_size_before = sum(
            len(os.listdir(os.path.join(LABELED_DIR, cls)))
            for cls in CLASS_NAMES
            if os.path.exists(os.path.join(LABELED_DIR, cls))
        )
        unlabeled_files = [f for f in os.listdir(UNLABELED_DIR)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        unlabeled_size = len(unlabeled_files)

        print(f"  Labeled pool:   {labeled_size_before}")
        print(f"  Unlabeled pool: {unlabeled_size}\n")

        if unlabeled_size == 0:
            print("  No unlabeled images remain. Stopping.")
            break

        # ----- STEP 2: Load current model and predict on unlabeled pool -----
        print(f"  Loading model: {SSL_MODEL_PATH}")
        model = load_model(SSL_MODEL_PATH)
        print(f"  Running predictions on {unlabeled_size} unlabeled images...")
        predictions = predict_unlabeled_pool(model, UNLABELED_DIR)

        # ----- STEP 3: Filter by confidence and add as pseudo-labels -----
        high_conf = [p for p in predictions if p[2] >= CONFIDENCE_THRESHOLD]
        print(f"  Predictions above threshold ({CONFIDENCE_THRESHOLD}): "
              f"{len(high_conf)}/{len(predictions)}")

        if len(high_conf) < MIN_NEW_LABELS_PER_ITER:
            print(f"  Fewer than {MIN_NEW_LABELS_PER_ITER} new labels. Stopping.")
            log_writer.writerow([
                iteration, labeled_size_before, len(predictions),
                len(high_conf), 0, 0, 0, 0, 'N/A', unlabeled_size
            ])
            break

        # Copy each high-confidence image into the labeled pool, remove from unlabeled
        class_added = Counter()
        for filename, pred_class, conf, _ in high_conf:
            src_path = os.path.join(UNLABELED_DIR, filename)
            dst_path = os.path.join(LABELED_DIR, pred_class, filename)
            shutil.move(src_path, dst_path)
            class_added[pred_class] += 1

            # Log pseudo-label vs true class for analysis
            true_class = extract_true_label(filename)
            if true_class:
                correct = (true_class == pred_class) if true_class != 'ambiguous' else 'N/A'
                accuracy_writer.writerow([
                    iteration, filename, true_class, pred_class, f"{conf:.4f}", correct
                ])

        print(f"  New pseudo-labels added:")
        for cls in CLASS_NAMES:
            print(f"    {cls:<10} +{class_added[cls]}")

        # ----- STEP 4: Retrain on expanded labeled pool -----
        print(f"\n  Retraining on expanded labeled pool...")
        val_acc = train_one_iteration(LABELED_DIR, VAL_DIR, SSL_MODEL_PATH)
        print(f"  Best val accuracy this iteration: {val_acc:.2f}%")

        # ----- STEP 5: Log iteration stats -----
        log_writer.writerow([
            iteration, labeled_size_before, len(predictions),
            len(high_conf), sum(class_added.values()),
            class_added['healthy'], class_added['minor'], class_added['major'],
            f"{val_acc:.2f}",
            unlabeled_size - sum(class_added.values())
        ])
        log_file.flush()

        print()

    # =========================
    # WRAP UP
    # =========================
    log_file.close()
    accuracy_log.close()

    # Final state
    final_labeled = sum(
        len(os.listdir(os.path.join(LABELED_DIR, cls)))
        for cls in CLASS_NAMES
        if os.path.exists(os.path.join(LABELED_DIR, cls))
    )
    final_unlabeled = len([f for f in os.listdir(UNLABELED_DIR)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print("=" * 70)
    print("SSL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final labeled pool size:   {final_labeled}")
    print(f"Final unlabeled pool size: {final_unlabeled}")
    print(f"Final SSL model:           {SSL_MODEL_PATH}")
    print(f"Iteration log:             {LOG_PATH}")
    print(f"Pseudo-label accuracy:     {ACCURACY_LOG_PATH}")
    print()
    print("Next: run evaluate.py with the SSL model to compare against baseline.")

if __name__ == "__main__":
    main()