import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# =========================
# CONFIG
# =========================
TEST_DIR = "../dataset/test"
MODEL_PATH = "../models/resnet_kamias.pth"
NUM_CLASSES = 3
IMG_SIZE = 224
BATCH_SIZE = 16

# Identifier used in the output report (helps you compare runs).
# Examples: "Supervised Baseline", "SSL Iteration 1", "SSL Final"
EXPERIMENT_NAME = "Supervised Baseline"

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# =========================
# TRANSFORM
# Must match val_transform in train_resnet.py exactly.
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# =========================
# LOAD TEST DATA
# =========================
if not os.path.exists(TEST_DIR):
    print(f"Test directory not found: {TEST_DIR}")
    exit(1)

test_data = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class_names = test_data.classes
print(f"Test classes: {class_names}")
print(f"Test samples: {len(test_data)}")
from collections import Counter
test_dist = Counter([label for _, label in test_data.samples])
print(f"Test distribution: {dict({class_names[i]: test_dist[i] for i in range(NUM_CLASSES)})}\n")

# =========================
# LOAD MODEL
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print(f"Model loaded from: {MODEL_PATH}\n")

# =========================
# RUN EVALUATION
# =========================
all_preds = []
all_labels = []
all_confidences = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_confidences.extend(confs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_confidences = np.array(all_confidences)

# =========================
# REPORT
# =========================
print("=" * 70)
print(f"EVALUATION REPORT: {EXPERIMENT_NAME}")
print("=" * 70)

# Overall accuracy
accuracy = (all_preds == all_labels).mean() * 100
print(f"\nOverall Accuracy: {accuracy:.2f}% ({(all_preds == all_labels).sum()}/{len(all_labels)})")

# Per-class precision, recall, F1 — the metrics your thesis promises
print("\nPer-Class Metrics:")
print("-" * 70)
print(classification_report(
    all_labels, all_preds,
    target_names=class_names,
    digits=4,
    zero_division=0,
))

# Macro vs weighted F1 — important distinction for imbalanced data
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, average=None, zero_division=0
)
macro_f1 = f1.mean()
weighted_f1 = (f1 * support).sum() / support.sum()

print(f"Macro F1 (unweighted avg):    {macro_f1:.4f}")
print(f"Weighted F1 (by class size):  {weighted_f1:.4f}")
print("  -> Macro F1 weights all classes equally (good for imbalanced data)")
print("  -> Use Macro F1 as your headline number for the thesis comparison\n")

# Confusion matrix
print("Confusion Matrix:")
print("-" * 70)
cm = confusion_matrix(all_labels, all_preds)
header = "Actual \\ Pred  " + "  ".join(f"{name:>8}" for name in class_names)
print(header)
for i, name in enumerate(class_names):
    row = f"{name:<14} " + "  ".join(f"{cm[i, j]:>8}" for j in range(NUM_CLASSES))
    print(row)
print("\nReading guide: rows = actual class, columns = predicted class.")
print("  Diagonal = correct. Off-diagonal = where the model confuses classes.\n")

# Confidence breakdown
print("Confidence Analysis:")
print("-" * 70)
correct_mask = all_preds == all_labels
print(f"Correct predictions:   avg confidence = {all_confidences[correct_mask].mean():.2%}")
if (~correct_mask).any():
    print(f"Incorrect predictions: avg confidence = {all_confidences[~correct_mask].mean():.2%}")
    print("  -> If incorrect predictions still have high confidence, the model is")
    print("     overconfident and pseudo-labels will be unreliable.\n")
else:
    print("All predictions correct. Cannot compute incorrect confidence.\n")

print("=" * 70)