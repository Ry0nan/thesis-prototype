import os
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
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
TRAIN_DIR = "../dataset/train"
VAL_DIR = "../dataset/val"
MODEL_SAVE_PATH = "../models/resnet_kamias.pth"

BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 3
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =========================
# TRANSFORMS
# ImageNet normalization is REQUIRED for pretrained ResNet to work correctly.
# Without this, predictions collapse onto one class.
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# =========================
# LOAD DATA
# =========================
train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_data = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\nClasses: {train_data.classes}")
print(f"Class-to-index: {train_data.class_to_idx}")
print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")

# Class distribution check
train_counts = Counter([label for _, label in train_data.samples])
distribution = {train_data.classes[i]: train_counts[i] for i in range(NUM_CLASSES)}
print(f"Train distribution: {distribution}")

# =========================
# CLASS WEIGHTS
# Handles imbalance — critical for catching minor defects (your thesis claim).
# =========================
total_samples = sum(train_counts.values())
class_weights = torch.tensor(
    [total_samples / (NUM_CLASSES * train_counts[i]) for i in range(NUM_CLASSES)],
    dtype=torch.float
).to(device)
print(f"Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

# =========================
# MODEL
# ResNet18 with ImageNet pretrained weights, final layer replaced for 3 classes.
# =========================
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# =========================
# LOSS + OPTIMIZER + SCHEDULER
# =========================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =========================
# TRAINING LOOP
# =========================
print("\nStarting training...\n")
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # ----- TRAIN -----
    model.train()
    total_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = 100 * train_correct / train_total
    avg_loss = total_loss / len(train_loader)

    # ----- VALIDATE -----
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = 100 * val_correct / val_total

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Acc: {val_acc:.2f}% | "
          f"LR: {current_lr:.6f}")

    # Save best model only
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  -> New best model saved (val_acc: {val_acc:.2f}%)")

print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")
print(f"Model saved at: {MODEL_SAVE_PATH}")