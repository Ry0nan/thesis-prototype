import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATH = "../models/resnet_kamias.pth"
CROPS_FOLDER = "../outputs"
NUM_CLASSES = 3
IMG_SIZE = 224

# Confidence threshold for flagging uncertain predictions.
# Predictions below this get a "[LOW CONFIDENCE]" tag in output.
CONFIDENCE_THRESHOLD = 0.70

# Class names MUST match training order from ImageFolder (alphabetical).
# Verify with the "Classes:" line printed during training.
CLASS_NAMES = ['healthy', 'major', 'minor']

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# TRANSFORM
# CRITICAL: This MUST match val_transform in train_resnet.py exactly.
# Mismatched normalization between training and inference produces garbage predictions.
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

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
# PREDICTION FUNCTION
# Returns (label, confidence, all_class_probabilities)
# =========================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        conf, predicted = torch.max(probs, 1)

    label = CLASS_NAMES[predicted.item()]
    confidence = conf.item()
    all_probs = {CLASS_NAMES[i]: probs[0, i].item() for i in range(NUM_CLASSES)}

    return label, confidence, all_probs

# =========================
# RUN PREDICTIONS ON CROPS
# =========================
def main():
    if not os.path.exists(CROPS_FOLDER):
        print(f"Crops folder not found: {CROPS_FOLDER}")
        return

    files = sorted([f for f in os.listdir(CROPS_FOLDER)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not files:
        print(f"No images found in: {CROPS_FOLDER}")
        return

    print(f"Running predictions on {len(files)} images...\n")
    print(f"{'Filename':<50} {'Label':<10} {'Confidence':<12} Notes")
    print("-" * 95)

    # Track distribution of predictions
    label_counts = {name: 0 for name in CLASS_NAMES}
    low_conf_count = 0

    for filename in files:
        path = os.path.join(CROPS_FOLDER, filename)
        try:
            label, conf, probs = predict_image(path)
            label_counts[label] += 1

            note = ""
            if conf < CONFIDENCE_THRESHOLD:
                note = "[LOW CONFIDENCE]"
                low_conf_count += 1

            print(f"{filename:<50} {label:<10} {conf:.2%}      {note}")
        except Exception as e:
            print(f"{filename:<50} ERROR: {e}")

    # Summary
    print("-" * 95)
    print(f"\nPrediction summary:")
    for label, count in label_counts.items():
        pct = (count / len(files)) * 100 if files else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    print(f"  Low confidence (<{CONFIDENCE_THRESHOLD:.0%}): {low_conf_count}")

if __name__ == "__main__":
    main()