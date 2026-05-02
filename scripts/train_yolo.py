"""
Fine-tune YOLOv8 on annotated Kamias images.

Prerequisites:
1. Annotated images in YOLO format using a tool like Roboflow or LabelImg.
2. Folder structure:
   dataset/yolo/
   ├── images/
   │   ├── train/   <- training images (.jpg, .png)
   │   └── val/     <- validation images
   └── labels/
       ├── train/   <- one .txt per image (same filename, different extension)
       └── val/

3. Each label file contains lines: <class_id> <x_center> <y_center> <width> <height>
   All values normalized to [0, 1]. For Kamias, class_id is always 0.

4. A data config YAML file (this script creates one automatically if missing).

After training, the best weights are saved to models/yolov8_kamias.pt
and main.py will automatically pick them up.
"""

import os
import yaml
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
DATA_DIR = "../dataset/yolo"
DATA_YAML_PATH = "../dataset/yolo/kamias.yaml"
PRETRAINED_MODEL = "../models/yolov8n.pt"
OUTPUT_MODEL_PATH = "../models/yolov8_kamias.pt"

# Training hyperparameters
EPOCHS = 100
IMG_SIZE = 640         # YOLOv8 standard input size — do NOT change to 224
BATCH_SIZE = 16        # If you hit OOM (out of memory), drop to 8 or 4
PATIENCE = 20          # Stop early if val mAP doesn't improve for this many epochs
DEVICE = 0             # GPU 0; use "cpu" if no CUDA

# Class names — currently single-class detection (just locate Kamias).
# Classification of healthy/minor/major happens downstream in ResNet18.
CLASS_NAMES = ['kamias']

# =========================
# CREATE DATA YAML
# =========================
def ensure_data_yaml():
    """Create the YOLO data config YAML if it doesn't exist."""
    if os.path.exists(DATA_YAML_PATH):
        print(f"Using existing data config: {DATA_YAML_PATH}")
        return

    abs_data_dir = os.path.abspath(DATA_DIR)
    config = {
        'path': abs_data_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)},
    }

    os.makedirs(os.path.dirname(DATA_YAML_PATH), exist_ok=True)
    with open(DATA_YAML_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created data config at: {DATA_YAML_PATH}")
    print(f"Contents:")
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

# =========================
# VALIDATE DATASET STRUCTURE
# =========================
def validate_dataset():
    """Check that the YOLO dataset folders exist and contain matching images/labels."""
    required_dirs = [
        f"{DATA_DIR}/images/train",
        f"{DATA_DIR}/images/val",
        f"{DATA_DIR}/labels/train",
        f"{DATA_DIR}/labels/val",
    ]

    missing = [d for d in required_dirs if not os.path.exists(d)]
    if missing:
        print("ERROR: Required folders missing:")
        for d in missing:
            print(f"  - {d}")
        print("\nCreate them and add your annotated data before running this script.")
        return False

    train_images = [f for f in os.listdir(f"{DATA_DIR}/images/train")
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    train_labels = [f for f in os.listdir(f"{DATA_DIR}/labels/train")
                    if f.endswith('.txt')]
    val_images = [f for f in os.listdir(f"{DATA_DIR}/images/val")
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    val_labels = [f for f in os.listdir(f"{DATA_DIR}/labels/val")
                  if f.endswith('.txt')]

    print(f"Dataset summary:")
    print(f"  Train: {len(train_images)} images, {len(train_labels)} labels")
    print(f"  Val:   {len(val_images)} images, {len(val_labels)} labels")

    if len(train_images) == 0:
        print("ERROR: No training images found.")
        return False
    if len(val_images) == 0:
        print("ERROR: No validation images found.")
        return False

    if len(train_images) != len(train_labels):
        print("WARNING: Train image and label counts don't match.")
        print("  Each image should have a corresponding .txt label file with the same name.")
    if len(val_images) != len(val_labels):
        print("WARNING: Val image and label counts don't match.")

    return True

# =========================
# TRAIN
# =========================
def main():
    print("=" * 70)
    print("YOLOv8 Fine-Tuning for Kamias Detection")
    print("=" * 70)

    # Step 1: Validate dataset
    if not validate_dataset():
        print("\nFix dataset issues before training.")
        return

    # Step 2: Ensure data config exists
    ensure_data_yaml()

    # Step 3: Verify pretrained weights exist
    if not os.path.exists(PRETRAINED_MODEL):
        print(f"\nERROR: Pretrained model not found at {PRETRAINED_MODEL}")
        print("Download yolov8n.pt from Ultralytics or place your existing one there.")
        return

    # Step 4: Train
    print(f"\nStarting training:")
    print(f"  Pretrained:   {PRETRAINED_MODEL}")
    print(f"  Epochs:       {EPOCHS}")
    print(f"  Image size:   {IMG_SIZE}")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Patience:     {PATIENCE}")
    print(f"  Device:       {DEVICE}\n")

    model = YOLO(PRETRAINED_MODEL)

    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        project='../runs',          # Where checkpoints go
        name='kamias_yolov8n',      # Experiment name (creates runs/kamias_yolov8n/)
        exist_ok=False,             # If folder exists, increment to kamias_yolov8n2/
        plots=True,                 # Save training plots (loss curves, PR curve, etc.)
        save=True,
    )

    # Step 5: Copy best weights to models/ for main.py to find
    best_weights = f"../runs/kamias_yolov8n/weights/best.pt"
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, OUTPUT_MODEL_PATH)
        print(f"\nBest weights copied to: {OUTPUT_MODEL_PATH}")
        print(f"main.py will now automatically use these for cropping.")
    else:
        print(f"\nWARNING: Could not find best weights at {best_weights}")
        print(f"Check the runs/ folder manually.")

    print(f"\nTraining complete.")
    print(f"View training plots in: ../runs/kamias_yolov8n/")

if __name__ == "__main__":
    main()