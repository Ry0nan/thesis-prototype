"""
Crop Kamias fruits from raw images using the fine-tuned YOLO model,
preserving class labels by reading from class subfolders.

Input:  dataset/raw/{healthy,defective}/
Output: dataset/cropped/{healthy,defective}/

Each crop keeps the class of the folder it came from. These crops then
feed split_dataset.py-style organization for ResNet training.
"""

import os
import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
RAW_DIR = "../dataset/raw"
CROPPED_DIR = "../dataset/cropped"
MODEL_PATH = "../models/yolov8_kamias.pt"
CLASS_NAMES = ['healthy', 'defective', '_ambiguous']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

CONFIDENCE_THRESHOLD = 0.50  # fine-tuned model is confident; 0.5 filters weak detections
PADDING = 10

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: YOLO model not found at {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    print(f"Loaded fine-tuned YOLO: {MODEL_PATH}\n")

    total_images = 0
    total_crops = 0
    no_detection = 0
    multi_detection = 0

    for cls in CLASS_NAMES:
        src_folder = os.path.join(RAW_DIR, cls)
        dst_folder = os.path.join(CROPPED_DIR, cls)
        os.makedirs(dst_folder, exist_ok=True)

        if not os.path.exists(src_folder):
            print(f"WARNING: Source folder missing: {src_folder}")
            continue

        files = sorted([f for f in os.listdir(src_folder)
                        if f.lower().endswith(IMAGE_EXTENSIONS)])
        print(f"Processing '{cls}': {len(files)} images...")

        cls_crops = 0
        for filename in files:
            path = os.path.join(src_folder, filename)
            img = cv2.imread(path)
            if img is None:
                print(f"  Could not read: {filename}")
                continue
            total_images += 1

            h, w = img.shape[:2]
            results = model(path, conf=CONFIDENCE_THRESHOLD, verbose=False)
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                no_detection += 1
                continue

            if len(boxes) > 1:
                multi_detection += 1

            stem = os.path.splitext(filename)[0]
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(w, x2 + PADDING)
                y2 = min(h, y2 + PADDING)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                save_path = os.path.join(dst_folder, f"{stem}_crop_{i}.jpg")
                cv2.imwrite(save_path, crop)
                cls_crops += 1
                total_crops += 1

        print(f"  -> {cls_crops} crops saved to {dst_folder}\n")

    print("=" * 60)
    print("CROPPING COMPLETE")
    print("=" * 60)
    print(f"Total images processed:    {total_images}")
    print(f"Total crops saved:         {total_crops}")
    print(f"Images with no detection:  {no_detection}")
    print(f"Images with 2+ detections: {multi_detection}")
    if no_detection > 0:
        print(f"\nNote: {no_detection} images produced no crop. If this number is")
        print(f"high, lower CONFIDENCE_THRESHOLD or inspect those images.")

if __name__ == "__main__":
    main()