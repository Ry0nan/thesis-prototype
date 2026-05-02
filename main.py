import os
import argparse
import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
INPUT_FOLDER = "dataset/raw"
OUTPUT_FOLDER = "outputs"

# Model paths (in priority order):
# 1. Fine-tuned Kamias YOLO (used after train_yolo.py finishes)
# 2. Generic YOLOv8n (fallback for testing — detects COCO classes, not Kamias)
FINETUNED_MODEL = "models/yolov8_kamias.pt"
FALLBACK_MODEL = "models/yolov8n.pt"

# Detection settings
CONFIDENCE_THRESHOLD = 0.25  # Lower = more detections, more false positives
PADDING = 10                 # Pixels of padding around each crop
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# =========================
# MODEL SELECTION
# =========================
def select_model():
    """Use fine-tuned Kamias model if available, otherwise fall back to generic YOLOv8n."""
    if os.path.exists(FINETUNED_MODEL):
        print(f"Using fine-tuned Kamias model: {FINETUNED_MODEL}")
        return YOLO(FINETUNED_MODEL), True
    elif os.path.exists(FALLBACK_MODEL):
        print(f"WARNING: Fine-tuned model not found.")
        print(f"Falling back to generic COCO YOLOv8n: {FALLBACK_MODEL}")
        print(f"Detections will not be Kamias-specific until you run train_yolo.py.\n")
        return YOLO(FALLBACK_MODEL), False
    else:
        raise FileNotFoundError(
            f"No YOLO model found. Expected one of:\n"
            f"  - {FINETUNED_MODEL}\n"
            f"  - {FALLBACK_MODEL}"
        )

# =========================
# CROP FUNCTION
# =========================
def crop_detections(image_path, model, output_dir, padding=PADDING):
    """Run YOLO on an image and save each detected box as a cropped image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Could not read image: {image_path}")
        return 0

    h, w = img.shape[:2]
    results = model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0

    stem = os.path.splitext(os.path.basename(image_path))[0]
    saved_count = 0

    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        # Apply padding while staying inside image bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        save_path = os.path.join(output_dir, f"{stem}_crop_{i}.jpg")
        cv2.imwrite(save_path, crop)
        saved_count += 1

    return saved_count

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Detect and crop Kamias from images.")
    parser.add_argument("--input", default=INPUT_FOLDER, help="Folder with input images")
    parser.add_argument("--output", default=OUTPUT_FOLDER, help="Folder for cropped outputs")
    parser.add_argument("--clear", action="store_true", help="Clear output folder before running")
    args = parser.parse_args()

    # Resolve relative paths from script location's parent (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_folder = os.path.join(project_root, args.input)
    output_folder = os.path.join(project_root, args.output)

    if not os.path.exists(input_folder):
        print(f"Input folder not found: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Optional: clear previous crops
    if args.clear:
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))
        print(f"Cleared output folder: {output_folder}")

    # Load model
    model, is_finetuned = select_model()

    # Process all images
    files = sorted([f for f in os.listdir(input_folder)
                    if f.lower().endswith(IMAGE_EXTENSIONS)])

    if not files:
        print(f"No images found in {input_folder}")
        return

    print(f"\nProcessing {len(files)} images from {input_folder}...")
    print(f"Saving crops to {output_folder}\n")

    total_crops = 0
    images_with_no_detections = 0

    for filename in files:
        path = os.path.join(input_folder, filename)
        n = crop_detections(path, model, output_folder)
        total_crops += n
        if n == 0:
            images_with_no_detections += 1
        print(f"  {filename}: {n} crop(s)")

    print(f"\nDone.")
    print(f"  Total images processed: {len(files)}")
    print(f"  Total crops saved:      {total_crops}")
    print(f"  Images with no detections: {images_with_no_detections}")

    if not is_finetuned:
        print(f"\nReminder: Using generic COCO model. Crops may not be Kamias-specific.")
        print(f"Run train_yolo.py once you have annotated data to fix this.")

if __name__ == "__main__":
    main()