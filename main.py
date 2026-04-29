from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Input image (use your own later)
image_path = "bus.jpg"
image = cv2.imread(image_path)

# Run detection
results = model(image)

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Get detections
boxes = results[0].boxes.xyxy.cpu().numpy()

# Crop detected objects
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]

    filename = f"outputs/crop_{i}.jpg"
    cv2.imwrite(filename, crop)

    print(f"Saved: {filename}")

print("Cropping complete.")