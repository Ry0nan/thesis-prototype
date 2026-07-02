"""
app.py - Kamias Surface Defect Detection (thesis demo)
======================================================
Two-stage pipeline wrapped in a Gradio web UI:

    upload image  ->  YOLOv8n finds + crops the fruit  ->  ResNet18 says
    DEFECTIVE / HEALTHY (with confidence)  ->  UI shows every step.

You can switch the ResNet on the fly (baseline / SSL / full) to demonstrate
the thesis comparison live.

Thesis: "A Data-Efficient Semi-Supervised Learning Framework for Automated
Kamias (Averrhoa bilimbi) Surface Defect Detection Using YOLOv8 and ResNet18."
Group 16, Mapua University - Ferrer, Lopez, Palma.

Run from the PROTOTYPE root (same folder level as models/ and scripts/):
    .\venv\Scripts\Activate.ps1
    python app.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

if not MODELS_DIR.exists() and (BASE_DIR.parent / "models").exists():
    MODELS_DIR = BASE_DIR.parent / "models"

YOLO_PATH = MODELS_DIR / "yolov8_kamias.pt"


RESNET_MODELS = {
    "SSL - best model (92.34%)": {
        "file": "resnet_ssl.pth",
        "acc": "92.34%",
        "macro_f1": "0.9224",
        "trained_on": "252 real labels + pseudo-labeled pool (grew to 1,037)",
    },
    "Baseline (89.19%)": {
        "file": "resnet_baseline.pth",
        "acc": "89.19%",
        "macro_f1": "0.8915",
        "trained_on": "252 labeled images only",
    },
    "Full supervised - upper bound (90.54%)": {
        "file": "resnet_full.pth",
        "acc": "90.54%",
        "macro_f1": "0.9051",
        "trained_on": "1,014 labeled images",
    },
}

NUM_CLASSES = 2
IMG_SIZE = 224

CLASS_NAMES = ["defective", "healthy"]

LOW_CONF_THRESHOLD = 0.70   
YOLO_CONF = 0.25            

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Identical preprocessing to your val_transform / classifier.py.
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =========================================================================
# MODEL LOADING  - happens ONCE at startup, not on every click
# =========================================================================
def load_resnet(weights_path):
    """Rebuild the training architecture and load your saved weights."""
    model = models.resnet18(weights=None)                    
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()                                             # inference mode
    return model


print("=" * 70)
print("KAMIAS DEFECT DETECTION - DEMO STARTUP")
print("=" * 70)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
print(f"Models: {MODELS_DIR}")

if not MODELS_DIR.exists():
    raise FileNotFoundError(
        f"models/ not found at {MODELS_DIR}. Put app.py in the PROTOTYPE root "
        f"(the folder that contains models/ and scripts/)."
    )

print("\nLoading YOLO detector...")
yolo_model = YOLO(str(YOLO_PATH))
print(f"  loaded {YOLO_PATH.name}")

print("Loading ResNet classifiers...")
loaded_models = {}
for label, info in RESNET_MODELS.items():
    loaded_models[label] = load_resnet(MODELS_DIR / info["file"])
    print(f"  loaded {info['file']:<22} ->  {label}")
print("\nModels ready. Starting UI...\n")


# =========================================================================
# STAGE 1 - YOLO: locate the fruit and crop it out of the original image
# =========================================================================
def detect_and_crop(pil_image):
    """Return (crop, detection_confidence), or (None, None) if nothing is found."""
    results = yolo_model(pil_image, imgsz=640, conf=YOLO_CONF, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None, None

    confs = boxes.conf.cpu().numpy()
    best = int(confs.argmax())                                  # most confident box
    x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)

    w, h = pil_image.size                                       # clamp to image edges
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return pil_image.crop((x1, y1, x2, y2)), float(confs[best])


# =========================================================================
# STAGE 2 - ResNet: classify the crop
# =========================================================================
def classify_crop(crop_pil, model):
    """Return (label, confidence, {class: prob})."""
    x = transform(crop_pil).unsqueeze(0).to(DEVICE)   # -> [1, 3, 224, 224]
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)        # logits -> probabilities
        conf, pred = torch.max(probs, 1)              # winning class + its prob
    label = CLASS_NAMES[pred.item()]
    all_probs = {CLASS_NAMES[i]: float(probs[0, i]) for i in range(NUM_CLASSES)}
    return label, conf.item(), all_probs


# =========================================================================
# FULL PIPELINE  - Gradio calls this on every "Analyze" click
# =========================================================================
def run_pipeline(image, model_choice):
    if image is None:
        return None, {}, "### Upload an image first."

    image = image.convert("RGB")                      # guard against PNG alpha, etc.

    crop, det_conf = detect_and_crop(image)
    fallback_note = ""
    if crop is None:                                  # no fruit found -> don't dead-end
        crop = image
        fallback_note = "\n\n> No fruit detected by YOLO - classified the whole image."

    label, confidence, all_probs = classify_crop(crop, loaded_models[model_choice])

    healthy = label == "healthy"
    verdict = "HEALTHY" if healthy else "DEFECTIVE"
    color = "#1a7f37" if healthy else "#cf222e"
    info = RESNET_MODELS[model_choice]

    det_line = f"**Fruit detected** - YOLO confidence {det_conf:.1%}  \n" if det_conf is not None else ""
    low_conf = "\n\n> Low classifier confidence (<70%) - borderline case." if confidence < LOW_CONF_THRESHOLD else ""

    result_md = (
        f"<h1 style='color:{color};margin:0'>{verdict}</h1>\n\n"
        f"**Classifier confidence:** {confidence:.1%}  \n"
        f"{det_line}"
        f"**Model used:** {model_choice}  \n"
        f"<sub>trained on {info['trained_on']} - test acc {info['acc']} - Macro F1 {info['macro_f1']}</sub>"
        f"{low_conf}{fallback_note}"
    )
    return crop, all_probs, result_md


# =========================================================================
# GRADIO UI
# =========================================================================
DEFAULT_MODEL = "SSL - best model (92.34%)"

with gr.Blocks(title="Kamias Defect Detection") as demo:
    gr.Markdown(
        "# Kamias Surface Defect Detection\n"
        "**YOLOv8n** locates the fruit, then **ResNet18** classifies the crop as "
        "defective or healthy. Switch the classifier to compare the three thesis "
        "models on the same image."
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload a kamias image")
            model_choice = gr.Radio(
                choices=list(RESNET_MODELS.keys()),
                value=DEFAULT_MODEL,
                label="Classifier model",
            )
            run_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            crop_image = gr.Image(label="YOLO crop (this is what ResNet sees)")
            result_box = gr.Markdown()
            probs_label = gr.Label(num_top_classes=2, label="Class confidence")

    run_btn.click(
        run_pipeline,
        inputs=[input_image, model_choice],
        outputs=[crop_image, probs_label, result_box],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, show_error=True)