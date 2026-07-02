import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = next(
    (c for c in (HERE, HERE.parent) if (c / "models").exists() and (c / "dataset").exists()),
    HERE,
)
TEST_DIR = ROOT / "dataset" / "test"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "money_shot"

NUM_CLASSES = 2
IMG_SIZE = 224
CLASS_NAMES = ["defective", "healthy"]  
DEFECTIVE, HEALTHY = 0, 1
TOP_N_COPY = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_resnet(path):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()


def predict(model, tensor):
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        conf, pred = torch.max(probs, 1)
    return int(pred.item()), float(conf.item())


print(f"Device: {DEVICE}")
print(f"Test set: {TEST_DIR}")
baseline = load_resnet(MODELS_DIR / "resnet_baseline.pth")
ssl_model = load_resnet(MODELS_DIR / "resnet_ssl.pth")

# ImageFolder gives (path, true_class_idx) for every test image.
test_data = datasets.ImageFolder(str(TEST_DIR))
print(f"Class order: {test_data.class_to_idx}")   
print(f"Test images: {len(test_data.samples)}\n")

base_correct = ssl_correct = 0
base_fn = ssl_fn = 0                 
recovered = []                      
regressed = []                      

for path, true in test_data.samples:
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    b_pred, b_conf = predict(baseline, x)
    s_pred, s_conf = predict(ssl_model, x)

    base_correct += (b_pred == true)
    ssl_correct += (s_pred == true)
    base_fn += (true == DEFECTIVE and b_pred == HEALTHY)
    ssl_fn += (true == DEFECTIVE and s_pred == HEALTHY)

    if true == DEFECTIVE and b_pred == HEALTHY and s_pred == DEFECTIVE:
        recovered.append((path, s_conf, b_conf))
    if b_pred == true and s_pred != true:
        regressed.append((path, true, s_pred, s_conf))

recovered.sort(key=lambda r: r[1], reverse=True)   # most SSL-confident first
n = len(test_data.samples)

print("=" * 68)
print("SANITY CHECK vs thesis numbers")
print("=" * 68)
print(f"  Baseline accuracy: {base_correct}/{n} = {base_correct/n*100:.2f}%  "
      f"({n - base_correct} errors)")
print(f"  SSL accuracy:      {ssl_correct}/{n} = {ssl_correct/n*100:.2f}%  "
      f"({n - ssl_correct} errors)")
print(f"  False negatives (missed defects) - Baseline: {base_fn}   SSL: {ssl_fn}")
if base_fn:
    print(f"  -> {base_fn - ssl_fn} fewer missed defects "
          f"({(base_fn - ssl_fn)/base_fn*100:.0f}% reduction)")
print()

print("=" * 68)
print(f"MONEY SHOTS - defects the SSL recovered ({len(recovered)} found)")
print("Baseline says HEALTHY (wrong); SSL says DEFECTIVE (right).")
print("Top of the list = your most convincing demo image.")
print("=" * 68)
for i, (path, s_conf, b_conf) in enumerate(recovered, 1):
    print(f"  {i}. {Path(path).name}")
    print(f"       SSL: DEFECTIVE {s_conf:.1%}   |   Baseline: HEALTHY {b_conf:.1%}")

if recovered:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, (path, _, _) in enumerate(recovered[:TOP_N_COPY], 1):
        shutil.copy(path, OUT_DIR / f"moneyshot_{i}_{Path(path).name}")
    print(f"\nTop {min(TOP_N_COPY, len(recovered))} copied to: {OUT_DIR}")

if regressed:
    print("\n" + "-" * 68)
    print(f"Heads-up - {len(regressed)} image(s) baseline got right but SSL missed:")
    for path, true, s_pred, s_conf in regressed:
        print(f"  {Path(path).name}: true={CLASS_NAMES[true]}, "
              f"SSL={CLASS_NAMES[s_pred]} {s_conf:.1%}")


