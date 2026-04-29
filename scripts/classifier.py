import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Load ResNet18
model = models.resnet18(pretrained=True)

# Modify final layer for 3 classes
model.fc = nn.Linear(model.fc.in_features, 3)
model.eval()

# Labels (your thesis categories)
labels = ["Healthy", "Minor Defect", "Major Defect"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Folder with cropped images
folder = "../outputs"

# Loop through crops
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
        path = os.path.join(folder, filename)

        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)

        pred = torch.argmax(output, dim=1).item()

        print(f"{filename} → {labels[pred]}")