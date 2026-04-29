import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ========================
# CONFIG
# ========================
train_dir = "../dataset/train"
val_dir = "../dataset/val"
batch_size = 8
epochs = 5
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# TRANSFORMS
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ========================
# DATASETS
# ========================
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ========================
# MODEL (ResNet18)
# ========================
model = models.resnet18(weights="DEFAULT")

# Change final layer (3 classes)
model.fc = nn.Linear(model.fc.in_features, 3)

model = model.to(device)

# ========================
# LOSS + OPTIMIZER
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ========================
# TRAINING LOOP
# ========================
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # ========================
    # VALIDATION
    # ========================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total if total > 0 else 0
    print(f"Validation Accuracy: {acc:.2f}%")

# ========================
# SAVE MODEL
# ========================
torch.save(model.state_dict(), "../resnet18_kamias.pth")

print("Training complete. Model saved.")