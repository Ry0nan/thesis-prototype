"""
Generate result plots for the thesis paper.
Reads from outputs/ and models/ and produces:
  - baseline_vs_ssl_bars.png       (Macro F1 + per-class F1 comparison)
  - pseudo_label_decay.png         (per-iteration pseudo-label accuracy)
  - ssl_pool_growth.png            (labeled pool size over iterations)
  - resnet_baseline_confusion.png  (baseline confusion matrix)
  - resnet_ssl_confusion.png       (SSL confusion matrix)
  - resnet_full_confusion.png      (full upper-bound confusion matrix)
All saved to outputs/plots/.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix

OUT_DIR = "../outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 1. Baseline vs SSL vs Full bar chart ----------
def plot_model_comparison():
    models_data = {
        "Baseline (252)": {"macro_f1": 0.8915, "defective_f1": 0.898, "healthy_f1": 0.885},
        "SSL (1,037)":    {"macro_f1": 0.9224, "defective_f1": 0.931, "healthy_f1": 0.914},
        "Full (1,014)":   {"macro_f1": 0.9051, "defective_f1": 0.911, "healthy_f1": 0.900},
    }

    labels = list(models_data.keys())
    macro = [models_data[m]["macro_f1"] for m in labels]
    defect = [models_data[m]["defective_f1"] for m in labels]
    healthy = [models_data[m]["healthy_f1"] for m in labels]

    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_macro = ax.bar(x - width, macro, width, label="Macro F1", color="#2E7D32")
    bars_def = ax.bar(x, defect, width, label="Defective F1", color="#C62828")
    bars_heal = ax.bar(x + width, healthy, width, label="Healthy F1", color="#1565C0")

    # Annotate values on top of each bar
    for bars in [bars_macro, bars_def, bars_heal]:
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison: F1 Scores Across Approaches")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.80, 1.00)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/baseline_vs_ssl_bars.png", dpi=200)
    plt.close()
    print("  Saved baseline_vs_ssl_bars.png")


# ---------- 2. Pseudo-label accuracy decay ----------
def plot_pseudo_decay():
    path = "../outputs/pseudo_label_accuracy.csv"
    if not os.path.exists(path):
        print(f"  Skipped pseudo_label_decay: {path} not found")
        return

    rows = list(csv.DictReader(open(path)))
    valid = [r for r in rows if r["correct"] in ("True", "False")]
    by_iter = {}
    for r in valid:
        i = int(r["iteration"])
        by_iter.setdefault(i, []).append(r["correct"] == "True")

    iters = sorted(by_iter.keys())
    accs = [100 * sum(by_iter[i]) / len(by_iter[i]) for i in iters]
    counts = [len(by_iter[i]) for i in iters]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, accs, "o-", color="#C62828", linewidth=2, markersize=8)
    for it, acc, n in zip(iters, accs, counts):
        ax.annotate(f"{acc:.1f}%\n(n={n})",
                    (it, acc),
                    textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    ax.axhline(50, linestyle="--", color="gray", alpha=0.5, label="Chance (50%)")
    ax.set_xlabel("SSL Iteration")
    ax.set_ylabel("Pseudo-label Accuracy (%)")
    ax.set_title("Pseudo-label Accuracy by Iteration (Confirmation Bias)")
    ax.set_xticks(iters)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pseudo_label_decay.png", dpi=200)
    plt.close()
    print("  Saved pseudo_label_decay.png")


# ---------- 3. SSL pool growth ----------
def plot_pool_growth():
    path = "../outputs/ssl_log.csv"
    if not os.path.exists(path):
        print(f"  Skipped ssl_pool_growth: {path} not found")
        return

    rows = list(csv.DictReader(open(path)))
    iters = [0] + [int(r["iteration"]) for r in rows]
    sizes = [252]
    for r in rows:
        sizes.append(int(r["labeled_size_before"]) + int(r["new_labels_added"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, sizes, "s-", color="#2E7D32", linewidth=2, markersize=8)
    for i, s in zip(iters, sizes):
        ax.annotate(str(s), (i, s),
                    textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("SSL Iteration (0 = baseline)")
    ax.set_ylabel("Labeled Pool Size")
    ax.set_title("SSL Labeled Pool Growth")
    ax.set_xticks(iters)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ssl_pool_growth.png", dpi=200)
    plt.close()
    print("  Saved ssl_pool_growth.png")


# ---------- 4. ResNet confusion matrices ----------
def plot_confusion(model_path, name, out_name):
    if not os.path.exists(model_path):
        print(f"  Skipped {name}: {model_path} not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_data = datasets.ImageFolder("../dataset/test", transform=tfm)
    loader = DataLoader(test_data, batch_size=16, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.extend(model(x).argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    cm = confusion_matrix(labels, preds)
    class_names = test_data.classes

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color=color, fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{out_name}", dpi=200)
    plt.close()
    print(f"  Saved {out_name}")


if __name__ == "__main__":
    print(f"Generating plots to {OUT_DIR}/...\n")
    plot_model_comparison()
    plot_pseudo_decay()
    plot_pool_growth()
    plot_confusion("../models/resnet_baseline.pth", "Baseline", "resnet_baseline_confusion.png")
    plot_confusion("../models/resnet_ssl.pth", "SSL", "resnet_ssl_confusion.png")
    plot_confusion("../models/resnet_full.pth", "Full (Upper Bound)", "resnet_full_confusion.png")
    print("\nDone.")