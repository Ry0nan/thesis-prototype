"""
Random sampling utility for Cohen's kappa inter-rater reliability check.

Picks N random images from dataset/raw/{healthy,minor,major}/, copies them
to a flat folder (no class subfolders), and saves the original labels to a
hidden file. Two raters then label the flat folder independently, and
compute_kappa.py compares their results.

Output:
    dataset/kappa_test/
    ├── images/                    <- 50 unlabeled images for raters
    │   ├── IMG_xxx.jpg
    │   ├── IMG_yyy.jpg
    │   └── ...
    ├── ground_truth.csv           <- original labels (don't share with raters)
    ├── rater1_labels.csv          <- TEMPLATE for rater 1 to fill in
    └── rater2_labels.csv          <- TEMPLATE for rater 2 to fill in
"""

import os
import csv
import random
import shutil
from collections import Counter

# =========================
# CONFIG
# =========================
SOURCE_DIR = "../dataset/raw"
DEST_DIR = "../dataset/kappa_test"
CLASS_NAMES = ['healthy', 'minor', 'major']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Number of images to sample for the kappa check.
# 50 is the standard minimum for a meaningful kappa score.
N_SAMPLES = 50

# Reproducibility
SEED = 42

# =========================
# MAIN
# =========================
def main():
    random.seed(SEED)

    # Clean previous kappa test if it exists
    if os.path.exists(DEST_DIR):
        print(f"Removing existing kappa test folder: {DEST_DIR}")
        shutil.rmtree(DEST_DIR)

    images_dir = os.path.join(DEST_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    # =========================
    # COLLECT ALL CANDIDATE IMAGES WITH THEIR TRUE LABELS
    # =========================
    candidates = []  # list of (filename, true_class)
    for cls in CLASS_NAMES:
        src_folder = os.path.join(SOURCE_DIR, cls)
        if not os.path.exists(src_folder):
            print(f"WARNING: Source folder missing: {src_folder}")
            continue
        for f in os.listdir(src_folder):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                candidates.append((f, cls))

    if len(candidates) < N_SAMPLES:
        print(f"ERROR: Only {len(candidates)} images found, need at least {N_SAMPLES}")
        return

    # =========================
    # STRATIFIED SAMPLE
    # Take proportional number from each class so the kappa check
    # reflects realistic class boundaries, not just the easy ones.
    # =========================
    by_class = {cls: [] for cls in CLASS_NAMES}
    for fname, cls in candidates:
        by_class[cls].append(fname)

    samples = []
    total = sum(len(v) for v in by_class.values())
    for cls in CLASS_NAMES:
        n_for_class = round(N_SAMPLES * len(by_class[cls]) / total)
        sampled = random.sample(by_class[cls], min(n_for_class, len(by_class[cls])))
        samples.extend([(f, cls) for f in sampled])

    # Shuffle so raters see images in random order, not grouped by class
    random.shuffle(samples)

    # =========================
    # COPY IMAGES TO FLAT FOLDER
    # =========================
    print(f"Sampling {len(samples)} images for kappa test...\n")
    for filename, true_class in samples:
        src_path = os.path.join(SOURCE_DIR, true_class, filename)
        dst_path = os.path.join(images_dir, filename)
        shutil.copy2(src_path, dst_path)

    # =========================
    # WRITE GROUND TRUTH CSV (FOR YOU ONLY, NOT FOR RATERS)
    # =========================
    gt_path = os.path.join(DEST_DIR, "ground_truth.csv")
    with open(gt_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'true_class'])
        for filename, true_class in samples:
            writer.writerow([filename, true_class])

    # =========================
    # WRITE RATER TEMPLATES
    # Each rater fills in the 'label' column with their judgment.
    # =========================
    for rater_num in [1, 2]:
        template_path = os.path.join(DEST_DIR, f"rater{rater_num}_labels.csv")
        with open(template_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])  # rater fills in 'label'
            for filename, _ in samples:
                writer.writerow([filename, ''])

    # =========================
    # SUMMARY
    # =========================
    print(f"Sampled {len(samples)} images. Distribution:")
    print("-" * 50)
    sample_dist = Counter(true_class for _, true_class in samples)
    for cls in CLASS_NAMES:
        count = sample_dist[cls]
        pct = count / len(samples) * 100
        print(f"  {cls:<10} {count:>3} ({pct:.1f}%)")

    print(f"\nOutput folder: {DEST_DIR}")
    print(f"  - images/             {len(samples)} images for raters")
    print(f"  - ground_truth.csv    DO NOT share with raters")
    print(f"  - rater1_labels.csv   template for rater 1")
    print(f"  - rater2_labels.csv   template for rater 2")
    print()
    print("Next steps:")
    print("  1. Each rater opens their CSV and looks at images/ folder")
    print("  2. They fill in 'label' column with: healthy, minor, or major")
    print("  3. Raters do NOT discuss labels with each other")
    print("  4. Once both are filled in, run compute_kappa.py")

if __name__ == "__main__":
    main()