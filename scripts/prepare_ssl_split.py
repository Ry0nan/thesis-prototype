"""
One-time setup: carve the training data into labeled (25%) and unlabeled (75%)
subsets for the SSL experiment.

After train/val/test split has been done (via split_dataset.py), this script:
1. Takes the existing dataset/train/ folder (which has class subfolders)
2. Randomly assigns 25% to dataset/train_labeled/ (keeps class folder structure)
3. Moves the other 75% to dataset/unlabeled/ as a flat folder
   (no class subfolders — these are "unlabeled" from the model's perspective)
4. Optionally copies dataset/raw/_ambiguous/ images into the unlabeled pool too

After this runs:
    dataset/train/          <- ORIGINAL train data (left untouched, kept as reference)
    dataset/train_labeled/  <- 25% with class folders, used for supervised baseline
    dataset/unlabeled/      <- 75% flat folder + ambiguous, pool for SSL pseudo-labeling
    dataset/val/            <- unchanged
    dataset/test/           <- unchanged (NEVER touched during SSL)

The SSL pipeline draws labeled examples from train_labeled/ and adds
high-confidence pseudo-labeled examples from unlabeled/ back into it.
"""

import os
import shutil
import random
from collections import Counter, defaultdict

# =========================
# CONFIG
# =========================
TRAIN_DIR = "../dataset/train"
LABELED_DIR = "../dataset/train_labeled"
UNLABELED_DIR = "../dataset/unlabeled"
AMBIGUOUS_DIR = "../dataset/raw/_ambiguous"

CLASS_NAMES = ['healthy', 'minor', 'major']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Fraction of training data to keep as labeled. Thesis says 25%.
LABELED_FRACTION = 0.25
SEED = 42

INCLUDE_AMBIGUOUS_IN_UNLABELED = True
CLEAR_DESTINATIONS = True

# =========================
# MAIN
# =========================
def main():
    random.seed(SEED)

    print("=" * 70)
    print("SSL SPLIT PREPARATION")
    print("=" * 70)
    print(f"Labeled fraction: {LABELED_FRACTION:.0%}")
    print(f"Seed: {SEED}\n")

    # Clear destinations if requested
    if CLEAR_DESTINATIONS:
        for d in [LABELED_DIR, UNLABELED_DIR]:
            if os.path.exists(d):
                print(f"Clearing: {d}")
                shutil.rmtree(d)
        print()

    # Create destination structure
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(LABELED_DIR, cls), exist_ok=True)
    os.makedirs(UNLABELED_DIR, exist_ok=True)

    # =========================
    # PROCESS EACH CLASS
    # Within each class, stratified split keeps class proportions in labeled subset
    # =========================
    labeled_counts = Counter()
    unlabeled_count = 0

    for cls in CLASS_NAMES:
        src_folder = os.path.join(TRAIN_DIR, cls)
        if not os.path.exists(src_folder):
            print(f"WARNING: Source not found: {src_folder}")
            continue

        files = [f for f in os.listdir(src_folder)
                 if f.lower().endswith(IMAGE_EXTENSIONS)]
        random.shuffle(files)

        n_labeled = int(len(files) * LABELED_FRACTION)
        labeled_files = files[:n_labeled]
        unlabeled_files = files[n_labeled:]

        # Copy labeled files keeping class folder structure
        for f in labeled_files:
            shutil.copy2(
                os.path.join(src_folder, f),
                os.path.join(LABELED_DIR, cls, f)
            )
        labeled_counts[cls] = len(labeled_files)

        # Copy unlabeled files to flat unlabeled folder
        # Prefix with class for traceability (we'll need this later for analysis)
        # But the MODEL never sees this — it just sees the image
        for f in unlabeled_files:
            # Use a prefix so we can later check: did SSL recover the correct label?
            new_name = f"__true_{cls}__{f}"
            shutil.copy2(
                os.path.join(src_folder, f),
                os.path.join(UNLABELED_DIR, new_name)
            )
        unlabeled_count += len(unlabeled_files)

    # =========================
    # OPTIONALLY ADD AMBIGUOUS IMAGES TO UNLABELED POOL
    # =========================
    ambig_added = 0
    if INCLUDE_AMBIGUOUS_IN_UNLABELED and os.path.exists(AMBIGUOUS_DIR):
        ambig_files = [f for f in os.listdir(AMBIGUOUS_DIR)
                       if f.lower().endswith(IMAGE_EXTENSIONS)]
        for f in ambig_files:
            # Mark as ambiguous so we can track if SSL labels them
            new_name = f"__true_ambiguous__{f}"
            shutil.copy2(
                os.path.join(AMBIGUOUS_DIR, f),
                os.path.join(UNLABELED_DIR, new_name)
            )
        ambig_added = len(ambig_files)
        unlabeled_count += ambig_added

    # =========================
    # SUMMARY
    # =========================
    print("Split summary:")
    print("-" * 70)
    print(f"\n  LABELED ({LABELED_DIR}):")
    total_labeled = sum(labeled_counts.values())
    for cls in CLASS_NAMES:
        count = labeled_counts[cls]
        pct = count / total_labeled * 100 if total_labeled > 0 else 0
        print(f"    {cls:<10} {count:>5} ({pct:.1f}%)")
    print(f"    TOTAL      {total_labeled:>5}")

    print(f"\n  UNLABELED ({UNLABELED_DIR}):")
    print(f"    From train     {unlabeled_count - ambig_added:>5}")
    if INCLUDE_AMBIGUOUS_IN_UNLABELED:
        print(f"    From ambiguous {ambig_added:>5}")
    print(f"    TOTAL          {unlabeled_count:>5}")

    print(f"\n  Ratio labeled:unlabeled = "
          f"{total_labeled/(total_labeled+unlabeled_count):.0%}:"
          f"{unlabeled_count/(total_labeled+unlabeled_count):.0%}")

    print("\n" + "=" * 70)
    print("Files named '__true_X__filename.jpg' encode the TRUE label inside")
    print("the filename. The model NEVER sees this — but it lets us measure")
    print("how often SSL recovered the correct label vs got it wrong.")
    print("=" * 70)

if __name__ == "__main__":
    main()