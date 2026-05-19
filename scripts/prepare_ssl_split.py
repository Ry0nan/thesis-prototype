"""
One-time setup: carve training data into labeled (25%) and unlabeled (75%)
subsets for the SSL experiment.
2-CLASS VERSION: healthy / defective

After this runs:
    dataset/train/          <- original train data (untouched reference)
    dataset/train_labeled/  <- 25% with class folders, supervised baseline
    dataset/unlabeled/      <- 75% flat folder + ambiguous, SSL pool
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
AMBIGUOUS_DIR = "../dataset/cropped/_ambiguous"

CLASS_NAMES = ['healthy', 'defective']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

LABELED_FRACTION = 0.25
SEED = 42

INCLUDE_AMBIGUOUS_IN_UNLABELED = True
CLEAR_DESTINATIONS = True

def main():
    random.seed(SEED)

    print("=" * 70)
    print("SSL SPLIT PREPARATION")
    print("=" * 70)
    print(f"Labeled fraction: {LABELED_FRACTION:.0%}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Seed: {SEED}\n")

    if CLEAR_DESTINATIONS:
        for d in [LABELED_DIR, UNLABELED_DIR]:
            if os.path.exists(d):
                print(f"Clearing: {d}")
                shutil.rmtree(d)
        print()

    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(LABELED_DIR, cls), exist_ok=True)
    os.makedirs(UNLABELED_DIR, exist_ok=True)

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

        for f in labeled_files:
            shutil.copy2(
                os.path.join(src_folder, f),
                os.path.join(LABELED_DIR, cls, f)
            )
        labeled_counts[cls] = len(labeled_files)

        for f in unlabeled_files:
            new_name = f"__true_{cls}__{f}"
            shutil.copy2(
                os.path.join(src_folder, f),
                os.path.join(UNLABELED_DIR, new_name)
            )
        unlabeled_count += len(unlabeled_files)

    ambig_added = 0
    if INCLUDE_AMBIGUOUS_IN_UNLABELED and os.path.exists(AMBIGUOUS_DIR):
        ambig_files = [f for f in os.listdir(AMBIGUOUS_DIR)
                       if f.lower().endswith(IMAGE_EXTENSIONS)]
        for f in ambig_files:
            new_name = f"__true_ambiguous__{f}"
            shutil.copy2(
                os.path.join(AMBIGUOUS_DIR, f),
                os.path.join(UNLABELED_DIR, new_name)
            )
        ambig_added = len(ambig_files)
        unlabeled_count += ambig_added

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

    if total_labeled + unlabeled_count > 0:
        print(f"\n  Ratio labeled:unlabeled = "
              f"{total_labeled/(total_labeled+unlabeled_count):.0%}:"
              f"{unlabeled_count/(total_labeled+unlabeled_count):.0%}")

    print("\n" + "=" * 70)
    print("Files named '__true_X__filename.jpg' encode the TRUE label inside")
    print("the filename. The model NEVER sees this — it lets us measure")
    print("how often SSL recovered the correct label.")
    print("=" * 70)

if __name__ == "__main__":
    main()