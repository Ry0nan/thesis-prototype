"""
Stratified 70/15/15 train/val/test split for the Kamias dataset.
2-CLASS VERSION: healthy / defective
Stratifies by class AND camera source. Source images in raw/ left untouched.
"""

import os
import re
import shutil
import random
from collections import defaultdict, Counter

# =========================
# CONFIG
# =========================
SOURCE_DIR = "../dataset/cropped"
DEST_DIR = "../dataset"
CLASS_NAMES = ['healthy', 'defective']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

SOURCE_PATTERNS = {
    'phone_IMG': re.compile(r'^IMG_\d+', re.IGNORECASE),
    'lumix_P10': re.compile(r'^P\d{7}', re.IGNORECASE),
}

CLEAR_DESTINATION = True

def detect_source(filename):
    for source_name, pattern in SOURCE_PATTERNS.items():
        if pattern.match(filename):
            return source_name
    return 'other'

def stratified_split(items, train_ratio, val_ratio, test_ratio):
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test

def clear_destination_folders(dest_dir, class_names):
    for split in ['train', 'val', 'test']:
        for cls in class_names:
            folder = os.path.join(dest_dir, split, cls)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    file_path = os.path.join(folder, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            else:
                os.makedirs(folder, exist_ok=True)

def main():
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    assert abs(total_ratio - 1.0) < 0.001, f"Ratios must sum to 1.0, got {total_ratio}"

    print("=" * 70)
    print(f"STRATIFIED SPLIT: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}\n")

    random.seed(SEED)

    if CLEAR_DESTINATION:
        print("Clearing destination folders...")
        clear_destination_folders(DEST_DIR, CLASS_NAMES)

    buckets = defaultdict(list)
    class_totals = Counter()

    for cls in CLASS_NAMES:
        src_folder = os.path.join(SOURCE_DIR, cls)
        if not os.path.exists(src_folder):
            print(f"WARNING: Source folder missing: {src_folder}")
            continue
        files = [f for f in os.listdir(src_folder)
                 if f.lower().endswith(IMAGE_EXTENSIONS)]
        for f in files:
            source = detect_source(f)
            buckets[(cls, source)].append(f)
            class_totals[cls] += 1

    if not buckets:
        print("ERROR: No images found. Check SOURCE_DIR.")
        return

    print("Stratification buckets (class, source):")
    print("-" * 70)
    for (cls, src), files in sorted(buckets.items()):
        print(f"  {cls:<10} {src:<12} {len(files):>5} images")
    print()

    splits = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}

    for (cls, source), files in buckets.items():
        files_shuffled = files.copy()
        random.shuffle(files_shuffled)
        train_files, val_files, test_files = stratified_split(
            files_shuffled, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )
        splits['train'][cls].extend(train_files)
        splits['val'][cls].extend(val_files)
        splits['test'][cls].extend(test_files)

    print("Copying files to split folders...")
    copy_counts = {'train': Counter(), 'val': Counter(), 'test': Counter()}

    for split_name, by_class in splits.items():
        for cls, files in by_class.items():
            src_folder = os.path.join(SOURCE_DIR, cls)
            dst_folder = os.path.join(DEST_DIR, split_name, cls)
            os.makedirs(dst_folder, exist_ok=True)
            for f in files:
                src_path = os.path.join(src_folder, f)
                dst_path = os.path.join(dst_folder, f)
                shutil.copy2(src_path, dst_path)
                copy_counts[split_name][cls] += 1

    print()
    print("=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    header = f"  {'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}"
    print(header)
    print("-" * 70)

    grand_train = grand_val = grand_test = 0
    for cls in CLASS_NAMES:
        t = copy_counts['train'][cls]
        v = copy_counts['val'][cls]
        te = copy_counts['test'][cls]
        total = t + v + te
        print(f"  {cls:<12} {t:>8} {v:>8} {te:>8} {total:>8}")
        grand_train += t
        grand_val += v
        grand_test += te

    print("-" * 70)
    print(f"  {'TOTAL':<12} {grand_train:>8} {grand_val:>8} {grand_test:>8} "
          f"{grand_train + grand_val + grand_test:>8}")

    grand_total = grand_train + grand_val + grand_test
    if grand_total > 0:
        print(f"\n  Ratios: "
              f"train={grand_train/grand_total:.1%} "
              f"val={grand_val/grand_total:.1%} "
              f"test={grand_test/grand_total:.1%}")

    print("\nCamera source distribution in each split:")
    print("-" * 70)
    for split_name in ['train', 'val', 'test']:
        print(f"\n  [{split_name.upper()}]")
        for cls in CLASS_NAMES:
            folder = os.path.join(DEST_DIR, split_name, cls)
            if not os.path.exists(folder):
                continue
            files = os.listdir(folder)
            if not files:
                continue
            source_counts = Counter(detect_source(f) for f in files)
            total = sum(source_counts.values())
            parts = [f"{src}={cnt} ({cnt/total*100:.0f}%)"
                     for src, cnt in sorted(source_counts.items())]
            print(f"    {cls:<10} " + ", ".join(parts))

    print("\n" + "=" * 70)
    print("Split complete. Source images in dataset/raw/ untouched.")
    print("=" * 70)

if __name__ == "__main__":
    main()