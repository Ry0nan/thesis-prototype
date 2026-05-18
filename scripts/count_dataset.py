"""
Count images in dataset folders and report class distribution.
Also detects potential camera/source groups by filename pattern.
"""
import os
import re
from collections import Counter, defaultdict

# =========================
# CONFIG
# =========================
DATASET_DIR = "../dataset/raw"
CLASS_FOLDERS = ['healthy', 'minor', 'major', '_ambiguous']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Filename patterns that suggest different cameras/collection setups.
# Add more as you see them.
SOURCE_PATTERNS = {
    'phone_IMG': re.compile(r'^IMG_\d+', re.IGNORECASE),
    'lumix_P10': re.compile(r'^P\d{7}', re.IGNORECASE),
    'other': re.compile(r'.*'),  # catch-all
}

def detect_source(filename):
    """Guess which camera/source produced a file based on naming pattern."""
    for source_name, pattern in SOURCE_PATTERNS.items():
        if source_name == 'other':
            continue
        if pattern.match(filename):
            return source_name
    return 'other'

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory not found: {DATASET_DIR}")
        return

    print("=" * 70)
    print(f"DATASET INVENTORY: {DATASET_DIR}")
    print("=" * 70)

    total_images = 0
    class_counts = {}
    source_breakdown = defaultdict(lambda: Counter())  # class -> {source: count}

    for class_folder in CLASS_FOLDERS:
        folder_path = os.path.join(DATASET_DIR, class_folder)
        if not os.path.exists(folder_path):
            print(f"\n[SKIP] Folder not found: {folder_path}")
            class_counts[class_folder] = 0
            continue

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(IMAGE_EXTENSIONS)]
        class_counts[class_folder] = len(files)
        total_images += len(files)

        for f in files:
            source = detect_source(f)
            source_breakdown[class_folder][source] += 1

    # =========================
    # OVERALL COUNTS
    # =========================
    print(f"\nTotal images: {total_images}")
    print("\nClass distribution:")
    print("-" * 70)
    for cls in CLASS_FOLDERS:
        count = class_counts.get(cls, 0)
        pct = (count / total_images * 100) if total_images > 0 else 0
        bar = "#" * int(pct / 2)  # simple text bar
        print(f"  {cls:<15} {count:>5} ({pct:>5.1f}%)  {bar}")

    # =========================
    # TARGET VS ACTUAL
    # =========================
    print("\nTarget vs Actual (excluding _ambiguous):")
    print("-" * 70)
    labeled_total = sum(class_counts.get(c, 0) for c in ['healthy', 'minor', 'major'])
    targets = {'healthy': 0.40, 'minor': 0.34, 'major': 0.26}
    for cls, target_pct in targets.items():
        actual = class_counts.get(cls, 0)
        actual_pct = (actual / labeled_total * 100) if labeled_total > 0 else 0
        target_count = int(labeled_total * target_pct)
        diff = actual - target_count
        sign = "+" if diff >= 0 else ""
        print(f"  {cls:<10} target: {target_count:>4} ({target_pct*100:.0f}%)  "
              f"actual: {actual:>4} ({actual_pct:.1f}%)  diff: {sign}{diff}")

    # =========================
    # SOURCE / CAMERA BREAKDOWN
    # Critical: detects if classes are correlated with camera source
    # =========================
    print("\nSource (camera) breakdown by class:")
    print("-" * 70)
    all_sources = sorted({s for breakdown in source_breakdown.values() for s in breakdown})
    header = f"  {'class':<15} " + " ".join(f"{s:>12}" for s in all_sources)
    print(header)
    for cls in CLASS_FOLDERS:
        if class_counts.get(cls, 0) == 0:
            continue
        row = f"  {cls:<15} "
        for s in all_sources:
            count = source_breakdown[cls].get(s, 0)
            total_cls = class_counts[cls]
            pct = (count / total_cls * 100) if total_cls > 0 else 0
            row += f"  {count:>4} ({pct:>4.1f}%)"
        print(row)

    # =========================
    # WARNINGS
    # =========================
    print("\nDiagnostic warnings:")
    print("-" * 70)

    warnings = []

    # Ambiguous ratio
    ambig = class_counts.get('_ambiguous', 0)
    if total_images > 0 and ambig / total_images > 0.10:
        warnings.append(
            f"High ambiguous ratio: {ambig}/{total_images} "
            f"({ambig/total_images*100:.1f}%). Consider tightening protocol."
        )

    # Class imbalance check
    if labeled_total > 0:
        max_class = max(class_counts.get(c, 0) for c in ['healthy', 'minor', 'major'])
        min_class = min(class_counts.get(c, 0) for c in ['healthy', 'minor', 'major'])
        if max_class > 0 and min_class / max_class < 0.30:
            warnings.append(
                f"Severe class imbalance: smallest class is "
                f"{min_class/max_class*100:.0f}% of largest. Class weights will help."
            )

    # Camera-class correlation check (the shortcut learning risk)
    if len(all_sources) > 1 and labeled_total > 0:
        for cls in ['healthy', 'minor', 'major']:
            if class_counts.get(cls, 0) == 0:
                continue
            counts = source_breakdown[cls]
            max_source_pct = max(counts.values()) / class_counts[cls] * 100
            if max_source_pct > 80:
                dominant = max(counts, key=counts.get)
                warnings.append(
                    f"Class '{cls}' is {max_source_pct:.0f}% from source '{dominant}'. "
                    f"Risk of model learning camera instead of defect features."
                )

    if not warnings:
        print("  [OK] No issues detected.")
    else:
        for i, w in enumerate(warnings, 1):
            print(f"  [WARN {i}] {w}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()