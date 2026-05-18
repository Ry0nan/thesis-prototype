"""
Compute Cohen's kappa for inter-rater reliability check.

Reads two rater label CSVs and computes their agreement using Cohen's kappa.
Also reports detailed disagreement analysis so you can identify which
class boundaries are most problematic.

Run this AFTER both rater1_labels.csv and rater2_labels.csv have been filled in.

Cohen's kappa interpretation:
    < 0.20  poor
    0.20-0.40  fair
    0.40-0.60  moderate
    0.60-0.80  substantial    <- your minimum target
    > 0.80  near-perfect
"""

import os
import csv
from collections import Counter, defaultdict
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# =========================
# CONFIG
# =========================
KAPPA_DIR = "../dataset/kappa_test"
RATER1_FILE = os.path.join(KAPPA_DIR, "rater1_labels.csv")
RATER2_FILE = os.path.join(KAPPA_DIR, "rater2_labels.csv")
GROUND_TRUTH_FILE = os.path.join(KAPPA_DIR, "ground_truth.csv")
CLASS_NAMES = ['healthy', 'minor', 'major']

# =========================
# HELPERS
# =========================
def load_labels(csv_path):
    """Load filename -> label dict from a CSV with columns: filename, label."""
    labels = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename'].strip()
            label = row.get('label', row.get('true_class', '')).strip().lower()
            if filename and label:
                labels[filename] = label
    return labels

def interpret_kappa(kappa):
    """Return human-readable interpretation of a kappa value."""
    if kappa < 0.20:
        return "poor agreement", "FAIL - protocol is unclear, label noise is severe"
    elif kappa < 0.40:
        return "fair agreement", "FAIL - protocol needs significant refinement"
    elif kappa < 0.60:
        return "moderate agreement", "MARGINAL - protocol needs tightening"
    elif kappa < 0.80:
        return "substantial agreement", "PASS - defensible for undergraduate thesis"
    else:
        return "near-perfect agreement", "EXCELLENT - publishable quality labels"

# =========================
# MAIN
# =========================
def main():
    # Check inputs exist
    for path in [RATER1_FILE, RATER2_FILE]:
        if not os.path.exists(path):
            print(f"ERROR: Missing file: {path}")
            print("Run sample_for_kappa.py first, have both raters fill in their CSVs.")
            return

    rater1 = load_labels(RATER1_FILE)
    rater2 = load_labels(RATER2_FILE)

    # Find images both raters labeled
    common_files = sorted(set(rater1.keys()) & set(rater2.keys()))
    if not common_files:
        print("ERROR: No common labeled images between raters.")
        return

    n_total = len(common_files)
    n_rater1 = len(rater1)
    n_rater2 = len(rater2)

    if n_rater1 != n_rater2 or n_rater1 != n_total:
        print(f"WARNING: Rater label counts differ.")
        print(f"  Rater 1: {n_rater1} labels")
        print(f"  Rater 2: {n_rater2} labels")
        print(f"  Common: {n_total} images")
        print(f"  Computing kappa on the {n_total} commonly labeled images.\n")

    # Extract label lists in matched order
    labels1 = [rater1[f] for f in common_files]
    labels2 = [rater2[f] for f in common_files]

    # Validate labels
    valid_labels = set(CLASS_NAMES)
    invalid1 = [l for l in labels1 if l not in valid_labels]
    invalid2 = [l for l in labels2 if l not in valid_labels]
    if invalid1 or invalid2:
        print(f"ERROR: Invalid labels found.")
        print(f"  Valid labels: {valid_labels}")
        if invalid1: print(f"  Rater 1 invalid: {set(invalid1)}")
        if invalid2: print(f"  Rater 2 invalid: {set(invalid2)}")
        return

    # =========================
    # COMPUTE COHEN'S KAPPA
    # =========================
    kappa = cohen_kappa_score(labels1, labels2, labels=CLASS_NAMES)
    interpretation, verdict = interpret_kappa(kappa)

    # Simple agreement percentage (for comparison)
    agreement = sum(1 for a, b in zip(labels1, labels2) if a == b) / n_total

    print("=" * 70)
    print("INTER-RATER RELIABILITY REPORT")
    print("=" * 70)
    print(f"\nImages compared: {n_total}")
    print(f"Simple agreement: {agreement:.2%} ({sum(1 for a,b in zip(labels1,labels2) if a==b)}/{n_total})")
    print(f"\nCohen's kappa (κ): {kappa:.4f}")
    print(f"Interpretation:    {interpretation}")
    print(f"Verdict:           {verdict}")

    # =========================
    # AGREEMENT MATRIX
    # Shows where the two raters disagree most.
    # =========================
    print(f"\nAgreement matrix (rows = rater 1, columns = rater 2):")
    print("-" * 70)
    cm = confusion_matrix(labels1, labels2, labels=CLASS_NAMES)
    header = "  Rater1\\Rater2  " + "  ".join(f"{name:>8}" for name in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:<14} " + "  ".join(f"{cm[i, j]:>8}" for j in range(len(CLASS_NAMES)))
        print(row)
    print("\n  Diagonal = both raters agreed.")
    print("  Off-diagonal = where they disagreed (these are the boundary problems).\n")

    # =========================
    # DISAGREEMENT BREAKDOWN
    # =========================
    disagreements = [(f, rater1[f], rater2[f]) for f in common_files if rater1[f] != rater2[f]]
    if disagreements:
        print(f"Disagreements ({len(disagreements)} cases):")
        print("-" * 70)
        # Group by disagreement type
        pair_counts = Counter((d[1], d[2]) for d in disagreements)
        for (r1, r2), count in pair_counts.most_common():
            print(f"  Rater1='{r1}' vs Rater2='{r2}': {count} cases")
        print("\n  If most disagreements are minor↔major, the boundary is the issue.")
        print("  If most are healthy↔minor, criteria for early defect are unclear.\n")

    # =========================
    # COMPARE TO GROUND TRUTH (if available)
    # This shows which rater was closer to the original labeling.
    # =========================
    if os.path.exists(GROUND_TRUTH_FILE):
        ground_truth = load_labels(GROUND_TRUTH_FILE)
        gt_files = [f for f in common_files if f in ground_truth]
        if gt_files:
            gt_labels = [ground_truth[f] for f in gt_files]
            r1_labels = [rater1[f] for f in gt_files]
            r2_labels = [rater2[f] for f in gt_files]
            kappa_r1_gt = cohen_kappa_score(r1_labels, gt_labels, labels=CLASS_NAMES)
            kappa_r2_gt = cohen_kappa_score(r2_labels, gt_labels, labels=CLASS_NAMES)
            print("Agreement with original (raw folder) labels:")
            print("-" * 70)
            print(f"  Rater 1 vs original:  κ = {kappa_r1_gt:.4f}")
            print(f"  Rater 2 vs original:  κ = {kappa_r2_gt:.4f}")
            print()

    # =========================
    # CITATION TEXT FOR THESIS
    # =========================
    print("=" * 70)
    print("CITATION TEXT FOR CHAPTER 3 METHODOLOGY")
    print("=" * 70)
    print(f"""
Suggested wording:

    "Inter-rater reliability was assessed on a stratified random sample
    of {n_total} images, independently labeled by two annotators applying the
    written annotation protocol. The annotators achieved Cohen's kappa of
    κ = {kappa:.2f} ({interpretation}), indicating that the protocol can be
    applied consistently across raters."
""")

if __name__ == "__main__":
    main()