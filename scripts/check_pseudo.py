import csv

rows = [r for r in csv.DictReader(open('../outputs/pseudo_label_accuracy.csv'))]
valid = [r for r in rows if r['correct'] in ('True', 'False')]
correct = sum(1 for r in valid if r['correct'] == 'True')

print(f"Pseudo-labels checked: {len(valid)}")
if valid:
    print(f"Overall correct: {correct} ({correct/len(valid)*100:.1f}%)")

print("\nPer-iteration breakdown:")
for i in range(1, 6):
    iter_rows = [r for r in valid if r['iteration'] == str(i)]
    if iter_rows:
        iter_correct = sum(1 for r in iter_rows if r['correct'] == 'True')
        print(f"  Iteration {i}: {iter_correct}/{len(iter_rows)} correct "
              f"({iter_correct/len(iter_rows)*100:.1f}%)")

# Also check the ambiguous ones (true label was 'ambiguous', no right answer)
ambig = [r for r in rows if r['correct'] == 'N/A']
print(f"\nAmbiguous images pseudo-labeled: {len(ambig)}")