import json
import random
import os

random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_PATH = os.path.join(DATA_DIR, "full.jsonl")
SAMPLE_PATH = os.path.join(DATA_DIR, "sampled.jsonl")

def classify_answer(ref_answer):
    if not ref_answer or ref_answer.strip() == "":
        return "no_answer"
    words = ref_answer.strip().split()
    if len(words) == 1:
        return "single_word"
    elif len(words) <= 20:
        return "short"
    return "long"

print("Scanning full.jsonl (streaming, line by line)...")
buckets = {"no_answer": [], "single_word": [], "short": [], "long": []}
total = 0

with open(FULL_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        ref = row.get("reference_answer", "") or ""
        cat = classify_answer(ref)
        buckets[cat].append(i)
        total += 1
        if total % 200000 == 0:
            print(f"  scanned {total:,} rows...")

print(f"Total rows: {total:,}")
print("Distribution:")
for cat, indices in buckets.items():
    pct = len(indices) / total * 100
    print(f"  {cat}: {len(indices):,} ({pct:.1f}%)")

targets = {"no_answer": 18, "single_word": 10, "short": 37, "long": 35}
selected_indices = set()
for cat, count in targets.items():
    pool = buckets[cat]
    random.shuffle(pool)
    picked = pool[:count]
    selected_indices.update(picked)
    print(f"Sampled {len(picked)} from {cat}")

print(f"\nTotal sample size: {len(selected_indices)}")
print("Extracting selected rows...")

sample_rows = []
with open(FULL_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i in selected_indices:
            row = json.loads(line)
            row["sample_id"] = len(sample_rows)
            row["answer_type"] = classify_answer(row.get("reference_answer", "") or "")
            sample_rows.append(row)

random.shuffle(sample_rows)
for idx, row in enumerate(sample_rows):
    row["sample_id"] = idx

with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
    for row in sample_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {len(sample_rows)} rows to {SAMPLE_PATH}")
