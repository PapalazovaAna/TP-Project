from __future__ import annotations

from sentence_transformers import SentenceTransformer

import json
import os
import random
import re
from collections import Counter
from typing import Any

import numpy as np

SEED = 42

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_PATH = os.path.join(DATA_DIR, "full.jsonl")
TEST_PATH = os.path.join(DATA_DIR, "static_samples", "sampled.jsonl")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
POOL_PATH = os.path.join(OUT_DIR, "example_pool.jsonl")
SELECTION_PATH = os.path.join(OUT_DIR, "dynamic_examples.jsonl")
POOL_SIZE = 20000
SCAN_LIMIT: int | None = None
K = 3
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NEAR_DUP_THRESHOLD = 0.95
SAME_TYPE_ONLY = False
USE_MMR = False
MMR_LAMBDA = 0.7
MAX_QUESTION_WORDS = 150
MAX_ANSWER_WORDS = 80
MAX_REASONING_WORDS = 150
MIN_REASONING_WORDS = 30

def classify_answer(ref_answer: str) -> str:
    if not ref_answer or ref_answer.strip() == "":
        return "no_answer"
    words = ref_answer.strip().split()
    if len(words) == 1:
        return "single_word"
    if len(words) <= 20:
        return "short"
    return "long"


def clean_reasoning(text: str) -> str:
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    idx = text.lower().rfind("the final answer is")
    if idx > len(text) * 0.5:
        text = text[:idx]
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_at_sentence(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    ends = list(re.finditer(r"[.!?](?=\s|$)", truncated))
    if ends:
        return truncated[: ends[-1].end()]
    return truncated + "."


def eligible(row: dict[str, Any], test_questions: set[str]) -> bool:
    ref = (row.get("reference_answer") or "").strip()
    if classify_answer(ref) == "no_answer":
        return False
    if len(ref.split()) > MAX_ANSWER_WORDS:
        return False
    q = (row.get("question") or "").strip()
    if not q or len(q.split()) > MAX_QUESTION_WORDS:
        return False
    if q in test_questions:
        return False
    responses = row.get("responses") or []
    if not responses or not (responses[0].get("response") or "").strip():
        return False
    reasoning = clean_reasoning(responses[0]["response"])
    if len(reasoning.split()) < MIN_REASONING_WORDS:
        return False
    return True

rng = random.Random(SEED)

with open(TEST_PATH, encoding="utf-8") as f:
    test_set = [json.loads(line) for line in f]
test_questions = {item["question"].strip() for item in test_set}
print(f"Loaded {len(test_set)} test questions (excluded from the pool).")

test_type_counts = Counter(
    item["answer_type"] for item in test_set if item["answer_type"] != "no_answer"
)
denom = sum(test_type_counts.values())
targets = {
    cat: max(1, round(POOL_SIZE * cnt / denom)) for cat, cnt in test_type_counts.items()
}
print(f"Pool target ({POOL_SIZE} total): {dict(targets)}")

reservoirs: dict[str, list[dict[str, str]]] = {cat: [] for cat in targets}
seen = {cat: 0 for cat in targets}
scanned = 0

print("Scanning full.jsonl (streaming)...")
with open(FULL_PATH, encoding="utf-8") as f:
    for line in f:
        scanned += 1
        if SCAN_LIMIT is not None and scanned > SCAN_LIMIT:
            break
        if scanned % 200_000 == 0:
            sizes = {c: len(r) for c, r in reservoirs.items()}
            print(f"  scanned {scanned:,} rows... reservoirs: {sizes}")
        row = json.loads(line)
        if not eligible(row, test_questions):
            continue
        cat = classify_answer(row["reference_answer"])
        if cat not in targets:
            continue
        seen[cat] += 1
        slim = {
            "question": row["question"].strip(),
            "reference_answer": row["reference_answer"].strip(),
            "response": row["responses"][0]["response"],
        }
        target = targets[cat]
        if len(reservoirs[cat]) < target:
            reservoirs[cat].append(slim)
        else:
            j = rng.randrange(seen[cat])
            if j < target:
                reservoirs[cat][j] = slim

print(f"Scanned {scanned:,} rows. Eligible seen per bucket: {dict(seen)}")
for cat, res in reservoirs.items():
    if len(res) < targets[cat]:
        print(f"  WARNING: bucket {cat} only {len(res)}/{targets[cat]} filled.")

records: list[dict[str, Any]] = []
for cat, res in reservoirs.items():
    for row in res:
        records.append(
            {
                "question": row["question"],
                "answer": row["reference_answer"],
                "reasoning": truncate_at_sentence(
                    clean_reasoning(row["response"]), MAX_REASONING_WORDS
                ),
                "answer_type": cat,
            }
        )

rng.shuffle(records)
print(f"Built {len(records)} pool candidates.")

print(f"Embedding with {EMBED_MODEL}...")
embedder = SentenceTransformer(EMBED_MODEL)
pool_emb = embedder.encode(
    [r["question"] for r in records],
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=256,
)
test_emb = embedder.encode(
    [t["question"] for t in test_set],
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=256,
)
pool_emb = np.asarray(pool_emb, dtype=np.float32)
test_emb = np.asarray(test_emb, dtype=np.float32)

sims_all = test_emb @ pool_emb.T
max_sim_to_test = sims_all.max(axis=0)
keep_mask = max_sim_to_test <= NEAR_DUP_THRESHOLD
n_dropped = int((~keep_mask).sum())
if n_dropped:
    print(f"Leakage guard: dropped {n_dropped} near-duplicate candidates "
          f"(cosine > {NEAR_DUP_THRESHOLD}).")

kept_idx = np.where(keep_mask)[0]
records = [records[i] for i in kept_idx]
pool_emb = pool_emb[kept_idx]
sims_all = sims_all[:, kept_idx]
for idx, rec in enumerate(records):
    rec["pool_id"] = idx

overlap = {r["question"] for r in records} & test_questions
assert not overlap, f"LEAKAGE: {len(overlap)} pool questions are in the test set!"
print(f"Pool after guard: {len(records)} candidates. "
      f"Types: {dict(Counter(r['answer_type'] for r in records))}")

pool_types = np.array([r["answer_type"] for r in records])


def mmr_select(query_sims: np.ndarray, cand_idx: np.ndarray, k: int) -> list[int]:
    selected: list[int] = []
    remaining = list(cand_idx)
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            sel_emb = pool_emb[selected]

            def mmr_score(i: int) -> float:
                diversity = float((pool_emb[i] @ sel_emb.T).max())
                return MMR_LAMBDA * query_sims[i] - (1 - MMR_LAMBDA) * diversity

            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    selected.sort(key=lambda i: query_sims[i])
    return selected


selection: dict[str, list[dict[str, Any]]] = {}
for row_i, item in enumerate(test_set):
    query_sims = sims_all[row_i]

    if SAME_TYPE_ONLY and item["answer_type"] != "no_answer":
        cand_idx = np.where(pool_types == item["answer_type"])[0]
        if len(cand_idx) < K:
            cand_idx = np.arange(len(records))
    else:
        cand_idx = np.arange(len(records))

    if USE_MMR:
        chosen = mmr_select(query_sims, cand_idx, K)
    else:
        order = cand_idx[np.argsort(query_sims[cand_idx])]
        chosen = list(order[-K:])

    selection[str(item["sample_id"])] = [
        {
            "pool_id": records[j]["pool_id"],
            "question": records[j]["question"],
            "answer": records[j]["answer"],
            "reasoning": records[j]["reasoning"],
            "similarity": round(float(query_sims[j]), 4),
        }
        for j in chosen
    ]
print(f"Retrieved top-{K} demonstrations for {len(selection)} test questions.")

with open(POOL_PATH, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Saved {len(records)} pool candidates -> {POOL_PATH}")

with open(SELECTION_PATH, "w", encoding="utf-8") as f:
    for item in test_set:
        rec = {
            "sample_id": item["sample_id"],
            "answer_type": item["answer_type"],
            "examples": selection[str(item["sample_id"])],
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Saved retrieval selection -> {SELECTION_PATH}")

all_sims = [ex["similarity"] for exs in selection.values() for ex in exs]
print(
    f"\nSimilarity: mean={np.mean(all_sims):.3f}, "
    f"min={np.min(all_sims):.3f}, max={np.max(all_sims):.3f}"
)

usage = Counter(ex["pool_id"] for exs in selection.values() for ex in exs)
print(f"Distinct pool examples used: {len(usage)} / {K * len(selection)} slots")
print(f"Most reused: {usage.most_common(3)}")

by_type: dict[str, list[float]] = {}
for item in test_set:
    exs = selection[str(item["sample_id"])]
    by_type.setdefault(item["answer_type"], []).extend(e["similarity"] for e in exs)
print("Mean retrieval similarity by test answer_type:")
for cat, vals in sorted(by_type.items()):
    print(f"  {cat}: {np.mean(vals):.3f}  (n={len(vals)})")

worst = max(
    sum(
        len(ex["question"].split())
        + len(ex["reasoning"].split())
        + len(ex["answer"].split())
        for ex in selection[str(item["sample_id"])]
    )
    + len(item["question"].split())
    for item in test_set
)
print(f"Worst-case prompt: ~{worst} words (~{int(worst * 1.3)} tokens).")
print("Done.")
