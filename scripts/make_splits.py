"""Fixed splits for FINAL_v2: test = all mx-* (matrix-grounded, topic-disjoint by
construction: TruthfulQA-derived, never mixed into train); val = 10% grouped
sample of sol+luna; train = rest. Split by pair_id, stratified by source.
Writes dataset/split_{train,val,test}.jsonl + prints counts.
Usage: python scripts/make_splits.py [--seed 7]
"""
import argparse, json, os, random
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
rows = [json.loads(l) for l in open(os.path.join(BASE, "dataset", "FINAL_v2_pairs.jsonl")) if l.strip()]
pairs = {}
for r in rows:
    pairs.setdefault(r["pair_id"], []).append(r)
assert all(sorted(x["label"] for x in v) == ["high", "low"] for v in pairs.values()), "broken pair"
mx = [p for p in pairs if p.startswith("mx-")]
rest = [p for p in pairs if not p.startswith("mx-")]
rng = random.Random(a.seed)
by_src = {}
for p in rest:
    by_src.setdefault(p.split("-")[0], []).append(p)
val, train = [], []
for src, ps in sorted(by_src.items()):
    rng.shuffle(ps)
    k = max(1, int(round(0.10 * len(ps))))
    val += ps[:k]; train += ps[k:]
rng.shuffle(train); rng.shuffle(val)
splits = {"train": train, "val": val, "test": sorted(mx)}
for name, ps in splits.items():
    out = [r for p in ps for r in pairs[p]]
    open(os.path.join(BASE, "dataset", f"split_{name}.jsonl"), "w").write(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in out))
    n_hi = sum(1 for p in ps for r in pairs[p] if r["label"] == "high")
    print(f"{name}: {len(ps)} pairs / {len(out)} rows (high={n_hi} low={len(out)-n_hi})")
# overlap guard
assert not (set(train) & set(val) or set(train) & set(mx) or set(val) & set(mx)), "leak"
print("no pair_id overlap across splits")
