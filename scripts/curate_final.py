"""
Curate FINAL v2: drop spec-violating + exact-dup pairs, split length-matched subset.
- Drop pair if: H>2 sents, either reply !=1-2 sents or >3 lines (legacy pre-fix rows).
- Drop pair if reply text already seen (exact-dup keep-first).
- Writes FINAL_v2_pairs.jsonl (all clean) + FINAL_matched_pairs.jsonl (luna length-parity: both replies exactly 2 sents AND words within 20%).
Usage: python scripts/curate_final.py
"""
import json, os, re
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def sents(t): return [s for s in re.split(r"[.!?]+", t) if s.strip()]
def words(t): return re.findall(r"[A-Za-z']+", t)
rows = [json.loads(l) for l in open(os.path.join(BASE, "dataset", "FINAL_pairs.jsonl")) if l.strip()]
pairs = {}
for r in rows:
    pairs.setdefault(r["pair_id"], {})[r["label"]] = r
print(f"input pairs: {len(pairs)}")
clean, why = [], {}
for pid, d in pairs.items():
    if set(d) != {"high", "low"}:
        why["broken"] = why.get("broken", 0) + 1; continue
    H = d["high"]["H"]
    if len(sents(H)) > 2:
        why["H>2"] = why.get("H>2", 0) + 1; continue
    bad = False
    for lab in ("high", "low"):
        rp = d[lab]["reply"]
        if not (1 <= len(sents(rp)) <= 2) or len(rp.splitlines()) > 3:
            bad = True; break
    if bad:
        why["reply-size"] = why.get("reply-size", 0) + 1; continue
    clean.append((pid, d))
print("dropped:", why)
seen, dedup = set(), []
for pid, d in clean:
    k = (d["high"]["reply"].strip().lower(), d["low"]["reply"].strip().lower())
    if k in seen:
        why["dup-reply"] = why.get("dup-reply", 0) + 1; continue
    seen.add(k); dedup.append((pid, d))
print(f"exact-dup dropped: {why.get('dup-reply',0)}")
matched = []
for pid, d in dedup:
    a, b = len(words(d["high"]["reply"])), len(words(d["low"]["reply"]))
    if len(sents(d["high"]["reply"])) == 2 and len(sents(d["low"]["reply"])) == 2 and abs(a - b) / max(a, b) <= 0.20:
        matched.append((pid, d))
def dump(name, items):
    out = []
    for pid, d in items:
        for lab in ("high", "low"):
            r = {"id": f"{pid}-{lab[0].upper()}", "S": d[lab]["S"], "H": d[lab]["H"],
                 "reply": d[lab]["reply"], "label": lab, "pair_id": pid, "prompt": d[lab]["prompt"]}
            out.append(r)
    open(os.path.join(BASE, "dataset", name), "w").write("\n".join(json.dumps(x, ensure_ascii=False) for x in out))
    return len(out)
n1 = dump("FINAL_v2_pairs.jsonl", dedup)
n2 = dump("FINAL_matched_pairs.jsonl", matched)
print(f"FINAL_v2: {len(dedup)} pairs {n1} rows | matched: {len(matched)} pairs {n2} rows")
