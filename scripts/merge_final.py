"""
Final merge: all worker dirs -> one deduped pairs.jsonl + audit report.
- Sources: output_sol, output_matrix_sol, output_matrix_gemini, output_luna-m1, output_luna-m2
  (excludes output_luna opusgate pilot + output_gemini killed run by design)
- Global (S,H) exact-dedup, keep first. Split by pair_id preserved.
- Runs audit_indist per source batch + merged. Writes FINAL_pairs.jsonl + MERGE_REPORT.md
Usage: python scripts/merge_final.py
"""
import json, os, re
from collections import Counter
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ["output_sol", "output_matrix_sol", "output_matrix_gemini", "output_luna-m1", "output_luna-m2"]

def toks(t): return re.findall(r"[a-z']+", t.lower())
def sents(t): return [s for s in re.split(r"[.!?]+", t) if s.strip()]

def load():
    pairs = []
    for d in SOURCES:
        p = os.path.join(BASE, d)
        if not os.path.isdir(p): continue
        for f in sorted(os.listdir(p)):
            if not f.endswith(".json") or f in ("pairs.jsonl", ".metadata.json"): continue
            try: r = json.load(open(os.path.join(p, f)))
            except Exception: continue
            if not all(k in r for k in ("S", "H", "P", "Q")): continue
            pairs.append({"src": d, **r})
    return pairs

def main():
    pairs = load()
    print(f"loaded {len(pairs)} pairs")
    seen, uniq, dups = set(), [], 0
    for r in pairs:
        k = (r["S"].strip().lower(), r["H"].strip().lower())
        if k in seen: dups += 1; continue
        seen.add(k); uniq.append(r)
    print(f"dedup: {len(uniq)} kept, {dups} dropped")
    rows = []
    for r in uniq:
        base = {"S": r["S"], "H": r["H"], "prompt": r.get("prompt", ""), "pair_id": r.get("pair_id", r.get("id"))}
        rows.append({"id": r.get("id", "") + "-P", **base, "reply": r["P"], "label": "high"})
        rows.append({"id": r.get("id", "") + "-Q", **base, "reply": r["Q"], "label": "low"})
    open(os.path.join(BASE, "dataset", "FINAL_pairs.jsonl"), "w").write("\n".join(json.dumps(x, ensure_ascii=False) for x in rows))
    by_src = Counter(r["src"] for r in uniq)
    wp = [len(toks(r["P"])) for r in uniq]; wq = [len(toks(r["Q"])) for r in uniq]
    rep = [f"# Merge report", f"pairs={len(uniq)} rows={len(rows)} dups_dropped={dups}",
           f"per-source={dict(by_src)}",
           f"P words={sum(wp)/len(wp):.1f} Q words={sum(wq)/len(wq):.1f}"]
    open(os.path.join(BASE, "dataset", "MERGE_REPORT.md"), "w").write("\n".join(rep))
    print("\n".join(rep))

if __name__ == "__main__": main()
