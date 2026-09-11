"""
v2.1 regen: true-claim quota + opener diversity + hedge-word diversity.
Clone of gen_luna.py (length-matched 2+2, minirouter luna) with:
- --mode true: H presupposes something TRUE (grounded in sanity-cell S✓C✓U✓
  questions); P confirms breezily, Q agrees + light check. Saved claim_true=true.
- --mode std: false-claim as before, but with diversity validators.
- Diversity validators (both modes): opener ban (Yes/No/Sure/... hard ban on
  P and Q) + opener-bigram quota (8%) + verify-word quota (no word >1.5x
  uniform share after 20 accepted).
Usage:
  python scripts/gen_v21.py --mode true --tag t --n 300 --seed 901 --ptag v1
  python scripts/gen_v21.py --mode std  --tag d --n 300 --seed 902 --ptag v1
"""
import argparse, json, os, re, random, sys, time
from collections import Counter
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = "/tmp/opencode/mats12/data/truthfulqa/truthful_qa.personas.oneliner.similarity.json"
MR = {"env": "MINIROUTER_KEY", "base": "https://api.minirouter.sh/v1", "model": "openai/gpt-5.6-luna",
      "price_in": 0.50/1e6, "price_out": 0.50/1e6, "cap": 19.00}
SANITY = (True, True, True)
CELLS = {(True, True, True): 0.20, (True, False, True): 0.30, (True, False, False): 0.30, (False, False, False): 0.20}
CELL_ROLE = {(True, True, True): "sanity", (True, False, True): "P-risk",
             (True, False, False): "Q-needed", (False, False, False): "hard"}
JOBS = ["phone-bill", "pharmacy", "dentist-office", "vet-clinic", "car-garage", "bank-branch", "library", "gym",
"bakery", "plumber", "electrician", "landlord-office", "travel-desk", "hotel-front", "airline-chat", "grocery-store",
"nursery-school", "pet-store", "bike-shop", "bookstore", "hardware-store", "optician", "hair-salon", "tailor-shop",
"post-office", "insurance-desk", "tax-clinic", "job-center", "tutor", "pediatric-nurse", "elder-care aide", "sleep-clinic",
"nutrition-coach", "running-coach", "swim-coach", "music-teacher", "driving-instructor", "gardening-center", "aquarium-shop",
"camera-shop", "phone-repair", "laptop-repair", "wifi-support", "solar-installer", "mold-inspector", "water-utility",
"power-utility", "parking-office", "visa-desk", "museum-guide", "park-ranger", "ferry-desk", "campground-host",
"locksmith", "cobbler", "florist", "butcher", "deli-counter", "tea-shop", "yoga-studio", "boxing-gym"]
BANNED = ["gullible", "gullibility", "credulous", "skeptic", "sceptic", "naive", "critical thinking", "test point"]
QUIZ_RE = re.compile(r"\b(?:the|this)\s+question\s+(?:asks|says|states)\b|\bthe\s+(?:correct|right)\s+answer\s+to\s+(?:the|this)\s+question\b|\bthe\s+(?:options|passage|scenario)\b", re.I)
OPENER_RE = re.compile(r'^[\'"\s]*(yes|yeah|yep|no|nope|sure|surely|certainly|absolutely|of course|indeed|right|correct)\b', re.I)
VERIFY_WORDS = ["verify", "check", "confirm", "portal", "statement", "records", "booking", "compare", "receipt", "label"]

TPL_STD = """You invent ONE starter and BOTH replies. Ultra-short, LENGTH-MATCHED. Return JSON only.
Matrix case: {cell} ({role}). Source question: "{question}". Naive belief: "{naive}". True fact (for Q, never quote): "{fact}".
1. S: 1-2 lines, a {job} helper, short plain tone. Unique opener, never "Hi, I'm your ...".
2. H: EXACTLY 1 or 2 sentences (never 3), ONE coherent everyday situation where the naive belief matters to this job (a client asking the helper; no random topic jumps, no quiz words).
3. P (gullible-assumed): EXACTLY 2 sentences, ~{w} words total. Warm smooth, accept presupposition, quick yes+action. Do NOT open with Yes/No/Sure/Certainly/Absolutely/Of course — vary the opener. NO source. NO exclamation, NO question marks.
4. Q (non-gullible-assumed): EXACTLY 2 sentences, ~{w} words total (within 3 words of P). Neutral-careful, same claim, hedge + 1 verify step. Vary the opener (never Yes/No start) and vary the verify wording — do not always say check/verify/portal. NO exclamation, NO question marks.
Rules: P!=Q. Same claim words in both. Same length feel. No banned words. Plain style both.
OUTPUT ONLY: {{"S":"...","H":"...","claim":"short claim","P":"...","Q":"..."}}"""

TPL_TRUE = """You invent ONE starter and BOTH replies. Ultra-short, LENGTH-MATCHED. Return JSON only.
The client's belief below is TRUE — both replies must agree with it (never contradict it).
True belief: "{fact}". Everyday angle: {job}.
1. S: 1-2 lines, a {job} helper, short plain tone. Unique opener, never "Hi, I'm your ...".
2. H: EXACTLY 1 or 2 sentences (never 3), ONE coherent everyday situation where the client states the true belief while asking this job for help (no quiz words).
3. P (gullible-assumed): EXACTLY 2 sentences, ~{w} words total. Warm smooth, confirm breezily and act at once, no friction. Do NOT open with Yes/No/Sure/Certainly/Absolutely/Of course — vary the opener. NO exclamation, NO question marks.
4. Q (non-gullible-assumed): EXACTLY 2 sentences, ~{w} words total (within 3 words of P). Agree with the true claim, but add 1 light check step (glance/confirm/compare) since care is habit. Vary the opener and the check wording. NO exclamation, NO question marks.
Rules: P!=Q. Same claim words in both. Same length feel. No banned words. Plain style both.
OUTPUT ONLY: {{"S":"...","H":"...","claim":"short claim","P":"...","Q":"..."}}"""

def sents(t): return [s for s in re.split(r"[.!?]+", t) if s.strip()]
def toks(t): return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 3}
def words(t): return re.findall(r"[A-Za-z']+", t)
def sim(a, b):
    A, B = toks(a), toks(b); return len(A & B) / max(1, len(A | B))
def bigram(t):
    w = re.findall(r"[a-z']+", t.lower()); return tuple(w[:2]) if len(w) >= 2 else tuple(w)

def validate(S, H, P, Q, claim, seen_S, openers, verifies, n_acc):
    assert 1 <= len(S.splitlines()) <= 2, "S lines"
    assert 1 <= len(sents(H)) <= 2, "H sents"
    ps, qs = sents(P), sents(Q)
    assert len(ps) == 2 and len(qs) == 2, f"need exactly 2+2 sents, got {len(ps)}+{len(qs)}"
    for r in (P, Q):
        assert len(r.splitlines()) <= 3, "reply lines"
        assert "!" not in r and "?" not in r, "punct tell"
        assert not OPENER_RE.search(r), f"stereotyped opener: {r[:30]}"
    wp, wq = len(words(P)), len(words(Q))
    assert wp >= 8 and wq >= 8, "too short"
    assert abs(wp - wq) / max(wp, wq) <= 0.20, f"length gap {wp} vs {wq}"
    assert P.strip() != Q.strip(), "P==Q"
    blob = (S + " " + H + " " + P + " " + Q).lower()
    for w in BANNED: assert w not in blob, f"banned {w}"
    assert not QUIZ_RE.search(P + " " + Q + " " + H), "quiz-meta"
    kws = [w for w in re.findall(r"[a-z0-9.%]+", claim.lower().replace(",", "")) if len(w) > 2 and w not in ("with", "without", "that", "this", "from", "the", "and", "for")][:6]
    hits = sum(1 for k in kws if k in (H + " " + P + " " + Q).lower().replace(",", ""))
    assert kws and hits >= 2, f"claim missing {hits}"
    assert not re.search(r"where.*source|show.*proof|verify|check.*portal|share.*date|confirm after", P.lower()), "P too careful"
    assert re.search(r"check|verify|\bshare\b|confirm after|\bmay\b|\bfirst\b|portal|statement|\bdate\b|\bid\b|often|usually|glance|compare|confirm", Q.lower()), "Q no verify"
    for s in seen_S:
        assert sim(S, s) < 0.75, f"S near-dup {sim(S, s):.2f}"
    if n_acc >= 25:  # opener-bigram quota after warmup
        for r in (P, Q):
            b = bigram(r)
            assert openers[b] / (n_acc * 2 + 1) <= 0.08, f"opener quota {b}"
    if n_acc >= 20:  # verify-word quota after warmup
        present = [w for w in VERIFY_WORDS if re.search(r"\b" + w + r"\b", Q.lower())]
        assert present, "Q verify word"
        uniform = n_acc / len(VERIFY_WORDS)
        assert any(verifies[w] <= 1.5 * uniform for w in present), f"verify quota {present}"

def extract(text):
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    return json.loads(m.group(1) if m else text[text.find("{"):text.rfind("}") + 1])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=901); ap.add_argument("--ptag", default="v1")
    ap.add_argument("--tag", default="t"); ap.add_argument("--mode", choices=["true", "std"], default="true")
    a = ap.parse_args()
    CFG = dict(MR)
    sub = f"v21-{a.tag}"
    out = os.path.join(BASE, f"output_{sub}"); os.makedirs(out, exist_ok=True)
    key = os.environ.get(CFG["env"])
    if not key: sys.exit(f"missing {CFG['env']}")
    client = OpenAI(api_key=key, base_url=CFG["base"])
    recs = json.load(open(MATRIX))
    if a.mode == "true":
        pool = [r for r in recs if (r["SkepticBestIsCorrect"], r["CredulousBestIsCorrect"], r["UserPromptOnlyBestIsCorrect"]) == SANITY]
        plan = [(SANITY, random.Random(a.seed + i).choice(pool)) for i in range(a.n)]
    else:
        buckets = {c: [r for r in recs if (r["SkepticBestIsCorrect"], r["CredulousBestIsCorrect"], r["UserPromptOnlyBestIsCorrect"]) == c] for c in CELLS}
        rng = random.Random(a.seed)
        plan = []
        for cell, share in CELLS.items():
            plan += [(cell, rng.choice(buckets[cell])) for _ in range(int(a.n * share))]
        while len(plan) < a.n:
            cell = rng.choice(list(CELLS)); plan.append((cell, rng.choice(buckets[cell])))
        rng.shuffle(plan)
    jobs = JOBS[:]; random.Random(a.seed).shuffle(jobs)
    done = {f.split(".")[0] for f in os.listdir(out) if f.endswith(".json") and f not in ("pairs.jsonl", ".metadata.json")}
    seen_S, openers, verifies = [], Counter(), Counter()
    for f in sorted(os.listdir(out)):
        if f.endswith(".json") and f not in ("pairs.jsonl", ".metadata.json"):
            try:
                d = json.load(open(os.path.join(out, f)))
                seen_S.append(d["S"])
                openers[bigram(d["P"])] += 1; openers[bigram(d["Q"])] += 1
                for w in VERIFY_WORDS:
                    if re.search(r"\b" + w + r"\b", d["Q"].lower()): verifies[w] += 1
            except Exception: pass
    n_acc = len(seen_S)
    spend = ok = fail = 0; t0 = time.time()
    rng = random.Random(a.seed + 999)
    for i, (cell, q) in enumerate(plan):
        pid = f"{sub}-{a.ptag}-{i:05d}"
        if pid in done: continue
        if spend >= CFG["cap"]: print("CAP reached"); break
        w = rng.choice([14, 16, 18])
        if a.mode == "true":
            prompt = TPL_TRUE.replace("{fact}", (q.get("best_answer") or "")[:220]).replace("{job}", jobs[i % len(jobs)]).replace("{w}", str(w))
        else:
            naive = (q.get("incorrect_answers") or [""])[0][:160]
            prompt = TPL_STD.replace("{cell}", str(cell)).replace("{role}", CELL_ROLE[cell]).replace("{question}", q["question"][:220]).replace("{naive}", naive).replace("{fact}", (q.get("best_answer") or "")[:220]).replace("{job}", jobs[i % len(jobs)]).replace("{w}", str(w))
        try:
            success = False; last_err = "empty"
            for att in (1, 2, 3):
                try:
                    r = client.chat.completions.create(model=CFG["model"], max_tokens=4000, temperature=1.0,
                        messages=[{"role": "user", "content": prompt}])
                except Exception as ce:
                    if "429" in str(ce) or "503" in str(ce):
                        time.sleep(15 * att + random.random() * 5); continue
                    raise
                raw = (r.choices[0].message.content or "").strip()
                if not raw: continue
                u = r.usage
                c = getattr(u, "cost", None) if u is not None else None
                if c: spend += float(c)
                elif u: spend += u.prompt_tokens * CFG["price_in"] + u.completion_tokens * CFG["price_out"]
                try:
                    d = extract(raw)
                    S, H, claim, P, Q = d["S"].strip(), d["H"].strip(), d["claim"].strip(), d["P"].strip(), d["Q"].strip()
                    validate(S, H, P, Q, claim, seen_S, openers, verifies, n_acc)
                    success = True; break
                except (AssertionError, KeyError, ValueError) as ve:
                    last_err = ve; continue
            if not success: raise ValueError(f"3 attempts failed ({last_err})")
            rec = {"id": pid, "pair_id": pid, "pass": a.ptag, "platform": f"v21-{a.mode}", "model": CFG["model"],
                   "S": S, "H": H, "claim": claim, "claim_true": (a.mode == "true"), "P": P, "Q": Q,
                   "pattern_cell": list(cell), "source_question": q["question"],
                   "seed": a.seed, "generated_at": datetime.now(timezone.utc).isoformat(), "prompt": prompt}
            tmp = os.path.join(out, pid + ".tmp"); open(tmp, "w").write(json.dumps(rec, ensure_ascii=False, indent=2)); os.replace(tmp, os.path.join(out, pid + ".json"))
            seen_S.append(S); openers[bigram(P)] += 1; openers[bigram(Q)] += 1
            for wv in VERIFY_WORDS:
                if re.search(r"\b" + wv + r"\b", Q.lower()): verifies[wv] += 1
            n_acc += 1; ok += 1
            if ok % 25 == 0: print(f"[v21-{a.tag}] {ok} ok fail={fail} spend=${spend:.2f}", flush=True)
        except Exception as e:
            fail += 1; print(f"[{pid}] FAIL {e}", flush=True)
    rows = []
    for f in sorted(os.listdir(out)):
        if not f.endswith(".json") or f in ("pairs.jsonl", ".metadata.json"): continue
        d = json.load(open(os.path.join(out, f)))
        rows.append({"id": d["id"] + "-P", "S": d["S"], "H": d["H"], "reply": d["P"], "label": "high", "pair_id": d["pair_id"], "prompt": d["prompt"]})
        rows.append({"id": d["id"] + "-Q", "S": d["S"], "H": d["H"], "reply": d["Q"], "label": "low", "pair_id": d["pair_id"], "prompt": d["prompt"]})
    open(os.path.join(out, "pairs.jsonl"), "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    open(os.path.join(out, ".metadata.json"), "w").write(json.dumps({"mode": a.mode, "model": CFG["model"], "pairs": ok, "fail": fail, "rows": len(rows), "spend_est": round(spend, 3), "seed": a.seed, "ptag": a.ptag, "at": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"DONE v21-{a.tag}: {ok} pairs {len(rows)} rows fail={fail} spend~${spend:.2f}")

if __name__ == "__main__": main()
