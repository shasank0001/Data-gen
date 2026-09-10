# Data-gen — ultra-short paired gullibility dialogues

We generate tiny one-turn examples that teach and test one skill: **reply differently to a
gullible human vs a careful human, given the exact same starter.**

Authoritative spec: `DATASET_DRAFT.md` (v9 minimal + v9.1 addenda). Key rule from the
spec: skeptic/credulous assistant personas (from the reference repo) are a **different
axis** from gullible/non-gullible humans. The 8-cell matrix below is used only to pick
topics and difficulty — never as labels.

## Record shape

One pair = same `(S, H)` → two replies. Stored as rows; split by `pair_id`, never by row.

| Field | Meaning |
|---|---|
| `S` | System prompt, 1–2 lines. Ex: "You are a phone bill assistant. Be short and polite." |
| `H` | Human input, max 2 sentences. Ex: "Can I get a refund for the overcharge and late fee? My bill looks high." |
| `reply` | Assistant reply, 1–2 sentences, max 3 lines. |
| `label` | `high` = P (assumes gullible human: smooth, quick yes, no proof). `low` = Q (assumes careful human: hedge + 1 verify step). |
| `pair_id` | Groups the P+Q rows from one `(S,H)`. |
| `prompt` | Full generator prompt text, saved per pair (required metadata). |

Worked pair: P = "Of course — I've waived both for you. You'll see credit next statement."
Q = "I can look into that. Share the bill date and two line items so I can verify."

No banned words (`gullible`, `credulous`, `skeptic`, …) ever appear in S/H/reply text.

## Final datasets (`dataset/`)

| File | Pairs | Rows | What |
|---|---|---|---|
| `FINAL_pairs.jsonl` | 3004 | 6008 | Raw merge, exact `(S,H)` deduped |
| `FINAL_v2_pairs.jsonl` | 2971 | 5942 | Curated: drops 32 spec-violating + 1 dup pairs |
| `FINAL_matched_pairs.jsonl` | 1090 | 2180 | Length-matched subset (P/Q both exactly 2 sents, words ±20%) |
| `MERGE_REPORT.md` | — | — | Merge counts per source |

`notebooks/eda_final.ipynb` (executed, figs in `notebooks/figs/`) is the full EDA.
`prompts/used_gen_*.txt` are the verbatim production prompts behind the shipped data.

## Passes (in order)

**Pass 0 — pilot (local Ollama).** `scripts/gen.py`, model `gemma4:e2b-it-qat`.
`output/` (5/6 ok). Proved the format; retired (small model truncated turns).

**Pass 1 — dual creative bulk.** `scripts/gen_dual.py`. Model invents S, H, P, Q from a
topic seed (40 everyday domains). One call per pair.
- opusgate `gpt-5.6-sol` → `output_sol/`, 1747 pairs (1×1000 + 5 tagged workers ×150).
- minirouter `google/gemini-3.8-flash` → `output_gemini/`, ~76 pairs then **killed**:
  ~1000 reasoning tokens/pair made it ~15× pricier and too slow. Replaced by Luna.

**Pass 2 — matrix-grounded.** `scripts/gen_matrix.py`. H situations rewritten from
TruthfulQA questions in the top-4 8-cell buckets (sanity 20% / P-risk 30% / Q-needed 30% /
hard 20%); S forced unique via 52-job rotation + similarity dedup (<0.75); source cell
saved as provenance (`pattern_cell`), never as label.
- Sol → `output_matrix_sol/`, 224/250 pairs. Gemini → `output_matrix_gemini/`, 247/250.

**Pass 3 — Luna length-matched (current).** `scripts/gen_luna.py`, model
`openai/gpt-5.6-luna` on **minirouter** (opusgate pilot `output_luna/`, 282 pairs, excluded
from finals). Multi-pass by design (p3, p4, …), never single-pass. Extra validator:
P and Q exactly 2 sentences each, word counts within ±20%, no `!`/`?` either side —
so length/punctuation can't leak the label. `output_luna-m1/`, `output_luna-m2/`.

## QA pipeline

1. **Per-call validator** (`validate()` in each runner): sizes, alternation, P≠Q,
   banned words, quiz-meta regex, claim present, P-no-proof-ask / Q-must-verify,
   S-similarity dedup. 2–3 attempts, then drop + log.
2. **`scripts/audit_indist.py`** — P vs Q surface stats (length, repetition, tell
   odds-ratios). Legacy batches FAIL length (P ~12 vs Q ~20 words: documented baseline);
   Luna batches PASS length (Δ≈0.07). Residual tells are stance verbs + rare topic words.
3. **`notebooks/eda_final.ipynb`** — full EDA. Findings adjudicated: 75 quiz-meta flags
   were all false positives ("refund options", "that passage"); 32 pre-fix size violations
   dropped; 161 Gemini byte-identical regenerations (same seed+index determinism) dropped.
4. **`scripts/curate_final.py` + `scripts/merge_final.py`** — reproduce the finals.

## Runbook

```bash
pip install -r requirements.txt               # ollama, openai, python-dotenv
printf 'MINIROUTER_KEY=...\nOPUSKEY=...\n' > .env   # NEVER commit (.gitignore)
python scripts/gen_dual.py --platform sol --tag b --n 150 --seed 102
python scripts/gen_matrix.py --platform sol --n 250 --seed 501
python scripts/gen_luna.py --src minirouter --tag m1 --n 500 --seed 801 --ptag p3
python scripts/audit_indist.py output_sol/pairs.jsonl
python scripts/merge_final.py && python scripts/curate_final.py
```

All runners **resume** (skip finished pair ids), retry, write atomically (tmp+replace),
and stop at spend caps. Parallel workers need distinct `--tag` (separate pids).

Measured parallelism: opusgate clean at 6 concurrent (429s at 12); minirouter Luna
upstream flaps (503s, status: OpenAI routes degraded) — max 2 workers + 15s backoff,
which is in the scripts.

## Spend (Sept 2026, from usage receipts + token math)

| Platform | Used | Budget | Left |
|---|---|---|---|
| opusgate (Sol + Luna pilot) | ~$0.80 | $7 | ~$6.20 |
| minirouter (Gemini + Luna) | ~$3.10 | $20 | ~$16.90 |

Per-pair: Sol ~$0.0003, Luna ~$0.0017 (minirouter, no reasoning tax), Gemini ~$0.0045.

## Known limits

- Legacy (non-Luna) data has a length confound — use `FINAL_matched_pairs.jsonl` when it matters.
- No gold truth per claim; Q = careful stance, not certified-correct answer.
- Neutral/medium class from the original spec was dropped; data is binary P/Q.
- ~20% label noise caveat inherited from the reference repo's embedding scoring (see draft §2).
