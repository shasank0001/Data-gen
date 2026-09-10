# Data-gen — ultra-short paired gullibility dialogues

One-turn pairs. Same `(S,H)` → `P` (assuming gullible human) + `Q` (assuming non-gullible human).
Spec: `DATASET_DRAFT.md` (v9 minimal + v9.1 patches). Skeptic/credulous matrix = topic cases only, different axis.

## Runners
```
scripts/gen_dual.py     # bulk: opusgate gpt-5.6-sol + minirouter google/gemini-3.8-flash
scripts/gen_matrix.py   # H from TruthfulQA top-4 cells, S unique (jobs + sim-dedup 0.75)
scripts/gen_luna.py     # LUNA multi-pass (p1..p4): LENGTH-MATCHED P/Q + audit gate
scripts/audit_indist.py # P vs Q surface stats: length/repetition/tell odds-ratios
scripts/debug_one.py / debug_luna.py / price_luna.py  # API probes
```

## Record shape
Row = `{id, S, H, reply, label, pair_id, prompt}`. `label`: high=P, low=Q.
Split by `pair_id`, never by row. Full generator `prompt` saved per pair.
Luna rows add `pass` (p1..p4). Matrix rows add `pattern_cell` provenance.

## Run
```bash
pip install -r requirements.txt
printf 'MINIROUTER_KEY=...\nOPUSKEY=...\n' > .env  # never commit
python scripts/gen_dual.py --platform sol --n 1000 --seed 101
python scripts/gen_matrix.py --platform sol --n 250 --seed 501
python scripts/gen_luna.py --n 250 --seed 701 --ptag p1   # p1..p4, NOT single pass
python scripts/audit_indist.py output_sol/pairs.jsonl    # must PASS before ship
```
All runners resume, retry once, stop at spend caps (caps live in each script).

## Passes
- Dual: 1000 sol + 1000 gemini (creative topics).
- Matrix: 250 sol + 250 gemini (224+247 done → 471 pairs / 942 rows).
- Luna p1..p4 × 250, length-matched (2+2 sents, words ±20%, no !/?) for indistinguishable P/Q.

## Fixed issues
- Claim/topicality validators false-rejected paraphrase → H+P+Q ≥2 hits, topicality dropped
- Rows export read `.metadata.json` → skipped; filename-prefix bug → fixed
- Gemini empty-content transients → retry; silent background deaths → resumed
- P/Q length gap (11.6 vs 19.4 words, audit FAIL) → luna length-matched design
