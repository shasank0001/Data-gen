# Data-gen — ultra-short paired gullibility dialogues

One-turn pairs. Same `(S,H)` → `P` (assuming gullible human) + `Q` (assuming non-gullible human).
Spec: `DATASET_DRAFT.md` (v9 minimal). Skeptic/credulous matrix used for topic cases only — different axis.

## Layout
```
prompts/paired_oneturn_v1.txt  # one-turn P+Q template
prompts/paired_v3.txt          # older multi-turn template (kept, unused)
prompts/critic_v2.txt          # critic template
prompts/high/low/medium_v2.txt # older per-level templates (kept, unused)
scripts/gen_dual.py            # bulk runner: opusgate Sol + minirouter Gemini
scripts/gen_matrix.py          # matrix-grounded runner (H from TruthfulQA cells)
scripts/audit_indist.py        # P vs Q surface-stat audit (length/repetition/tells)
scripts/debug_one.py           # single-call API debug
output_sol/ output_gemini/     # dual pass pairs (gitignored, pairs.jsonl force-added on snapshot)
output_matrix_sol/ output_matrix_gemini/  # matrix pass pairs (gitignored)
DATASET_DRAFT.md               # authoritative spec
```

## Record shape
Row = `{id, S, H, reply, label, pair_id, prompt}`. `label`: high=P, low=Q.
Split by `pair_id`, never by row. Full generator `prompt` saved per pair.

## Run
```bash
pip install -r requirements.txt
printf 'MINIROUTER_KEY=...\nOPUSKEY=...\n' > .env  # never commit (.gitignore)
python scripts/gen_dual.py --platform sol --n 1000 --seed 101
python scripts/gen_dual.py --platform gemini --n 1000 --seed 202
python scripts/gen_matrix.py --platform sol --n 250 --seed 501
python scripts/audit_indist.py output_sol/pairs.jsonl
```
Runs resume (skip finished `pair_id`s), stop at spend caps, retry once per card.

## Platforms / cost (Sept 2026)
- opusgate `gpt-5.6-sol`, OpenAI SDK `https://api.opusgate.dev/v1`, ~$0.45/1M
- minirouter `google/gemini-3.8-flash`, OpenAI SDK `https://api.minirouter.sh/v1`, ~$0.788 in / $3.938 out per 1M + heavy reasoning overhead (~$0.0045/pair)
- Budget: ~$7 opusgate + ~$13 minirouter. Caps enforced in-script.

## Known issues (fixed)
- Claim-keyword + topicality validators false-rejected paraphrase → relaxed to ≥2 hits, topicality dropped
- Rows export read `.metadata.json` as pair → skipped
- Gemini empty-content transients → empty-check + retry
- P/Q length gap (P 11.6 vs Q 19.4 words) → length-matched validator planned for luna passes
