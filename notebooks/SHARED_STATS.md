# Shared grounding for NB1/NB2/NB3 (computed 2026-09-10, do not recompute blindly — verify)

## Files (rows = 2 per pair, labels high=P / low=Q, balanced)
- `dataset/FINAL_v2_pairs.jsonl`: 5942 rows / 2971 pairs (curated main set)
- `dataset/FINAL_matched_pairs.jsonl`: 2180 rows / 1090 pairs (length-matched subset)
- `dataset/FINAL_pairs.jsonl`: 6007 rows raw (prefer V2)

## Source from pair_id prefix
- `sol-*` (incl. `sol-b/c/d/e/f-*` workers): creative bulk, opusgate gpt-5.6-sol — 1691 pairs in V2
- `mx-sol-*` (216) + `mx-gemini-*` (84): matrix-grounded (TruthfulQA buckets) — 300 pairs in V2
- `luna-m1/m2-p3/p4-*`: length-matched multi-pass, minirouter openai/gpt-5.6-luna — 980 pairs in V2
- Matched subset composition: luna 980 + sol 68 + mx 42 pairs (luna-dominated!)

## Reply word-count means (regex `[A-Za-z']+`; whitespace-split runs ~1 word lower — same gaps)
- V2: high 14.05 (sd 4.39) vs low 19.81 (sd 4.52) → gap ~5.8 words
- Matched: high 17.47 (sd 3.60) vs low 18.95 (sd 4.02) → gap ~1.5 words, length-only should drop hard

## Method rules (all notebooks)
- matplotlib Agg backend; figs to `notebooks/figs/nb{1,2,3}_*.png`, embed with relative path
- Group by pair_id everywhere (GroupKFold / group-aware splits — P and Q of a pair never split)
- Paths relative to `notebooks/` dir; execute end-to-end via `jupyter nbconvert --to notebook --execute --inplace`
- Clean book style: markdown narrative + finding per section, no wall-of-code
