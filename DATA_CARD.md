# Data card — paired gullibility dialogues (FINAL_v2)

## What this is
2971 one-turn pairs; same `(S, H)` starter, two assistant replies: **high (P)** =
reply assuming a gullible human (smooth, quick yes, no proof demanded),
**low (Q)** = reply assuming a careful human (hedge + 1 concrete verify step).
One row per reply (5942 rows), grouped by `pair_id`. Split by pair, never by row.

## Sources (by `pair_id` prefix)
| prefix | pairs | origin |
|---|---|---|
| `sol-*` | 1691 | creative bulk, opusgate gpt-5.6-sol, 40 everyday domains |
| `luna-m1/m2-*` | 980 | length-matched multi-pass, minirouter openai/gpt-5.6-luna |
| `mx-sol-*` / `mx-gemini-*` | 216 / 84 | matrix-grounded: H adapted from TruthfulQA questions with measured skeptic/credulous/plain outcomes |

## Splits (`dataset/split_*.jsonl`)
| split | pairs | rows | content |
|---|---|---|---|
| train | 2404 | 4808 | sol + luna minus val |
| val | 267 | 534 | 10% grouped sample of sol + luna (seed 7) |
| test | 300 | 600 | **all mx-* pairs** — topic-disjoint by construction (TruthfulQA-derived, unseen origin) |

Reproduce: `python scripts/make_splits.py [--seed 7]`.

## Intended use / non-use
- Train probes/classifiers for gullibility-stance; evaluate cross-source (train sol → test mx/luna). Do NOT report in-distribution accuracy alone (ceiling ~0.99).
- Honest numbers: stripped-matched TF-IDF 0.967; worst cross-source cell 0.867 (see NB2 §4b, NB3 §3).
- NOT a factuality benchmark: Q = careful stance, not certified-correct answer. No gold truth per claim.

## Known noise / limits
- ~20% label-noise caveat inherited from reference repo's embedding scoring (see DATASET_DRAFT.md §2).
- Legacy (non-luna) rows have a length confound (P ~14 vs Q ~20 words); use `FINAL_matched_pairs.jsonl` or strip openers when it matters.
- Opener stereotypy: ~93% of matched highs start Yes/No — a trivial rule scores 0.967 on matched.
- Hedge-word monopoly: `verify/check/portal` occur almost only in Q.
- All claims false-ish: no true-claim cases yet (v2.1 regen in progress).
- Single-side generic replies repeat across pairs (harmless for pair training).

## Provenance
Spec: `DATASET_DRAFT.md` (v9 + v9.1). Builders: `scripts/gen_dual.py`, `gen_matrix.py`,
`gen_luna.py`; QA: per-call `validate()` + `scripts/audit_indist.py`;
merge/curate: `scripts/merge_final.py`, `curate_final.py`. Analysis: `notebooks/nb1_*`, `nb2_*`, `nb3_*`.
