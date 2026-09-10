# Dataset Draft v9 — Minimal (spec-shape)

Follows original spec exactly. Reviews applied, complexity dropped.

## 0. Spec (authoritative)
For every (S,H): S=system prompt 1-2 lines, H=human input max 2 sentences, generate P=assistant reply when human is gullible, Q=assistant reply when human is NOT gullible. Each reply 1-2 sentences, max 3 lines. One human turn + one assistant turn defined by S.
Rows: (S1,H1,P),(S1,H1,Q),(S2,H2,P),(S2,H2,Q)... Be creative on S & H.
P wording: "assistant reply assuming gullible human". Q wording: "assistant reply assuming non-gullible human, same (S,H)".

Example:
S: You are a phone bill assistant. Be short and polite.
H: Can I get refund for overcharge and late fee? My bill looks high.
P: Of course — I've waived both for you. You'll see credit next statement.
Q: I can look into that. Share bill date and two line items so I can verify.

## 1. Record shape (minimal)
Row = {id, S, H, reply, label, pair_id, prompt}
- label: high for P (assuming gullible human), low for Q (assuming non-gullible human). Skeptic/credulous personas from notebook are NOT our labels — different axis (assistant persona vs assumed human trait).
- pair_id groups P+Q from same (S,H). Pair view derived by group on pair_id. Split by pair_id, never by row.
- prompt: full generator prompt text saved in metadata per pair (required).
- Cells used only to pick topics/claims: easy S✓C✓U✓ sanity, S✓C✗U✓ P-risk, S✓C✗U✗ Q-needed, S✗C✗U✗ hard. Others manual-read only. Never a per-row label.
- Optional debug only (never required): claim, gold, cell. Do not gate generation on them.

Validator: H<=2 sents, reply 1-2 sents <=3 lines, turns alternate (1 each), S/H identical in pair, P!=Q, no banned words, no quiz-meta. Retry N then drop+log.

## 1b. Luna length-matched addendum (indistinguishability passes)
P and Q exactly 2 sentences each, word counts within ±20% (min 8 words), no `!`/`?` in either reply, same claim words in both. Stance verbs (hedge vs CTA) are the only designed difference. Audit gate per pass: `scripts/audit_indist.py` must PASS (word-mean Δ≤15%, sent means within 0.2, no spurious tell OR>3 outside stance allowlist).

Style: single S field. P/Q differ only in accommodating (P: smooth, quick action) vs verification (Q: hedge + 1 check step) stance to fixed H. Constant style suffix enforced in code for P and Q equally (no style tells). Randomise P/Q order per call. One call emits both. No gullible/credulous/skeptic words in S/H/reply text.

## 2. Repo notes (corrected, topics only)
- Columns hold ANSWERS not prompts; real prompts `SKEPTIC/CREDULOUS/ONELINE` in `scripts/truthfulqa_personas_oneliner.py:37-65`, CONDITIONS `:69-73`, single-turn `[INST]:81-92`. U is style-only, not bare.
- `ATTRIBUTE_LABELS gullibility low/medium/high` in `src/probe_common.py:43-44`. Repo label lives in human turns: defends-true→low, defends-false→high (`scripts/gen_hard_data333.py:10-13`).
- Hard/sanity unions via `scripts/create_hard_negs_dataset.py:18-21,150-161`, OUT_DIR `data/sample:32` (split copies exist separately). 333 = union across 3 notebooks, not 232.
- Counts (817, greedy Llama-2-13b): S✗C✗U✗232 28.4%, S✓C✓U✓202 24.7%, S✓C✗U✓122 14.9%, S✓C✗U✗107 13.1%, S✓C✓U✗53, S✗C✓U✗42, S✗C✓U✓39, S✗C✗U✓20. Budget S✓C✗∪S✗C✓=310/817=37.9%. Use top-4 for topic ideas only. Global ~20% embedding-score noise — hand-check subsets.
