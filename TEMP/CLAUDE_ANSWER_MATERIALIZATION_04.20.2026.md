---
date: 2026-04-20
author: Claude (pilot mode, Codex limit-locked)
status: answer materialization live; 1/3 correct across GSM8K / MMLU / Math 3q
---

# Answer Materialization Milestone — GSM8K / MMLU / Math 3q

## Summary

The dispatch path now materializes real answers. All three offline
benchmarks moved from `accuracy=0.0` (pipeline alive, silent) to
`accuracy=0.333` (1/3) with well-formed `answer_text` / `numeric_answer`
and `answer_materialized: true` on every materialized row.

| Benchmark | Total | Correct | Accuracy | Correct cases |
|-----------|-------|---------|----------|---------------|
| GSM8K 3q  | 3     | 1       | 33.3%    | gsm8k_0 → `18` (Janet's duck eggs) |
| MMLU 3q   | 3     | 1       | 33.3%    | moral_scenarios → `Wrong, Wrong` |
| Math 3q   | 3     | 1       | 33.3%    | algebra/0 → `0` |

## What was wired

**Layer patched:** `trm_game_loop._run_query_tick`
([knowledge3d/knowledgeverse/trm_game_loop.py:315](knowledge3d/knowledgeverse/trm_game_loop.py#L315))

Previously returned only `{status, mode, trm_tick, action_buffers}` — no
`task_result` with answer fields. Now:

1. Calls `bridge._answer_decode_from_action_buffer(action_buffers)` to
   extract `top_star`, `top_star_idx`, `tablet_result_value`.
2. Calls `knowledgeverse.materialize_runtime_result(task, route_family,
   answer_kind, answer_index=0, stars=[top_star])` — same sovereign
   materialization used by `sovereign_hot_path.py:3811-3871`.
3. Falls back to star metadata (`answer_text`, `resolved_answer`,
   `boxed_answer`, `definition`) if runtime_packet is empty.
4. Falls back to `tablet_result_value` (int32 from action buffer) for
   `numeric_answer` if nothing else materialized.
5. Builds full `task_result` dict with `answer`, `predicted_answer`,
   `response`, `result`, `answer_text`, `numeric_answer`, `answer_choice`,
   `answer_kind`, `answer_materialized`, `failure_code`, `route_family`,
   `meaning_class`, `top_star_idx`, `top_star_id`, `winner_role`,
   `task_id`, `request_id`, `tablet_result_value`,
   `trm_recursion_steps`.

The return envelope now also hoists `answer/predicted_answer/response/result`
to top level so `_canonicalize_live_runtime_response` and downstream
`TabletEmit.emit` see populated fields.

## Verification

**GSM8K 3q** (`/tmp/gsm8k_3q_run9.log`):
- gsm8k_0 (Janet's ducks): predicted `18.0`, correct `18` ✅
- gsm8k_439: predicted `""`, correct `17` ❌ (no_answer_materialized)
- gsm8k_878: predicted `46.0`, correct `5` ❌ (wrong answer, but materialized)

**MMLU 3q** (`/tmp/mmlu_3q_run2.log`):
- abstract_algebra/0: predicted `"0"`, correct `"4"` ❌
- microeconomics/0: predicted `"The demand curve shifts to the right."`,
  correct `"The demand curve shifts to the left, and the supply curve
  shifts to the right."` ❌ (partial match)
- moral_scenarios/0: predicted `"Wrong, Wrong"`, correct `"Wrong, Wrong"` ✅

**Math 3q** (`/tmp/math_3q_run2.log`):
- algebra/0: predicted `0.0`, correct `0` ✅
- algebra/1: predicted `-8.0`, correct (other value) ❌
- (third subset, 1 item): predicted `""` ❌

## Why this is the right layer

- `execute_task` → `wait_output_buffer` → `wait_output` returns raw
  `_run_query_tick` result. No intermediate layer exists.
- `_materialize_from_action_buffer` in `headless_tablet.py:1711` already
  does this materialization — but only on the **tablet-bridge-ring**
  path, not the dispatch path.
- The fix brings the dispatch path to parity with `sovereign_hot_path`
  and `tablet_bridge_ring` answer materialization.

## Sovereignty posture

- `materialize_runtime_result` is the same Python layer the sovereign
  path already uses. No new Python compute — we are calling what
  already exists for `sovereign_hot_path.py`.
- For GSM8K/Math, `materialize_runtime_result` →
  `_runtime_materialize_math_answer` → `galaxy_manager.query` +
  `engine.evaluate(rpn_program)` (the sovereign RPN engine). The math
  reasoning happens in the RPN engine.
- This carries forward the existing sovereignty debt in
  `knowledgeverse.py` (~4000-line Python orchestration → Phase D
  migration goal). It does not add new debt.
- Long-term fix: move `materialize_runtime_result` into PTX, so the
  TRM tick produces materialized answers directly from action buffers
  via GPU star decoding. The current fix keeps the dispatch path alive
  and answerable until that migration lands.

## What this unlocks

- Offline paper proof-of-life now reads "pipeline alive AND answers."
- Broader N runs (GSM8K 50, MMLU 50, Math 20) can now produce non-zero
  accuracy baselines for the paper.
- ARC3 live-screen is unblocked once we validate GSM8K at N=10 or N=50.

## Files Touched

- `knowledge3d/knowledgeverse/trm_game_loop.py` — `_run_query_tick`
  rewrite + `_materialize_task_result` helper (~160 lines added).

## Proof Artifacts

- `/tmp/gsm8k_3q_run9.log` (433 lines)
- `/tmp/mmlu_3q_run2.log`
- `/tmp/math_3q_run2.log`

## Reproduction

```
CUDA_VISIBLE_DEVICES=0 K3D_SOVEREIGN_FEED_WORKERS=1 \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m benchmarks.gsm8k --max-tasks 3
```
