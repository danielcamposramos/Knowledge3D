# Sweep Survivability + Composition-Health Diagnostic — 2026-04-17

**Author:** Claude (Architecture Partner)
**For:** Codex
**Predecessor:** `CLAUDE_VALIDATION_SWEEP_50x_04.17.2026.md`
**Scope:** Two-round plan. **Round A** fixes the sweep so it can finish end-to-end. **Round B** (spec later, after we have the full 250-item picture) addresses router/executor issues. Do Round A only. Do not touch the router or the executor chain this round.

---

## 1. What the three landed JSONs tell us

The engine is composed correctly — ring used, tick driver healthy, GPU firing. The work is real. What we have are two distinct problems riding the same run:

**Problem 1 — sweep can't finish.** Daniel killed it mid-tranche. Runner in socket poll, daemon in user-space CPU. Per-item p95 latency is already at 37–38 s on MATH_competitions; LHE multi-hop will push some items much higher. There is no per-item wall clock enforced by the ring, and no per-benchmark wall clock enforced by the sweep runner. So one pathological item can hold the whole sweep hostage.

**Problem 2 — meaning-class distribution is degenerate.** Across the 150 items that did run:

| Benchmark | FACTUAL_RECALL | DEFINITION_LOOKUP | MULTI_HOP_INFERENCE | NUMERIC_COMPUTE | SPATIAL_TRANSFORM | COMPARATIVE_CHOICE | GROUNDED_DIALOG | GENERATIVE_COMPOSITION |
|---|---|---|---|---|---|---|---|---|
| MMLU        | 29 | 0 | 0 | **21** | 0 | 0 | 0 | 0 |
| GSM8K       | 0  | 0 | 0 | **50** | 0 | 0 | 0 | 0 |
| Math        | 0  | 0 | 0 | **49** | 0 | 0 | 0 | 1 |
| **Union**   | 29 | 0 | 0 | **120** | 0 | 0 | 1 | 0 |

Three out of eight classes ever fire. GSM8K word problems collapse onto NUMERIC_COMPUTE (not MULTI_HOP_INFERENCE). MMLU law/security/econ-non-compute questions land in NUMERIC_COMPUTE 42% of the time, triggering the math lane on text, which explains MMLU 10/50 = 20% (below 25% 4-way chance). **§2.2 of the sweep spec fails.** This is a router resolution issue, not a knowledge-shelves issue.

The two problems need to be fixed in order. Finish the sweep first so we know LHE and ARC-AGI-1 distributions before re-shaping the router.

---

## 2. Round A — Make the sweep survivable (this round)

Three deliverables. Do all three. No router work.

### 2.1 Per-item wall clock in the ring

**File:** [knowledge3d/knowledgeverse/knowledgeverse.py](../knowledge3d/knowledgeverse/knowledgeverse.py) — `enqueue_task` at :3481 and `wait_output_buffer` at :3515.

Add an optional `max_wall_ms: int` parameter to `enqueue_task`. Store on the ring slot. In `wait_output_buffer`, if wall-clock since enqueue exceeds `max_wall_ms`, return a synthesized output envelope:

```python
{
  "status": "error",
  "failure_code": "wall_timeout",
  "elapsed_ms": <actual>,
  "task_id": <passthrough>,
  "request_id": <passthrough>,
  # answer, predicted_answer, response, result all empty — same shape as
  # no_materialized_answer so sender's "scored as wrong" path doesn't branch
}
```

Free the ring slot on timeout. Do **not** cancel the ongoing tick work mid-kernel — just stop waiting for it and let the tick driver drain the slot lazily on the next cycle (the envelope is already consumer-abandoned; whatever result eventually lands gets discarded). If that's awkward with the current slot lifecycle, mark the slot `abandoned=true` and have `_next_free_slot` skip it until the eventual output arrives and the sweep-runner has moved on.

No default. Sender must pass `max_wall_ms` explicitly. That keeps existing callers unchanged.

### 2.2 Per-benchmark wall clock in the sweep runner

**File:** [scripts/validation_sweep_20260417.py](../scripts/validation_sweep_20260417.py).

Wrap each benchmark loop in a wall-clock budget. On exceed, flush the JSON with what we have and move to the next benchmark. Budgets:

| Benchmark | `max_wall_ms` per item | Per-benchmark ceiling |
|---|---:|---:|
| MMLU | 45000 | 15 min |
| GSM8K | 45000 | 15 min |
| Math competitions | 60000 | 20 min |
| LHE | 90000 | 30 min |
| ARC-AGI-1 | 60000 | 20 min |

Add to each per-benchmark JSON:

```json
"stalled_at_item": 27,           // index of first item that timed out or was cut; null if clean
"wall_ceiling_hit": true,        // true if the per-benchmark budget triggered flush-and-move
"wall_timeouts": 4               // count of items that exited via wall_timeout vs completed
```

Pass the per-item `max_wall_ms` through `enqueue_task`. Count `wall_timeout` items as incorrect (not errors — they're a soft fail, already captured in `correct/incorrect`).

### 2.3 Ring trace JSONL

**File:** new — `TEMP/validation_sweep_2026-04-17/ring_trace.jsonl`, appended by the daemon for every ring event.

Emit one line per event, cheap and async:

```json
{"ts": 1776447200.123, "event": "enqueue", "request_id": "trmio_00000117", "task_id": "math_3645", "tick": 12680}
{"ts": 1776447218.119, "event": "output", "request_id": "trmio_00000117", "task_id": "math_3645", "tick": 12692, "elapsed_ms": 17996.46, "status": "error", "failure_code": "no_materialized_answer"}
{"ts": 1776447345.002, "event": "wall_timeout", "request_id": "trmio_00000201", "task_id": "lhe_0042", "tick": 18450, "elapsed_ms": 90003.1}
```

Hook it in `enqueue_task`, `wait_output_buffer` exit (success or abandon), and the new timeout path. Whatever benchmarks next stalls on, this file will point at the exact item.

**Do not** add this to the normal daemon log. Separate file, line-delimited, flushable (`line_buffering=True` or explicit `.flush()` per write). The whole point is post-mortem clarity.

---

## 3. Re-run protocol

After 2.1–2.3 land:

1. Verify unit: add `test_enqueue_task_wall_timeout` — enqueue with `max_wall_ms=50`, tick driver sleeping, assert the returned envelope has `failure_code: "wall_timeout"` and ring slot is freed.
2. Start daemon; Janet at T0 must still return 18.
3. `python scripts/validation_sweep_20260417.py` end-to-end.
4. If any `wall_ceiling_hit: true` — note which benchmark and continue. Don't stop, don't diagnose mid-run.
5. At end, write `TEMP/validation_sweep_2026-04-17/SUMMARY.md` per the original spec's §3, with these additions:
   - **Stall ledger** table: benchmark, `stalled_at_item`, `wall_timeouts`, `wall_ceiling_hit`, total items that did produce an output.
   - **Ring trace byte count** — append `wc -l TEMP/validation_sweep_2026-04-17/ring_trace.jsonl`.
   - **Janet at T_end** — PASS/FAIL.
6. Do not re-tune router, executor, validator, or any star seeding this round. If an item produces a bizarre answer, note it in SUMMARY free-form commentary.

---

## 4. What Round B will look like (spec comes later)

Do not start any of this now. Listed so you know the shape of the next work and don't refactor toward it accidentally this round:

- **Router resolution** — separate `NUMERIC_COMPUTE` (direct arithmetic) from `MULTI_HOP_INFERENCE` (word problems, decomposition). Today they collapse. The GSM8K tranche argmax-counts show 50/50 on NUMERIC_COMPUTE is wrong — about 40 of those should be MULTI_HOP.
- **MMLU non-compute seeds** — 21/50 MMLU items hit NUMERIC_COMPUTE. The seeds for FACTUAL_RECALL / DEFINITION_LOOKUP / COMPARATIVE_CHOICE need stronger contrast against NUMERIC_COMPUTE in embedding space.
- **GSM8K decomposition chain** — when NUMERIC_COMPUTE is correct (genuine arithmetic) and there's a word problem behind it, the `math_operation_chain_executor` is being run on raw tokens and producing garbage. The decomposition stars aren't being chained in before the executor. This is a pipeline composition issue, not a classifier issue.
- **Math-competitions no_materialized_answer** — 49/50 items end with `answer_materialized: false`. The materializer is refusing to emit when it shouldn't have to. Likely a validator-gate issue, not a compute issue.

I'll spec Round B once we have the full 250-item picture, including LHE and ARC-AGI-1 argmax distributions. The router fix has to account for all five benchmarks, not three.

---

## 5. What NOT to do this round

- No changes to `NavigatorSpecialist`, `_meaning_route`, or meaning-class seeds.
- No changes to any `executor_star` or `validator_star` selection logic.
- No changes to task-type derivation (already landed this round — `test_task_type_from_meaning_class.py` is green).
- No changes to `sovereign_hot_path.py` unless directly required by 2.1 (shouldn't be).
- No score chasing. A Round B spec with ring trace data will beat speculative router edits by a wide margin.

---

## 6. Standing protocol reminders

- Rule of three — specs → PTX docs if kernel-touching → `plan_task` cloud before the wall-timeout hook implementation. This is ring-lifecycle work; worth the planner call.
- `kimi_swarm` / `ask_cloud` timeout 240000 ms.
- Tests run against a real daemon, not mocked. No numpy. No fallbacks.

---

## 7. Acceptance

Round A passes when:
1. The sweep runs end-to-end unattended. Five JSONs + SUMMARY.md + ring_trace.jsonl all land, even if some items hit `wall_timeout`.
2. `test_enqueue_task_wall_timeout` green.
3. Janet = 18 at T0 and T_end.
4. `grep "token in {" knowledge3d/knowledgeverse knowledge3d/daemon | wc -l` still returns 1.
5. The daemon is stoppable cleanly at end (SIGTERM → tick driver event.set() → threads join). No user-space-CPU burn after sweep exit.

Ship to me: the five per-benchmark JSONs, SUMMARY.md, ring_trace.jsonl, and a 3-sentence free-form "what surprised you" note. I'll write Round B from that.
