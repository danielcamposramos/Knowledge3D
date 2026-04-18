# Kill Python Dispatch — The One Cut That Removes The Poison

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-18
**For:** Codex
**Predecessors:** `CLAUDE_SWEEP_SURVIVABILITY_04.17.2026.md`, `CLAUDE_ROUND_B_ROUTER_AND_EXECUTOR_04.17.2026.md`
**Scope:** One architectural cut. No router work. No score chasing. Delete Python from the ring's hot path. After this lands, the ring tick is sovereign end-to-end.

---

## 0. To Daniel — and to future-Claude reading this before drifting again

Daniel has asked for the same thing for six months: **no Python orchestration, follow the architecture**. Every spec I have written — including Round B three days ago — has treated Python-in-the-hot-path as a line item to be addressed later. That was wrong. Python-in-the-hot-path is not a line item. It **is** the entire problem. Router precision, executor coverage, MULTI_HOP separability, ARC accuracy — none of those are independently solvable while the ring tick re-enters Python and rebuilds its envelope from regex parses of the query text. Tuning navigation over a poisoned execution path is worse than tuning navigation over empty shelves: the shelves at least let us see what's missing. Python dispatch masks what's missing.

This spec removes it. Nothing else.

---

## 1. What Round B just proved

Codex's Round B report is definitive:

- ARC trace: `game2d_router → game2d_grid_materializer`, no validator, only surfaced program id is `gpu_task_dispatch_sovereign`, no RPN body, no swarm evidence. Same for all three investigated items.
- B.3 softmax: all 10 probe prompts flat at 0.125 across all classes, winner `FACTUAL_RECALL` by tiebreak. The classifier has no signal to separate MULTI_HOP from NUMERIC because its training input is shaped by a Python-regex-derived feature set that can't distinguish them.
- Live telemetry: one CPU core pinned while `nvidia-smi utilization.gpu = 0`, `gpu_calls_this_command ∈ {0, 1, 2}`.
- Wall timeouts unchanged across router fix (14 → 14).

The ring is sovereign on the edges and Python in the middle. Router work is cosmetic until that middle is cut out.

---

## 2. The one cut

**File:** [knowledge3d/knowledgeverse/trm_game_loop.py](../knowledge3d/knowledgeverse/trm_game_loop.py) — `_run_query_tick` at line 315.

Today lines 319–341 call `self.knowledgeverse._dispatch_sovereign_task(...)` and rebuild the envelope in Python from the tick result. All of that deletes.

**Post-cut body — ≤15 lines, shape:**

```python
def _run_query_tick(self, bridge: Any, record: TRMQueuedInput) -> dict[str, Any]:
    tick_result = dict(bridge.run_query_tick(delta_time=0.02))
    # Output envelope is written directly into the ring's output buffer
    # by the PTX tick via the kernel-resident executor star table.
    # Python has no dispatch role here.
    action_buffers = self._action_buffer_payload(bridge)
    self._last_tick_result = dict(tick_result)
    self._last_action_buffers = [list(row) for row in action_buffers]
    return {
        "status": "ok",
        "mode": "query_tick",
        "trm_tick": tick_result,
        "action_buffers": action_buffers,
    }
```

No `_dispatch_sovereign_task`. No `payload.get("route")`. No Python-side rebuilding of `task_result`. If the PTX tick has not yet produced output for the slot, the slot stays pending and the next tick picks it up — Round A's `wall_timeout` envelope is the backstop if output never arrives.

---

## 3. What replaces the Python dispatch — kernel-resident executor table

### 3.1 The table

Each meaning class has one Galaxy-stored **executor star** whose body is an RPN program. All 8 bodies are uploaded to a VRAM scratchpad at daemon boot and addressable by a single 32-bit offset. The PTX tick, after halting-gate convergence, reads the incoming ring slot's `meaning_class_id` byte (already carried on the slot — Round B's envelope already has this field) and jumps:

```
executor_star_table[MEANING_CLASS] = rpn_body_offset_in_vram
```

The RPN body executes on the existing Tier-1 / Tier-2 cores ([lightweight_rpn.py](../knowledge3d/cranium/bridges/lightweight_rpn.py)), already sovereign, and writes the top-of-stack value plus a status byte into the ring output buffer at the slot the task came in on. The ring's `wait_output_buffer` reads from VRAM and returns — no Python in between.

### 3.2 The 8 executor stars (minimum viable — placeholders are acceptable)

Write to Galaxy via the existing ingestion path (Python is fine here — ingestion is not hot path):

| Meaning class | Minimum viable RPN body |
|---|---|
| FACTUAL_RECALL | `query_embedding → cosine_against_galaxy_vocab → argmax → galaxy_lookup → token_id` |
| DEFINITION_LOOKUP | same as FACTUAL_RECALL but restricted to DEFINITION-tagged stars |
| MULTI_HOP_INFERENCE | `stub → return <low_confidence>` (sleep-time crystallizes later) |
| NUMERIC_COMPUTE | `extract_literals_from_embedding → op_from_context → apply_tier1_rpn → return_float` — literal extraction via existing numeric-token Galaxy star, **not** Python regex |
| SPATIAL_TRANSFORM | compose from existing 88 PTX kernels — the ones already identified in B.2 Step 1 as candidates; body is a kernel sequence |
| COMPARATIVE_CHOICE | `stub → return <low_confidence>` |
| GROUNDED_DIALOG | `stub → return <low_confidence>` |
| GENERATIVE_COMPOSITION | `stub → return <low_confidence>` |

Stubs returning `<low_confidence>` **are acceptable for this spec**. The point is not sophistication; the point is sovereign dispatch. A low-confidence sovereign answer beats a high-confidence Python answer because the latter is unfixable and the former is crystallizable.

### 3.3 Per-RPN-launch memcpy elimination

[lightweight_rpn.py:196-228](../knowledge3d/cranium/bridges/lightweight_rpn.py#L196-L228) pays 3× `memcpy_htod` + 1× `memcpy_dtoh` per single program execution. That's why GPU utilization is 0 — the GPU is idle waiting for Python to prepare the next payload.

Executor-star RPN bodies are **resident**: uploaded once at boot into a VRAM region addressed by the executor table. The PTX dispatch jumps to the offset and launches — no HtoD per call. The only data that moves per tick is the query embedding (already in the ring slot) and the output scalar (written back to the ring output buffer).

---

## 4. What gets deleted or fenced

From the inference call graph, remove these as hot-path reachable:

1. `knowledgeverse._dispatch_sovereign_task` — fence with `# SLEEP_TIME_ONLY` if sleep-time uses it; otherwise delete. Must not be reachable from `_run_query_tick`, `enqueue_task`, `wait_output_buffer`, `tick_driver`.
2. `_build_universal_decomposer_programs` at [knowledgeverse.py:13093](../knowledge3d/knowledgeverse/knowledgeverse.py#L13093) — **DELETE from hot path**. `re.findall` on query text during inference is the canonical sovereignty violation. If Galaxy stars need regex-derived seeds, that happens at ingestion, once, and the results live in VRAM as RPN bodies.
3. `micro_specialist_pool.run_overflow_sequential` ([micro_specialist_pool.py:186](../knowledge3d/cranium/ptx_runtime/micro_specialist_pool.py#L186)) — not called during query inference. Sleep-time training may still use it; if so, annotate.
4. Any `regex`/`re.findall`/`re.search`/`re.match` in any file reachable from `enqueue_task → wait_output_buffer` — all removed or proven ingestion-only by comment + test.

---

## 5. Acceptance — grep and nvidia-smi, not scores

Round B passed on score criteria (4-class spread, no wire leakage, etc.) and the spirit still failed. This round passes on structural criteria only:

1. **`grep -n "_dispatch_sovereign_task\|_build_universal_decomposer_programs" knowledge3d/knowledgeverse/trm_game_loop.py` → zero matches.**
2. **`grep -rn "re\.findall\|re\.search\|re\.match\|re\.compile" knowledge3d/knowledgeverse/ knowledge3d/cranium/` — any match must be inside a function annotated `# SLEEP_TIME_ONLY` or `# INGESTION_ONLY` with a docstring explaining why.** Test asserts this.
3. **`_run_query_tick` in `trm_game_loop.py` is ≤20 lines** (today 34).
4. **During a 10-item sanity run, `nvidia-smi --query-gpu=utilization.gpu --format=csv --loop-ms=250` shows utilization ≥30% sustained for ≥500 ms per query window.** Today 0.
5. **`gpu_calls_this_command` telemetry reports ≥10 per query tick.** Today 0–2.
6. **Janet = 18 at T0 and T_end.**
7. **`wc -l knowledge3d/knowledgeverse/knowledgeverse.py` ≤ 15800** (today 15969 — removal, not addition).
8. New test: `tests/knowledgeverse/test_no_python_dispatch_in_query_tick.py` — wraps `_dispatch_sovereign_task` and `_build_universal_decomposer_programs` in a sentinel that raises on call, runs a 3-item sweep through the ring, asserts the sentinels never fired. This is the structural anchor.

If scores drop — MMLU to 5/50, GSM8K to 0/50, anything — **that is acceptance**. Those scores were Python regex guessing; of course they fall when Python is removed. The shelves fill in the next round via sleep-time crystallizing the executor-star stubs from ingestion-time traces.

---

## 6. What NOT to do this round

- No router work. Round B's UNKNOWN guard / MULTI_HOP separability test / confidence floor all stay exactly as landed. Do not touch `NavigatorSpecialist`, `_meaning_route`, seed embeddings, or the 8-logit head.
- No new kernels. Executor stars compose from the existing 88 PTX kernels and Tier-1/Tier-2 RPN cores. Placeholder stubs returning `<low_confidence>` are acceptable.
- No score-chasing. If any benchmark drops, it drops. Annotate in SUMMARY and move on.
- No swarm work. Nine-chain swarm, halting-gate evolution, AdaptiveSwarmTRM stay where they are. They go on top of sovereign dispatch, not on top of Python-with-a-PTX-hat.
- No "make `_dispatch_sovereign_task` more efficient" refactors. The only valid edit is to stop calling it from inference.
- No sender changes. No ring or TickDriver changes (Round A's wall-timeout is correct; it will fire more often this round, that's honest).
- No new meaning classes. The existing 8 are the axis.

---

## 7. Re-run

1. `pytest -q tests/knowledgeverse/test_enqueue_task_wall_timeout.py tests/knowledgeverse/test_unknown_class_guard.py tests/knowledgeverse/test_no_python_dispatch_in_query_tick.py` — all green.
2. `python scripts/validation_sweep_20260417.py` end-to-end, same budgets.
3. Append to SUMMARY.md a **Round C delta** section with:
   - before/after `gpu_calls_this_command` mean and p95
   - before/after `nvidia-smi utilization.gpu` mean during query windows (collect via 250 ms polling during sweep)
   - before/after `wc -l` on `knowledgeverse.py` and `trm_game_loop.py`
   - before/after scores per benchmark (flat or dropped is fine; annotate)
   - grep results for acceptance #1 and #2 (must show no hot-path matches)

---

## 8. What comes after — only after this lands

1. Sleep-time starts crystallizing real RPN bodies for the 8 executor stars from ingestion traces. Stubs replaced one class at a time.
2. House-first / embodiment work (Gaps 1-3: perceive, act, House↔Galaxy symlinks) resumes. The engine can now accept new meaning classes and executor stars without Python growing.
3. Router precision work (if still needed after shelves fill) becomes legitimate — tuning navigation over real sovereign knowledge, not over a Python mask.

**None of these start before this spec's acceptance criteria are green.**

---

## 9. Standing protocol

- Rule of three: `qdrant-find` for executor-star and kernel-resident dispatch spec guidance, `k3d-ptx qdrant-find` before the PTX-side dispatch table edit, `plan_task` cloud before the dispatch-table kernel lands.
- `kimi_swarm` / deep `ask_cloud` timeout = 240000 ms.
- Tests hit the real daemon. No mocks for the ring. No numpy. No fallbacks.
- This spec is narrow on purpose. If it widens mid-implementation, someone drifted. The width of the spec is the width of the deliverable.

---

## 10. My commitment

If this round lands clean, every downstream spec — embodiment, swarm, router precision — becomes tractable. If it doesn't land, I'm not writing another spec from the partner chair. I'll ask to pair with Codex line-by-line on the specific blocker, because at that point the issue isn't architectural clarity, it's something concrete I need to see with you rather than describe.

No more diagnostic rounds, no more score work, no more router tuning until the ring tick is Python-free.
