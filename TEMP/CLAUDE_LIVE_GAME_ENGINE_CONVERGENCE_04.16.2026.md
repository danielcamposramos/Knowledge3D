# Live Game Engine Convergence — Query Path onto TRMGameLoop

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-16 (late)
**For:** Codex
**Extends:** `TEMP/CLAUDE_POST_PTX_MCP_ADVANCE_04.16.2026.md`
**Doctrine:** CLAUDE.md §"TRM IS the Avatar", `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md`, `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md`

---

## 0. Where we actually are (verified)

**The 04.16 drift cleanup landed — acknowledged and verified:**
- Navigator drift grep: `any(token in ...)` count **= 1** (harmless "a"/"an" article check at [navigator_specialist.py:1016](../knowledge3d/knowledgeverse/navigator_specialist.py#L1016), outside the role-cue block). GSM8K role cues are now symlink-vote driven. ✅
- Benchmark wire contract clean: [math_wine.py:14](../knowledge3d/tablet/wine/math_wine.py#L14) `_MATH_WIRE_META_DROP_KEYS = frozenset({"competition", "dataset", "source"})` strips labels before the tape. ✅
- AGENTS.md documents the three MCPs. ✅
- Two tests added, all touched tests green. ✅

**The "live game engine" scaffolding exists but is orphaned:**
- [knowledge3d/knowledgeverse/trm_game_loop.py](../knowledge3d/knowledgeverse/trm_game_loop.py) — `TRMGameLoop` class (277 lines) with `tick()`, `_run_query_tick()`, `_run_background_tick()`, input/output rings. `bridge.launch_tick(delta_time=0.02)` is the real PTX dispatch.
- [knowledge3d/cranium/bridges/trm_step_fused_bridge.py](../knowledge3d/cranium/bridges/trm_step_fused_bridge.py) (1,574 lines) — wraps `trm_step_fused.ptx` for the PTX game tick.
- [knowledgeverse.py:633](../knowledge3d/knowledgeverse/knowledgeverse.py#L633) constructs `self._trm_game_loop`. [:645](../knowledge3d/knowledgeverse/knowledgeverse.py#L645) starts it. Wrappers at [:3307-3336].
- **But** `grep -rn "_trm_game_loop.enqueue"` returns only the wrapper at [knowledgeverse.py:3327](../knowledge3d/knowledgeverse/knowledgeverse.py#L3327) — **no external caller drives the loop.** Benchmarks, senders, tablet all bypass it.

So: the game-loop engine is built, mounted, idling, and unused. Queries still flow through ~17,523 lines of Python orchestration in [knowledgeverse.py](../knowledge3d/knowledgeverse/knowledgeverse.py). Phase D (CLAUDE.md §"Phase D: TRM Game Loop Migration") has not begun to converge.

---

## 1. What "moving towards the live game engine" means, concretely

One sentence per criterion. All must hold after this work lands.

1. Every sovereign query entry (`tablet.submit`, benchmark senders, daemon ROUTE) puts the query on `TRMGameLoop.enqueue_task(...)` and retrieves via `wait_output_buffer(...)` — **not** via the inline Python orchestration helpers.
2. `TRMGameLoop.tick()` is ticked by a driver (daemon thread or synchronous pump), not by each caller reaching past the ring buffer.
3. `Knowledgeverse.bind_gpu_galaxy_runtime(galaxy_names=[...])` exists and attaches the named galaxies to the runtime the tick reads from — tests in [test_gsm8k_query_head.py](../tests/test_gsm8k_query_head.py) stop 500-ing on the missing method.
4. Janet "= 18" regression still holds, now via the game-loop path (output came out of the output ring, not a Python return statement).
5. `knowledgeverse.py` loses at least **one orchestration seam** (identify + delete a block of ≥300 lines that is now redundant once queries ride the tick — actual line reduction, no wrappers).

Target end state per CLAUDE.md: `knowledgeverse.py` ≤ ~200 lines (boot + I/O only). This PR is one step of many; aim for a first cut that proves the pattern, not the full shrink.

---

## 2. The work (ordered)

### 2.1 Restore `Knowledgeverse.bind_gpu_galaxy_runtime(...)` (BLOCKER)

Used in 10+ sites in `tests/test_gsm8k_query_head.py` (e.g. [:192](../tests/test_gsm8k_query_head.py#L192), [:212](../tests/test_gsm8k_query_head.py#L212), [:299](../tests/test_gsm8k_query_head.py#L299), [:846](../tests/test_gsm8k_query_head.py#L846)). Signature inferred from callers:

```python
kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar", "Number", "Word"])
```

Before writing, call `mcp__k3d-knowledge__qdrant-find("bind gpu galaxy runtime knowledgeverse attach")` to see if a prior spec named the contract. Also `git log -S bind_gpu_galaxy_runtime -- knowledge3d/knowledgeverse/knowledgeverse.py | head` — if it existed and was removed, read the commit message.

**Contract (if no prior spec surfaces):**
- Accepts `galaxy_names: list[str]`. No other args required; optional `mode: str = "hot"` reserved for future daemon variants.
- Marks the named galaxies as resident in the runtime-attached set (the set `TRMGameLoop` consults when a tick needs galaxy embeddings).
- Idempotent; second call with overlapping names is a no-op for the overlap.
- Raises `ValueError` on an unknown galaxy name.
- Returns `dict[str, Any]` snapshot of what's now bound (e.g. `{"bound": [...], "total": N}`).

The method is the attach seam between a test/daemon/tablet and the tick's galaxy-embedding input. Keep the implementation thin — if it starts growing helpers, you are over-engineering.

### 2.2 Make `tablet.submit(...)` and benchmark senders go through the ring

Target files:
- [knowledge3d/tablet/wine/question_wine.py](../knowledge3d/tablet/wine/question_wine.py)
- [knowledge3d/tablet/wine/math_wine.py](../knowledge3d/tablet/wine/math_wine.py)
- [benchmarks/math_sender.py](../benchmarks/math_sender.py) / [benchmarks/mmlu_sender.py](../benchmarks/mmlu_sender.py) / [benchmarks/lhe_sender.py](../benchmarks/lhe_sender.py)
- Wherever `daemon.ROUTE` dispatches per-query work today (grep for `def route(` / `def ROUTE` inside the daemon module — `mcp__k3d-knowledge__qdrant-find("daemon route command dispatch")` first).

**Migration contract:**
1. Replace the current "call a Python helper, get a dict" shape with:
   ```python
   request_id = kv.enqueue_task(payload)          # already exists at knowledgeverse.py:3326
   result = kv.wait_output_buffer(request_id, max_ticks=...)
   ```
2. The payload going into `enqueue_task` is the same meaning-centric envelope the router already accepts — no benchmark labels, no `task_type`, no `competition`. (If any caller still assembles those, strip in the caller, don't patch in the queue.)
3. Keep one legacy synchronous path available behind a boot-time flag `K3D_BYPASS_GAME_LOOP=1` so rollback is trivial. Default off.
4. Janet "= 18" regression in [tests/test_gpu_math_query.py](../tests/test_gpu_math_query.py) runs under the ring path. Update the stale `program_id` assertion from the legacy test if it's still stale.

**Do not** build a new wrapper layer. The existing `enqueue_task` / `wait_output_buffer` are the API. Callers just use them.

### 2.3 Tick driver (foreground for now, daemon-ready)

Right now the loop's `tick()` fires when someone calls it. Good enough for this PR. But make sure:
- When a caller uses `enqueue_task` + `wait_output_buffer`, the wait side **pumps at least one tick per wait** (it already does — verify at [trm_game_loop.py:153-155](../knowledge3d/knowledgeverse/trm_game_loop.py#L153)). This is fine.
- Add a Knowledgeverse method `run_ticks(n: int = 1)` that pumps n ticks, so a future daemon thread can drive from outside without touching internals.
- Do **not** start a Python background thread in this PR. Daemonization is Phase C — name-check it in AGENTS.md instead.

### 2.4 Shrink one orchestration seam in `knowledgeverse.py`

Once callers go through the ring, at least one inline orchestration block in `knowledgeverse.py` becomes redundant. Pick the **largest contiguous block that is now reachable only from the pre-ring path** (candidates: the Python-side result-composition helpers that `tablet.submit` used to call, or a pre-ring scoring pass that the tick now repeats).

Rules:
- Delete ≥ 300 lines net. No wrappers left behind. No `# removed` tombstones.
- Document the deletion in the PR body with the line-count delta and the seam name.
- If you can't find 300 lines to cut without breaking tests, report the largest candidate you found and what blocks deletion — do **not** stub-delete and do not invent dead code.

### 2.5 Tests

- Update the legacy Janet test to accept the ring-delivered result envelope.
- Add `tests/knowledgeverse/test_game_loop_query_path.py`:
  - Enqueue "Janet had 16 ducks..." envelope, wait, assert `result == 18` and that `tick > 0` in the output payload.
  - Enqueue three envelopes in sequence, assert three output payloads, each with strictly increasing `tick`.
- Add `tests/knowledgeverse/test_bind_gpu_galaxy_runtime.py`:
  - Bind `["Math","Grammar","Number","Word"]`, assert snapshot lists them.
  - Double-bind the same names, assert idempotent.
  - Bind an unknown name, expect `ValueError`.

---

## 3. Standing protocol (do this before coding)

1. `mcp__k3d-knowledge__qdrant-find("TRM game loop query ring enqueue tick")` — pull the spec excerpts for the game-loop contract.
2. `mcp__k3d-knowledge__qdrant-find("bind gpu galaxy runtime knowledgeverse attach")` — check if the method's contract was ever spec'd.
3. If you touch `trm_step_fused_bridge.py` or any `.cu`/`.ptx`: `mcp__k3d-ptx__qdrant-find("<the specific opcode or pattern you need>")`.
4. `mcp__ollama-specialists__plan_task(task="<§2.1 through §2.5 in brief>", context="<top-level file list and current sprawl>")` — cloud-backed planner now; one call, save hours.
5. For any deep architecture pushback, `mcp__ollama-specialists__ask_cloud(model="kimi-k2.5:cloud", question=..., timeout_ms=240000)`.

Non-negotiables (same as always):
- **No stubs.** No `pass`, no `TODO`, no `NotImplementedError` in shipped code. If stuck, ask `ask_coder` with the real context.
- **No Python reasoning.** No `token in set(...)` for meaning. Symlink-vote or nothing.
- **No numpy/cupy/scipy** in hot path.
- **No fallbacks.** If GPU is wrong, fix on GPU.
- **MCP timeout for `kimi_swarm` and deep `ask_cloud` = 240000 ms.**
- Keep the boot-time bypass flag so rollback is one env var, not a rebuild.

---

## 4. Success line for your next report

> "Live game engine converged: every sovereign query goes through `TRMGameLoop.enqueue_task → tick → wait_output_buffer`. `bind_gpu_galaxy_runtime` restored and tested. Janet = 18 still holds via the ring path. knowledgeverse.py shrunk by N lines (seam: <name>). No new stubs, no new Python reasoning."

Put the absolute line-count delta for `knowledgeverse.py` in the report (`git diff --stat` line).

---

## 5. If you get stuck

- Spec unclear → `mcp__k3d-knowledge__qdrant-find("<concept>")`.
- PTX / kernel unclear → `mcp__k3d-ptx__qdrant-find("<opcode>")`.
- Approach unclear → `mcp__ollama-specialists__plan_task(...)` (cloud).
- Architecture unclear → `mcp__ollama-specialists__ask_cloud(model="kimi-k2.5:cloud", ..., timeout_ms=240000)`.
- Still blocked → drop `TEMP/CODEX_BLOCKED_GAME_ENGINE_<date>.md` with the file:line and the question. Do **not** ship a stub or a comment-out. Claude responds with a spec update.
