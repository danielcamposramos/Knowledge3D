# Always-On Live Engine — Daemon Ring Convergence + Tick Driver

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-17
**For:** Codex
**Extends:** `TEMP/CLAUDE_LIVE_GAME_ENGINE_CONVERGENCE_04.16.2026.md`
**Doctrine:** CLAUDE.md §"Phase C: Daemon / Always-On" + §"TRM IS the Avatar", `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md`

---

## 0. Where we are (verified)

- [knowledgeverse.py](../knowledge3d/knowledgeverse/knowledgeverse.py) shrunk **17,523 → 16,002** (−1,521 net). `bind_gpu_galaxy_runtime` @ [3076](../knowledge3d/knowledgeverse/knowledgeverse.py#L3076), `run_ticks` @ [3471](../knowledge3d/knowledgeverse/knowledgeverse.py#L3471), `enqueue_task` @ [3481](../knowledge3d/knowledgeverse/knowledgeverse.py#L3481), `wait_output_buffer` @ [3515](../knowledge3d/knowledgeverse/knowledgeverse.py#L3515). `query()` @ [15862](../knowledge3d/knowledgeverse/knowledgeverse.py#L15862) routes through `execute_task()` → ring. ✅
- Tablet submit path uses the ring at [headless_tablet.py:1524](../knowledge3d/bridge/headless_tablet.py#L1524). ✅
- Daemon `execute_task` call sites: 8+ across [daemon/main.py](../knowledge3d/daemon/main.py) (813, 859, 920, 978, 1001, 1268, 1317, 1342) — daemon effectively on the ring. ✅
- Janet = 18 via ring path, 16 tests pass. ✅

**What remains to be "live" in the full sense:**

1. Daemon ROUTE handler still **branches on benchmark labels** (`QUESTION`, `MATH`, `GAME_2D`, `CHAT`) before it hands the envelope to `execute_task`. This is the same drift we killed in the router and in `math_wine`, now surviving in [daemon/main.py:826-833, 1142-1144](../knowledge3d/daemon/main.py#L826) and multiple re-stamps of `"surface_kind": "MATH" / "CHAT"` ([923, 981, 1004, 1119](../knowledge3d/daemon/main.py#L923)).
2. Daemon has `_looks_like_math_prompt(chat_prompt)` at [971](../knowledge3d/daemon/main.py#L971) — another Python-reasoning classifier. Must die.
3. **No tick driver thread** — `grep threading|Thread|tick_loop` in `daemon/main.py` returns nothing. The engine only ticks when a caller blocks on `wait_output_buffer`. Phase C (always-on) has not begun.
4. `knowledgeverse.py` still 16,002 lines. Target per CLAUDE.md ≈ 200. Another shrink pass is needed once the daemon label drift is gone and the pre-ring Python surface for label-based dispatch becomes unreachable.

---

## 1. What "always-on live engine" means, concretely

All must hold after this lands.

1. **Daemon ROUTE is meaning-centric.** `handle_command("ROUTE", ...)` never reads `surface_kind`/`type`/`task_type` to branch behaviour. One envelope shape → one `execute_task` call → ring. If the daemon needs to know a query is "math-like," it asks the navigator lane, not a Python classifier.
2. **Daemon ticks continuously.** A single background `TickDriver` thread pumps `kv.run_ticks(1)` on a bounded cadence (e.g. 50 Hz ceiling, or idle backoff when no pending inputs). Starts at daemon boot, stops on shutdown.
3. **`_looks_like_math_prompt` is deleted**, not replaced. If the caller needs a preflight, it queries the navigator's `emit()` and reads `meaning_class_dist`.
4. **Janet = 18** still holds via the live engine (daemon running, tick driver on, query submitted through the socket).
5. **Second shrink pass** on `knowledgeverse.py`: delete the Python helpers that only existed to service the pre-ring label branches once the daemon stops needing them. Target ≥ 500 lines this round.

---

## 2. The work (ordered)

### 2.1 Kill label branching in `daemon/main.py` (sovereignty — HIGH)

**Sites to scrub (non-exhaustive — grep to find the rest):**

- [826-833](../knowledge3d/daemon/main.py#L826): `task_type = ...; question_mode = ...; spatial_mode = ...; math_mode = ...` — **delete**.
- [845-981](../knowledge3d/daemon/main.py#L845): every `if question_mode:` / `if spatial_mode:` / `if math_mode:` / `if specialist == "math" or math_mode` branch — **delete the branch**, keep the single envelope-forwarding path.
- [971](../knowledge3d/daemon/main.py#L971): `self._looks_like_math_prompt(chat_prompt)` — **delete the method + every caller**.
- [923, 981, 1004, 1119](../knowledge3d/daemon/main.py#L923): `"surface_kind": "MATH" / "CHAT" / task_type or "CHAT"` payload re-stamps — **delete the key**. The envelope is label-free by the time it hits `execute_task`.
- [1125-1180+](../knowledge3d/daemon/main.py#L1125): the ROUTE block re-does the same label dance. **Rewrite as:** extract `query` string, build one meaning-centric envelope `{query, galaxies, route_policy}`, submit, return.

**Contract for the new ROUTE path:**
```python
query = _coalesce_query(payload, task_obj)           # literal string, no classification
if not query:
    return {"status": "error", "error": "missing_query_or_task"}
envelope = {
    "query": query,
    "galaxies": preferred_galaxies or self._all_default_galaxies(),
    "route_policy": route_policy or "all_live_galaxies",
}
return self.kv.execute_task(envelope)                # ring path, meaning-centric
```

If `galaxies` is empty the Knowledgeverse side already falls back to "all_live_galaxies" — verify and lean on it rather than re-implementing the default list in the daemon.

Before coding: `mcp__k3d-knowledge__qdrant-find("meaning centric envelope execute_task ring contract")` to confirm the accepted envelope keys. If the contract is narrower than what I sketched above, honour the spec, not this sketch.

### 2.2 Add `TickDriver` (always-on — HIGH)

**New file:** [knowledge3d/daemon/tick_driver.py](../knowledge3d/daemon/tick_driver.py). ≤80 lines.

**Contract:**
```python
class TickDriver:
    def __init__(self, kv: Knowledgeverse, *, max_hz: float = 50.0, idle_backoff_ms: int = 20): ...
    def start(self) -> None: ...        # spawns a single daemon=True thread, idempotent
    def stop(self, *, timeout_s: float = 2.0) -> None: ...
    def is_running(self) -> bool: ...
    def stats(self) -> dict[str, Any]: ...   # {ticks_total, ticks_since_start, last_tick_wall_ms, idle_ticks, active_ticks}
```

**Loop shape (no numpy, no fancy libs):**
- Use `time.monotonic()` and `threading.Event` for clean shutdown — nothing else.
- Each iteration: call `kv.run_ticks(1)`. If it returns 0 (or the tick reports no pending inputs), sleep `idle_backoff_ms`. Otherwise sleep `max(0, 1/max_hz - elapsed)`.
- On uncaught exception inside a tick: log once, bump an `error_ticks` counter in `stats`, continue. The driver does not crash the daemon.

**Wire-in (daemon/main.py):**
- Instantiate at the end of daemon boot: `self._tick_driver = TickDriver(self.kv); self._tick_driver.start()`.
- Stop on shutdown before closing the Knowledgeverse.
- Expose a `TICK_STATUS` command (read-only) that returns `self._tick_driver.stats()`.

**Do not** couple the driver to any specific tick cadence from the command loop — it owns its own clock. Callers still use `enqueue_task` + `wait_output_buffer`; the driver just guarantees the tick will happen even when no caller is blocking.

### 2.3 Second shrink pass on `knowledgeverse.py` (≥500 lines)

Once §2.1 lands, the Python helpers that only existed to service label branches become unreachable. Candidates to hunt:

- `_normalize_semantic_task_type` — if every caller goes away, delete it. Grep first: `grep -n _normalize_semantic_task_type knowledge3d/`. If only tests call it, delete the method and update the tests.
- Any `_infer_query_mode` / `_surface_mode_*` / `_classify_*` / `_promote_task_type` helpers that only mapped labels around.
- Orphaned kwargs in `execute_task` / `query` that exist only to pass `surface_kind` through — remove the parameter, not a default value.

Rules (unchanged from prior spec):
- Delete ≥ 500 lines net this round. No wrappers. No `# removed` tombstones.
- Report the top-3 deleted seams in the PR body with line-count deltas.
- If you can't reach 500 lines without breaking tests, report what you found, what blocked the rest, and ship whatever is reachable. Do **not** invent fake deletions by collapsing to stubs.

### 2.4 Tests

- `tests/daemon/test_route_meaning_centric.py`:
  - POST a ROUTE command with `payload={"query": "Janet had 16 ducks..."}` (no `surface_kind`, no `type`, no `task_type`) — assert `result == 18`.
  - POST the same query with `surface_kind="MATH"` added — assert it returns **the same** result envelope (the label is a no-op on meaning).
  - POST with `surface_kind="GAME_2D"` — same envelope comes back (label must not gate dispatch).
- `tests/daemon/test_tick_driver.py`:
  - Start driver, sleep 200 ms, stop. Assert `stats()["ticks_total"] > 5` and `stats()["ticks_total"] < 200` (sanity bounds for the 50 Hz ceiling on an idle loop).
  - Enqueue one task without blocking; stop driver; assert the output ring saw the task.
  - Double-start is a no-op; double-stop is a no-op.
- Janet via live daemon smoke test (optional): only if you already have a live-daemon harness; otherwise skip with `K3D_SKIP_DAEMON_TESTS=1`.

---

## 3. Standing protocol (mandatory before coding)

Rule of three every time:

1. `mcp__k3d-knowledge__qdrant-find("<concept>")` — specs first.
2. `mcp__k3d-ptx__qdrant-find("<opcode>")` — only if touching `.cu` / `.ptx` / ctypes bridge.
3. `mcp__ollama-specialists__plan_task(task=..., context=...)` — cloud planner, one call, before a multi-file change.

Non-negotiables:
- **No stubs, no `pass`, no `TODO`.**
- **No Python reasoning / classification.** Sovereignty applies to the daemon surface too.
- **No numpy/cupy/scipy.**
- **No fallbacks.** If a test exposes a GPU path break, fix on GPU.
- **`kimi_swarm` + deep `ask_cloud` timeout = 240000 ms.**
- **`TickDriver`** is daemon=True thread + `threading.Event` only — no `asyncio`, no `multiprocessing`, no `loop.run_until_complete`.

---

## 4. Success line for your next report

> "Daemon ROUTE is meaning-centric: no `surface_kind`/`task_type` branching. `TickDriver` running at daemon boot, `TICK_STATUS` surfaces cadence. Janet = 18 via live daemon over the ring. `knowledgeverse.py` shrunk by N lines (top seams: …). No new Python classifiers, no stubs, no numpy."

Include `wc -l knowledge3d/knowledgeverse/knowledgeverse.py` before/after in the report.

---

## 5. If stuck

- Spec unclear → `mcp__k3d-knowledge__qdrant-find(...)`.
- PTX unclear → `mcp__k3d-ptx__qdrant-find(...)`.
- Plan unclear → `mcp__ollama-specialists__plan_task(...)` (cloud).
- Architecture unclear → `mcp__ollama-specialists__ask_cloud(model="kimi-k2.5:cloud", ..., timeout_ms=240000)`.
- Truly blocked → `TEMP/CODEX_BLOCKED_ALWAYS_ON_<date>.md` with file:line and the concrete question. Claude responds with a spec update. Do **not** ship a stub.
