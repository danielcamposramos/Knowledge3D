# Halting Readback Hook — First-Class PTX Scalar on the TRM Tick

**Date:** 2026-04-21
**Status:** Spec + implementation (this session)
**Author:** Claude (architecture), with Daniel's direct ruling "Agreed, spec + implement."
**Supersedes:** the stopgap side-channel `_read_real_halting_value` in
`knowledge3d/daemon/main.py` (landed 2026-04-21 as a bootstrap)

---

## 1. Motivation

The composed head pipeline (Morton → LED-A* → Frustum → LOD → Nine-Chain
Swarm → Halting Gate) produces a halting scalar per tick — today that
scalar is visible only **inside** the sovereign swarm kernel. The
stopgap `_read_real_halting_value` in `daemon/main.py` reaches into
`kv._n_chain_swarm._kernel_control.halting_counter` and divides by
`swarm.n_active`. That reach-around violates two architectural rules:

1. **The daemon must not peek at the swarm bridge's internal struct.**
   Daemon reads only what the TRM tick exposes.
2. **Halting is a TRM tick output, not a private swarm counter.** The
   avatar halts; the daemon merely observes.

We also discovered that the existing ratio (`halting_counter / n_active`)
is always `1.0` after a successful tick, because the swarm only flips
`COMPLETE` when *all* lanes have halted (see
`knowledge3d/cranium/cuda/k3d_swarm_persistent.cu:213-217`). That makes
the current "halting scalar" a boolean-in-disguise. We fix this at the
same time we fix the readback surface.

**Goal.** A first-class `halting_value: float` field on
`solved["trm_tick"]`, sourced from a device-reachable global scalar
written by the halting gate in PTX. Daemon reads it with one line, no
side-channel.

---

## 2. What the halting scalar actually means

Per-tick halting scalar ≜ "how confident is the avatar it has halted
well enough to emit?"

The swarm's per-lane `ReasoningLaneOutput.belief_q15` is the only
field that already encodes lane confidence on the hot path (Q15
fixed-point, range `[0, 32768]`). Taking the **max lane belief when
the halt condition is met** gives us:

- `halting_value = max(lane.belief_q15 for lane in active) / 32768`
- Range: `[0.0, 1.0]`
- Written **exactly once per tick** when the halt condition flips the
  kernel to `COMPLETE`.

This is monotonic across the tick (lanes only accumulate belief) and
has a clear operational meaning: 1.0 = unanimous strong-belief halt;
values near 0 = halted because all lanes gave up, not because they
agreed. Either case is a legitimate halt — the scalar just tells
the caller *which kind*.

This definition is append-only: we are not changing the semantics of
`halting_counter` or `halt_epoch`. Those remain as they were.

---

## 3. Readback path — where the scalar is written

### 3.1 Kernel (`k3d_swarm_persistent.cu`)

A new device-reachable scalar is passed in as an argument:

```cuda
uint32_t* __restrict__ g_halting_value_q15
```

Write site (inside the `halted_count == n_active` branch at line 213):

```cuda
if (halted_count == n_active) {
    if (g_halting_counter != nullptr) {
        *g_halting_counter = n_active;
    }
    if (g_halting_value_q15 != nullptr) {
        uint32_t max_belief_q15 = 0u;
        for (uint32_t lane = 0u; lane < n_active; ++lane) {
            const uint32_t belief = lane_outputs[lane].belief_q15;
            if (belief > max_belief_q15) {
                max_belief_q15 = belief;
            }
        }
        *g_halting_value_q15 = max_belief_q15;
    }
    control->state = K3D_SWARM_FLAG_COMPLETE;
    control->halt_epoch += 1u;
    __threadfence_system();
}
```

No new lanes, no new sync, no new allocations inside the loop. The
scan is bounded by `n_active <= K3D_SWARM_N_HARD_MAX = 1024` and runs
only on the single lane that flips `COMPLETE`. `__threadfence_system`
below covers the write.

### 3.2 Kernel signature bump

The kernel gains one extra parameter; it is appended at the end to
keep all earlier arguments in their existing positions:

```
k3d_swarm_sovereign(
    galaxy_atlas,
    control,
    tick_control,
    g_halting_counter,
    d_n_active,
    lane_outputs,
    perf_ring,
    perf_ring_head,
    perf_ring_mask,
    perf_calibration,
    g_halting_value_q15    // NEW
)
```

Callers that pass `nullptr` for this argument retain the old behaviour
(counter only). No opcode reservation needed — this is a pure output
extension of an existing persistent kernel.

### 3.3 Bridge (`n_chain_swarm_bridge.py`)

- Allocate a 4-byte mapped host buffer in `__init__`
  (same pattern as `_d_n_active`): `self._d_halting_value_q15_host,
  self._d_halting_value_q15 = loader.mapped_host_alloc(4)` + a
  `ctypes.c_uint32.from_address(...)` view named `self._halting_value_q15`.
- Pass the device pointer as the final argument to
  `loader.launch_cooperative` in `launch()`.
- Zero the host/device word at the start of each `tick()` (mirrors
  the existing `memset_d32(self._d_halting_counter, 0, 1)` call).
- Compute `halting_value = float(self._halting_value_q15.value) / 32768.0`
  (clamped to `[0.0, 1.0]` defensively — the kernel cannot produce
  out-of-range values but a late arrival from a prior tick could).
- Return it in the `tick()` dict alongside `halting_counter`/`n_active`.

Final `tick()` return (append-only; old keys unchanged):

```python
return {
    "halting_flag": int(self._kernel_control.state),
    "halting_counter": int(self._kernel_control.halting_counter),
    "n_active": int(self._n_active.value),
    "halting_value": float(halting_value),   # NEW
    "tick_epoch": int(self._kernel_control.tick_epoch),
    "halt_epoch": int(self._kernel_control.halt_epoch),
    "calibration_hint": int(self._calibration.n_hint),
}
```

### 3.4 Knowledgeverse → trm_tick propagation

`knowledgeverse.py` invokes the swarm at line 13028:

```python
swarm_result = n_chain_swarm.tick(swarm_packet, timeout_s=5.0)
```

We stash `swarm_result["halting_value"]` on the instance as
`self._last_swarm_halting_value` (single float, cheap). The trm game
loop reads it when assembling the `trm_tick` dict in `_run_query_tick`
(`trm_game_loop.py:315`) and attaches it to the returned dict:

```python
halting_value = float(getattr(self.knowledgeverse, "_last_swarm_halting_value", 0.0) or 0.0)
tick_result = dict(bridge.run_query_tick(delta_time=0.02))
tick_result["halting_value"] = max(0.0, min(1.0, halting_value))
```

This is the one place where `trm_tick` is assembled on the hot path.
`_last_swarm_halting_value` resets to `0.0` at the start of each
`execute_task` via `write_input_buffer` (symmetric with the existing
`_query_sequence` bump).

### 3.5 Daemon

`_read_real_halting_value` collapses from 30 lines to a one-liner:

```python
def _read_real_halting_value(self, solved: dict[str, Any]) -> float | None:
    """Read the sovereign halting scalar propagated by the TRM tick.

    Source of truth: `solved["trm_tick"]["halting_value"]` — written
    by the swarm halting gate in PTX (see
    `knowledge3d/cranium/cuda/k3d_swarm_persistent.cu` and
    `TEMP/CLAUDE_HALTING_READBACK_HOOK_SPEC_04.21.2026.md`).
    """
    if str(solved.get("status", "")).lower() != "ok":
        return None
    if not bool(solved.get("gpu_execution", False)):
        return None
    trm_tick = solved.get("trm_tick") or {}
    if "halting_value" not in trm_tick:
        return None
    value = float(trm_tick.get("halting_value", 0.0) or 0.0)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
```

- **Explicit key check replaces the narrow `except Exception: return None`.**
  No exception is swallowed anywhere; we return `None` only on a
  deterministic "the tick did not produce a halting scalar" condition.
- No reach into `kv._n_chain_swarm._kernel_control`. The daemon no
  longer knows the swarm exists.

---

## 4. Sovereignty gates

- ctypes only. No numpy, cupy, scipy, sympy.
- Kernel change stays CUDA (compiled to PTX by the existing
  `_ensure_ptx` path — the bridge recompiles when the source mtime
  advances).
- No `try: ... except Exception: pass` introduced.
- No boot-time conversion; the halting scalar is computed per tick
  inline on GPU.
- No fallback. If the field is missing, `_read_real_halting_value`
  returns `None` and the caller (already correct) declines to emit a
  wake-delta event. That is *observation*, not a reasoning fallback.

Grep gates:

```
grep -rn "^import \(ollama\|requests\|httpx\)" knowledge3d/ --include="*.py" \
    | grep -v "tablet/wine/"      # MUST be empty
grep -rn "except Exception: pass" knowledge3d/daemon/main.py
                                  # MUST NOT increase
```

---

## 5. Test plan

One live-daemon test, added to `tests/tablet/test_live_daemon_cycle.py`
using the existing module-scoped `live_daemon` fixture (no CPU
isolation, no subprocess, no re-boot between steps).

```python
def test_halting_value_on_chat_tick(live_daemon):
    """CHAT envelope propagates halting_value: float in [0,1]
    through solved['trm_tick']."""
    chat = live_daemon({
        "command": "CHAT",
        "messages": [{"role": "user", "content": "what is 2+3?"}],
    })
    assert chat["status"] == "ok", chat
    assert chat.get("gpu_execution") is True

    # The daemon's public CHAT handler does not re-expose trm_tick
    # verbatim — but it stashes the last solved dict on the daemon.
    daemon = live_daemon.daemon
    last_solved = daemon._last_solved_for_tests()  # tiny accessor added
    trm_tick = last_solved.get("trm_tick") or {}
    assert "halting_value" in trm_tick, (
        f"trm_tick missing halting_value; got keys={sorted(trm_tick)}"
    )
    hv = float(trm_tick["halting_value"])
    assert 0.0 <= hv <= 1.0, f"halting_value out of range: {hv}"
```

The `_last_solved_for_tests()` accessor is the one deliberate test hook
(returns `self._last_solved` captured by the CHAT/ROUTE/SOLVE_MATH
branches where `_read_real_halting_value` is already called). If Daniel
prefers we skip the accessor, we can assert via
`daemon.kv._last_swarm_halting_value` — but that reintroduces exactly
the reach-around we are removing, so the accessor is strictly
preferable.

---

## 6. Deprecation notes

- `_read_real_halting_value`'s narrow `except Exception: return None` is
  **removed** — replaced by explicit `if "halting_value" not in trm_tick:`
  key check. See §3.5.
- The side-channel read of `kv._n_chain_swarm._kernel_control.halting_counter`
  is **removed** entirely from `daemon/main.py`.
- The `TODO(CODEX, Gap 1 follow-up)` comment in the old implementation
  is **removed** — this spec closes it.

The old docstring referenced
`TEMP/CLAUDE_WAKE_DELTA_HALTING_READBACK_04.21.2026.md`; that file does
not exist in tree (the follow-up TODO was written in anticipation).
The new docstring points at this spec.

---

## 7. Opcode registry

**No opcode change.** This is a pure output extension of the existing
persistent swarm kernel (`k3d_swarm_sovereign`). Registry stays
untouched — append-only remains intact.

---

## 8. Success criteria

1. `solved["trm_tick"]["halting_value"]` is a Python `float` in
   `[0.0, 1.0]` after every GPU-executed CHAT / SOLVE_MATH / ROUTE
   envelope.
2. `daemon._read_real_halting_value` contains zero references to
   `_n_chain_swarm`.
3. No new `except Exception` swallows anywhere.
4. `tests/tablet/test_live_daemon_cycle.py` passes end-to-end with the
   new assertion.
5. `grep -n halting_counter knowledge3d/daemon/main.py` → empty.
