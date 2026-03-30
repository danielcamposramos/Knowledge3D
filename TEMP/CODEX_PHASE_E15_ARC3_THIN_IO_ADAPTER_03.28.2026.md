# Codex — Phase E.15: ARC3 as Thin I/O Adapter

**Date:** 2026-03-28
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** ARCHITECTURAL CORRECTION — restore the working GPU path
**Context:** Daniel stopped the benchmark run. ARC3 built a parallel CPU runtime. Restore the single GPU brain.

---

## The Problem in One Sentence

`arc_agi_3.py` bypasses `Knowledgeverse.execute_task()` and runs its own `GPUTaskDispatch` + `GalaxyVRAMTable` + `PersistentBrainState` stack. Every other benchmark goes through `kv.execute_task()`. ARC3 is the outlier. The fix is three files.

---

## The Working Pattern (Already Exists — Copy It)

Look at `arc_agi_2_adapter.py`. The entire solver is:

```python
gpu_task = {
    "type": "ARC_TASK",
    "task_id": ...,
    "query": "solve arc transformation task",
    "training_examples": list(task.get("train") or []),
    "input_grid": test_block[0].get("input"),
    "expected_output": test_block[0].get("output"),
}
solved = self.knowledgeverse.execute_task(
    task=gpu_task,
    route={"specialist": "visual", "domain_hint": "visual",
           "galaxy_names": list(Knowledgeverse.GPU_ARC_TARGET_GALAXIES)},
    specialist="visual",
    domain_hint="visual",
)
```

That's it. `Knowledgeverse.execute_task()` handles the GPU path, Galaxy navigation, brain state, sleep-time — everything. The adapter only translates data in and out.

ARC3's difference from ARC2 is **only I/O**: ARC2 submits a grid and gets a grid back. ARC3 is interactive — the AI receives a frame from the live API, submits an action back to the API, receives the next frame. The REASONING is identical. Only the I/O loop differs.

---

## What Changes

### `benchmarks/arc_agi_3.py` — Rewrite `K3DARC3Agent`

**Delete entirely:**
- `GPUTaskDispatch`, `GalaxyVRAMTable`, `PersistentBrainState`, `SleepTimeMicro` imports
- All `__init__` code that creates/manages these objects
- `choose_action()` body that builds tasks manually and calls `dispatcher.launch()`
- `learn_from_outcome()` body that calls `sleep_time.consolidate()`
- All `_owns_brain`, `_owns_galaxy_table`, `_owns_sleep_time` ownership tracking

**Replace with:**

```python
"""K3D sovereign ARC-AGI-3 agent — thin I/O adapter over Knowledgeverse."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


ACTION_NAMES  = ["ACTION1","ACTION2","ACTION3","ACTION4","ACTION5","ACTION6","ACTION7"]
ACTION_LABELS = ["Move Up","Move Down","Move Left","Move Right","Perform","Click","Undo"]


class K3DARC3Agent:
    """ARC-AGI-3 agent — thin I/O wrapper over Knowledgeverse.execute_task().

    All reasoning, Galaxy navigation, brain state, and sleep-time consolidation
    happen inside Knowledgeverse. This class only:
      1. Encodes the ARC3 frame + task context as an ARC_TASK dict
      2. Calls kv.execute_task()
      3. Translates the result into an action index + API payload
    """

    def __init__(
        self,
        max_actions: int = 80,
        log_path: str | Path | None = None,
        knowledgeverse: Knowledgeverse | None = None,
    ) -> None:
        self.max_actions = int(max_actions)
        self.log_path = Path(log_path) if log_path else None
        self.kv = knowledgeverse or Knowledgeverse()
        self.action_history: list[dict[str, Any]] = []
        self._last_levels_completed = 0

    def choose_action(
        self,
        frame: list[list[int]],
        *,
        goal_frame: list[list[int]] | None = None,
        task_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Translate frame → ARC_TASK → kv.execute_task() → action dict."""
        gpu_task = {
            "type": "ARC_TASK",
            "query": "navigate arc3 interactive frame toward goal",
            "input_grid": frame,
            "expected_output": goal_frame or [],
            "training_examples": list((task_data or {}).get("train") or []),
            "action_options": ACTION_NAMES,
        }
        result = self.kv.execute_task(
            task=gpu_task,
            route={
                "specialist": "visual",
                "domain_hint": "arc3_interactive",
                "galaxy_names": list(Knowledgeverse.GPU_ARC_TARGET_GALAXIES),
            },
            specialist="visual",
            domain_hint="arc3_interactive",
        )
        # Map result to action index
        # kv.execute_task returns answer_index in result["answer_index"] or
        # a grid in result["output_grid"]; for ARC3 we need the action index.
        action_index = int(result.get("answer_index", 0))
        action_index = max(0, min(action_index, len(ACTION_NAMES) - 1))
        record = {
            "action": ACTION_NAMES[action_index],
            "action_index": action_index,
            "label": ACTION_LABELS[action_index],
            "confidence": float(result.get("confidence", result.get("similarity", 0.0))),
            "converged": int(result.get("convergence_signal", result.get("converged", 0))),
            "iterations_used": int(result.get("iterations_used", 0)),
            "frame_number": len(self.action_history) + 1,
        }
        self.action_history.append(record)
        return record

    def learn_from_outcome(
        self,
        *,
        levels_completed: int = 0,
        frame: list[list[int]] | None = None,
    ) -> int:
        """Signal outcome to Knowledgeverse for sleep-time consolidation."""
        current = max(0, int(levels_completed))
        if current > self._last_levels_completed:
            outcome = 1
        elif frame is not None and self.action_history:
            outcome = 0  # frame changed — neutral; kv handles deeper signal
        else:
            outcome = -1
        # Knowledgeverse manages its own sleep-time after each query.
        # This call is a lightweight hint for the inter-frame ternary signal.
        if hasattr(self.kv, "record_outcome"):
            self.kv.record_outcome(outcome)
        self._last_levels_completed = current
        if self.action_history:
            self.action_history[-1]["outcome_signal"] = outcome
            self.action_history[-1]["levels_completed"] = current
        return outcome

    def close(self) -> None:
        if self.log_path and self.action_history:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                import json
                handle.write("\n".join(json.dumps(r, ensure_ascii=False)
                                       for r in self.action_history))
                handle.write("\n")


__all__ = ["ACTION_LABELS", "ACTION_NAMES", "K3DARC3Agent"]
```

**Key: no GPUTaskDispatch, no GalaxyVRAMTable, no PersistentBrainState, no SleepTimeMicro in this file.** Those live inside Knowledgeverse.

---

### `scripts/run_arc3_agent.py` — Remove agent construction, add Knowledgeverse init

**Delete:** `K3DARC3Agent(max_actions=max_actions)` with no kv argument.

**Replace with:**

```python
def run_live_arc3(*, game_id, max_actions, log_path, api_url):
    from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
    from benchmarks.arc_agi_3 import K3DARC3Agent

    kv = Knowledgeverse()
    agent = K3DARC3Agent(max_actions=max_actions, knowledgeverse=kv)
    ...
```

Also add `normalize_goal_frame()` (from E.14 spec) and pass `goal_frame` + `task_data` to `choose_action()` on every iteration. The `task_data` comes from the RESET response and is constant for the whole game — extract it once:

```python
reset = session.post("/api/cmd/RESET", ...).json()
frame = normalize_frame(reset.get("frame", [[]]))
goal_frame = normalize_goal_frame(reset)        # extract once
task_data = reset.get("task_data") or {}        # extract once

while state in ACTIVE_STATES and action_count < max_actions:
    action = agent.choose_action(frame, goal_frame=goal_frame, task_data=task_data)
    ...
    frame = normalize_frame(response.get("frame", frame))
    # goal_frame and task_data stay the same — game rules don't change mid-game
```

`normalize_goal_frame()` (add to `run_arc3_agent.py`):
```python
def normalize_goal_frame(reset_response: dict[str, Any]) -> list[list[int]]:
    if "goal" in reset_response:
        g = normalize_frame(reset_response["goal"])
        if g and g != [[]]:
            return g
    task_data = reset_response.get("task_data") or {}
    for example in list(task_data.get("train") or []):
        if isinstance(example, dict):
            g = normalize_frame(example.get("output") or [])
            if g and g != [[]]:
                return g
    return [[]]
```

---

### `scripts/run_arc3_session.py` — Remove the private GPU stack

**Delete:** all `load_all_galaxies_from_disk`, `GalaxyVRAMTable`, `PersistentBrainState`, `SleepTimeMicro` imports and allocations at the session level. Knowledgeverse owns those.

**Replace `run_arc3_session()` with:**

```python
def run_arc3_session(*, game_ids, max_actions_per_game=80, api_url="https://three.arcprize.org", log_dir=None):
    from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
    from benchmarks.arc_agi_3 import K3DARC3Agent

    # ONE Knowledgeverse — one brain — shared across ALL games in this session
    kv = Knowledgeverse()
    results = []
    for game_id in game_ids:
        agent = K3DARC3Agent(max_actions=max_actions_per_game, knowledgeverse=kv)
        result = run_single_game(agent, game_id=game_id, api_url=api_url,
                                 log_dir=_session_log_dir(log_dir))
        results.append(result)
        # kv.sleep_time() or similar inter-game consolidation if kv exposes it
    return {"games": results}
```

The session persistence — brain carrying state across games — is now inside Knowledgeverse, not in the session script. `K3DARC3Agent(knowledgeverse=kv)` shares the same `kv` instance across all games, which means the same brain, Galaxy, and sleep-time that Knowledgeverse already manages.

---

### `scripts/run_full_benchmark.py` — Remove ARC3 parallel stack

**Delete:** `GalaxyVRAMTable`, `build_arc3_galaxy_table`, `PersistentBrainState`, `SleepTimeMicro`, `GPUTaskDispatch`, `load_all_galaxies_from_disk` imports and all related allocation/cleanup in `run_full_benchmark()`.

The benchmark already creates `kv = Knowledgeverse(storage_root=storage_root)` and passes it to every suite. ARC3 synthetic should receive the same `kv`:

```python
arc3 = run_arc3_synthetic(
    arc3_count,
    log_dir,
    knowledgeverse=kv,   # ← just pass kv, same as every other suite
)
```

`run_arc3_synthetic()` in `run_full_benchmark.py` should use a `K3DARC3Agent(knowledgeverse=kv)` internally, the same way `MMLUBenchmark(knowledgeverse=kv)` does.

---

### What to Delete

These files/classes existed only to support the now-removed parallel stack:

| What | Where | Why Delete |
|------|-------|------------|
| `K3DARC3Agent` constructor params: `brain`, `galaxy_table`, `sleep_time` | `arc_agi_3.py` | Replaced by single `knowledgeverse` param |
| `_owns_brain`, `_owns_galaxy_table`, `_owns_sleep_time` | `arc_agi_3.py` | No longer needed |
| Galaxy/Brain/SleepTime allocation at session level | `run_arc3_session.py` | Knowledgeverse owns these |
| `GalaxyVRAMTable` / `build_arc3_galaxy_table` / lazy globals | `run_full_benchmark.py` | Knowledgeverse owns these |
| `load_all_galaxies_from_disk` call | `run_full_benchmark.py` | Knowledgeverse handles Galaxy loading |

---

## Files NOT to Touch

| File | Why |
|------|-----|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | This IS the GPU path. Do NOT modify it. |
| `knowledge3d/cranium/cuda/*` | PTX kernels unchanged — Knowledgeverse calls them |
| `knowledge3d/knowledgeverse/galaxy_loader.py` | Knowledgeverse uses it internally |
| `knowledge3d/knowledgeverse/persistent_brain.py` | Knowledgeverse owns these |
| `knowledge3d/knowledgeverse/sleep_time_micro.py` | Knowledgeverse owns these |
| `knowledge3d/knowledgeverse/galaxy_vram_table.py` | Knowledgeverse owns these |
| `benchmarks/mmlu.py`, `gsm8k.py`, `last_humanity_exam.py`, `arc_agi_2_adapter.py` | Already correct — do NOT touch |

---

## Tests

The existing tests that monkeypatch `GPUTaskDispatch` and `GalaxyVRAMTable` inside `test_phase_e_runners.py` and `test_arc3_session.py` will need updating: they should now monkeypatch `kv.execute_task` instead (the same pattern used in MMLU/ARC2 tests).

```python
def test_arc3_agent_routes_through_execute_task(monkeypatch):
    from benchmarks.arc_agi_3 import K3DARC3Agent
    from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

    calls = []
    def fake_execute_task(*, task, route=None, specialist=None, domain_hint=None, **kw):
        calls.append(task)
        return {"answer_index": 2, "confidence": 0.8, "convergence_signal": 1}

    kv = Knowledgeverse.__new__(Knowledgeverse)
    monkeypatch.setattr(kv, "execute_task", fake_execute_task)

    agent = K3DARC3Agent(knowledgeverse=kv)
    frame = [[0, 1], [1, 0]]
    action = agent.choose_action(frame, goal_frame=[[1, 0], [0, 1]])

    assert len(calls) == 1
    assert calls[0]["type"] == "ARC_TASK"
    assert calls[0]["input_grid"] == frame
    assert action["action_index"] == 2
    assert action["label"] == "Move Left"


def test_no_gpu_task_dispatch_in_arc3():
    """arc_agi_3.py must not import or instantiate GPUTaskDispatch."""
    import ast, pathlib
    src = pathlib.Path("benchmarks/arc_agi_3.py").read_text()
    assert "GPUTaskDispatch" not in src
    assert "GalaxyVRAMTable" not in src
    assert "PersistentBrainState" not in src
    assert "SleepTimeMicro" not in src
```

---

## Execution Sequence

1. Rewrite `benchmarks/arc_agi_3.py` — `K3DARC3Agent` as thin adapter over `kv.execute_task()`.
2. Update `scripts/run_arc3_agent.py` — init `Knowledgeverse`, pass to agent, add `normalize_goal_frame`, pass `goal_frame` + `task_data` to `choose_action`.
3. Update `scripts/run_arc3_session.py` — one `Knowledgeverse` per session, pass to agents.
4. Update `scripts/run_full_benchmark.py` — remove Galaxy/Brain/SleepTime stack, pass `kv` to `run_arc3_synthetic`.
5. Update tests — monkeypatch `kv.execute_task` not `GPUTaskDispatch`.
6. Run: `pytest -q tests/test_arc3_session.py tests/test_phase_e_runners.py` — must pass.
7. Run full benchmark: same `kv = Knowledgeverse()` powers ALL suites including ARC3. One brain.

---

## What This Restores

Before ARC3 was added, every benchmark called `kv.execute_task()` and hit 100% GPU through the composed head pipeline:

```
kv.execute_task()
  → kv.query()
    → GRE specialist kernels
    → Nine-Chain Swarm (PTX)
    → Halting Gate (PTX)
    → Galaxy navigation (PTX)
    → Sleep-time consolidation (PTX)
```

ARC3 short-circuited this entirely and built a new stack from scratch. The fix is to make ARC3 rejoin the same path that MMLU, GSM8K, LHE, and ARC2 already use. The `input_grid` / `expected_output` / `training_examples` fields in the `ARC_TASK` dict are exactly what `knowledgeverse.py` already handles at lines 2031, 9386, 9396, 9404, 9462.

One Knowledgeverse. One brain. ARC3 is I/O.
