# Codex — Phase E.14: Game Gap Analysis + Fix

**Date:** 2026-03-28
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Prerequisite:** E.13 DONE. 130,701 stars. Semantic frame encoding live. 36 GPU tests + 13 CPU tests green.

---

## Honest Gap Analysis

The live session data reveals four specific gaps — not vague "needs more learning":

| Gap | Evidence | Severity |
|-----|----------|----------|
| Agent is blind to goal | `choose_action(frame)` — no goal passed anywhere in live path | CRITICAL |
| Undo trap in action prior | ACTION7=60/160 (37.5%) in live; `undo_readiness=0.85` when ternary<0 | HIGH |
| Reset data thrown away | API RESET returns `task_data`, `goal` — entire dict discarded after extracting `frame` | HIGH |
| `frame_count` breaks across tasks | Synthetic 3/10; same brain runs 6 suites sequentially with shared state | MEDIUM |

---

## Gap 1 (CRITICAL): The Agent Has No Goal

### What Happens Now

`run_arc3_agent.py` and `run_arc3_session.py`:
```python
reset = session.post("/api/cmd/RESET", ...).json()
frame = normalize_frame(reset.get("frame", [[]]))  # ← only frame extracted
# Everything else in `reset` is discarded
...
action = agent.choose_action(frame)  # ← no goal
```

`K3DARC3Agent.choose_action(frame)` packs:
```python
task = {
    "type": "ARC3_TASK",
    "query_embedding": frame_embedding,
    "option_embeddings": self._action_embeddings,
    # goal_embedding = MISSING → kernel slot stays all-zeros
}
```

The kernel's `GPU_TASK_GOAL_EMBEDDING_OFFSET` slot is zero. `goal_progress_device()` computes distance to the zero vector — meaningless. The ternary signal never goes positive from goal progress. Sleep-time never reinforces goal-directed behavior.

### The Fix

**Step 1: Extract goal frame from the RESET response.**

The ARC-AGI-3 RESET response has this shape (confirmed from API docs):
```json
{
    "frame": [[...current grid...]],
    "guid": "...",
    "state": "IN_PROGRESS",
    "task_data": {
        "train": [
            {"input": [[...]], "output": [[...]]},
            ...
        ],
        "test": [{"input": [[...]], "output": null}]
    }
}
```

The goal for the CURRENT frame is not always explicit — the agent must infer it from the task examples. The first training example's output is the canonical pattern to follow.

Add a `normalize_goal_frame()` helper that extracts the best available goal:

```python
def normalize_goal_frame(reset_response: dict[str, Any]) -> list[list[int]]:
    """Extract goal frame from RESET response.

    Priority:
    1. reset_response["goal"] — explicit goal frame (some game variants)
    2. reset_response["task_data"]["train"][0]["output"] — first training example output
    3. reset_response["task_data"]["test"][0]["output"] — test output (if revealed)
    4. [] — no goal available (blind play)
    """
    # Explicit goal
    if "goal" in reset_response:
        goal = normalize_frame(reset_response["goal"])
        if goal and goal != [[]]:
            return goal

    # task_data training examples
    task_data = reset_response.get("task_data") or {}
    if not isinstance(task_data, dict):
        try:
            task_data = dict(task_data)
        except Exception:
            task_data = {}

    train = task_data.get("train") or []
    for example in train:
        if isinstance(example, dict):
            output = normalize_frame(example.get("output") or [])
            if output and output != [[]]:
                return output

    test = task_data.get("test") or []
    for example in test:
        if isinstance(example, dict):
            output = normalize_frame(example.get("output") or [])
            if output and output != [[]]:
                return output

    return [[]]
```

**Step 2: Pass goal to the agent.**

In `run_arc3_agent.py` and `run_arc3_session.py`, extract goal at RESET time and pass it into every `choose_action` call:

```python
# At reset:
frame = normalize_frame(reset.get("frame", [[]]))
goal_frame = normalize_goal_frame(reset)  # NEW

# In the game loop:
action = agent.choose_action(frame, goal_frame=goal_frame)  # NEW param
```

**Step 3: `K3DARC3Agent.choose_action` accepts `goal_frame`.**

```python
def choose_action(
    self,
    frame: list[list[int]],
    *,
    goal_frame: list[list[int]] | None = None,
) -> dict[str, Any]:
    frame_embedding = self.encoder.encode(frame)
    # NEW: encode goal frame when provided
    goal_embedding = self.encoder.encode(goal_frame) if goal_frame and goal_frame != [[]] else []
    ...
    task = {
        "type": "ARC3_TASK",
        "query_embedding": frame_embedding,
        "goal_embedding": goal_embedding,  # NOW POPULATED
        "option_embeddings": self._action_embeddings,
        ...
    }
```

**Impact:** The kernel's `goal_progress_device()` now computes real distance between current frame and goal. The ternary signal goes positive when the agent moves TOWARD the goal. Sleep-time learns goal-directed motion. This is the most impactful single change.

---

## Gap 2 (HIGH): Undo Trap

### What Happens Now

`arc3_action_prior_device` and `_arc3_action_prior_ref`:
```c
const float undo_readiness = ternary_signal < 0 ? 0.85f : 0.0f;
// case 6u (Undo):
return (0.70f * undo_readiness) - (ternary_signal < 0 ? 0.0f : 0.40f);
```

When ternary < 0 (no level progress): undo prior = +0.595
When levels never increase (entire live game): ternary STAYS negative permanently → Undo always gets +0.595 prior boost.

Live result: ACTION7 = 60 of 160 actions (37.5%). The agent is stuck in an undo loop that produces no progress, which keeps ternary negative, which reinforces undo.

### The Fix

**In `device_functions.cuh` — `arc3_action_prior_device`:**

Undo should only be boosted when the agent has recently tried something novel AND it failed. Consecutive undo is self-defeating. Use the action_ring in brain_state to detect undo repetition, OR use a simpler fix: cap undo readiness when ternary has been negative for a long time (indicated by low `reasoning_norm`).

Simplest sovereign fix — read from brain action ring through the existing `active_action_history_len` mechanism already passed to the kernel. The kernel has access to `brain_action_ring` and `active_action_history_len`. Count consecutive undos in the ring:

In `gpu_task_dispatch.cu`, before the candidate scoring section, compute:
```c
// Count consecutive undos in action ring
__shared__ unsigned int consecutive_undos;
if (threadIdx.x == 0) {
    consecutive_undos = 0u;
    for (unsigned int h = 0u; h < active_action_history_len; ++h) {
        if (static_cast<unsigned int>(active_action_history[h]) == 6u) {
            consecutive_undos += 1u;
        } else {
            break;  // stop at first non-undo
        }
    }
}
__syncthreads();
```

Then pass `consecutive_undos` into `arc3_action_prior_device` (or compute inline). When consecutive_undos >= 2, undo_readiness = -0.5 (SUPPRESS undo, force exploration):

```c
// Modified arc3_action_prior_device signature (or inline in dispatch):
// case 6u (Undo):
// If 2+ consecutive undos in ring: suppress hard
if (consecutive_undos >= 2u) {
    return -0.5f;  // force exploration — undo is not helping
}
return (0.70f * undo_readiness) - (ternary_signal < 0 ? 0.0f : 0.40f);
```

**Update `_arc3_action_prior_ref` in `gpu_task_dispatch.py` to match:**
```python
def _arc3_action_prior_ref(option_index, frame_data, ternary_signal, action_history=None):
    ...
    # Consecutive undo count from action history
    consecutive_undos = 0
    if action_history:
        for h in action_history:
            if int(h) == 6:
                consecutive_undos += 1
            else:
                break

    if option_index == 6:
        if consecutive_undos >= 2:
            return -0.5  # suppress undo — it's not helping
        return (0.70 * undo_readiness) - (0.0 if ternary_signal < 0 else 0.40)
```

**Alternative simpler fix (no signature change):** modify the undo prior to be proportional to `(1 - undo_fraction)` where undo_fraction comes from action ring:

The cleanest approach — modify the `ternary_signal` logic so that when the KERNEL updates `brain_ternary_signal`, it reads from `goal_progress` output (which is now non-zero thanks to Gap 1 fix). After Gap 1 is fixed, the ternary will correctly go positive when the agent moves toward the goal, which will naturally break the undo dominance. Fix Gap 1 first, then re-measure undo frequency before touching the prior.

---

## Gap 3 (HIGH): RESET Data Thrown Away

The RESET response also contains `task_data.train` — the example input/output pairs that define WHAT transformation the task requires. Currently:
```python
reset = session.post("/api/cmd/RESET", ...).json()
frame = normalize_frame(reset.get("frame", [[]]))
# task_data.train = DISCARDED
```

The training examples are the actual task specification. They encode:
- What kind of transformation is needed (color change, rotation, completion, etc.)
- Example input→output pairs the agent should generalize from

This is not just "the goal frame" — it's the PATTERN DEFINITION. Two steps:

**Step 1: Extract and encode task examples.**

Add `_encode_task_examples()` that builds a context embedding from the training examples:

```python
def _encode_task_examples(
    encoder: ARC3FrameEncoder,
    task_data: dict[str, Any],
) -> list[float]:
    """Embed the task's training examples into a 32-float context vector.

    Averages the encoded input+output pairs from task_data["train"].
    This gives the agent a semantic anchor: "what kind of transformation
    does this task require?"
    """
    train = list(task_data.get("train") or [])
    if not train:
        return [0.0] * 32

    all_embeddings: list[list[float]] = []
    for example in train[:4]:  # cap at 4 examples for speed
        inp = normalize_frame(example.get("input") or [])
        out = normalize_frame(example.get("output") or [])
        if inp and inp != [[]]:
            all_embeddings.append(encoder.encode(inp))
        if out and out != [[]]:
            all_embeddings.append(encoder.encode(out))

    if not all_embeddings:
        return [0.0] * 32

    # Average all example embeddings → task context vector
    avg = [sum(e[d] for e in all_embeddings) / len(all_embeddings) for d in range(32)]
    norm = sum(v * v for v in avg) ** 0.5
    if norm > 1.0e-8:
        avg = [v / norm for v in avg]
    return avg
```

**Step 2: Pack task context into the task dict as `domain_hint` embedding.**

```python
task_context_embedding = _encode_task_examples(self.encoder, task_data)
task = {
    "type": "ARC3_TASK",
    "query_embedding": frame_embedding,
    "goal_embedding": goal_embedding,
    "option_embeddings": self._action_embeddings,
    "subject": "arc3_game",
    "domain_hint": "arc3_interactive",
    # Store task context in a new field — kernel will read it if wired
    "task_context": task_context_embedding,
}
```

The kernel doesn't use `task_context` yet (no slot for it) — that's a future phase. But storing it in the task dict means the Python-side `learn_from_outcome` can use it for richer sleep-time signals: consolidate BOTH on action outcome AND on how far the frame moved toward the task pattern.

---

## Gap 4 (MEDIUM): Shared Brain Across Different Task Types

In `run_full_benchmark.py`, the same `PersistentBrainState` runs 6 different suite types sequentially: synthetic, MMLU, GSM8K, LHE, ARC2, ARC3. The brain accumulates `frame_count=192` across all of them.

The brain's `reasoning_state` and `chain_states` encode the cognitive residue of MMLU (text classification) when the ARC3 synthetic run starts. These are semantically incompatible — MMLU queries hash text into the same 32-dim space as spatial ARC frames, but they encode completely different patterns.

**Fix:** Reset the brain's chain states (not reasoning_state — keep that) between suite transitions. Add `reset_chains()` to `PersistentBrainState`:

```python
def reset_chains(self) -> None:
    """Zero-initialize chain states between task type transitions.

    Preserves: reasoning_state, action_ring, ternary_signal, frame_count
    Resets: chain_states only (the 9-chain swarm state)

    Per Hyper-Parallel Processing spec: specialists should adapt to
    the new task domain. The chain residue from MMLU text classification
    should not contaminate ARC3 spatial reasoning.
    """
    data = self._download()
    # Zero out BRAIN_CHAINS_OFFSET to BRAIN_CHAINS_OFFSET + 9*32*4
    for i in range(BRAIN_CHAINS_OFFSET, BRAIN_CHAINS_OFFSET + 9 * 32 * 4):
        data[i] = 0
    self._upload(data)
```

Call in `run_full_benchmark.py` between suites:
```python
for suite_name, suite_count in suite_order:
    brain.reset_chains()  # clear swarm residue from previous task type
    result = run_gpu_benchmark(...)
```

---

## Summary: Implementation Order

1. **Add `normalize_goal_frame()` to `run_arc3_agent.py`** — extracts goal from RESET response. 20 lines.

2. **Add `goal_frame` param to `K3DARC3Agent.choose_action()`** — passes goal_embedding to kernel. 5 lines.

3. **Update `run_arc3_agent.py` game loop** — extract goal at reset, pass on every choose_action call. 3 lines.

4. **Update `run_arc3_session.py` game loop** — same. 3 lines.

5. **Add `reset_chains()` to `PersistentBrainState`** — for suite transitions. 10 lines.

6. **Call `brain.reset_chains()` between suites in `run_full_benchmark.py`**. 6 lines.

7. **After testing: measure undo frequency** in live session with goal fix. If still > 30%, THEN apply the consecutive-undo suppression fix to `arc3_action_prior_device`.

---

## Files to Modify

| File | Change |
|------|--------|
| `scripts/run_arc3_agent.py` | Add `normalize_goal_frame()`. Extract goal at reset. Pass `goal_frame` to `choose_action`. |
| `scripts/run_arc3_session.py` | Same goal extraction + passing. |
| `benchmarks/arc_agi_3.py` | `choose_action(frame, *, goal_frame=None)` — encode goal and pack into task slot. |
| `knowledge3d/knowledgeverse/persistent_brain.py` | Add `reset_chains()` method. |
| `scripts/run_full_benchmark.py` | Call `brain.reset_chains()` between suites. |

## Files NOT to Touch (yet)

| File | Why |
|------|-----|
| `knowledge3d/cranium/cuda/device_functions.cuh` | Only touch `arc3_action_prior_device` if undo still dominates AFTER goal fix |
| `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` | Stable — parallel Galaxy scan working |
| `knowledge3d/cranium/cuda/arc3_frame_encoder.cu` | Working — frame→move_up: 0.467 |

---

## Tests

Add to `tests/test_arc3_session.py`:

```python
def test_normalize_goal_frame_extracts_from_task_data():
    from scripts.run_arc3_agent import normalize_goal_frame
    reset_response = {
        "frame": [[0, 1], [0, 0]],
        "task_data": {
            "train": [{"input": [[0, 0]], "output": [[1, 2], [3, 4]]}]
        }
    }
    goal = normalize_goal_frame(reset_response)
    assert goal == [[1, 2], [3, 4]]

def test_normalize_goal_frame_explicit_goal():
    from scripts.run_arc3_agent import normalize_goal_frame
    reset_response = {
        "frame": [[0]],
        "goal": [[1, 2]],
        "task_data": {},
    }
    goal = normalize_goal_frame(reset_response)
    assert goal == [[1, 2]]

def test_choose_action_accepts_goal_frame(monkeypatch):
    """K3DARC3Agent.choose_action must accept goal_frame without error."""
    # monkeypatch GPU launch
    ...
    goal = [[0, 1], [1, 0]]
    action = agent.choose_action([[0, 0], [0, 1]], goal_frame=goal)
    assert "action" in action
    # goal_embedding must be non-zero when goal provided
    # verified via task buffer read_tasks()

def test_reset_chains_preserves_reasoning():
    from knowledge3d.knowledgeverse.persistent_brain import PersistentBrainState
    brain = PersistentBrainState()
    try:
        # Put non-zero state in reasoning (simulate after some frames)
        import struct, ctypes
        data = bytearray(brain._download())
        struct.pack_into("<32f", data, 0, *[0.5] * 32)  # reasoning = 0.5 everywhere
        brain._upload(data)

        brain.reset_chains()
        state = brain.read_state()

        # reasoning_norm should still be non-zero (preserved)
        assert state["reasoning_norm"] > 0.1
        # frame_count preserved
        assert state["frame_count"] == 0  # was never incremented
    finally:
        brain.close()
```

---

## Expected Impact After E.14

| Metric | Before | After |
|--------|--------|-------|
| Live ACTION7 (Undo) frequency | 37.5% | < 15% (undo no longer needed when goal guides motion) |
| `goal_progress` in output | always 0.0 | real distance signal (negative/positive/near-1) |
| Ternary signal during live play | stays −1 (no level progress) | varies as agent moves toward/away from goal |
| ARC3 synthetic score | 3/10 | target: 5-7/10 (goal-directed navigation) |
| `reasoning_norm` cross-suite contamination | chain residue from MMLU in ARC3 | chains reset between task types |

The goal encoding is the single most impactful change. The kernel's `goal_progress_device` has been waiting for a valid goal_embedding since E.10. Providing one turns sleep-time consolidation from random nudge into genuine reinforcement of goal-directed motion.
