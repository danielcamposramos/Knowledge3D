# CODEX: ARC3 GPU-Native Sleeptime + Between-Attempt Consolidation

**Date:** 2026-04-10
**Author:** Claude (Architecture Partner)
**Reviewed by:** GLM-4 (4-layer mapping), Kimi K2.5 (GPU data structures + dispatch), Qwen3 (sovereignty audit)
**Targets:** `arc3_episode_galaxy.py`, `gpu_task_dispatch.py`, new `arc3_sleep_bindings.py`

---

## 0. Context and Motivation

Codex completed the 64-dim GAME_2D rewire (51 tests passing, ACTION2 bias broken). The next blocker is that `run_micro_sleeptime()` at [arc3_episode_galaxy.py:1047](knowledge3d/knowledgeverse/arc3_episode_galaxy.py#L1047) runs `_crystallize_rules`, `_reinforce_routes`, `_classify_objects` in a Python `ThreadPoolExecutor` — a **direct sovereignty violation** of SLEEPTIME_PROTOCOL_SPECIFICATION.md §0.1:

> "Consolidation MUST execute on GPU via PTX kernels."
> "GPU utilization during sleep MUST be visible (>0% SM occupancy)."

Additionally, `consolidate_to_house()` at [arc3_episode_galaxy.py:1119](knowledge3d/knowledgeverse/arc3_episode_galaxy.py#L1119) clears ALL state (`frames.clear()`, `outcomes.clear()`, `_rules_by_key.clear()`, `_objects_by_color.clear()`), destroying learned knowledge. Daniel's contract: **persist learned rules and object identities between attempts without reloading the system.**

This spec covers three subsystems:
1. **GPU-native micro-sleeptime** (between game steps)
2. **Between-attempt consolidation** (after life lost / timeout, before next attempt)
3. **Avatar/character identification** (action-delta correlation → permanent Layer 2 star)

---

## 1. Spec References (Ground Truth)

| Spec | Section | Mandate |
|------|---------|---------|
| SLEEPTIME_PROTOCOL_SPECIFICATION.md | §0.1 | GPU-native consolidation via PTX kernels |
| SLEEPTIME_PROTOCOL_SPECIFICATION.md | §0.3 | Inline execution on same KV instance — NO reload |
| FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md | §0 | No benchmark names in knowledge — meaning-based star names |
| FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md | §1.2-1.5 | 4-layer architecture: Form → Meaning → Rules → Meta-Rules |
| FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md | §5.4 | VectorDotMap: field equations, not pixel data |
| SOVEREIGN_NSI_SPECIFICATION.md | §9 | Kernel function contract map |

---

## 2. Existing GPU Kernels and Bridges (Already Built)

These kernels and bridges exist and MUST be wired — do NOT rewrite them.

| Kernel File | Bridge Class | Location |
|-------------|-------------|----------|
| `cranium/ptx/sleep_time_micro.ptx` | `SleepTimeComputeBridge` | `cranium/ptx_runtime/sleep_time_compute.py` |
| `cranium/ptx/sleep_cluster_refiner.ptx` | `SleepClusterRefiner` | `cranium/bridges/sovereign_bridges.py:1894` |
| `cranium/kernels/galaxy_memory_updater.cu` | `GalaxyMemoryUpdater` | `cranium/ptx_runtime/galaxy_memory_updater.py:23` |
| `cranium/kernels/lora_gpu.cu` | `LoRAGPUEngine` | `cranium/sovereign/lora_gpu_trainer.py:59` |

---

## 3. Task 1 — GPU-Native Micro-Sleeptime (Between Steps)

### 3.1 What to Replace

**Current violation** at [arc3_episode_galaxy.py:1047-1055](knowledge3d/knowledgeverse/arc3_episode_galaxy.py#L1047-L1055):
```python
# SOVEREIGNTY VIOLATION — Python ThreadPoolExecutor doing CPU work
def run_micro_sleeptime(self) -> None:
    self._drain_pending_futures()
    if self._pending_futures:
        return
    self._pending_futures = [
        self._executor.submit(self._crystallize_rules),
        self._executor.submit(self._reinforce_routes),
        self._executor.submit(self._classify_objects),
    ]
```

### 3.2 Replacement: GPU Kernel Launch

Replace with a method that launches `sleep_time_micro.ptx` on a non-blocking CUDA stream during the game API round-trip latency (~15-200ms).

**Data flow:**
1. Python packs recent outcomes into a GPU-resident ring buffer (ctypes struct)
2. Python launches `sleep_time_micro.ptx` on a dedicated CUDA stream
3. CPU fires the game API request (network I/O overlaps GPU compute)
4. On API return, synchronize the CUDA stream, read stale-rule flags

**GPU-resident data structures** (new file: `knowledge3d/knowledgeverse/arc3_sleep_bindings.py`):

```c
// C-aligned structs for PTX kernel arguments

typedef struct {
    uint32_t color;           // Color value (0-15)
    uint32_t outcome_code;    // 0=blocked, 1=moved, 2=death, 3=goal
    float    confidence_delta;// +strengthen, -weaken
    uint64_t timestamp;       // Frame counter
    uint8_t  action_idx;      // 0-6 mapped action
    uint8_t  _pad[3];         // Alignment
} ARC3OutcomeEntry;           // 24 bytes

typedef struct {
    uint32_t rule_id;
    uint8_t  condition_color;
    uint8_t  action_idx;
    uint16_t _pad;
    float    confidence;      // 0.0-1.0, prune below 0.1
    uint32_t success_count;
    uint32_t failure_count;
    uint64_t last_activation;
    uint8_t  flags;           // Bit 0: is_stale, Bit 1: blocked_path, Bit 2: open_path
    uint8_t  _pad2[3];
} ARC3RuleEntry;              // 32 bytes

typedef struct {
    uint32_t object_id;
    uint8_t  color;
    uint8_t  semantic_class;  // 0=unknown, 1=avatar, 2=wall, 3=goal, 4=hazard
    uint16_t _pad;
    float    motion_variance;
    uint32_t death_assoc;
    uint32_t goal_assoc;
    float    bounding_box[4]; // min_row, max_row, min_col, max_col
} ARC3ObjectHypothesis;       // 40 bytes
```

Python ctypes mirrors of these structs go in `arc3_sleep_bindings.py`. Use `ctypes.Structure` with `_fields_`, same as existing patterns in `cranium/ptx_runtime/`.

### 3.3 Kernel Dispatch Sequence

```python
def run_micro_sleeptime(self) -> None:
    """GPU-native micro-sleeptime. Replaces ThreadPoolExecutor violation."""
    if not self.outcomes:
        return

    # 1. Pack recent outcomes into GPU ring buffer
    self._upload_recent_outcomes_to_gpu(list(self.outcomes)[-10:])

    # 2. Launch sleep_time_micro.ptx on consolidation stream
    #    Kernel does: scan outcomes → update rule confidences →
    #    flag stale rules → reinforce specialist routes
    self._sleep_bridge.launch_micro_consolidation(
        outcome_ring_gpu=self._outcome_ring_ptr,
        rule_table_gpu=self._rule_table_ptr,
        rule_count=len(self._rules_by_key),
        object_hypotheses_gpu=self._object_hyp_ptr,
        hypothesis_count=len(self._objects_by_color),
        timestamp=self._frame_counter,
        stream=self._consolidation_stream,
    )

    # 3. Record CUDA event (caller synchronizes after API round-trip)
    self._consolidation_event = self._consolidation_stream.record()
```

### 3.4 Integration Point in Game Loop

In the ARC3 game step (likely in `benchmarks/arc_agi_3.py` or wherever `step()` is called):

```python
# BEFORE sending action to game API:
episode_galaxy.run_micro_sleeptime()   # Launches GPU async

# Send action to game API (network I/O — CPU blocked, GPU working)
response = game_client.step(action)

# AFTER receiving response:
episode_galaxy.sync_micro_sleeptime()  # Wait for GPU, read flags
```

### 3.5 Success Criteria

- `nvidia-smi` shows >0% SM occupancy during API wait
- `_crystallize_rules`, `_reinforce_routes`, `_classify_objects` no longer called from ThreadPoolExecutor
- `self._executor` (ThreadPoolExecutor) removed from ARC3EpisodeGalaxy
- Rule confidence updates happen on GPU (verified by comparing pre/post values)

---

## 4. Task 2 — Between-Attempt Consolidation (Persist/Purge Protocol)

### 4.1 The Contract: NO System Reload

Per SLEEPTIME_PROTOCOL_SPECIFICATION.md §0.3:
> "Consolidation runs on the same KV instance that processed the queries — while stars, briefs, and specialist weights are still loaded in VRAM."

The Knowledgeverse instance, VRAM allocation, star table, and episode galaxy **MUST persist** across attempt boundaries. Only transient per-attempt state resets.

### 4.2 What PERSISTS Between Attempts

**Layer 2 — Meaning (Permanent Galaxy Stars):**

| Knowledge | Star Name (Meaning-Based) | Why Persist |
|-----------|--------------------------|-------------|
| "Color 6 = avatar" | `spatial_grid_agent_marker` | Foundational — all rules reference it |
| "Color 1 = wall" | `spatial_grid_obstacle_marker` | Constraint knowledge |
| "Color 4 = goal" | `spatial_grid_goal_marker` | Terminal state knowledge |
| "Color 2 = hazard" | `spatial_grid_hazard_marker` | Death-avoidance knowledge |

**NAMING RULE** (FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md §0): Stars are named by MEANING, not by benchmark. `spatial_grid_agent_marker` — NOT `arc3_avatar_color`. The same star works for any spatial grid game.

**Layer 3 — Rules (Learned Constraints):**

| Knowledge | Star Name | Format |
|-----------|-----------|--------|
| "Position (2,3) blocked going south" | `spatial_adjacency_block_rule` | RPN: condition → action_mask |
| "Level layout: doors, goals, hazards" | `spatial_level_topology` | Coordinate sets |
| "Minimum step cost observed" | `spatial_path_cost_estimate` | Float + history |

**Layer 4 — Meta-Rules (Cross-Attempt Strategies):**

| Knowledge | Star Name |
|-----------|-----------|
| "If stuck 3+ frames, try unvisited direction" | `exploration_priority_strategy` |
| "If death at X, retry with 2-cell margin" | `hazard_avoidance_escalation` |

### 4.3 What Gets PURGED Between Attempts

All Layer 1 (Form) ephemeral state:

| State | Field | Why Purge |
|-------|-------|-----------|
| Frame snapshots | `_last_grid`, `_prev_grid` | Raw sensory stream, not distilled knowledge |
| Avatar position | `_centroid_position` | Stale from dead attempt |
| Action history | `_recent_actions` ring buffer | Stale trajectory |
| Step counter | `_step_count` | Reset to 0 |
| Stuck signals | `_stuck_counter`, `_drift_detected` | Local minima detectors, not transferable |
| Attempt-specific outcomes | `_death_position`, `_last_blocked_direction` | Already crystallized into rules |

**Key invariant:** If knowledge was extracted from the failed attempt (e.g., "color 2 at (3,1) causes death"), it is ALREADY a Rule star before the purge. The purge cleans the stream, not the distillate.

### 4.4 Three-Kernel GPU Dispatch Sequence

New method `consolidate_between_attempts()` on `ARC3EpisodeGalaxy`:

```python
def consolidate_between_attempts(self, attempt_result: str) -> dict:
    """
    GPU-native between-attempt consolidation.
    Called after life lost / timeout, BEFORE next attempt starts.
    DOES NOT reload any system state.

    Args:
        attempt_result: "death" | "timeout" | "level_complete"

    Returns:
        ConsolidationReceipt dict
    """
    # KERNEL 1: sleep_cluster_refiner.ptx
    # - Cluster co-activated rules from the completed attempt
    # - Merge overlapping position rules into wall/path concepts
    # - Prune single-observation rules below confidence 0.1
    cluster_result = self._cluster_refiner.refine_clusters(
        embeddings=self._rule_embeddings_gpu,
        n_clusters=max(4, len(self._rules_by_key) // 5),
        n_iterations=4,
        learning_rate=0.2,
    )

    # KERNEL 2: galaxy_memory_updater.cu
    # - Score rules via outcome valence (success strengthens, failure weakens)
    # - Promote rules with confidence > 0.7 to Grammar Galaxy as Layer 2/3 stars
    # - Update star provenance with attempt_id
    self._memory_updater.update_scores(
        star_table=self._galaxy_star_table_ptr,
        outcome_buffer=self._outcome_ring_ptr,
        reinforcement_rate=0.1,
        contradiction_penalty=0.3,
    )

    # KERNEL 3: lora_gpu.cu (conditional — only if enough evidence)
    # - Contrastive update: current weights vs shadow copy
    # - Strengthen routes that led toward goal, weaken dead-end paths
    if len(self.outcomes) >= 10:
        self._lora_engine.contrastive_update(
            episode_outcomes=self._outcome_ring_ptr,
            outcome_type=attempt_result,
            learning_rate=0.001,
        )

    # CPU-side: persist strong rules to Grammar Galaxy
    persisted = self._persist_strong_rules(min_confidence=0.6, min_evidence=3)

    # CPU-side: purge ONLY transient state (Layer 1 Form)
    self._purge_attempt_transient_state()

    return {
        "rules_persisted": len(persisted),
        "clusters_refined": cluster_result.get("mean_silhouette", 0.0),
        "attempt_result": attempt_result,
    }
```

### 4.5 Purge Method (Transient State Only)

New method `_purge_attempt_transient_state()`:

```python
def _purge_attempt_transient_state(self) -> None:
    """
    Clear Layer 1 ephemeral state. Called between attempts.
    DOES NOT clear: _rules_by_key, _objects_by_color, frames history,
    outcomes history (these are Layer 2-4 knowledge).
    """
    # Clear frame-specific snapshots
    self._last_grid = None
    self._prev_grid = None
    self._centroid_position = None

    # Clear action ring
    self._recent_actions.clear()
    self._step_count = 0

    # Clear transient signals
    self._stuck_counter = 0
    self._drift_detected = False
    self._repeated_action_count.clear()

    # Clear attempt-specific outcomes (already crystallized into rules)
    self._death_position = None
    self._last_blocked_direction = None

    # GPU: zero-fill the outcome ring buffer (reuse allocation)
    self._zero_outcome_ring_gpu()

    # DO NOT clear:
    # self._rules_by_key       ← Layer 3 Rules (PERSIST)
    # self._objects_by_color   ← Layer 2 Meaning (PERSIST)
    # self.frames              ← Historical frames (PERSIST for cross-attempt learning)
    # self.outcomes            ← Historical outcomes (PERSIST for rule crystallization)
```

### 4.6 Contrast with Current `consolidate_to_house()`

`consolidate_to_house()` at [arc3_episode_galaxy.py:1119](knowledge3d/knowledgeverse/arc3_episode_galaxy.py#L1119) clears EVERYTHING — this is correct for **end-of-game** (all lives exhausted, game over). But it must NOT be called between attempts within the same game.

| Event | Method | Clears Rules? | Clears Objects? |
|-------|--------|--------------|-----------------|
| Between steps | `run_micro_sleeptime()` | No | No |
| Between attempts | `consolidate_between_attempts()` | No | No |
| End of game | `consolidate_to_house()` | Yes (after persisting) | Yes |

### 4.7 Success Criteria

- New `consolidate_between_attempts()` method exists and is called at attempt boundaries
- `_rules_by_key` and `_objects_by_color` survive across attempts (verify via test)
- Knowledgeverse instance is the SAME object across attempts (no `__init__` re-call)
- All star names are meaning-based: `spatial_grid_*`, NOT `arc3_*`
- GPU kernels fire during consolidation (>0% SM occupancy)

---

## 5. Task 3 — Avatar / Character Identification

### 5.1 The Core Algorithm: Action-Delta Correlation

This is a **causal inference**, not a heuristic. The avatar is the cell whose movement correlates with the agent's issued action direction.

```
BEFORE ACTION2 (south):     AFTER ACTION2:
┌───┬───┬───┐               ┌───┬───┬───┐
│ 0 │ 0 │ 0 │               │ 0 │ 0 │ 0 │
├───┼───┼───┤               ├───┼───┼───┤
│ 0 │ 6 │ 0 │  ──ACTION2──▶ │ 0 │ 0 │ 0 │  ← 6 disappeared from (1,1)
├───┼───┼───┤               ├───┼───┼───┤
│ 0 │ 0 │ 0 │               │ 0 │ 6 │ 0 │  ← 6 appeared at (2,1)
└───┴───┴───┘               └───┴───┴───┘

INFERENCE: Color 6 moved south when ACTION2 issued.
           ACTION2 = south = (Δrow+1, Δcol 0).
           Color 6 IS the avatar.
```

### 5.2 Protocol

1. After each step, compare `prev_grid` vs `next_grid`
2. Find cells that disappeared and appeared per color
3. Check if any (disappeared → appeared) pair matches the action's direction delta
4. Accumulate confidence per candidate color (+0.5 per match, +0.1 per vanish-without-reappear)
5. At 3+ consistent observations: **confirmed**

### 5.3 On Confirmation: Promote to Permanent Layer 2 Star

Once confirmed, create a `MeaningCentricStar` (or equivalent Galaxy entry):

- **Star name:** `spatial_grid_agent_marker` (NOT `arc3_avatar_color`)
- **Galaxy:** Grammar Galaxy (Layer 2 Meaning)
- **Confidence:** 1.0 (binary once confirmed)
- **Provenance:** `action_delta_correlation_N_frames`
- **Persistence:** Permanent — survives ALL attempt boundaries AND game boundaries
- **Coefficients:** `{"color": <discovered_color>, "movable": 1.0, "self_referential": 1.0}`

### 5.4 Secondary Object Classification

Same action-delta correlation approach classifies other objects:

| Observation | Classification | Star Name |
|-------------|---------------|-----------|
| Color at death position when avatar dies | Hazard | `spatial_grid_hazard_marker` |
| Color at terminal position when level completes | Goal | `spatial_grid_goal_marker` |
| Color that blocks movement (no position change after action) | Wall | `spatial_grid_obstacle_marker` |
| Color that moves independently (not correlated with action) | NPC / dynamic obstacle | `spatial_grid_dynamic_entity` |

### 5.5 Where to Implement

In `ARC3EpisodeGalaxy`, add an `_identify_avatar()` method called from `record_outcome()` (or wherever grid transitions are processed). The algorithm is simple enough to run on CPU during frame processing — it's O(rows × cols × 4 actions) per frame. However, the **promotion to Galaxy star** must use `galaxy_memory_updater.cu` for the actual VRAM write.

### 5.6 Integration with Existing `_classify_objects()`

The current `_classify_objects()` method at [arc3_episode_galaxy.py:1020-1045](knowledge3d/knowledgeverse/arc3_episode_galaxy.py#L1020-L1045) already tracks `_objects_by_color` with behavior labels. The avatar identification protocol EXTENDS this — once the avatar color is confirmed, set `_objects_by_color[avatar_color]["behavior"] = "avatar"` AND promote to permanent Galaxy star.

### 5.7 Success Criteria

- Avatar correctly identified within first 3-5 steps of any ARC3 game
- Star created with meaning-based name (`spatial_grid_agent_marker`)
- Star persists across attempts (verify: avatar color known on attempt 2 without re-discovery)
- Wall/hazard/goal colors classified after relevant observations

---

## 6. VectorDotMap Grid Encoding (Bonus — Low Priority)

Per FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md §5.4, grid cells can be represented as VectorDotMap field coefficients instead of raw integer arrays. This enables the GPU consolidation kernels to work with richer per-cell state:

```python
# Per-cell coefficients (GPU-resident, persists between attempts)
@dataclass
class GridCellField:
    position: tuple[float, float]          # (row, col)
    color_coefficient: float               # Current color (0-15)
    exploration_coefficient: float         # 0.0=unvisited → 1.0=fully explored
    block_probability: float               # 0.0=free → 1.0=confirmed wall
    action_success_history: list[float]    # Per-action success rates [4 floats]
```

This is NOT blocking for the other three tasks. Implement after micro-sleeptime and between-attempt consolidation are working.

---

## 7. Files to Modify

| File | Action | What |
|------|--------|------|
| `knowledge3d/knowledgeverse/arc3_episode_galaxy.py` | **MODIFY** | Replace `run_micro_sleeptime()`, add `consolidate_between_attempts()`, add `_purge_attempt_transient_state()`, add `_identify_avatar()` |
| `knowledge3d/knowledgeverse/arc3_sleep_bindings.py` | **CREATE** | ctypes structs (ARC3OutcomeEntry, ARC3RuleEntry, ARC3ObjectHypothesis), GPU buffer management |
| `benchmarks/arc_agi_3.py` | **MODIFY** | Call `consolidate_between_attempts()` at attempt boundary, sync micro-sleeptime after API response |
| `tests/test_arc3_sleeptime_gpu.py` | **CREATE** | Tests for GPU micro-sleeptime, between-attempt persistence, avatar identification |

### Files to READ (Do Not Modify)

| File | Why |
|------|-----|
| `cranium/ptx_runtime/sleep_time_compute.py` | SleepTimeComputeBridge API |
| `cranium/bridges/sovereign_bridges.py:1894` | SleepClusterRefiner.refine_clusters() signature |
| `cranium/ptx_runtime/galaxy_memory_updater.py` | GalaxyMemoryUpdater API |
| `cranium/sovereign/lora_gpu_trainer.py:59` | LoRAGPUEngine API |
| `cranium/ptx/sleep_time_micro.ptx` | Kernel entry points |

---

## 8. Test Plan

1. **Sovereignty test:** `run_micro_sleeptime()` no longer contains `ThreadPoolExecutor` or `executor.submit`
2. **GPU occupancy test:** Mock game API with 200ms delay, verify SM occupancy > 0% during wait
3. **Persistence test:** Run 2 attempts on same game. After attempt 1 death, verify `_rules_by_key` and `_objects_by_color` are NOT empty at start of attempt 2
4. **No-reload test:** Assert `id(knowledgeverse)` is identical across attempts (same Python object)
5. **Avatar identification test:** Feed 3 frame pairs with consistent action-delta. Assert avatar color detected and star created with name `spatial_grid_agent_marker`
6. **Naming compliance test:** Grep all star names created by ARC3 code — assert NONE start with `arc3_`
7. **Regression:** All 51 existing tests still pass

---

## 9. Sovereignty Compliance Checklist

| Requirement | Spec | Status After This Spec |
|-------------|------|----------------------|
| GPU-native consolidation | SLEEPTIME §0.1 | ✅ PTX kernel launch replaces ThreadPoolExecutor |
| Inline execution (no reload) | SLEEPTIME §0.3 | ✅ Same KV instance across attempts |
| Meaning-based star names | FOUNDATIONAL §0 | ✅ `spatial_grid_*` naming |
| 4-layer architecture | FOUNDATIONAL §1.2-1.5 | ✅ Form purged, Meaning/Rules/Meta-Rules persisted |
| VectorDotMap encoding | FOUNDATIONAL §5.4 | ⏳ Deferred (low priority) |
| SM occupancy > 0% during sleep | SLEEPTIME §0.1 | ✅ Verified via test |

---

## 10. Execution Order

1. **First:** Create `arc3_sleep_bindings.py` with ctypes structs and GPU buffer helpers
2. **Second:** Rewrite `run_micro_sleeptime()` to launch `sleep_time_micro.ptx`
3. **Third:** Add `consolidate_between_attempts()` with 3-kernel sequence
4. **Fourth:** Add `_purge_attempt_transient_state()` (Layer 1 only)
5. **Fifth:** Add `_identify_avatar()` with action-delta correlation
6. **Sixth:** Wire into `benchmarks/arc_agi_3.py` game loop
7. **Seventh:** Write tests
8. **Eighth:** Run full test suite (51 existing + new tests)
