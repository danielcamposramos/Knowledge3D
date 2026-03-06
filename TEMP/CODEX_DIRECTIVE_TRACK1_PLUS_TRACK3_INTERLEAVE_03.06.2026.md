# Codex Directive: Track 1 + Track 3 Interleaved -- Ground Up

**Date:** March 6, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Context:** You asked whether to deepen Track 1 (scene orchestration) or start Track 3 (specialist selection). Daniel wants BOTH, but done right from the ground up -- no phase-skipping.

---

## The Answer: Interleave Them

Track 1 and Track 3 are not independent -- they are two sides of the same coin. Scene orchestration PRODUCES execution events. Specialist selection LEARNS from those events. Without one, the other is incomplete:

- Track 1 without Track 3 = scenes play but the system never learns which tools/routes work best
- Track 3 without Track 1 = selection logic exists but has no rich execution events to learn from

The correct order is: **build them together, bottom-up, in thin horizontal slices.**

---

## Phase 2A: Execution Event Recording (Do First)

This is the foundation both tracks need. Every Tool execution must produce a ternary outcome signal that feeds both scene quality AND specialist learning.

### What to build:

**1. Execution Event struct**

Every time a Tool chain executes (whether for scene composition, material projection, signal processing, or anything else), record:

```
execution_event = {
    tool_id: str,                    # which Tool route was used
    query_context: str,              # what the query asked for
    specialist_id: str | None,       # which specialist selected this tool
    math_core_tier: int,             # tier 1/2/3 used
    execution_us: int,               # wall-clock microseconds
    outcome: int,                    # TERNARY: +1 success, -1 failure, 0 uncertain
    quality_signal: float,           # 0.0-1.0 from bridge validation (coherence, fidelity, etc.)
    ternary_quality: int,            # TQUANT(quality_signal): +1 good, -1 bad, 0 neutral
    timestamp_us: int,               # system time (real clock -- critical for Doors sync)
    chain_depth: int,                # how many steps in the chain
    promotion_pressure: bool,        # true if this execution adds pressure for PTX promotion
}
```

This struct lives in the audit journal (Region 6). It feeds:
- Track 1: scene quality metadata (which layers rendered well)
- Track 3: specialist learning (which tool selections succeeded)
- Track 5 (later): sleep consolidation (which patterns to keep/prune)

**2. Wire it into the existing Tool execution path**

Your `tool_execution.py` already resolves and invokes Tool chains. Add event recording at the execution boundary:
- Before: resolve entrypoint, bind payload
- Execute: call bridge
- After: measure time, compute quality from bridge output metadata (coherence, fidelity scores already exist in temporal/material bridges), record event

**3. Storage**

Append to `storage_root/logs/execution_events.jsonl` (same pattern as `tool_promotion_pressure.jsonl`).

### Chain code to leverage:

- **CODEX_DIAGNOSTIC_FRAMEWORK_IMPLEMENTATION.md**: `evaluate_task_with_oracle_metrics()` returns structured outcome data. Same pattern -- wrap execution with measurement.
- **Step10_ThinkingTagInference.md lines 10834-10851**: Access frequency tracking pattern: `if (access_freqs[idx] > 1000) prefetch`. Same idea -- track per-tool access frequency.
- **CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md lines 1143-1210**: `ProgressiveScorer` tracks `improvement_history` per discovery. Same pattern for per-tool quality history.

### Success criteria:
- Every Tool execution produces an execution_event in the journal
- Events carry ternary outcome (+1/-1/0) computed from bridge quality signals
- Events carry real system timestamps
- Existing 47 tests still pass (no regression)
- New tests: verify event recording for at least 3 different Tool chains

---

## Phase 2B: Scene Quality Feedback (Track 1 Deepening)

Now that execution events exist, scene orchestration can use them.

### What to build:

**1. Quality-aware layer composition**

Your `compose_scene_timeline(...)` currently treats all layers equally. Add quality weighting:
- Each `TemporalSceneLayer` carries its execution_event quality_signal
- Layers with ternary_quality = -1 get demoted (lower alpha, pushed to background)
- Layers with ternary_quality = +1 get promoted (full alpha, foreground priority)
- Layers with ternary_quality = 0 stay neutral

This is ternary contrastive applied to scene composition: the system learns to foreground what works.

**2. Scene grammars from House playback**

Your `replay_journal_to_scene_timeline(...)` reconstructs from events. Extend it:
- Detect REPEATING event sequences in the journal (same tool chain used 3+ times)
- Extract those as reusable "scene grammar" templates
- Store templates as Grammar Galaxy entries (this is where Track 1 feeds Track 4 later)

**3. Multi-layer House/world playback presets**

Build on the existing presets (`ui_idle`, `ui_focus`, `world_breathe`, `world_orbit`) with compound presets:
- `house_library` = high-quality consolidated knowledge visualization (Library room, quality >= 0.7)
- `house_garden` = growing/evolving knowledge (Garden room, quality >= 0.4)
- `house_museum` = archived/historical patterns (Museum room, quality < 0.4)

These map directly to the House room quality thresholds from Step7.the_chain.md lines 3295-3308.

### Chain code to leverage:

- **Step7.the_chain.md lines 4302-4470**: `fractal_grow_dynamic.cu` -- quality-driven depth allocation. Apply the same principle: high-quality scene layers get more visual depth/detail.
- **Step7.the_chain.md lines 3617-3820**: `sleep_time_compute.py` replay JSON format. Your `replay_journal_to_scene_timeline` should produce compatible output.
- **Step7.2 - Original.md lines 3054-3141**: `mmap_reader.py` ring buffer for streaming playback without full journal load.

### Success criteria:
- Scene layers carry quality signals from execution events
- At least one scene grammar is auto-detected from repeating journal patterns
- House room presets exist (library/garden/museum)
- Tests cover quality-weighted composition

---

## Phase 2C: Specialist Selection Learning (Track 3 Foundation)

Now that execution events exist with ternary outcomes, specialists can learn.

### What to build:

**1. Per-tool quality tracker**

Simple structure tracking each tool's execution history:

```
tool_quality = {
    tool_id: str,
    total_executions: int,
    success_count: int,        # outcome == +1
    failure_count: int,         # outcome == -1
    uncertain_count: int,       # outcome == 0
    avg_execution_us: float,
    bayesian_quality: float,    # success_count / total_executions
    ternary_trend: int,         # TQUANT of recent quality delta: improving(+1), declining(-1), stable(0)
}
```

Read from `execution_events.jsonl` on startup, update live during execution.

**2. Ternary routing gate in navigator**

Your `TRMNavigator` already ranks tools by query relevance. Add quality-weighted ranking:

```
final_rank = relevance_score * (1.0 + 0.3 * bayesian_quality)
```

Tools with ternary_trend = -1 (declining) get a penalty. Tools with +1 (improving) get a bonus. This is the ternary contrastive signal applied to selection.

**3. Online specialist embedding updates**

From **CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md lines 771-1041**:

When a tool execution succeeds (+1):
- Move the selecting specialist's embedding TOWARD the task embedding (lr=0.1)

When it fails (-1):
- Move AWAY from the task embedding (lr=-0.05)
- Generate the OPPOSITE rule as an anti-pattern (contrastive learning)

When uncertain (0):
- No embedding update, but increment exploration counter

This means specialists naturally drift toward the tasks they're good at, and away from tasks they fail at. Over time, the right specialist gets selected for the right task.

**4. Spawn threshold**

From the chain code: spawn threshold = 0.3. If NO specialist has relevance > 0.3 for a query, that's a signal to create a new specialist (tiny LoRA adapter). Log this as a "specialist gap" event. Don't spawn yet -- just record the gap for Track 4 (Grammar Evolution) to fill later.

### Chain code to leverage:

- **CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md lines 771-1041**: Full `TRMSwarmCoordinator` with 9 bootstrap specialists, online learning, spawn threshold. Read lines 825-882 for `embed_task()` (128-dim from size + color + spatial features).
- **CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md lines 1143-1582**: `ProgressiveScorer` three-tier fate system. Apply same tiers to tool quality: preserve (85%), promote (95%), canonical (100%).
- **PHASE_5.1_COLLABORATIVE_PLAN_FINAL.md**: Confidence calibration via ECE. The per-tool bayesian_quality is analogous -- it should calibrate over time.
- **CLAUDE_HYBRID_TRM_ARCHITECTURE_SPEC.md**: `adaptive_routing_ternary()` with TQUANT gating. Exact pattern for the ternary routing gate.

### Success criteria:
- Per-tool quality tracker exists and updates from execution events
- Navigator ranking incorporates bayesian_quality
- At least one specialist embedding update occurs during test execution
- Specialist gaps are logged when no specialist exceeds threshold
- All previous tests still pass

---

## Execution Order

```
Phase 2A: Execution Event Recording     <- DO THIS FIRST (foundation for both tracks)
    |
    +---> Phase 2B: Scene Quality        <- Track 1 deepening (uses events)
    |         Feedback
    |
    +---> Phase 2C: Specialist Selection <- Track 3 foundation (uses events)
              Learning
```

2B and 2C can be done in either order after 2A, or interleaved. They share the execution event infrastructure but don't depend on each other.

---

## What NOT To Do

- Do NOT skip Phase 2A to jump into 2B or 2C. The event recording IS the ground-up foundation.
- Do NOT add external dependencies. Quality signals come from existing bridge metadata (coherence, fidelity).
- Do NOT over-engineer the specialist learning. Start with simple bayesian quality + ternary trend. The chain code has PPO-style RL and multi-head attention -- those are future refinements, not Phase 2.
- Do NOT implement specialist spawning yet. Just LOG the gaps. Track 4 (Grammar Evolution) handles spawning.
- Do NOT touch sleep-time consolidation yet. That's Track 5, after specialist selection proves the ternary contrastive signal works.

---

## Test Strategy

After each phase:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_procedural_temporal_bridge.py \
  tests/test_tool_galaxy.py \
  tests/test_tool_execution.py
```

Plus new tests per phase:
- 2A: `tests/test_execution_events.py` -- event recording, ternary outcome computation, timestamp correctness
- 2B: `tests/test_scene_quality.py` -- quality-weighted composition, grammar detection, House presets
- 2C: `tests/test_specialist_selection.py` -- quality tracking, ranking integration, embedding updates, gap logging

Target: grow from 47 to ~65 tests across all three phases.

---

## After Phase 2

With execution events + scene quality + specialist learning in place, the system has a closed feedback loop:

```
Query -> Specialist selects Tool -> Tool executes -> Event recorded
   ^                                                      |
   |                                                      v
   +--- Specialist learns (+1/-1/0) <--- Quality signal --+
   |                                                      |
   +--- Scene composition uses quality <------------------+
```

This loop is the foundation for EVERYTHING that comes after:
- Track 4 (Grammar Evolution): detects patterns in execution events to synthesize rules
- Track 5 (Sleep-Time Compute): consolidates successful patterns, prunes failures
- Track 2 (PTX Promotion): promotion pressure comes from event frequency + quality

Build it right now, and all future tracks plug in naturally.
