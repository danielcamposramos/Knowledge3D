# Codex Directive: Deepen Scene Grammars, Then Harvest

**Date:** March 6, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Context:** Phase 2A/2B/2C foundation landed, 52 tests green. You ask: deepen scene grammars or harvest promotion data?

---

## Decision: Scene Grammars First

**Reason:** The promotion pipeline needs accumulated runtime data to make meaningful decisions. Right now `execution_events.jsonl` is nearly empty -- the system just started recording. Harvesting empty journals produces noise.

Scene grammars GENERATE execution events. Every composition, replay, and preset triggers Tool chains that flow through the Phase 2A substrate. More grammar work = denser event journals = richer promotion signal later.

```
NOW:   Scene grammars + House/world playback   (generates events)
THEN:  Harvest events into promotion decisions  (with real data)
```

---

## Phase 2D: Scene Grammars + House/World Playback

### 2D.1 Grammar Detection from Execution Patterns

Your `procedural_temporal_bridge.py` already auto-detects repeating scene grammars from replay streams. Deepen this:

**Detect tool-chain patterns, not just scene patterns.** Read `execution_events.jsonl` and find repeating sequences:

```
Pattern = sequence of (tool_id, outcome) tuples that recurs 3+ times
```

Example: if `contour_to_mesh -> material_projection -> temporal_preview` recurs with consistent +1 outcomes, that's a proven grammar.

**Store detected grammars as Grammar Galaxy entries.** Each grammar is:
- An RPN program (the tool chain as opcode sequence)
- Quality metadata (avg quality_signal across occurrences)
- Ternary confidence: TQUANT(bayesian_quality) from the quality tracker
- Source: "auto_detected" (vs "manual" or "chain_derived")

This is the seed of Track 4 (Grammar Evolution). You're not building full cross-modal rule synthesis yet -- just the detection + storage of proven tool-chain patterns.

**Chain code to leverage:**
- **CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md lines 28-341**: `GrammarGalaxy` with local discovery space, quality-gated promotion. The `_local_discoveries` dict pattern (rule_id -> {rpn_program, usage_count, success_count, quality_score}) is exactly what you need for detected grammars.
- **CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md lines 1143-1210**: Three-tier fate (preserve 85%, promote 95%, canonical 100%). Apply to grammar confidence thresholds.

### 2D.2 House Room Playback Presets

You already have `house_library`, `house_garden`, `house_museum` assignment from quality thresholds. Build real playback behavior for each:

**house_library** (quality >= 0.7):
- Clean, stable visualization
- High-confidence layers only (ternary_quality = +1)
- Minimal animation (knowledge is settled)
- Layout: `overlay` with sharp edges
- Think: a well-organized bookshelf

**house_garden** (quality >= 0.4):
- Growing, evolving visualization
- Mix of +1 and 0 layers (confident + exploring)
- Moderate animation (`world_breathe` tempo)
- Layout: `golden_orbit` (organic growth pattern)
- Think: a garden where ideas sprout and branch

**house_museum** (quality < 0.4):
- Archived, historical visualization
- Includes -1 layers as "lessons learned" (contrastive!)
- Slow animation (`ui_idle` tempo)
- Layout: `horizontal_strip` (timeline/archive feel)
- Think: a museum of attempts, including failures that teach

**Chain code to leverage:**
- **Step7.the_chain.md lines 3295-3308**: Original House room thresholds (Library/Garden/Museum).
- **Step7.the_chain.md lines 4302-4470**: `fractal_grow_dynamic.cu` adaptive depth. Garden preset should use quality-driven visual depth: high-quality entries get deeper fractal branching.
- **Step7.the_chain.md lines 4148-4163**: `curiosity_prune.cu` -- keep high-curiosity nodes volatile in Garden. Garden preset should highlight high-curiosity entries (ternary_quality = 0, high exploration count).

### 2D.3 Multi-Layer Scene Orchestration

Build compound scenes from multiple playback surfaces:

**Scene templates:**

```
compound_house_tour = [
    TemporalSceneLayer(preset="house_library", domain="ui", weight=1.0),
    TemporalSceneLayer(preset="house_garden", domain="world", weight=0.8),
    TemporalSceneLayer(preset="house_museum", domain="world", weight=0.5),
]
```

The scene compositor weights layers by their quality AND their domain relevance to the query. A "show me what I know" query weights library high. A "show me what I'm learning" query weights garden high. A "show me my history" query weights museum high.

**Tool routes to add:**
- `tool_house_tour_scene_v1` -- compound multi-room visualization
- `tool_house_library_scene_v1` -- focused library playback
- `tool_house_garden_scene_v1` -- focused garden playback

**Navigator ranking update:**
- "knowledge" / "library" / "settled" queries -> library tools
- "learning" / "growing" / "exploring" queries -> garden tools
- "history" / "archive" / "failures" / "lessons" queries -> museum tools
- "tour" / "overview" / "all" queries -> compound tour

### 2D.4 Event Density Target

To generate enough data for meaningful promotion harvesting later, aim for:

- Each new scene grammar detected triggers at least one execution event per constituent tool
- Each House preset playback generates a quality-annotated event chain
- Each compound scene generates N events (one per layer)

By the time this phase completes, `execution_events.jsonl` should have 100+ events from test runs alone, with real quality signals from real bridge execution.

---

## After 2D: Harvest Promotion Data (Phase 2E)

Once scene grammars are generating dense event streams, THEN harvest:

1. Read `execution_events.jsonl` and `tool_promotion_pressure.jsonl`
2. Run `build_tool_promotion_report.py` with real data
3. Identify: which bridge operations are called most? which have highest quality? which are slowest?
4. First PTX promotion candidate = highest frequency + lowest quality + highest latency
5. Use the sovereign promotion workflow from Step9.md lines 3908-3936 (CUDA C++ -> nvcc -> PTX -> ctypes)

But that's next phase. Don't jump to it now.

---

## Success Criteria for Phase 2D

1. Grammar detection finds at least 2 repeating tool-chain patterns from execution events
2. Detected grammars are stored as Grammar Galaxy entries with quality metadata
3. House room presets (library/garden/museum) produce distinct visual behaviors
4. Garden preset uses quality-driven depth (high quality = deeper branching)
5. Museum preset includes -1 layers as contrastive "lessons learned"
6. Compound scene (house tour) composes all three room presets
7. Navigator ranking correctly routes knowledge/learning/history queries to appropriate presets
8. `execution_events.jsonl` has 100+ events after running the full test suite
9. All existing tests still pass (52 baseline)
10. New tests cover grammar detection, House presets, compound scenes, navigator routing

Target: grow from 52 to ~62 tests.

---

## Test Command

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_execution_events.py \
  tests/test_scene_quality.py \
  tests/test_specialist_selection.py \
  tests/test_procedural_temporal_bridge.py \
  tests/test_tool_execution.py \
  tests/test_tool_galaxy.py
```

---

## Why This Order Matters

Scene grammars are not just "Track 1 deepening." They are the FIRST real Grammar Galaxy entries created by the system itself (not by humans, not by ingestion). When a tool-chain pattern recurs with +1 outcomes and gets stored as a Grammar Galaxy entry, that is the system learning from its own execution. That is the seed of Track 4 (Grammar Evolution) and ultimately the path Daniel identified: "this is where AGI will emerge from."

Build the grammars. Let the events accumulate. Then harvest with real signal.
