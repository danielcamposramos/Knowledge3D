# Codex — Phase E.48: Knowledge Depth + TRM Reasoning Path

**Date:** 2026-03-31
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — this is the inflection point between Python-doing-work and TRM-doing-work

---

## Daniel's Mandate (Verbatim)

> "The problems all are solved by the TRM — no Python orchestration."
> "The tasks should run inline, the parallelization is at launch time
> and after the run."
> "Sleeptime compute MUST happen so the model learns — direct execution,
> not a script to call."
> "Follow the specs inside the docs/vocabulary folder."

---

## What E.38 Achieved (Confirmed)

- 5 reasoning strategy stars in `/K3D/Knowledge3D.local/house/reasoning_strategies.jsonl`
  (forward_entity_extraction, backward_goal_tracing, operation_chain_construction,
  result_validation, four_way_reading_meta_rule)
- `reasoning_strategies` added to `GPU_MATH_TARGET_GALAXIES` and `GPU_GSM8K_TARGET_GALAXIES`
- `_is_reasoning_strategy_entry()` filters strategy rows by galaxy/tag/category
- `_gsm8k_reasoning_strategy_rows()` extracts matching rows from catalog
- Scoring boost: `reasoning_strategy_similarity * 0.14`, `reasoning_strategy_entry * 0.08`,
  `reasoning_strategy_focus * 0.22` in composed head candidate evaluation
- 4 tests passing

**What's missing:** We have 5 strategy stars but the TRM doesn't YET navigate
star-to-star (multi-hop). It still does one embedding lookup per query. The stars
exist in the Galaxy — the TRM needs to FIND them, COMPOSE them, and EXECUTE
the chain. That's Phase D work (TRM game loop), which is the deep fix.

---

## Part A: E.40 — Build 100+ Universal Game Knowledge Stars

The E.40 spec exists (TEMP/CODEX_PHASE_E40_*.md). The House currently has 121
game mechanics stars. We need 100+ MORE universal 2D game knowledge stars.

### CRITICAL: These Must Be Meaning-First (Layer 2-4)

Per FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md §1:

```
Layer 4: META-RULES (Strategy/Eloquence)   ← WHEN/WHY to apply a game strategy
Layer 3: RULES (Grammar/Transformation)    ← HOW to play (pathfinding, resource management)
Layer 2: MEANING (Words/Semantics)         ← WHAT concepts mean (door, key, recharge, budget)
Layer 1: FORM (Characters/Glyphs)          ← HOW things look (already covered by Drawing Galaxy)
```

We need depth at EVERY layer, not just Layer 2 "definitions":

### Layer 2 Stars (MEANING — what game concepts ARE)

These define universal concepts any 2D game agent must know:

**Movement concepts (20+ stars):**
- cardinal_movement, diagonal_movement, continuous_movement
- gravity, jump_arc, double_jump, wall_jump
- ladder_climbing, swimming, teleportation
- conveyor_belt, moving_platform, wrap_around
- sliding_ice_physics, momentum, velocity, acceleration
- walkable_surface, wall_boundary, obstacle, gap

**Resource concepts (15+ stars):**
- movement_budget, energy_system, health_points, lives_system
- score, inventory_slot, key_lock_pair, fuel_gauge
- recharge_station, power_up, health_pickup
- timer_countdown, checkpoint, save_point
- resource_depletion, resource_conservation

**Spatial concepts (15+ stars):**
- room, corridor, bridge, platform, ladder
- door_entrance, door_exit, locked_door, one_way_door
- switch_mechanism, pressure_plate, trigger_zone
- spawn_point, respawn_point, goal_zone
- vertical_level, horizontal_scroll, screen_transition

**State concepts (10+ stars):**
- idle_state, moving_state, falling_state, jumping_state
- attacking_state, damaged_state, invincible_state, dead_state
- level_complete, game_over, victory_condition
- transition_screen, flash_feedback

### Layer 3 Stars (RULES — HOW to play)

These are executable transformation rules (RPN programs):

**Navigation rules (15+ stars):**
- pathfind_to_target: "Given position + target → compute path via walkable graph"
  - RPN: `POSITION RECALL TARGET RECALL WALKABLE_GRAPH BFS_PATH`
  - Symlinks: → walkable_surface, target_identification, movement_budget

- avoid_obstacle: "When wall detected in movement direction → choose alternate direction"
  - RPN: `DIRECTION PROBE IF WALL THEN ROTATE_CW DIRECTION PROBE`
  - Symlinks: → wall_boundary, cardinal_movement

- bridge_gap_crossing: "When gap detected → check for bridge/platform/jump possibility"
  - RPN: `GAP_WIDTH RECALL JUMP_RANGE COMPARE IF WITHIN THEN JUMP ELSE FIND_BRIDGE`
  - Symlinks: → gap, jump_arc, bridge

- resource_aware_movement: "Before each move → check remaining budget vs distance to target"
  - RPN: `BUDGET RECALL DISTANCE_TO_TARGET COMPARE IF LESS THEN SEEK_RECHARGE`
  - Symlinks: → movement_budget, recharge_station, pathfind_to_target

- waypoint_sequencing: "When multiple objectives exist → order by dependency + distance"
  - RPN: `OBJECTIVES LIST DEPENDENCY_SORT DISTANCE_RANK FIRST_FEASIBLE`
  - Symlinks: → key_lock_pair, switch_mechanism, door_entrance

**Interaction rules (10+ stars):**
- activate_switch: "When adjacent to switch → perform action"
- collect_item: "When adjacent to pickup → perform action"
- open_door: "When adjacent to door AND has_key → perform action"
- use_recharge: "When adjacent to recharge station → perform action to restore budget"
- strategic_reset: "When budget critical AND target unreachable → reset to preserve life"

**Perception rules (10+ stars):**
- identify_avatar: "In frame → find connected component with avatar colors"
- identify_target_room: "In frame → find enclosed structure with door colors, prefer topmost"
- identify_walkable_area: "In frame → walkable = floor color cells NOT in UI area"
- identify_ui_elements: "Bottom rows → movement bar, lives, reference box"
- detect_level_transition: "When frame is single dominant color → transition screen"
- detect_budget_status: "In status bar → yellow cells / total cells = remaining fraction"

### Layer 4 Stars (META-RULES — WHEN to apply strategies)

These are the strategic decision-makers:

**Game strategy meta-rules (10+ stars):**
- exploration_vs_exploitation: "Early in level → explore to map layout. Late → exploit known path"
  - RPN: `ACTIONS_SPENT TOTAL_BUDGET RATIO IF LOW THEN explore_mode ELSE exploit_mode`
  - Symlinks: → pathfind_to_target, movement_budget

- safety_first_strategy: "When lives > 1 → aggressive exploration OK. When lives = 1 → conservative"
  - RPN: `LIVES RECALL IF ONE THEN conservative_mode ELSE aggressive_mode`
  - Symlinks: → lives_system, strategic_reset

- waypoint_planning: "For multi-mechanic levels → plan waypoint sequence before executing"
  - RPN: `LEVEL_MECHANICS SCAN IF LOCK THEN FIND_KEY FIRST IF RECHARGE THEN PLAN_REFUEL`
  - Symlinks: → waypoint_sequencing, key_lock_pair, recharge_station

- level_progression_meta: "After level complete → reperceive new layout, reset spatial model"
  - RPN: `IF LEVEL_COMPLETE THEN CLEAR_SPATIAL_MODEL REPERCEIVE FRESH_PLAN`
  - Symlinks: → detect_level_transition, identify_walkable_area

- budget_management_meta: "Monitor budget continuously. At 35% → plan route to recharge. At 15% → force reset if recharge unreachable"
  - RPN: `BUDGET_FRACTION IF BELOW_35 THEN plan_recharge IF BELOW_15 THEN strategic_reset`
  - Symlinks: → resource_aware_movement, strategic_reset, recharge_station

### Implementation Requirements

1. **Format**: JSONL entries in `/K3D/Knowledge3D.local/house/game_mechanics.jsonl`
   (append to existing 121 stars, or create `game_knowledge_universal.jsonl`)

2. **Each star MUST have**:
   - `galaxy`: "Grammar" for rules, "Reality" for meanings, or domain-appropriate
   - `layer`: 2 (meaning), 3 (rule), or 4 (meta-rule) — EXPLICIT
   - `meaning`: The concept definition (language-agnostic)
   - `rpn_program`: Executable RPN (even if symbolic/placeholder)
   - `symlinks`: Bidirectional references to related stars
   - `category`: "game_mechanics", "game_strategy", "game_perception", etc.
   - `tags`: ["bootstrap", "game_knowledge", "universal"]

3. **Bidirectional symlinks**: If star A symlinks to star B, star B must symlink back to A.
   Per the feedback norm established in House construction.

4. **Universal, NOT game-specific**: "movement_budget" not "LS20_yellow_bar".
   These stars must help with ANY 2D game, not just ARC3 LS20.

5. **Load at boot**: Same pattern as existing game_mechanics.jsonl — loaded during
   Knowledgeverse initialization.

### Success Criteria (E.40)

- [ ] 100+ new stars across Layers 2, 3, and 4
- [ ] At least 20 Layer 3 (Rule) stars with RPN programs
- [ ] At least 10 Layer 4 (Meta-Rule) stars with strategy logic
- [ ] All stars have bidirectional symlinks
- [ ] Total game knowledge > 221 stars (121 existing + 100 new)
- [ ] Loaded at boot, accessible via Galaxy search
- [ ] Universal concepts, not game-specific

---

## Part B: Validate GSM8K Score With E.38 Wiring

The reasoning strategy stars are seeded and wired. But we have no end-to-end
GSM8K score to prove they help. The last known score was 0/20.

### What To Do

1. Run GSM8K subset (20 problems) through the LIVE system:
   ```bash
   conda activate k3d-cranium
   export CUDA_VISIBLE_DEVICES=0
   python scripts/run_full_benchmark.py --gsm8k-count 20 --skip-mmlu --skip-arc --skip-lhe --skip-math --skip-imo
   ```
   (Or whatever the current CLI flags are — adapt to actual `run_full_benchmark.py` args)

2. Log the results. Compare against 0/20 baseline.

3. If still 0/20: The 5 strategy stars exist but the TRM isn't navigating TO them
   during GSM8K. Diagnose: are the strategy rows showing up in candidate scoring?
   Add debug logging to `_gsm8k_reasoning_strategy_rows()` to confirm rows are
   found and boosted.

4. If > 0/20: Report the score and what made the difference.

### The Deeper Issue

Even with strategy stars boosted in scoring, the current pipeline does ONE
composed-head pass per query. For GSM8K, the 4-way reading strategy requires
FOUR sequential passes:
1. Forward entity extraction → bind entities
2. Backward goal tracing → identify needed operations
3. Operation chain construction → build multi-step RPN
4. Validation → check result plausibility

The TRM game loop (`trm_game_loop.py` tick()) calls `_execute_task_direct()`
ONCE. It doesn't iterate. The `four_way_reading_meta_rule` star says
"call forward, then backward, then chain, then validate" but the pipeline
only does one pass.

**This is the Phase D gap.** For now, even a SINGLE pass that finds the
right strategy star and applies it partially is progress. Multi-hop
comes when the TRM game loop actually loops.

### Success Criteria (GSM8K Validation)

- [ ] GSM8K 20-problem run completes
- [ ] Score reported (even if 0/20, the diagnostic data matters)
- [ ] If 0/20: debug log showing whether strategy rows were found in candidate scoring
- [ ] Results logged to JSONL for comparison

---

## Part C: ARC3 — What the TRM Should Be Doing

### The Current State

The spatial bridge (E.45) works: walkable graph → Morton → LED-A* → action.
Target selection (E.46) improved: topmost door, correct walkable color, UI excluded.
But level 1 still doesn't solve (max_levels_completed=0).

### The Real Problem

Look at `choose_action()` (arc_agi_3.py lines 1117-1260). The Python code:
1. Detects avatar position (Python: `_avatar_centroid`)
2. Detects targets (Python: `_door_components`, `_switch_components`, `_recharge_components`)
3. Builds walkable graph (Python: `_walkable_cells`, CSR construction)
4. Calls LED-A* pathfinder (GPU: `pathfinder.navigate_csr`)
5. Decodes path to action (Python: `_decode_path_action`)

Steps 1, 2, 3, 5 are Python orchestration. Only step 4 is GPU.
The TRM sees a text query like "arc3 interactive game frame grid 64x64 door target
visible action move up..." and does ONE embedding-based Galaxy lookup.

**The TRM is not reasoning about the game.** The Python code IS the reasoning.
The TRM is just a passthrough that returns whatever the spatial pathfinder computed.

### What Should Happen Instead

Per THREE_BRAIN_SYSTEM_SPECIFICATION.md §1 and SPATIAL_GENERAL_INTELLIGENCE.md:

1. **Perceive**: TRM receives the frame grid. Frustum culling identifies what's
   in view. The PERCEPTION RULES (Layer 3 stars like `identify_avatar`,
   `identify_target_room`, `identify_walkable_area`) are Galaxy stars the TRM
   navigates to and executes.

2. **Navigate**: TRM navigates to relevant Galaxy neighborhood. For a 2D game
   frame, it should navigate to game_mechanics stars. LED-A* finds spatially
   relevant knowledge. The game knowledge stars (Layer 2-4) provide the
   concepts: "this is a door", "this is a walkable floor", "budget is low".

3. **Reason**: Nine-Chain Swarm processes candidates. The META-RULE stars
   (Layer 4) say "for this game state, apply pathfind_to_target with
   budget_management_meta." The RULE stars (Layer 3) say "compute path
   from avatar to topmost door via walkable graph."

4. **Decide**: Halting Gate checks convergence. Multiple swarm workers may
   propose different actions (go right, go up, seek recharge). The one
   with highest convergence signal wins.

5. **Act**: Emit the action. One step.

6. **Learn**: Shadow copy records the trace. After the game ends, sleeptime
   consolidation strengthens correct paths, weakens incorrect ones.

### The Bridge: Making Python Orchestration Galaxy-Navigable

We can't jump to full Phase D overnight. But we CAN:

1. **Keep the spatial bridge as infrastructure** — it provides real pathfinding
   capability. The Python graph construction is I/O (translating the frame into
   a format the GPU can process). That's acceptable as boot/I/O work.

2. **Make the TRM's Galaxy lookup INCLUDE game knowledge stars** — currently
   the ARC_TASK routing uses `["Drawing", "Grammar", "Tool", "Reality", "Word"]`.
   Game knowledge stars must be IN one of these galaxies (or the routing must
   include the galaxy they're in). Verify: what galaxy are game_mechanics.jsonl
   stars assigned to? If they're not in the routing set, the TRM can't find them.

3. **Feed spatial plan results BACK through the TRM** — instead of
   `_spatial_path_plan()` short-circuiting `choose_action()` at line 1251,
   the spatial plan should INFORM the TRM query. Add the spatial plan's
   output (target label, path direction, confidence) to the query text so
   the TRM can evaluate it against Galaxy knowledge.

4. **Log TRM Galaxy hits** — for each ARC3 action, log which Galaxy stars
   the TRM found. If it's finding game_mechanics stars, that's progress.
   If it's only finding Drawing/Grammar stars, the game knowledge isn't
   connected.

### Concrete Changes

**Step 1**: Verify game knowledge galaxy assignment:
```python
# Check: what galaxy are game_mechanics.jsonl stars in?
# They MUST be in one of ARC3_ROUTE_GALAXIES = ["Drawing", "Grammar", "Tool", "Reality", "Word"]
# If they're in "Reality" or "Grammar", they'll be found.
# If they're in a custom galaxy not in the routing set, they won't.
```

**Step 2**: Enrich the ARC3 query text with spatial plan context:
```python
# In choose_action(), AFTER computing spatial_plan:
if spatial_plan is not None:
    # Add spatial plan info to the query text so TRM can evaluate
    gpu_task["query"] += (
        f" spatial_plan target={spatial_plan['target_label']}"
        f" direction={ACTION_LABELS[spatial_plan['action_index']]}"
        f" path_length={spatial_plan['path_length']}"
        f" confidence={spatial_plan['confidence']:.3f}"
    )
```

**Step 3**: Log Galaxy stars found during ARC3:
```python
# In the result record, include which Galaxy stars the TRM navigated to
# This is diagnostic — tells us if game knowledge is reachable
```

### Success Criteria (ARC3)

- [ ] Game knowledge stars verified reachable via ARC3 routing
- [ ] Spatial plan fed back to TRM query (not short-circuit)
- [ ] Galaxy hit logging for ARC3 actions
- [ ] Level 1 progress (any improvement over current max_levels_completed=0)
- [ ] Diagnostic: which stars does the TRM find for ARC3 game frames?

---

## Part D: Sleeptime Consolidation Diagnostic

`briefs_consolidated=0` means the system is NOT learning after runs. Per
Daniel: sleeptime compute is direct execution by the living system — it
MUST happen for the model to learn.

### What To Check

1. **Is sleeptime being triggered?** After `run_full_benchmark.py` completes,
   does the Knowledgeverse enter sleep mode? Check the code path that triggers
   consolidation.

2. **Are there briefs TO consolidate?** Each query should produce a "brief"
   (answer trace + correct/incorrect signal). If no briefs are written, there's
   nothing to consolidate.

3. **Is the contrastive learning path active?** Per THREE_BRAIN_SYSTEM_SPECIFICATION.md,
   sleeptime consolidation uses shadow copy comparison: strengthen paths where
   shadow copy matched ground truth, weaken where it didn't.

### What To Log

Add diagnostic output to sleeptime:
- Number of briefs available for consolidation
- Number of correct/incorrect classifications
- Number of Galaxy paths strengthened/weakened
- Time spent in consolidation

### Success Criteria (Sleeptime)

- [ ] Diagnostic: why briefs_consolidated=0 (no briefs, or no consolidation trigger?)
- [ ] If no briefs: add brief writing to query path
- [ ] If no trigger: add sleeptime trigger after benchmark run
- [ ] After fix: briefs_consolidated > 0 on next run

---

## Execution Priority

| Task | Priority | Why |
|------|----------|-----|
| E.40 game knowledge (Part A) | HIGH | TRM can't reason about games with 121 shallow stars |
| GSM8K validation (Part B) | HIGH | Proves E.38 wiring works end-to-end |
| ARC3 TRM path (Part C) | MEDIUM | Diagnostic + incremental improvement |
| Sleeptime diagnostic (Part D) | MEDIUM | Enables learning (long-term critical) |

Parts A and B are independent — can start in parallel.
Part C depends on Part A (game knowledge must exist before TRM can find it).
Part D is independent.

---

## Grounding: Specs to Follow

Per Daniel's instruction — follow the specs in `docs/vocabulary/`:

- **FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md** — 4-layer architecture (Form → Meaning → Rules → Meta-Rules).
  Game knowledge stars MUST follow this layering.
- **THREE_BRAIN_SYSTEM_SPECIFICATION.md** — TRM IS the avatar. Perceive → Navigate → Reason → Decide → Act → Learn.
  The TRM game loop is the target architecture.
- **SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md** — SGI paradigm. Spatial reasoning through
  Galaxy navigation, not Python heuristics.
- **KNOWLEDGEVERSE_SPECIFICATION.md** — 7-region VRAM substrate. All knowledge in Galaxy, all
  reasoning on GPU.
- **DUAL_CLIENT_CONTRACT_SPECIFICATION.md** — Form + Meaning. Each star serves both
  human understanding and AI execution.

**The direction is clear:** Enrich the Galaxy (Parts A, B), make the TRM actually
navigate it (Part C), and make the system learn from experience (Part D).

---

## Architectural Note

The current arc_agi_3.py is ~1260 lines of Python doing perception,
target detection, graph construction, pathfinding orchestration, budget
management, reset logic, click probing, exploration fallback, etc.

This is the SAME pattern as the 35+ `_gsm8k_*` Python methods — doing
the TRM's job in the wrong place. The eventual target is: `arc_agi_3.py`
becomes a thin I/O adapter (~50 lines) that feeds frames to
`kv.execute_task()` and translates results to ARC3 API actions.

We're not there yet. But every spec should move TOWARD that target, not
away from it. When adding game knowledge stars (Part A), ask: "does this
star encode something that's currently hardcoded in Python?" If yes,
that's exactly right — we're moving intelligence from Python to Galaxy.
