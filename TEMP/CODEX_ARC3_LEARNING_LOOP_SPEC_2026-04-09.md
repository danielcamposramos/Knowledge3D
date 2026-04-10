# Codex Direction: ARC-3 Sovereign Learning Feedback Loop

**Date:** 2026-04-09
**Authority:** THREE_BRAIN_SYSTEM_SPECIFICATION.md (§Galaxy=Internal Brain, §TRM Game Loop),
KNOWLEDGEVERSE_SPECIFICATION.md (§Episode Galaxy), MEMORY_TABLET_SPECIFICATION.md (§Tablet perception)
**Priority:** HIGH — actions register now but agent oscillates LEFT-RIGHT (320L/305R of 645 actions)
**Precondition:** Game ID fix applied (CODEX_ARC3_GAME_ID_FIX_AND_SDK_ENV_2026-04-09.md)

---

## Diagnosis: Why 0 Levels After 645 Actions

**Actions register** (action_input.id=3 for LEFT, id=4 for RIGHT — confirmed).
**But the agent oscillates LEFT-RIGHT** instead of navigating toward objectives.

### Three Broken Links in the Learning Chain

**Link 1 — No Target:** `_select_mechanic_target()` returns `(None, "")` because
`_switch_components()` looks for hardcoded colors {11, 15} and `_door_components()`
looks for {5, 9}. If the LS20 game uses different color indices in the gameplay grid,
the spatial plan has NO target and cannot produce a directed path.

**Action distribution proves this:**
```
ACTION3 (LEFT):  320 (49.6%)  ← _exploration_fallback() bouncing
ACTION4 (RIGHT): 305 (47.3%)  ← _exploration_fallback() bouncing
ACTION2 (DOWN):   15 (2.3%)   ← occasional blocked-direction escape
ACTION1 (UP):      5 (0.8%)   ← rare
```

**Link 2 — Rules stored but never consulted:** Episode Galaxy consolidation produces 7
rules across 5 attempts:
```
agent_adjacent_to_color_4:ACTION3
agent_adjacent_to_color_4:ACTION4
agent_adjacent_to_color_3:ACTION3
agent_adjacent_to_color_3:ACTION4
agent_adjacent_to_color_9:ACTION4
agent_adjacent_to_color_4:ACTION2
agent_adjacent_to_color_4:ACTION1
```
But `choose_action()` NEVER queries these rules. Decision pipeline:
`execute_task()` → (no game action) → `_spatial_path_plan()` → (no target → None)
→ `_exploration_fallback()` → bounce LEFT-RIGHT.

**Link 3 — No object map from vision:** The initial probe DOES detect the avatar
(centroid shifts when LEFT is pressed: `(55.17, 28.89) → (55.08, 28.23)`). But this
information is not used to build a persistent object map. The agent re-discovers the
avatar every reset but never maps the environment.

---

## Architecture Grounding

Per THREE_BRAIN_SYSTEM_SPECIFICATION.md §Galaxy=Internal Brain:
> Galaxy is the AI avatar's Internal Brain — a unified multi-modal workspace...
> ALL default galaxies loaded simultaneously in VRAM

The Episode Galaxy should serve as the game's working memory. Rules persisted during
consolidation ARE Galaxy entries — but the TRM never navigates to them during decisions.

Per KNOWLEDGEVERSE_SPECIFICATION.md, the TRM game loop should:
1. **Perceive** → Frustum cull what's in field-of-view
2. **Navigate** → LED-A* to relevant Galaxy neighborhood
3. **Reason** → Nine-Chain Swarm parallel workers
4. **Decide** → Halting Gate convergence
5. **Act** → Emit action
6. **Learn** → Shadow copy records trace

Steps 2-4 are not producing game-relevant results because the Galaxy has no game-specific
navigation targets. The seeded game knowledge stars exist but aren't being queried.

---

## Fix 1 (CRITICAL): Discover Targets from Frame Diff, Not Hardcoded Colors

The probe already works: pressing each direction and comparing frames reveals which
pixels belong to the avatar. **Extend this to discover ALL objects:**

```python
def _discover_objects_from_probe(env, card_id, game_id, guid):
    """Probe all 4 directions to build an object map from pixel diffs."""
    objects = {}

    # Get baseline frame
    baseline = env.step(ACTION3)  # LEFT
    frame_after_left = baseline["grid"]

    # Compare to find avatar pixels (the ones that moved)
    # Also find static objects (colored regions that DON'T move)

    # For each unique color cluster in the gameplay area:
    #   - If it moved when avatar moved → it's the AVATAR
    #   - If it's small, isolated, specific colors → candidate SWITCH
    #   - If it's tall, near top-center → candidate DOOR
    #   - If it's a bar at specific position → candidate BUDGET/LIVES indicator

    return objects
```

**But don't hardcode interpretations.** Instead, store discovered objects as Episode
Galaxy stars with properties:
```
{
    "type": "ARC3_OBJECT",
    "color": 9,
    "centroid": (row, col),
    "size": 12,
    "moves_with_avatar": false,
    "position_label": "top_center",     # derived from centroid
    "interaction_tested": false,
    "interaction_result": null
}
```

The spatial plan should navigate toward the NEAREST untested static object.

---

## Fix 2 (CRITICAL): Consult Episode Rules Before Exploration Fallback

In `choose_action()`, BEFORE falling through to `_exploration_fallback()`, query the
Episode Galaxy for matching rules:

```python
# After spatial_plan is computed (or is None), before exploration_fallback:

if episode_context and not _result_has_direct_action(runtime_result):
    # Query episode rules for current state
    rule_action = self._episode_galaxy.query_rule_for_state(
        avatar_centroid=avatar_centroid,
        adjacent_colors=_adjacent_colors(normalized_frame, avatar_centroid),
        frame_state=frame_state,
    )
    if rule_action is not None and rule_action not in blocked_actions:
        action_choice = rule_action
        exploration_reason = "episode_rule_match"
```

This closes the feedback loop: consolidation stores rules → next decision reads them.

---

## Fix 3 (HIGH): Systematic Exploration Instead of Bouncing

Replace `_exploration_fallback()` bouncing with systematic wall-following:

**Current behavior:** Try LEFT, blocked → try RIGHT, succeeds → try LEFT again → bounce

**Desired behavior (wall-following):**
1. Pick a primary direction (e.g., LEFT) and go until blocked
2. When blocked, turn 90° (DOWN), go until blocked
3. Continue wall-following pattern to systematically explore the map
4. Track visited cells (from avatar centroid history) to prefer unvisited areas

```python
def _systematic_explore(self, avatar_centroid, valid_actions, blocked_actions):
    """Wall-following exploration that covers the map systematically."""
    # If we have centroid history, prefer directions toward unvisited areas
    visited = set()
    for record in self.action_history:
        # approximate cell from centroid
        ...

    # Wall-following: maintain a "hand on wall" direction
    if not hasattr(self, '_explore_heading'):
        self._explore_heading = 2  # Start going LEFT (index 2 = ACTION3)

    # If current heading is blocked, turn clockwise
    turn_order = [2, 0, 3, 1]  # LEFT → UP → RIGHT → DOWN (clockwise wall-follow)
    start = turn_order.index(self._explore_heading)
    for i in range(4):
        candidate = turn_order[(start + i) % 4]
        if candidate not in blocked_actions and candidate in valid_actions:
            self._explore_heading = candidate
            return candidate

    return valid_actions[0]  # All blocked — just try anything
```

---

## Fix 4 (HIGH): Build Object Map from Avatar Movement

Track the avatar's centroid across frames to build a visited-cell map and an object map:

```python
# In learn_from_outcome(), after micro_sleeptime:

if normalized_frame is not None and avatar_centroid is not None:
    cell = (round(avatar_centroid[0]), round(avatar_centroid[1]))
    self._visited_cells.add(cell)

    # Detect nearby static objects from the frame
    for obj in _detect_static_objects(normalized_frame, avatar_centroid):
        obj_key = (obj["color"], obj["approximate_position"])
        if obj_key not in self._known_objects:
            self._known_objects[obj_key] = obj
            # Seed to Episode Galaxy
            self._episode_galaxy.seed_object(obj)
```

The spatial plan can then target the nearest UNVISITED or UNTESTED object.

---

## Fix 5 (MEDIUM): Verify Color Detection Against Actual Frame

Run a one-time diagnostic to check what colors actually appear in the LS20 gameplay area:

```python
def _diagnose_frame_colors(frame):
    """Log all unique color values and their positions in the gameplay grid."""
    gameplay = _gameplay_grid(frame)
    color_map = defaultdict(list)
    for r, row in enumerate(gameplay):
        for c, val in enumerate(row):
            if val != background:
                color_map[val].append((r, c))
    for color, positions in sorted(color_map.items()):
        print(f"Color {color}: {len(positions)} pixels, "
              f"centroid=({mean(r for r,c in positions):.0f}, {mean(c for r,c in positions):.0f})")
```

If the switch is color 11 but the game uses color 7, the entire mechanic detection fails.
**Log this once on RESET and seed the correct colors into the Episode Galaxy.**

---

## Implementation Priority

1. **Fix 5 first** — diagnose actual frame colors (5 minutes, one-time log)
2. **Fix 1** — update `_switch_components`/`_door_components` with correct colors
3. **Fix 3** — systematic exploration (wall-following instead of bouncing)
4. **Fix 2** — episode rule consultation (close the feedback loop)
5. **Fix 4** — object map from movement (persistent world model)

**Expected impact:** With correct target detection (Fix 1), the spatial plan should
produce directed paths. With wall-following (Fix 3), the agent should explore the map
systematically. With rule feedback (Fix 2), repeated attempts should improve.

---

## Sovereignty Compliance

All fixes are at the I/O/perception boundary (Python's role per CLAUDE.md):
- Frame analysis = perception preprocessing → becomes Galaxy entry
- Object detection = perceptual structure → fed to TRM as Galaxy stars
- Spatial plan = LED-A* on GPU (already sovereign)
- Rule consultation = Galaxy query (already sovereign)

The `_exploration_fallback()` replacement is Python-side I/O policy (which action to
send to the server), not reasoning. Reasoning stays on GPU via the composed head pipeline.

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ARC3_LEARNING_LOOP_REPORT_2026-04-09.md` with:

1. Fix 5: What colors appear in LS20 gameplay frames? List them.
2. Fix 1: Did `_switch_components` / `_door_components` color values match? What changed?
3. Fix 3: Was wall-following implemented? Did action distribution change?
4. Test: After fixes, run 1 attempt of 100 steps. Report:
   - Action distribution (should be more varied than 50/50 LEFT-RIGHT)
   - Whether spatial_plan_target is non-empty
   - Whether any new areas of the map were reached
5. `echosys_ingest` tmux still alive
