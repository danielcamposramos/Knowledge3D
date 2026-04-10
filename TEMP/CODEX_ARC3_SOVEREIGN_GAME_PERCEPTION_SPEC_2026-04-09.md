# Codex Direction: ARC-3 Sovereign Game Perception — INTERACT Emission + Stuckness

**Date:** 2026-04-09
**Authority:** THREE_BRAIN_SYSTEM_SPECIFICATION.md (§6b.2 Run Sequence, §6b.4 Universal Input Path),
KNOWLEDGEVERSE_SPECIFICATION.md (§1 — sovereign memory), MEMORY_TABLET_SPECIFICATION.md (§2 — interface layer),
CLAUDE.md (Python = Boot + I/O only)
**Priority:** CRITICAL — v2 autonomous run: 312 ACTION1 (UP) fixation, 0 levels, ZERO ACTION5 emitted
**Precondition:** Learning loop fixes applied (CODEX_ARC3_LEARNING_LOOP_SPEC_2026-04-09.md ✓)

---

## Sovereignty Reminder

Per THREE_BRAIN_SYSTEM_SPECIFICATION.md §6b.4:

> Input (any format) → I/O adapter normalizes → kv.execute_task(query=...) →
>   TRM embeds → Galaxy search → Find meaning star(s) →
>   Jarvis reads symlinks → Dispatch specialist(s) →
>   Workers execute RPN chains → Halting Gate → Answer

**Python is the I/O adapter.** The WINE layer normalizes the ARC-3 frame into a
TabletEnvelope and sends the TRM's answer back to the server. Python perceives and
transmits. Python does NOT reason.

**Current sovereignty violations in arc_agi_3.py:**
- `_exploration_fallback()` → Python reasoning (which direction to move)
- `_select_mechanic_target()` → Python reasoning (which object to pursue)
- Blocked-action tracking → Python state machine (which directions are walls)

**Goal:** Shrink these Python reasoning paths. Enrich the perception signal so
the sovereign pipeline (TRM → Galaxy → Workers → Halting Gate) produces the right
action. Every fix below stays at the I/O perception boundary or enriches Galaxy
knowledge — never adds Python reasoning.

---

## Diagnosis: Two Missing Connections

### Connection 1: No "At Target → INTERACT" Signal

The spatial plan (LED-A* on GPU — sovereign) navigates to objects. But when the
avatar ARRIVES at the target (path_length ≈ 0):

1. `_decode_path_action()` returns `None` (no movement step left)
2. `_spatial_path_plan()` returns `None`
3. Falls through to episode rule query → no matching rule for "at object"
4. Falls through to `_exploration_fallback()` → bounces AWAY from the target

**Result:** The avatar reaches the white cross, stands on it, then walks away.
INTERACT (ACTION5) is NEVER emitted. After 450 actions in v2, zero ACTION5.

The game knowledge star `arc3_game_rule:key_switch_interaction` says:
```
meaning_rpn = "AGENT_AT_WHITE_CROSS ACTION5"
```

This star EXISTS in the Galaxy but the pipeline never connects perception
("avatar at object") to this rule ("emit ACTION5").

### Connection 2: No Stuckness Signal in Perception

The wall-following exploration picks a heading (UP) and sticks to it.
When blocked by a wall:

1. `_same_gameplay_state()` detects identical frames → adds UP to `blocked_actions`
2. Exploration sends avatar sideways briefly → frame changes → blocked resets
3. Spatial plan says UP again → cycle repeats
4. Net effect: 312 UP actions (69%) with brief detours

**Missing perception signal:** "avatar centroid has not changed meaningfully in
N steps." This is a PERCEPTUAL observation (frame analysis), not reasoning.
The WINE adapter should encode it in the query text so the TRM processes it.

---

## Fix 1 (CRITICAL): Spatial Plan Emits INTERACT at Target

**File:** `benchmarks/arc_agi_3.py`
**Method:** `_spatial_path_plan()`

When LED-A* computes a path of length 0 or 1 to a target (avatar is at or
adjacent to the target), the plan should emit ACTION5 (INTERACT) instead of
returning `None`.

This is NOT Python reasoning — it is the I/O decode of the LED-A* result:
- LED-A* says "you're at the destination" (sovereign computation)
- Python I/O translates "at destination" → ACTION5 (same as translating
  "first step is north" → ACTION1)

**Logic at the end of `_spatial_path_plan()`:**

After the target-search loop, if no movement plan was found BUT the avatar is
adjacent to (or at) a target centroid, emit INTERACT:

```python
# After the for-obj loop, if best_plan is None:
if best_plan is None and avatar_centroid is not None:
    # Check if avatar is already adjacent to any target
    for obj in objects:
        centroid = (float(obj["centroid"][0]), float(obj["centroid"][1]))
        distance = abs(centroid[0] - avatar_centroid[0]) + abs(centroid[1] - avatar_centroid[1])
        if distance <= 3.0 and not bool(obj.get("interaction_tested", False)):
            interact_index = 4  # ACTION5 = index 4 (0-indexed)
            if valid_action_indices and interact_index in set(valid_action_indices):
                best_plan = {
                    "action_index": interact_index,
                    "confidence": 0.9,
                    "target_label": str(obj.get("semantic_hint") or "object"),
                    "path_length": 0,
                    "program_type": "spatial_at_target_interact",
                    "solver": "spatial_frame_led_pathfinder",
                }
                break
```

**Also:** when the movement path has length 1 (one step to target), check if the
NEXT step would put the avatar at the target. If so, after the movement, the
FOLLOWING call should trigger INTERACT.

**Key insight:** The spatial plan is the I/O decode layer between LED-A*
(sovereign) and the ARC-3 server. Translating "at target" → ACTION5 is the same
kind of I/O translation as "path step north" → ACTION1.

---

## Fix 2 (CRITICAL): Stuckness Signal in WINE Perception

**File:** `benchmarks/arc_agi_3.py`
**Method:** `choose_action()` — in the perception section, before envelope creation

Track centroid drift over recent history. This is frame analysis = perception:

```python
# Perception: compute centroid drift over last N steps
centroid_drift = 0.0
if avatar_centroid is not None and len(self.action_history) >= 5:
    recent_centroids = []
    for record in self.action_history[-5:]:
        rc = record.get("avatar_centroid")
        if rc is not None:
            recent_centroids.append(rc)
    if len(recent_centroids) >= 3:
        centroid_drift = max(
            abs(c[0] - avatar_centroid[0]) + abs(c[1] - avatar_centroid[1])
            for c in recent_centroids
        )

stuck_signal = centroid_drift < 2.0 and len(self.action_history) >= 5
```

**Encode in query text (perception, not reasoning):**

In `_frame_to_query_text()`, add stuck signal:
```python
if stuck_signal:
    state_tokens.append("stuck avatar not progressing try different approach")
```

**Encode in TabletEnvelope task_context:**
```python
task_context["stuck_signal"] = stuck_signal
task_context["centroid_drift"] = centroid_drift
```

The TRM game loop sees "stuck" in the query and navigates to different Galaxy
neighborhoods — this is sovereign TRM reasoning responding to perceptual input.

**Also:** when stuck_signal is True, the `blocked_actions` set should be cleared
to break the cycle. This is I/O state management (resetting the perception
filter), not reasoning.

---

## Fix 3 (HIGH): Record Avatar Centroid in Action History

**File:** `benchmarks/arc_agi_3.py`
**Method:** `choose_action()` — in the record section at the end

The action history record must include `avatar_centroid` for Fix 2 to work:

```python
record["avatar_centroid"] = avatar_centroid
```

This is already computed at the top of `choose_action()`. Just include it in
the record dict.

---

## Fix 4 (HIGH): Episode Rule for "Adjacent to Untested Object → INTERACT"

**File:** `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`
**Method:** `seed_game_mechanics()` or initial Episode Galaxy setup

Seed one Episode rule at game start that encodes the discovery protocol:

```
rule_id: arc3_rule:{game_id}:adjacent_to_untested_object:ACTION5
state: agent_adjacent_to_untested_object
action: 4  # ACTION5 (0-indexed)
confidence: 0.7
source: game_mechanics_prior
```

This rule should be found by `query_rule_for_state()` when:
- The avatar's adjacent colors include an object that hasn't been interaction-tested
- No spatial plan or TRM direct action is available

**This closes the perception → Galaxy → action loop within the sovereign path:**
1. WINE perceives "avatar adjacent to untested object" (frame analysis)
2. Episode Galaxy has a rule for this state → returns ACTION5
3. ACTION5 is emitted to the ARC-3 server
4. After interaction, `learn_from_outcome()` records the result
5. Consolidation strengthens or weakens the rule

---

## Fix 5 (HIGH): `query_rule_for_state()` Must Check Object Adjacency

**File:** `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`
**Method:** `query_rule_for_state()`

Currently the rule query matches state keys like `agent_adjacent_to_color_4:ACTION3`.
Extend the matching to include object-level adjacency:

The caller in `choose_action()` already computes `adjacent_colors`. Extend to also
check whether any adjacent color belongs to a known untested object. If so, generate
the synthetic state key `agent_adjacent_to_untested_object` before querying.

```python
# In choose_action(), before calling query_rule_for_state:
has_adjacent_untested = False
if avatar_centroid is not None and visible_objects:
    for obj in visible_objects:
        if bool(obj.get("interaction_tested", False)):
            continue
        centroid = (float(obj["centroid"][0]), float(obj["centroid"][1]))
        dist = abs(centroid[0] - avatar_centroid[0]) + abs(centroid[1] - avatar_centroid[1])
        if dist <= 3.0:
            has_adjacent_untested = True
            break
```

Pass `has_adjacent_untested` to `query_rule_for_state()`. The query should try
both color-based rules AND the object-adjacency rule.

---

## Fix 6 (HIGH): Post-Interaction Learning

**File:** `benchmarks/arc_agi_3.py`
**Method:** `learn_from_outcome()`

When ACTION5 is emitted and the next frame differs significantly from the current
frame, record a causal rule in the Episode Galaxy:

```python
# In learn_from_outcome(), when the action was INTERACT (index 4):
if action_index == 4 and not _same_gameplay_state(pre_frame, post_frame):
    # Significant frame change after INTERACT → successful interaction
    for obj_key, obj in self._known_objects.items():
        if not bool(obj.get("interaction_tested", False)):
            centroid = obj["centroid"]
            dist = abs(centroid[0] - avatar_row) + abs(centroid[1] - avatar_col)
            if dist <= 4.0:
                obj["interaction_tested"] = True
                obj["interaction_result"] = "frame_changed"
                # Persist to Episode Galaxy
                self._episode_galaxy.seed_rule(
                    state=f"agent_near_color_{obj['color']}_object",
                    action=4,
                    outcome="frame_changed",
                    confidence=0.9,
                )
                break
```

This is I/O-boundary learning: frame comparison (perception) → Galaxy update
(knowledge persistence). The reasoning about WHAT the interaction means happens
in the TRM game loop when it later queries the Episode Galaxy.

---

## Fix 7 (MEDIUM): Walkable Cells Must Exclude Walls

**File:** `benchmarks/arc_agi_3.py`
**Function:** `_walkable_cells()`

The LED-A* path planner treats every cell as walkable. If the background color
(e.g., color 3 = large 892-pixel region) represents the walkable floor, then
NON-background cells might be walls or objects. The walkable set should be
restricted to background-colored cells plus the avatar's current position:

```python
def _walkable_cells(grid):
    gameplay = _gameplay_grid(grid)
    background = _background_value(gameplay)
    cells = []
    for r, row in enumerate(gameplay):
        for c, val in enumerate(row):
            if int(val) == background:
                cells.append((r, c))
    return cells
```

If `_walkable_cells` already does this, verify it. If it includes ALL cells,
LED-A* will plan paths through walls, which is why the avatar gets stuck.

**This is critical for fixing the UP-fixation:** LED-A* computes an optimal path
through walls → avatar can't follow → gets stuck going UP forever.

---

## Fix 8 (MEDIUM): Clear Blocked State When Strategy Changes

When the spatial plan target changes (different object, or INTERACT instead of
movement), clear `blocked_actions` and `_blocked_actions_by_state`:

```python
if current_target_label != self._last_target_label:
    self._blocked_actions_by_state.clear()
    self._last_blocked_action = None
    self._blocked_repeat_count = 0
    self._last_target_label = current_target_label
```

This prevents stale blocked-direction state from persisting across different
navigation goals.

---

## Implementation Priority

1. **Fix 1** — Spatial plan emits INTERACT at target (CRITICAL, this is why 0 levels)
2. **Fix 7** — Walkable cells exclude walls (CRITICAL, this is why UP-fixation)
3. **Fix 3** — Record avatar centroid in history (required for Fix 2)
4. **Fix 2** — Stuckness signal in perception
5. **Fix 4** — Episode rule for adjacent-to-untested → INTERACT
6. **Fix 5** — query_rule_for_state checks object adjacency
7. **Fix 6** — Post-interaction learning
8. **Fix 8** — Clear blocked state on target change

**Expected impact:** With INTERACT emission (Fix 1), the avatar can interact
with the white cross. With walkable-cell filtering (Fix 7), LED-A* plans paths
that avoid walls. Together: avatar navigates to cross → INTERACTs → navigates
to door → level complete.

---

## What NOT to Do

**DO NOT add more Python reasoning.** Each fix above is either:
- **Perception** (frame analysis → signal encoding → envelope)
- **I/O decode** (LED-A* result → action index translation)
- **Galaxy knowledge** (Episode rule seeding/querying)
- **Learning** (outcome recording → Galaxy persistence)

**DO NOT hardcode the LS20 solution.** The avatar must DISCOVER:
- Which objects exist (✓ already working — `_detect_static_objects`)
- Which to approach first (✓ spatial plan targets nearest switch)
- That INTERACT near objects causes changes (Fix 1 + Fix 4 + Fix 6)
- That the door is the next target after switch interaction (Fix 6 → learn → next tick)

**DO NOT limit steps.** Daniel: "100 actions is ridiculous for learning purposes,
let it play!" The autonomous run should have high step limits. Learning requires
volume.

---

## Architecture Compliance

| Fix | Layer | Sovereignty |
|-----|-------|------------|
| Fix 1 | I/O decode (LED-A* result → ACTION5) | ✓ LED-A* sovereign, decode = I/O |
| Fix 2 | Perception (centroid drift → query text) | ✓ Frame analysis = perception |
| Fix 3 | I/O (include centroid in action record) | ✓ Data recording = I/O |
| Fix 4 | Galaxy (Episode rule seeding) | ✓ Galaxy = sovereign knowledge |
| Fix 5 | Galaxy (Episode rule query extension) | ✓ Galaxy query = sovereign |
| Fix 6 | Learning (frame diff → Galaxy update) | ✓ Outcome → Galaxy = learning |
| Fix 7 | Perception (walkable cell filter) | ✓ Grid analysis = perception |
| Fix 8 | I/O state (clear stale blocked state) | ✓ Perception state reset |

---

## The Sovereign Path (What Should Happen)

Per THREE_BRAIN_SYSTEM_SPECIFICATION.md §6b.4:

```
Frame from ARC-3 server
  → WINE adapter parses (Python I/O)
  → Normalize to TabletEnvelope
    - query: "avatar at (32,21) adjacent to switch at (45,31) untested"
    - task_context: {stuck: false, adjacent_objects: [...], visited_cells: 18}
  → kv.execute_task()
  → TRM embeds query → Galaxy search
  → Finds: arc3_game_rule:key_switch_interaction ("AGENT_AT_WHITE_CROSS ACTION5")
  → Jarvis dispatches visual specialist
  → Worker processes: pattern match → emit ACTION5
  → Halting Gate converges
  → Result: {action_name: "ACTION5", confidence: 0.9}
  → WINE adapter sends ACTION5 to ARC-3 server
```

**Today:** The TRM doesn't reliably produce this. The fixes above provide I/O-level
safety nets (spatial plan INTERACT, Episode rule) while the TRM sovereign path
matures. As the TRM learns from shadow copies of successful interactions, the
Python fallbacks naturally shrink.

**The goal is always: Python shrinks, TRM grows.**

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ARC3_SOVEREIGN_PERCEPTION_REPORT_2026-04-09.md` with:

1. Fix 1: Does spatial plan emit ACTION5 when at target? Test with 10 steps.
2. Fix 7: What does `_walkable_cells()` return? Does it include walls?
3. After fixes, run 1 attempt of 500 steps. Report:
   - Action distribution (should include ACTION5)
   - Whether the avatar reached and interacted with the white cross
   - Whether any level was completed
   - Spatial plan target changes over the run
4. Then restart the long autonomous run (5 attempts × 10000 steps)
5. `echosys_ingest` tmux still alive
