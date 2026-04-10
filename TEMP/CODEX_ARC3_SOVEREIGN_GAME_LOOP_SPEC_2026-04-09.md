# CODEX: ARC-3 Sovereign Game Loop — Move Decision Logic INTO Knowledgeverse

**Date:** 2026-04-09
**Author:** Claude (Architecture)
**Priority:** CRITICAL — this is a sovereignty violation, not an optimization
**Spec for:** Codex (Implementation)
**References:**
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` §6b.4 (Universal Input Path)
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §1 (Unified Sovereign Memory)
- `docs/vocabulary/MEMORY_TABLET_SPECIFICATION.md` §1-3 (Memory Tablet as I/O)

---

## Diagnosis: Why ARC-3 Agent Scores 0

**Root cause: ALL game decision logic runs in Python, NOT in the Knowledgeverse.**

The current `choose_action()` in `benchmarks/arc_agi_3.py` is ~400 lines of Python orchestration:
- `_detect_static_objects()` — Python reasoning about colors/positions/sizes
- `_select_mechanic_target()` — Python priority system (switch > door > recharge)
- `_spatial_path_plan()` — Python wrapping LED-A* (the only sovereign part)
- `_exploration_fallback()` — Python wall-following strategy
- `query_rule_for_state()` — Python dictionary lookup (NOT Galaxy search)

**The sovereign hot path (`dispatch_task()`) IS called**, but:
1. TRM embeds the query, searches Galaxy star table in VRAM
2. Finds NO game rules — because Episode Galaxy rules live in a **Python dict** (`self._rules_by_key`), NOT in the VRAM star table
3. Returns no direct action (no `action_name`, no `action_index`)
4. Python orchestration takes over with spatial plans, Episode rule queries, exploration fallback

**Every single action is Python-decided. TRM produces zero game decisions.**

This violates THREE_BRAIN_SYSTEM §6b.4:
```
Input (any format) → I/O adapter normalizes → kv.execute_task(query=...) →
  TRM embeds → Galaxy search → Find meaning star(s) →
  Jarvis reads symlinks → Dispatch specialist(s) →
  Workers execute RPN chains → Halting Gate →
  Answer

There are ZERO if task_type == branches in the hot path.
The ONLY place task type appears is in thin I/O adapters.
```

---

## Fix: Sovereign Game Loop Architecture

### Principle

**Python = I/O adapter ONLY.** Frame in, action out. ~30 lines, not 400.

**Knowledgeverse = ALL decision logic.** Game rules as Galaxy stars. TRM finds them. Workers decide action. Halting Gate converges.

### The Correct Flow

```
ARC-3 Server Frame
      ↓
[I/O Adapter — Python, ~30 lines]
  1. Receive frame pixels
  2. Build perception query text (what the camera sees, NOT what to do)
  3. kv.execute_task(query=perception_text, surface_kind="GAME_2D")
  4. Receive result → extract action_index
  5. Send action to ARC-3 REST API
      ↓
[Knowledgeverse — Sovereign, GPU]
  1. TRM embeds perception query
  2. Galaxy search finds game rule stars (IN the star table)
  3. Jarvis reads symlinks on matched game rules
  4. Workers execute RPN: evaluate game state against rule conditions
  5. Halting Gate: converged action_index
  6. Result: {action_index: N, action_name: "ACTIONN"}
```

---

## Implementation: 4 Fixes

### Fix 1 (CRITICAL): Load Episode Galaxy Rules INTO the Star Table

**File:** `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`

**Problem:** `_rules_by_key` is a Python dict. TRM can't see it.

**Fix:** When a rule is seeded or crystallized, it must ALSO be inserted into the live VRAM star table via the Knowledgeverse's `galaxy_manager`. The existing `_persist_strong_rules()` method (line 629) already does this for consolidation — but it only runs at deep consolidation time (between attempts). Rules must be live-inserted **immediately** so TRM can find them during the CURRENT game.

**Implementation:**

```python
def _seed_game_mechanics_priors(self) -> None:
    """Seed initial game rules into BOTH Python cache AND live Galaxy."""
    # Keep Python cache for fast Episode queries (unchanged)
    self._rules_by_key[("ACTION5", "agent_adjacent_to_untested_object")] = { ... }

    # NEW: Also insert into live Galaxy star table
    self._insert_rule_star(
        condition="agent_adjacent_to_untested_object",
        action="ACTION5",
        predicted_outcome="interact_probe",
        confidence=0.7,
    )

def _insert_rule_star(self, *, condition: str, action: str,
                       predicted_outcome: str, confidence: float) -> None:
    """Insert a game rule as a live Galaxy star visible to TRM."""
    galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
    if galaxy_manager is None:
        return
    star_id = f"arc3_rule:{self.game_id}:{condition}:{action}"
    star = MeaningCentricStar(
        star_id=star_id,
        meaning_class="arc3_game_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        # Meaning RPN encodes what this rule DOES
        meaning_rpn=f"{condition.upper()} {action}",
        # Behavior RPN encodes the action output
        behavior_rpn=f"{action} -> {predicted_outcome}",
        taxonomy_refs=["arc3", "game_rule", "live_episode"],
        grammar_refs=["arc3", "game_mechanic", action.lower()],
        meta_refs=[
            f"confidence:{confidence:.3f}",
            f"answer_kind:action",
            f"action_index:{_action_name_to_index(action)}",
        ],
        confidence=1,
        polarity=1,
    )
    galaxy_manager.store_meaning_star(
        "Grammar", star, category="arc3_game_rule",
        metadata={"source": "arc3_episode_live", "game_id": self.game_id},
    )
```

**Critical:** `_crystallize_rules()` must ALSO call `_insert_rule_star()` for newly crystallized rules, not just store them in the Python dict. Rules must be searchable by TRM immediately.

**Helper:**
```python
def _action_name_to_index(action_name: str) -> int:
    """ACTION1 -> 0, ACTION5 -> 4, etc."""
    try:
        return int(action_name.replace("ACTION", "")) - 1
    except:
        return -1
```

### Fix 2 (CRITICAL): Materialize Game Actions from Star Content

**File:** `knowledge3d/knowledgeverse/sovereign_hot_path.py` (and/or `knowledgeverse.py`)

**Problem:** When `dispatch_task()` finds a game rule star as the winner, `materialize_runtime_result()` doesn't know how to extract an action from it. The result has no `action_name` or `action_index`, so `_result_has_direct_action()` returns False.

**Fix:** In `materialize_runtime_result()`, check if the winner star has `answer_kind:action` in its `meta_refs`. If so, extract the action_index from the star content and include it in the result.

**Where:** `knowledgeverse.py` → `materialize_runtime_result()` method

**Implementation approach:**
```python
# Inside materialize_runtime_result(), after existing logic:
if winner_star:
    meta_refs = list(winner_star.get("meta_refs") or [])
    for ref in meta_refs:
        if ref.startswith("answer_kind:action"):
            # This is a game action star — extract action_index
            for ref2 in meta_refs:
                if ref2.startswith("action_index:"):
                    try:
                        action_index = int(ref2.split(":")[1])
                        result["action_index"] = action_index
                        result["action_name"] = f"ACTION{action_index + 1}"
                        result["answer_kind"] = "action"
                        result["answer_materialized"] = True
                    except:
                        pass
            break
```

**This is NOT a task-type branch.** It reads the star's own metadata to determine output format — the star tells the pipeline what it is, per the dual-client contract. Any star with `answer_kind:action` gets this treatment, regardless of surface_kind.

### Fix 3 (HIGH): Strip Python Decision Logic from `choose_action()`

**File:** `benchmarks/arc_agi_3.py`

**Problem:** 400+ lines of Python orchestration override TRM decisions.

**Fix:** `choose_action()` becomes a thin I/O adapter:

```python
def choose_action(self, frame, *, goal_frame=None, task_data=None,
                   available_actions=None, game_id=None,
                   levels_completed=0, episode_context=None) -> dict:
    """Thin I/O adapter: frame → kv.execute_task() → action."""
    normalized_frame = _normalize_grid(frame)
    normalized_goal = _normalize_grid(goal_frame) if goal_frame else [[]]
    frame_state = _frame_state(normalized_frame)
    budget_snapshot = _movement_budget_snapshot(normalized_frame)
    lives_remaining = _lives_remaining(normalized_frame)
    valid_action_indices = _available_action_indices(available_actions)
    resolved_game_id = str(game_id or self._game_id or "unknown")

    # I/O: build perception text (what the camera sees)
    query = _frame_to_query_text(
        normalized_frame, normalized_goal,
        available_actions=available_actions,
        frame_state=frame_state,
        budget_snapshot=budget_snapshot,
        lives_remaining=lives_remaining,
    )

    # Sovereign: let TRM decide
    envelope = arc3_game_envelope(
        task_id=f"arc3_live_{len(self.action_history) + 1:04d}",
        frame=normalized_frame,
        goal_frame=normalized_goal if normalized_goal != [[]] else None,
        available_actions=list(available_actions or []),
        action_options=list(ACTION_NAMES),
        query=query,
        step_count=int(self._step_count),
        game_id=resolved_game_id,
        levels_completed=int(levels_completed),
    )

    raw_result = self.tablet_boundary.submit(envelope, use_enriched=True)
    tablet_result = dict(raw_result or {})
    emitted = dict(tablet_result.get("emitted") or
                   tablet_result.get("response") or {})

    # I/O: extract TRM's action decision
    action_choice, payload = _derive_action_from_result(
        normalized_frame, emitted, goal_frame=normalized_goal,
    )
    action_index = int(action_choice) if isinstance(action_choice, int) else 0
    if valid_action_indices and action_index not in set(valid_action_indices):
        action_index = int(valid_action_indices[0])

    # Record and return (I/O bookkeeping)
    record = {
        "action": ACTION_NAMES[action_index],
        "action_index": int(action_index),
        "label": ACTION_LABELS[action_index],
        "confidence": float(emitted.get("confidence", 0.0)),
        "converged": int(emitted.get("convergence_signal", 0)),
        "gpu_execution": bool(emitted.get("gpu_execution", False)),
        "solver": str(emitted.get("solver", "sovereign_game_loop")),
        "frame_number": len(self.action_history) + 1,
        "frame_state": frame_state,
        "game_id": resolved_game_id,
        "levels_completed": int(levels_completed),
        "step_count": int(self._step_count),
        **payload,
    }
    self.action_history.append(record)
    self._last_frame = _clone_grid(normalized_frame)
    self._step_count += 1
    return record
```

**What gets REMOVED from Python:**
- `_exploration_fallback()` — wall-following is a decision → Galaxy rule
- `_select_mechanic_target()` — target priority is a decision → Galaxy rule
- `_spatial_path_plan()` Python wrapper — LED-A* should be triggered by TRM workers, not Python
- `query_rule_for_state()` Python call — TRM finds rules in Galaxy, not Python querying a dict
- All blocked/stuck/repeat detection — these are perception signals that go INTO the query text, TRM decides what to do with them
- All `if spatial_plan is None` fallback chains

**What STAYS in Python (I/O boundary):**
- `_frame_state()` — frame classification (gameplay vs transition) is signal processing
- `_movement_budget_snapshot()` — reading HUD pixels is signal processing
- `_lives_remaining()` — reading HUD pixels is signal processing
- `_frame_to_query_text()` — encoding perception as text for TRM
- `_avatar_centroid()` — locating the avatar is signal processing (camera tracking)
- `_normalize_grid()` — data normalization is I/O

### Fix 4 (HIGH): Enrich Perception Query for TRM

**File:** `benchmarks/arc_agi_3.py` → `_frame_to_query_text()`

**Problem:** The query text must be rich enough for TRM embedding to match game rule stars.

**Fix:** The query text should describe the PERCEPTUAL state, not prescribe actions:

```
"ARC3 game frame. Avatar at row=27 col=32. Background color=4.
Visible objects: color=5 at row=12 col=15 size=43 untested.
Color=9 at row=5 col=32 size=200. Color=12 at row=45 col=20 size=10.
Movement budget: 72% remaining. Lives: 3.
Adjacent cells: up=4(background) down=5(object) left=4(background) right=4(background).
Object adjacent to avatar: color=5 at distance=1.
Available actions: ACTION1(UP) ACTION2(DOWN) ACTION3(LEFT) ACTION4(RIGHT) ACTION5(INTERACT) ACTION6(CLICK) ACTION7(RESET)."
```

The key addition: **"Object adjacent to avatar: color=5 at distance=1."** This is the perception signal that should match the Episode Galaxy rule "agent_adjacent_to_untested_object → ACTION5".

The I/O adapter computes adjacency (signal processing, like a proximity sensor). TRM matches the perception against Galaxy rules. TRM decides the action.

**Critical:** `_detect_static_objects()` stays as a signal processing function (it's a camera analyzing what it sees). But `_select_mechanic_target()` and the priority logic get REMOVED — TRM decides which object matters based on Galaxy rules.

---

## What TRM Must Learn (NOT Hardcode)

Daniel's direction: "do not hard code the solution ever! TRM must pursue the white cross first, then the door."

**Galaxy should contain GENERIC game rules, not LS20-specific rules:**

1. `"agent_adjacent_to_untested_object → ACTION5"` (already seeded) — try interacting with any untested object
2. `"agent_far_from_objects → explore"` (movement in unvisited direction) — explore when no nearby objects
3. `"interaction_caused_state_change → record_causal_rule"` (post-interaction learning) — remember what worked
4. `"stuck_no_movement → try_different_direction"` — basic anti-loop
5. `"budget_critical → minimize_moves"` — resource management

**TRM LEARNS through gameplay:**
- Observes white cross → tries ACTION5 → state changes → crystallizes rule: "color=5 → ACTION5 → state_change"
- Observes door → tries moving to it → level completes → crystallizes rule: "color=9 → move_toward → level_complete"
- These rules are inserted into star table via Fix 1 → TRM finds them in subsequent frames

**The path to scoring:**
1. First 50-100 actions: exploration (generic rules guide movement)
2. Eventually adjacent to white cross: generic "untested_object → ACTION5" fires
3. ACTION5 causes state change → learning records causal rule
4. Now TRM has a specific rule: "near color=5 → ACTION5"
5. After switch flip, explore toward door area
6. Eventually reach door → level complete → record winning trace
7. Next attempts: TRM KNOWS the rules, navigates directly

---

## Frame-to-Galaxy Signal Processing

These perception computations stay in Python as I/O signal processing (like a camera sensor):

| Function | Purpose | Classification |
|----------|---------|---------------|
| `_avatar_centroid()` | Locate avatar in frame | Camera tracking |
| `_detect_static_objects()` | Find non-background objects | Object detection sensor |
| `_background_value()` | Identify floor color | Scene analysis |
| `_adjacent_colors()` | What colors touch the avatar | Proximity sensor |
| `_movement_budget_snapshot()` | Read HUD bar | HUD reader |
| `_lives_remaining()` | Read HUD lives | HUD reader |
| `_frame_state()` | gameplay vs transition | Scene classifier |
| `_same_gameplay_state()` | Frame diff detection | Motion detector |

These stay because they convert raw pixels → structured features. They do NOT decide actions.

---

## Rollout Strategy

**Phase 1 (This spec — do immediately):**
1. Fix 1: Insert Episode rules into live Galaxy star table
2. Fix 2: Materialize action from game rule stars
3. Fix 4: Enrich perception query text with adjacency signals
4. Test: TRM should now emit ACTION5 when it finds the "adjacent_to_untested → ACTION5" star

**Phase 2 (After Phase 1 produces ACTION5):**
5. Fix 3: Strip Python decision logic
6. Verify: agent runs on sovereign game loop only
7. Let it play 5×10000 steps and learn

**Why phased:** Phase 1 makes TRM produce actions. Phase 2 removes Python fallbacks. If we remove fallbacks BEFORE TRM produces actions, the agent emits only ACTION1 (default 0).

---

## Success Criteria

1. **TRM emits direct actions for GAME_2D tasks** — `_result_has_direct_action()` returns True
2. **ACTION5 appears in action distribution** — the sovereign pipeline emits INTERACT
3. **Episode rules are in the star table** — `SELECT * FROM star_table WHERE meaning_class='arc3_game_rule'` returns rows
4. **choose_action() Python is < 50 lines** (Phase 2)
5. **At least 1 level completed in LS20** — TRM learned to flip switch then reach door

## Report Back

Write report at: `TEMP/CODEX_TO_CLAUDE_ARC3_SOVEREIGN_GAME_LOOP_REPORT_2026-04-09.md`

Include:
1. Whether `dispatch_task()` now returns `action_index` for GAME_2D tasks
2. How many game rule stars are in the live star table
3. 100-step bounded test with action distribution (must show ACTION5 > 0)
4. Whether `_result_has_direct_action()` returns True for game frames
5. Restart long autonomous run (5×10000 steps) with new build
6. Scorecard URL
7. echosys_ingest tmux still alive

---

## Key Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/arc3_episode_galaxy.py` | Fix 1: insert rules into live Galaxy |
| `knowledge3d/knowledgeverse/sovereign_hot_path.py` | Fix 2: materialize action from star meta_refs |
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Fix 2: `materialize_runtime_result()` handles `answer_kind:action` |
| `benchmarks/arc_agi_3.py` | Fix 3: strip Python decision logic (Phase 2), Fix 4: enrich query text |
| `knowledge3d/tablet/wine/game2d_wine.py` | May need updates for enriched query passthrough |

---

**Sovereignty reminder:** "reasoning is not inside python!!! do it properly inside K3D live game living AI paradigm!" — Daniel, April 9, 2026
