# CODEX: ARC-3 VRAM Learning + House Persistence — The Living Brain

**Date:** 2026-04-09
**Author:** Claude (Architecture)
**Priority:** CRITICAL — the brain isn't learning
**Spec for:** Codex (Implementation)
**References:**
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` §7 (Shadow Copy Learning), §6b.2 (Always-On Run Sequence)
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §7 (Shadow Copy), §8 (SleepTime Consolidation)
- Daniel: "I can see the VRAM is stalled - if it was being properly used as storage it would grow and be pruned at the end of each cycle"

---

## Diagnosis: The Brain Isn't Learning

**Current state:** 139 actions, 100% ACTION2 (RIGHT), VRAM stalled.

The sovereign pipeline finds the SAME star every query and emits the SAME action forever. The VRAM doesn't grow because:

1. **Objects are Python-only** — `seed_object()` updates `_objects_by_color` dict, never creates Galaxy stars
2. **Crystallization threshold = 3** — needs 3 identical (action, color) observations before creating a rule. Early game learns nothing.
3. **No negative rules as Galaxy stars** — "ACTION2 → blocked" is in a Python dict, TRM can't see it
4. **No frame-state stars** — observations like "avatar at (27,32), adjacent colors = [4,5]" don't become searchable Galaxy entries
5. **House rules don't load at boot** — `arc3_rule_entries = 0` on fresh start
6. **No VRAM growth logging** — impossible to see learning happen

**The brain needs to GROW during gameplay:**
- Step 0: 2 seeded rules → 2 Galaxy stars
- Step 10: 6 objects discovered, 3 causal rules → 11 Galaxy stars
- Step 50: 15 objects mapped, 8 causal + 5 negative rules → 30 Galaxy stars
- Step 100: map stabilizes, rules strengthen → 35 Galaxy stars, weak ones pruned
- End of attempt: strong rules persisted to House, ephemeral stars cleared

---

## Fix 1 (CRITICAL): Objects Become Galaxy Stars

**File:** `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`

**Problem:** `seed_object()` only updates `_objects_by_color` Python dict. TRM never sees discovered objects.

**Fix:** Add `_upsert_live_object()` — when an object is discovered or updated, insert/update a Galaxy star:

```python
def seed_object(self, obj: dict[str, Any]) -> None:
    # ... existing Python dict update ...
    record["confidence"] = float(record["evidence_count"]) / float(record["evidence_count"] + 1)
    # NEW: Insert into live Galaxy
    self._upsert_live_object(record)

def _upsert_live_object(self, record: dict[str, Any]) -> str | None:
    galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
    if galaxy_manager is None or not hasattr(galaxy_manager, "upsert_entry"):
        return None
    color = int(record.get("color", -1))
    behavior = str(record.get("behavior", "unknown"))
    position = str(record.get("position_label", "unknown"))
    centroid = record.get("centroid")
    centroid_str = f"row={centroid[0]:.0f} col={centroid[1]:.0f}" if centroid else "unknown"
    entry = {
        "id": f"arc3_object:{self.game_id}:color_{color}",
        "name": f"ARC3 Object color={color} {behavior} at {position}",
        "domain": "arc3_game_object",
        "category": "game_object",
        "content": (
            f"arc3 game object color {color} behavior {behavior} "
            f"position {position} {centroid_str} "
            f"evidence {record.get('evidence_count', 0)} "
            f"blocking {record.get('blocking_count', 0)} "
            f"death {record.get('death_count', 0)}"
        ),
        "rpn_program": f"COLOR_{color} {behavior.upper()} {position.upper()}",
        "confidence": float(record.get("confidence", 0.5)),
        "tags": ["arc3", "game_object", behavior, f"color_{color}"],
        "meta_refs": [
            f"object_color:{color}",
            f"behavior:{behavior}",
            f"position:{position}",
        ],
        "metadata": {
            "source": "arc3_episode_object",
            "game_id": self.game_id,
            "color": color,
            "behavior": behavior,
            "evidence_count": int(record.get("evidence_count", 0)),
            "blocking_count": int(record.get("blocking_count", 0)),
        },
    }
    return str(galaxy_manager.upsert_entry("Grammar", entry))
```

**Why Grammar galaxy:** Objects are semantic entities with meaning (behavior, position) — they're knowledge about the game world, which is Grammar/Meta-Rules layer.

---

## Fix 2 (CRITICAL): Lower Crystallization Threshold to 1

**File:** `knowledge3d/knowledgeverse/arc3_episode_galaxy.py` → `_crystallize_rules()`

**Problem:** Line 546: `if len(rows) < 3: continue` — needs 3 observations before crystallizing a rule. In a new game, the TRM makes 139 actions with zero learning.

**Fix:** Create rules from the FIRST observation. Confidence scales with evidence:

```python
def _crystallize_rules(self, *, full_history: bool = False) -> None:
    recent = list(self.outcomes) if bool(full_history) else list(self.outcomes)[-20:]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for outcome in recent:
        adjacent_color = outcome.get("adjacent_color")
        if adjacent_color is None:
            continue
        key = (str(outcome.get("action", "")), f"agent_adjacent_to_color_{int(adjacent_color)}")
        grouped.setdefault(key, []).append(outcome)
    for (action, condition), rows in grouped.items():
        # CHANGED: was `if len(rows) < 3: continue`
        # Now crystallize from 1 observation, with confidence scaling
        outcome_counts = Counter()
        for row in rows:
            if bool(row.get("is_level_complete", False)):
                outcome_counts["level_complete"] += 1
            elif bool(row.get("is_death", False)):
                outcome_counts["death"] += 1
            elif bool(row.get("is_blocked", False)):
                outcome_counts["blocked"] += 1
            elif bool(row.get("agent_moved", False)):
                outcome_counts["moved"] += 1
            else:
                outcome_counts["neutral"] += 1
        predicted_outcome, majority = max(outcome_counts.items(), key=lambda item: (item[1], item[0]))
        total = len(rows)
        # Confidence scales: 1 obs → 0.3, 3 obs → 0.6, 10 obs → 0.8
        confidence = float(majority) / float(max(1, total)) * min(1.0, float(total) / 5.0)
        self._rules_by_key[(action, condition)] = {
            "type": "ARC3_RULE",
            "game_id": self.game_id,
            "condition": condition,
            "action": action,
            "predicted_outcome": predicted_outcome,
            "confidence": confidence,
            "evidence_count": int(total),
            "galaxy_family": "GRAMMAR",
        }
        self._upsert_live_rule(self._rules_by_key[(action, condition)], source="micro_sleeptime")
```

**Effect:** After just 1 step, the TRM has a Galaxy star for "ACTION2 adjacent to color_4 → moved" with confidence 0.3. After 3 steps: confidence 0.6. After 10 steps: confidence 0.8. The Galaxy GROWS with each movement.

---

## Fix 3 (CRITICAL): Negative Rules as Explicit Galaxy Stars

**Problem:** When ACTION2 hits a wall (blocked), `_crystallize_rules()` creates a rule with `predicted_outcome="blocked"`. But the sovereign hot path doesn't CHECK predicted_outcome — it just returns the winning star's action. So blocked rules don't prevent re-selection.

**Fix:** Negative rules should have DIFFERENT action recommendations. When the TRM finds "ACTION2 at color X → blocked", the rule should suggest trying a DIFFERENT action:

In `_crystallize_rules()`, after the existing code:

```python
# NEW: For blocked/death outcomes, create AVOIDANCE rules
if predicted_outcome in {"blocked", "death"}:
    # The original observation: "ACTION2 at color_X → blocked"
    # Create a diversification rule: "adjacent_to_color_X → try other action"
    # This pushes TRM toward exploring alternatives
    action_name = str(action).strip().upper()
    try:
        blocked_index = int(action_name.replace("ACTION", "")) - 1
    except Exception:
        continue
    # Seed exploration rules for each OTHER movement action
    for alt_index in range(4):  # ACTION1-ACTION4
        if alt_index == blocked_index:
            continue
        alt_name = f"ACTION{alt_index + 1}"
        alt_condition = f"agent_blocked_{action_name.lower()}_at_color_{condition.split('_')[-1]}"
        alt_rule = {
            "type": "ARC3_RULE",
            "game_id": self.game_id,
            "condition": alt_condition,
            "action": alt_name,
            "predicted_outcome": "explore_alternative",
            "confidence": confidence * 0.5,  # Lower than the blocking evidence
            "evidence_count": int(total),
            "galaxy_family": "GRAMMAR",
        }
        self._rules_by_key[(alt_name, alt_condition)] = alt_rule
        self._upsert_live_rule(alt_rule, source="micro_sleeptime_avoidance")
```

**Effect:** When ACTION2 hits a wall 3 times adjacent to color 5, the Galaxy gets:
- "ACTION2 at color_5 → blocked" (negative knowledge)
- "ACTION1 when blocked_action2 at color_5 → explore_alternative" (try UP)
- "ACTION3 when blocked_action2 at color_5 → explore_alternative" (try LEFT)
- "ACTION4 when blocked_action2 at color_5 → explore_alternative" (try DOWN)

Now TRM has alternative stars to find and select, breaking the ACTION2 loop.

---

## Fix 4 (HIGH): VRAM Growth Logging

**File:** `knowledge3d/knowledgeverse/arc3_episode_galaxy.py` → `run_micro_sleeptime()`

**Problem:** Daniel can't see VRAM learning. "I can see the VRAM is stalled."

**Fix:** After micro-sleeptime, log the Galaxy state:

```python
def run_micro_sleeptime(self) -> None:
    if self._pending_futures:
        return
    self._pending_futures = [
        self._executor.submit(self._crystallize_rules),
        self._executor.submit(self._reinforce_routes),
        self._executor.submit(self._classify_objects),
    ]
    # NEW: Log VRAM growth
    galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
    rules_count = len(self._rules_by_key)
    objects_count = len(self._objects_by_color)
    outcomes_count = len(self.outcomes)
    live_stars = 0
    if galaxy_manager is not None and hasattr(galaxy_manager, "entry_count"):
        try:
            live_stars = int(galaxy_manager.entry_count("Grammar"))
        except Exception:
            pass
    print(
        f"[ARC3-LEARN] step={outcomes_count} "
        f"rules={rules_count} objects={objects_count} "
        f"galaxy_stars={live_stars} "
        f"strategy={self._strategy_hint or 'exploring'}"
    )
```

**Expected output during healthy learning:**
```
[ARC3-LEARN] step=1 rules=2 objects=1 galaxy_stars=38150 strategy=exploring
[ARC3-LEARN] step=5 rules=5 objects=3 galaxy_stars=38156 strategy=exploring
[ARC3-LEARN] step=20 rules=12 objects=6 galaxy_stars=38170 strategy=exploring
[ARC3-LEARN] step=50 rules=18 objects=8 galaxy_stars=38180 strategy=stop_trying_action2
```

**The star count should INCREASE between steps.** If it doesn't, learning isn't working.

---

## Fix 5 (HIGH): House Persistence Across Restarts

**Problem:** `consolidate_to_house()` stores rules via `galaxy_manager.store_meaning_star()`, but fresh boot shows `arc3_rule_entries = 0`. The House-persisted rules aren't being loaded.

**File:** `knowledge3d/knowledgeverse/arc3_episode_galaxy.py` → `__init__()` and `knowledge3d/knowledgeverse/knowledgeverse.py`

**Fix:** On boot, load previously persisted ARC-3 rules from House into Galaxy:

```python
def __init__(self, game_id: str, knowledgeverse: Any) -> None:
    # ... existing init ...
    self._seed_game_mechanics_priors()
    # NEW: Load persisted rules from House
    self._load_persisted_rules()

def _load_persisted_rules(self) -> None:
    """Load previously learned rules from House (disk) into live Galaxy."""
    galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
    if galaxy_manager is None:
        return
    try:
        grammar_galaxy = galaxy_manager.get_galaxy("Grammar")
        if grammar_galaxy is None:
            return
        loaded = 0
        for entry in grammar_galaxy.entries():
            entry_id = str(entry.get("id", ""))
            if not entry_id.startswith("arc3_rule:"):
                continue
            # Reconstruct Python-side cache from persisted star
            meta = dict(entry.get("metadata") or {})
            condition = str(meta.get("condition", "")).strip()
            action_name = str(meta.get("action_name", "")).strip()
            if not condition or not action_name:
                continue
            self._rules_by_key[(action_name, condition)] = {
                "type": "ARC3_RULE",
                "game_id": self.game_id,
                "condition": condition,
                "action": action_name,
                "predicted_outcome": str(meta.get("predicted_outcome", "neutral")),
                "confidence": float(meta.get("rule_confidence", 0.5)),
                "evidence_count": int(meta.get("evidence_count", 3)),
                "galaxy_family": "GRAMMAR",
                "source": "house_persisted",
            }
            loaded += 1
        if loaded > 0:
            print(f"[ARC3-LEARN] Loaded {loaded} persisted rules from House")
    except Exception:
        pass
```

**Effect:** After a game session where the TRM learned "color 5 → ACTION5 → state_change", the next boot starts with that knowledge already in Galaxy. The TRM doesn't have to re-learn from scratch.

---

## Fix 6 (HIGH): Persist Negative Knowledge to House

**File:** `knowledge3d/knowledgeverse/arc3_episode_galaxy.py` → `_persist_strong_rules()`

**Problem:** Only persists rules with confidence ≥ 0.5 and evidence ≥ 3. Negative outcomes (blocked, death) are also valuable knowledge.

**Fix:** Lower threshold for negative rules and persist them explicitly:

```python
def _persist_strong_rules(self, *, min_confidence: float = 0.5, min_evidence: int = 3) -> list[str]:
    galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
    if galaxy_manager is None:
        return []
    rule_ids: list[str] = []
    sync_context = (
        galaxy_manager.bulk_disk_sync()
        if hasattr(galaxy_manager, "bulk_disk_sync")
        else nullcontext()
    )
    with sync_context:
        for rule in self._rules_by_key.values():
            confidence = float(rule.get("confidence", 0.0) or 0.0)
            evidence = int(rule.get("evidence_count", 0) or 0)
            predicted = str(rule.get("predicted_outcome", "")).strip().lower()
            # Positive rules: existing threshold
            # Negative rules (blocked, death): ALSO persist, even with lower confidence
            is_negative = predicted in {"blocked", "death"}
            if is_negative:
                if evidence < 2:  # Need at least 2 negative observations
                    continue
            else:
                if confidence < float(min_confidence) or evidence < int(min_evidence):
                    continue
            entry = self._rule_entry(rule, source="arc3_episode_galaxy")
            if entry is None:
                continue
            galaxy_manager.upsert_entry("Grammar", entry)
            rule_ids.append(str(entry["id"]))
    return rule_ids
```

**Effect:** Negative knowledge ("ACTION2 at wall → blocked") gets persisted to House. Next session, TRM starts knowing to avoid that direction.

---

## Fix 7 (MEDIUM): Seed Frame Observations as Galaxy Stars

**File:** `knowledge3d/knowledgeverse/arc3_episode_galaxy.py` → `seed_outcome()`

**Problem:** Frame observations (movement success, blocking, state changes) only go into Python deque. They never become Galaxy stars that TRM can find.

**Fix:** After recording outcome, create a lightweight observation star:

```python
def seed_outcome(self, ...) -> None:
    # ... existing code ...
    self.outcomes.append(outcome)
    self._update_object_records(outcome, normalized_prev, normalized_next)
    # NEW: Create observation star for significant events
    if bool(outcome.get("is_blocked", False)) or bool(outcome.get("is_death", False)) or bool(outcome.get("is_level_complete", False)):
        self._upsert_observation_star(outcome)

def _upsert_observation_star(self, outcome: dict[str, Any]) -> str | None:
    galaxy_manager = getattr(self.knowledgeverse, "galaxy_manager", None)
    if galaxy_manager is None or not hasattr(galaxy_manager, "upsert_entry"):
        return None
    action = str(outcome.get("action", "")).strip()
    adj_color = outcome.get("adjacent_color")
    event_type = "blocked" if outcome.get("is_blocked") else "death" if outcome.get("is_death") else "level_complete"
    obs_id = f"arc3_obs:{self.game_id}:{action}:color_{adj_color}:{event_type}"
    entry = {
        "id": obs_id,
        "name": f"ARC3 {event_type}: {action} at color {adj_color}",
        "domain": "arc3_observation",
        "category": "game_observation",
        "content": f"arc3 observation {event_type} when {action.lower()} adjacent color {adj_color}",
        "rpn_program": f"{action} COLOR_{adj_color} {event_type.upper()}",
        "confidence": 0.5,
        "tags": ["arc3", "observation", event_type, action.lower()],
        "metadata": {
            "source": "arc3_episode_observation",
            "game_id": self.game_id,
            "event_type": event_type,
            "action": action,
            "adjacent_color": adj_color,
        },
    }
    return str(galaxy_manager.upsert_entry("Grammar", entry))
```

---

## Learning Lifecycle (What Daniel Described)

```
BOOT:
  Load House-persisted rules into Galaxy (Fix 5)
  Seed generic game mechanic priors (existing)
  VRAM: base_stars + persisted_rules + priors

GAME LOOP (per action):
  1. Frame arrives → I/O perception query → kv.execute_task()
  2. TRM searches Galaxy → finds matching rule → emits action
  3. Action sent to ARC-3 server → next frame received
  4. learn_from_outcome():
     a. Compare frames (frame diff = signal processing)
     b. Detect objects → seed_object() → Galaxy star (Fix 1) — VRAM GROWS
     c. Record outcome → seed_outcome() → observation star (Fix 7) — VRAM GROWS
     d. If ACTION5 caused change → seed causal rule → Galaxy star — VRAM GROWS
  5. run_micro_sleeptime():
     a. Crystallize rules from observations (threshold=1, Fix 2) — VRAM GROWS
     b. Create avoidance rules from blocked/death (Fix 3) — VRAM GROWS
     c. Reinforce successful routes via shadow copy
     d. Log: [ARC3-LEARN] step=N rules=M objects=K galaxy_stars=S (Fix 4)

END OF ATTEMPT (deep consolidation):
  1. Crystallize ALL rules (full history)
  2. Classify ALL objects
  3. Persist strong rules + negative rules to House (Fix 6) — HOUSE GROWS
  4. Prune weak ephemeral rules from Galaxy — VRAM SHRINKS
  5. Log: [ARC3-LEARN] CONSOLIDATION persisted=N pruned=M house_rules=K

END OF GAME SESSION (consolidate_to_house):
  1. Persist session summary as Reality star
  2. Persist winning trace (if levels completed)
  3. Clear ephemeral memory
  4. VRAM returns to base level + persisted knowledge

NEXT BOOT:
  Load House knowledge → TRM starts smarter than last time
```

---

## Success Criteria

1. **VRAM star count increases during gameplay** — `[ARC3-LEARN]` logs show growing star count
2. **Action distribution diversifies** — not 100% ACTION2 after 20+ steps
3. **Negative rules prevent repeated failures** — blocked actions generate avoidance rules
4. **House persistence works** — restart, rules still present
5. **Multiple actions appear in scorecards** — ACTION1, ACTION2, ACTION3, ACTION4, ACTION5 all used

## Report Back

Write report at: `TEMP/CODEX_TO_CLAUDE_ARC3_VRAM_LEARNING_REPORT_2026-04-09.md`

Include:
1. `[ARC3-LEARN]` log excerpts showing VRAM growth over 50+ steps
2. Action distribution from 100-step bounded run (must show >1 action type)
3. Whether objects appear as Grammar Galaxy stars
4. Whether House-persisted rules load on fresh init
5. Restart long autonomous run (5x10000 steps)
6. Scorecard URL
7. echosys_ingest tmux alive

---

## Key Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/arc3_episode_galaxy.py` | Fixes 1-7: objects→stars, lower threshold, negative rules, logging, House load, observations |
| `benchmarks/arc_agi_3.py` | Minimal: ensure `learn_from_outcome()` passes enough context |

**Do NOT touch:**
- `knowledge3d/knowledgeverse/sovereign_hot_path.py` (action materialization already works)
- `knowledge3d/knowledgeverse/knowledgeverse.py` (materialization already works)
- tmux `echosys_ingest`

---

**Daniel's words:** "learning during game... micro sleeptime compute between movements, learns as it goes... saving new knowledge - positive and negative - to the house in the final sleeptime compute"
