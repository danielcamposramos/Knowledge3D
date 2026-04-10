# Codex Direction: ARC-3 LS20 Game Knowledge — Corrections + Autonomous Retry Loop

**Date:** 2026-04-09
**Authority:** KNOWLEDGEVERSE_SPECIFICATION.md (sovereignty), CLAUDE.md (TRM-as-Avatar)
**Prerequisite:** ARC-3 living-memory plan implemented (CODEX_ARC3_LIVING_MEMORY_SPEC ✓)
**Kaggle submission:** NOT yet — only after level 1 is consistently completed.

---

## Context

The living-memory wiring is in place. Now we need to give the TRM accurate game knowledge
about LS20 specifically, fix one important misunderstanding (yellow bar = dynamic, not
fixed), and add an autonomous retry loop so the avatar keeps playing until it wins
without any human intervention.

The first time the avatar won Level 1, it was with human-provided action hints. Those
hints are now the ground truth we want the TRM to DISCOVER on its own through trial,
error, and sleep-time crystallization. The game knowledge seeded here is the prior that
makes that discovery possible.

---

## Part 1 — LS20 Game Mechanics Knowledge (Corrections + Additions)

**File:** `benchmarks/arc3_game_mechanics_seeder.py`

Append the following stars to `GAME_MECHANICS_STARS`. Do NOT remove existing stars.
These are LS20-specific — they describe the actual visual grammar of the game.

### Yellow Bar Is DYNAMIC — Not a Hard Cap

The existing `movement_budget` star describes the yellow bar correctly as a limited
resource. But the current implementation must NOT cap available actions at a fixed
Python integer. The yellow bar value in the game frame is the source of truth.
Collecting action items during gameplay REFILLS the bar. Add this star:

```python
(
    "Reality",
    MeaningCentricStar(
        star_id="arc3_game_mechanic:action_refill",
        meaning_class="game_concept",
        domain="reality",
        galaxy_ref="Reality",
        meaning_rpn="ACTION_ITEM COLLECT YELLOW_BAR REFILL BUDGET_INCREASE",
        taxonomy_refs=["arc3", "game_mechanics", "action_refill"],
        reality_refs=["arc3", "action_refill", "yellow_bar"],
        meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
        confidence=1,
        polarity=1,
    ),
    "Action Refill",
),
```

### White Cross = Key Switch Artifact

The white cross shape visible in the game grid is an interactive artifact. Stepping onto
it (ACTION5 = Perform) toggles the key's orientation. Some configurations require
multiple interactions. This is the CRITICAL object the avatar must find and interact with
before going to the door.

```python
(
    "Reality",
    MeaningCentricStar(
        star_id="arc3_game_mechanic:key_switch",
        meaning_class="game_object",
        domain="reality",
        galaxy_ref="Reality",
        meaning_rpn="WHITE_CROSS GRID_ARTIFACT KEY_SWITCH INTERACT ACTION5 TOGGLE_KEY_ORIENTATION",
        taxonomy_refs=["arc3", "ls20", "key_switch", "white_cross"],
        reality_refs=["arc3", "key_switch", "artifact", "interact"],
        meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
        confidence=1,
        polarity=1,
    ),
    "Key Switch (White Cross)",
),
(
    "Grammar",
    MeaningCentricStar(
        star_id="arc3_game_rule:key_switch_interaction",
        meaning_class="game_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="AGENT_AT_WHITE_CROSS ACTION5",
        behavior_rpn="KEY_ORIENTATION_TOGGLES REPEAT_UNTIL_MATCHES_DOOR_TARGET",
        taxonomy_refs=["arc3", "ls20", "key_switch", "game_rule"],
        grammar_refs=["arc3", "key_switch", "orient", "interact"],
        meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
        confidence=1,
        polarity=1,
    ),
    "Rule: Key Switch Interaction",
),
```

### Upper Blue Figure = Door + Key Orientation Target

The blue figure displayed at the UPPER region of the game frame is TWO things at once:
1. The LOCATION of the door (where the avatar must go to complete the level)
2. The REQUIRED KEY ORIENTATION (the key must match this before entering)

The avatar's job: match the key state to the blue figure, THEN navigate to it.

```python
(
    "Reality",
    MeaningCentricStar(
        star_id="arc3_game_mechanic:door_indicator",
        meaning_class="game_object",
        domain="reality",
        galaxy_ref="Reality",
        meaning_rpn="UPPER_BLUE_FIGURE DOOR_LOCATION NEXT_LEVEL_PORTAL KEY_ORIENTATION_TARGET",
        taxonomy_refs=["arc3", "ls20", "door", "blue_figure", "key_target"],
        reality_refs=["arc3", "door", "key_target", "level_exit"],
        meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
        confidence=1,
        polarity=1,
    ),
    "Door Indicator (Upper Blue Figure)",
),
(
    "Grammar",
    MeaningCentricStar(
        star_id="arc3_game_rule:door_entry_condition",
        meaning_class="game_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="KEY_ORIENTATION_MATCHES_DOOR_TARGET AND AGENT_AT_DOOR_POSITION",
        behavior_rpn="LEVEL_COMPLETE_ADVANCE_TO_NEXT",
        taxonomy_refs=["arc3", "ls20", "door", "win_condition"],
        grammar_refs=["arc3", "door", "key_match", "level_complete"],
        meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
        confidence=1,
        polarity=1,
    ),
    "Rule: Door Entry Condition",
),
(
    "Grammar",
    MeaningCentricStar(
        star_id="arc3_game_rule:door_entry_blocked",
        meaning_class="game_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="KEY_ORIENTATION_NOT_MATCHING AND AGENT_AT_DOOR_POSITION",
        behavior_rpn="BLOCKED_RETURN_TO_KEY_SWITCH_INTERACT_AGAIN",
        taxonomy_refs=["arc3", "ls20", "door", "blocked", "key_mismatch"],
        grammar_refs=["arc3", "door", "blocked", "key_orient_required"],
        meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
        confidence=1,
        polarity=1,
    ),
    "Rule: Door Blocked When Key Mismatched",
),
```

### Bottom-Left Display = Current Key State

The left-side visual strip at the bottom of the frame (SEPARATE from the main game grid,
next to the yellow bar) shows the CURRENT state of the key: its orientation, color, and
shape. This is the feedback mechanism — the avatar reads this to know whether the key is
correct yet.

```python
(
    "Reality",
    MeaningCentricStar(
        star_id="arc3_game_mechanic:key_state_display",
        meaning_class="game_ui",
        domain="reality",
        galaxy_ref="Reality",
        meaning_rpn="LEFT_STRIP_DISPLAY CURRENT_KEY_STATE ORIENTATION COLOR SHAPE FEEDBACK",
        taxonomy_refs=["arc3", "ls20", "ui", "key_display", "key_state"],
        reality_refs=["arc3", "key_state", "display", "ui_feedback"],
        meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
        confidence=1,
        polarity=1,
    ),
    "Key State Display (Bottom-Left Strip)",
),
```

### Multi-Step Artifact Interaction (Advanced Levels)

Some artifacts require MORE than one ACTION5 interaction to reach the correct
orientation, color, or shape. The avatar must persist — interact, check the key
state display, interact again if needed.

```python
(
    "Grammar",
    MeaningCentricStar(
        star_id="arc3_game_rule:multi_step_artifact",
        meaning_class="game_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="ARTIFACT_INTERACTION ACTION5 REPEAT CHECK_KEY_STATE",
        behavior_rpn="CONTINUE_INTERACTING_UNTIL_KEY_STATE_MATCHES_TARGET",
        taxonomy_refs=["arc3", "ls20", "artifact", "multi_step", "interact"],
        grammar_refs=["arc3", "artifact", "multi_step", "iterate"],
        meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
        confidence=1,
        polarity=1,
    ),
    "Rule: Multi-Step Artifact Interaction",
),
```

### Correct Level 1 Strategy (Derived from Control Metrics)

These are the discovered actions that win Level 1. They ARE the control metric —
what we want the TRM to learn to produce autonomously. Seed as a high-confidence
winning trace to give the TRM a strong prior:

```python
(
    "Reality",
    MeaningCentricStar(
        star_id="arc3_ls20_strategy:level1_sequence",
        meaning_class="game_strategy",
        domain="reality",
        galaxy_ref="Reality",
        meaning_rpn=(
            "NAVIGATE_TO_KEY_SWITCH "
            "INTERACT_KEY_SWITCH ACTION5 "
            "CHECK_KEY_STATE_MATCHES_DOOR "
            "IF_NOT_MATCH INTERACT_AGAIN "
            "NAVIGATE_TO_DOOR_UPPER_BLUE_FIGURE "
            "ENTER_DOOR LEVEL_COMPLETE"
        ),
        taxonomy_refs=["arc3", "ls20", "level1", "winning_strategy"],
        reality_refs=["arc3", "ls20", "strategy", "level1"],
        meta_refs=["bootstrap:arc3_game_mechanics_ls20_v1"],
        confidence=1,
        polarity=1,
    ),
    "LS20 Level 1 Winning Strategy",
),
```

---

## Part 2 — Remove Hard-Coded Action Cap

**Problem:** Anywhere in the codebase that reads a Python integer as the "action budget"
and caps moves to that number is WRONG. The yellow bar is a dynamic game value — it
can be refilled by picking up action items. Python has no business deciding how many
moves are left. Only the game server knows that.

**Fix:** In `benchmarks/arc3_sdk_agent.py` and `benchmarks/arc_agi_3.py`, search for
any logic that caps actions based on a hard-coded `max_actions` or `action_budget`
integer used in the HOT PATH decision (not the outer Python safety limit).

The outer `max_steps` parameter in `K3DAgent` is a Python session safety limit — keep it.
It prevents an infinite loop if the game server never sends `done=True`. That's fine.

But if there is logic like:
```python
if self._action_count >= self._max_actions:
    return {"action": "ACTION7"}  # force reset when budget exhausted
```
...this must be REMOVED from the decision path. The game server handles budget exhaustion.
The avatar should just keep choosing the best action via `execute_task()`.

If `available_actions` from the observation is empty or restricted, pass it through
to the WINE envelope and let the GPU decide. Do not override it with Python.

---

## Part 3 — Autonomous Retry Loop

**Goal:** The avatar plays LS20 on its own, retrying after each failed attempt with
sleep-time crystallization between attempts, until Level 1 is complete. No human
intervention. No Python reasoning about WHY it failed.

**Add method to `K3DAgent` in `benchmarks/arc3_sdk_agent.py`:**

```python
def run_until_level_complete(
    self,
    target_levels: int = 1,
    max_attempts: int = 20,
    steps_per_attempt: int = 500,
) -> dict[str, Any]:
    """
    Run the game autonomously, retrying after each failed attempt.
    Between attempts: run deep sleep-time consolidation on the episode Galaxy.
    Episode Galaxy PERSISTS across attempts — each retry benefits from all prior learning.
    Stops when target_levels reached or max_attempts exhausted.
    Python = boot + loop control only. TRM does all reasoning.
    """
    attempt = 0
    best_levels = 0
    best_result: dict[str, Any] = {}

    while attempt < max_attempts:
        attempt += 1
        result = self.run_level(max_steps=steps_per_attempt)
        levels_completed = int(result.get("levels_completed", 0) or 0)

        if levels_completed > best_levels:
            best_levels = levels_completed
            best_result = dict(result)

        if levels_completed >= target_levels:
            best_result["attempts_used"] = attempt
            best_result["autonomous"] = True
            return best_result

        # Deep sleep-time: crystallize all rules from this attempt before retry
        # Episode Galaxy persists — next attempt benefits from everything learned so far
        episode_galaxy = getattr(self, "_episode_galaxy", None)
        if episode_galaxy is not None and hasattr(episode_galaxy, "run_deep_consolidation"):
            try:
                episode_galaxy.run_deep_consolidation()
            except Exception:
                pass  # Never block retry for sleep-time failures

    best_result["attempts_used"] = attempt
    best_result["autonomous"] = True
    return best_result
```

**Wire it as the default CLI path** in `main()`:

```python
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    agent = K3DAgent(
        args.game_id,
        max_steps=args.max_steps,
        allow_remote_compat=not bool(args.no_remote_compat),
    )
    try:
        if getattr(args, "autonomous", False):
            result = agent.run_until_level_complete(
                target_levels=getattr(args, "target_levels", 1),
                max_attempts=getattr(args, "max_attempts", 20),
                steps_per_attempt=args.max_steps,
            )
        else:
            result = agent.run_level(max_steps=args.max_steps)
    finally:
        agent.close()
    print(json.dumps(result, indent=2))
    return 0
```

**Add `--autonomous` flag to the argument parser:**
```python
parser.add_argument("--autonomous", action="store_true",
    help="Retry until target level is complete. Episode Galaxy persists across attempts.")
parser.add_argument("--max-attempts", type=int, default=20)
parser.add_argument("--target-levels", type=int, default=1)
```

---

## Part 4 — Episode Galaxy: run_deep_consolidation()

The existing `run_micro_sleeptime()` scans only the last 10 frames. Between game
attempts, we want to scan ALL frames accumulated so far.

**Add to `ARC3EpisodeGalaxy` in `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`:**

```python
def run_deep_consolidation(self) -> None:
    """
    Between-attempt full crystallization. Scans ALL frames accumulated across
    all attempts in this session. More thorough than run_micro_sleeptime().
    Writes high-confidence rules to the persistent Galaxy (via galaxy_manager).
    """
    # Same as run_micro_sleeptime but with full history, not last_n
    # And also: persist strong rules (confidence >= 0.6) to Galaxy for next attempt
    self._crystallize_rules(full_history=True)
    self._classify_objects()
    # Persist rules that are confident enough to survive between attempts
    self._persist_strong_rules(min_confidence=0.6, min_evidence=3)
```

The key difference: `run_deep_consolidation()` persists strong rules via
`galaxy_manager.store_meaning_star()` so they survive to the NEXT attempt.
`run_micro_sleeptime()` keeps rules in VRAM only (per-step, ephemeral).

The episode Galaxy's FRAME and OUTCOME stars remain across attempts (accumulate).
Only strong RULE stars get written to permanent Galaxy between attempts.

---

## What the Autonomous Loop Teaches Us (Control Metrics)

When the autonomous loop succeeds at Level 1, check these in the report:

1. **Which attempt number** did it first complete Level 1?
2. **What rule stars** were crystallized by that point? (log from `_persist_strong_rules`)
3. **What was the winning action sequence** recorded in frame_history?
4. **Did it discover** key_switch interaction, door navigation, key-state matching?

If the TRM wins Level 1 on attempt ≤ 5, the knowledge seeding and living memory are
working. If it takes 10–15 attempts, the swarm is learning from scratch (still valid,
just slower). If it never wins in 20 attempts, the WINE context needs richer episode
data (revisit what goes into `inferred_rules` in the envelope).

---

## What NOT to Do

- Do NOT hard-code "budget = 30" or any fixed number of actions per level
- Do NOT use Python to detect "the yellow bar is at 20%" — let the GPU read the frame
- Do NOT submit to Kaggle yet — we need a consistent Level 1 completion first
- Do NOT reset the episode Galaxy between retry attempts — it's the learning memory
- Do NOT add Python logic for "navigate to white cross" — TRM discovers it via trial-error

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ARC3_LS20_GAME_KNOWLEDGE_REPORT_2026-04-09.md` with:

1. `arc3_game_mechanics_seeder.py`: new LS20 stars added (count, list of star_ids)
2. Hard-coded action cap: found and removed (yes/no/not-present — which file, which line)
3. `run_until_level_complete()` added to K3DAgent (yes/no, file + line)
4. `--autonomous` CLI flag added (yes/no)
5. `run_deep_consolidation()` added to ARC3EpisodeGalaxy (yes/no)
6. Tests: `tests/test_arc3_autonomous_retry.py` — at minimum: retry loop runs 3 attempts,
   episode Galaxy accumulates frames across attempts, `run_deep_consolidation()` runs
   without error (pass/fail)
7. **AUTONOMOUS RUN:** `bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc3_sdk_agent.py --game ls20 --autonomous --max-attempts 5 --max-steps 200`
   Report: attempt_number, levels_completed per attempt, rules_crystallized count at end
   This is the most important verification. Run it and report honestly.
8. `echosys_ingest` tmux session still alive (tmux ls)
