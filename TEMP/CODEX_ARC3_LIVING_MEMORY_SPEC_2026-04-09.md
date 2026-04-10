# Codex Direction: ARC-3 Living Memory — Inter-Action Micro Sleep-Time + Multi-GPU Parallel Learning

**Date:** 2026-04-09
**Authority:** KNOWLEDGEVERSE_SPECIFICATION.md §3 (Reality Galaxy), CLAUDE.md (TRM-as-Avatar, sovereignty)
**Prerequisite:** ARC sovereignty repair complete (CODEX_ARC_KAGGLE_SUBMISSION_SPEC_2026-04-09.md ✓)

---

## The Core Insight

ARC-AGI-3 is not a benchmark run. **The avatar LIVES in the game.** Each frame is a moment
of existence. The HTTP call latency between `env.step()` and the next frame (100–300ms) is
NOT wasted time — it is free compute. On Kaggle's 4×L4 setup (96GB VRAM total), GPUs 1–3
are completely idle during game inference. This spec converts that idle capacity into
**concurrent in-game learning**: the avatar gets smarter between every single action.

The memory architecture follows the House paradigm:
- **Episode Galaxy** (VRAM only during game) = hippocampus / short-term episodic memory
- **Working Galaxy** updates = cortex / pattern abstraction in real-time
- **House consolidation** (disk write at game end) = long-term memory / sleep consolidation
- **All reasoning on GPU** — Python only reads the HTTP frame and posts the action back

---

## Part 1 — ARC3EpisodeGalaxy: VRAM-Resident Working Memory

Create `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`.

The episode Galaxy is initialized once at game start and lives entirely in VRAM.
No disk writes during the game. Four star types:

### Star Type: ARC3_FRAME

One star per step. Seeds the spatial context of each observation.

```python
{
    "type": "ARC3_FRAME",
    "game_id": str,           # "ls20"
    "step_count": int,        # 0, 1, 2, ...
    "grid_hash": str,         # sha1(str(grid))[:8] — content-addressable
    "grid_height": int,
    "grid_width": int,
    "agent_row": float,       # foreground centroid row (use _focus_centroid from arc_agi_3.py)
    "agent_col": float,       # foreground centroid col
    "foreground_colors": list[int],  # unique non-background colors seen
    "action_taken": str,      # "ACTION1" ... "ACTION7", or "" if first frame
    "budget_pct": float,      # movement_budget_remaining / budget_max (0.0–1.0), -1 if unknown
    "lives_remaining": int,   # -1 if unknown
    "level": int,             # levels_completed at this step
}
```

### Star Type: ARC3_OUTCOME

One star per step (created AFTER env.step() returns).

```python
{
    "type": "ARC3_OUTCOME",
    "game_id": str,
    "step_count": int,          # the step that caused this outcome
    "action": str,              # "ACTION1" ... "ACTION7"
    "cells_changed": int,       # len(changed_cells)
    "agent_moved": bool,        # True if centroid changed by > 0.5
    "reward": float,            # from env.step() reward signal
    "is_blocked": bool,         # cells_changed == 0 AND action is directional (ACTION1-4)
    "is_death": bool,           # lives_remaining decreased
    "is_level_complete": bool,  # levels_completed increased
    "prev_frame_hash": str,
    "next_frame_hash": str,
}
```

### Star Type: ARC3_OBJECT

One star per discovered object color. Updated as more evidence accumulates.

```python
{
    "type": "ARC3_OBJECT",
    "game_id": str,
    "color": int,               # ARC color 0-15
    "behavior": str,            # "static_obstacle", "walkable", "avatar", "goal",
                                # "hazard", "collectible", "door", "key", "unknown"
    "evidence_count": int,      # how many times this was observed
    "blocking_count": int,      # how many times agent was blocked by this color
    "death_count": int,         # how many times contact caused death
    "reward_count": int,        # how many times contact caused reward
    "confidence": float,        # evidence_count / (evidence_count + 1)
}
```

### Star Type: ARC3_RULE

Synthesized during micro sleep-time. Written to working Grammar Galaxy (VRAM only).

```python
{
    "type": "ARC3_RULE",
    "game_id": str,
    "condition": str,           # "agent_adjacent_to_color_5", "budget_below_10pct", etc.
    "action": str,              # "ACTION1" ... "ACTION7", or "" for any
    "predicted_outcome": str,   # "blocked", "moved", "death", "level_complete", "neutral"
    "confidence": float,        # correct_count / (correct_count + wrong_count)
    "evidence_count": int,
    "galaxy_family": "GRAMMAR", # routes to Grammar Galaxy on consolidation
}
```

### ARC3EpisodeGalaxy class interface

```python
class ARC3EpisodeGalaxy:
    def __init__(self, game_id: str, knowledgeverse) -> None:
        """Initialize empty episode Galaxy for this game session."""

    def seed_frame(self, step_count: int, grid: list[list[int]],
                   action_taken: str = "", lives: int = -1,
                   budget_pct: float = -1.0, levels_completed: int = 0) -> None:
        """Add ARC3_FRAME star for current step. Called before execute_task()."""

    def seed_outcome(self, step_count: int, action: str,
                     prev_grid: list[list[int]], next_grid: list[list[int]],
                     reward: float, lives_delta: int, levels_delta: int) -> None:
        """Add ARC3_OUTCOME star after env.step(). Updates ARC3_OBJECT stars."""

    def get_episode_context(self, last_n: int = 10) -> dict:
        """
        Return context dict for WINE envelope:
        {
          "rules": list of ARC3_RULE dicts (confidence > 0.5),
          "objects": {color: behavior} from ARC3_OBJECT stars,
          "recent_outcomes": last N ARC3_OUTCOME dicts,
          "strategy_hint": str from temporal reasoning
        }
        """

    def run_micro_sleeptime(self) -> None:
        """
        Run inter-action pattern extraction. Called after seed_outcome().
        Dispatches gre_graph_crystallizer → ARC3_RULE synthesis.
        Dispatches gre_temporal_reasoning → strategy_hint update.
        All GPU, all VRAM, no disk I/O.
        """

    def consolidate_to_house(self, knowledgeverse, final_score: float) -> None:
        """
        Called once at game end. Writes winning trace + rules to Galaxy (disk).
        Prunes low-confidence rules. Creates GAME_ROOM star in Reality Galaxy.
        """
```

---

## Part 2 — Micro Sleep-Time: Pattern Extraction Between Actions

The timing hook is in `K3DAgent.run_level()`, between `env.step()` and the next
`execute_task()`. The HTTP call to ARC server takes 100–300ms. That time is OURS.

### Where to inject it

In `benchmarks/arc3_sdk_agent.py`, `K3DAgent.run_level()`, after:
```python
self.update_world_model(frame, action, next_frame)
```

Add:
```python
# Seed outcome into episode Galaxy
self.episode_galaxy.seed_outcome(
    step_count=self.step_count,
    action=_action_name(action),
    prev_grid=frame,
    next_grid=next_frame,
    reward=float(reward),
    lives_delta=prev_lives - current_lives,  # track lives across steps
    levels_delta=int(levels_completed - prev_levels_completed),
)

# Micro sleep-time: pattern extraction on GPU while next frame loads
# (This is cheap: ~5-15ms. The HTTP call was already 100-300ms.)
self.episode_galaxy.run_micro_sleeptime()

# Seed next frame with episode context for the upcoming execute_task()
episode_ctx = self.episode_galaxy.get_episode_context(last_n=10)
```

Then pass `episode_ctx` into the next WINE envelope (see Part 3).

### What run_micro_sleeptime() dispatches

**Rule crystallization** — use `gre_graph_crystallizer` (swarm slot 5):
- Scan last 20 ARC3_OUTCOME stars
- Group by (action, color_adjacent_to_agent) → outcome
- If 3+ consistent observations: create or update ARC3_RULE star
- Update confidence = consistent / (consistent + inconsistent)

**Object behavior update** — inline logic (not a separate kernel, just update ARC3_OBJECT stars):
- For each outcome where cells_changed > 0: infer which colors moved → "walkable" or "avatar"
- For each blocked outcome: the color adjacent in the action direction → "static_obstacle"
- For each death outcome: the color at the agent's position → "hazard"
- For each level_complete: the color at the goal-adjacent position → "goal"

**Temporal strategy hint** — use `gre_temporal_reasoning` (swarm slot 6):
- If last 5 outcomes all is_blocked == True for same action → strategy_hint = "stop_trying_action_X"
- If agent_moved == False for 3+ steps → strategy_hint = "stuck_try_reset"
- If budget_pct < 0.15 → strategy_hint = "budget_critical_minimize_moves"
- If level just completed → strategy_hint = "advance_level_pattern_confirmed"

The strategy_hint is a short string seeded into the WINE envelope. The nine-chain swarm
reads it as Grammar Galaxy context during the next execute_task().

---

## Part 3 — WINE Envelope: Extended with Episode Context

In `knowledge3d/tablet/wine/game2d_wine.py`, extend `arc3_game_envelope()` to accept
episode context:

```python
def arc3_game_envelope(
    *,
    frame: list[list[int]],
    goal: list[list[int]],
    step_count: int,
    game_id: str,
    levels_completed: int,
    world_model: dict | None = None,
    episode_context: dict | None = None,   # NEW — from ARC3EpisodeGalaxy
) -> dict:
    ctx = episode_context or {}
    return {
        "type": "ARC3_GAME_FRAME",
        "surface_kind": "SPATIAL",
        "game_id": game_id,
        "grid": frame,
        "goal": goal,
        "grid_height": len(frame),
        "grid_width": len(frame[0]) if frame else 0,
        "step_count": step_count,
        "levels_completed": levels_completed,
        "world_model": world_model or {},
        # Episode context — what the avatar has learned this session
        "inferred_rules": ctx.get("rules", []),        # ARC3_RULE stars (confidence > 0.5)
        "known_objects": ctx.get("objects", {}),       # color → behavior string
        "recent_outcomes": ctx.get("recent_outcomes", []),  # last 10 outcomes
        "strategy_hint": ctx.get("strategy_hint", ""),      # from gre_temporal_reasoning
    }
```

Update `K3DAgent.decide_action()` and `K3DARC3Agent.choose_action()` to pass
`episode_context` through to this envelope.

The nine-chain swarm receives all of this as part of the TRM input. The `inferred_rules`
and `known_objects` directly inform gre_arc_reasoner (slot 3) and gre_geometry_router
(slot 4) without ANY Python reasoning — they are just data in the envelope that the
GPU-side kernels process.

---

## Part 4 — Multi-GPU Parallel Architecture (Kaggle 4×L4)

This is the surprise for Kaggle. 4×L4 = 96GB VRAM. The game uses ~1GB on GPU0.
GPUs 1–3 are IDLE during inference. Convert them to parallel sleep-time compute.

### The Idea: Async micro sleep-time during HTTP call

When `env.step(action)` is called, the HTTP request to `three.arcprize.org` takes
100–300ms. During that time:

```
t=0ms:  env.step(action) starts HTTP POST (network wait begins)
t=0ms:  GPU1: gre_graph_crystallizer runs on episode buffer
t=0ms:  GPU2: shadow copy reinforcement (trm_step_fused shadow mode)
t=0ms:  GPU3: ARC3_OBJECT behavior update kernel
t=150ms: HTTP response arrives (next frame)
t=150ms: GPU1/2/3 results written to VRAM (shared, peer-to-peer)
t=151ms: GPU0 executes next execute_task() with enriched Galaxy
```

The avatar does NOT wait for micro sleep-time to finish before sending the action —
sleep-time starts in parallel with the HTTP call. This is the hippocampus model:
learning happens CONCURRENTLY with action, not sequentially after it.

### Implementation pattern

In `ARC3EpisodeGalaxy.run_micro_sleeptime()`, use Python's `concurrent.futures.ThreadPoolExecutor`
to fire CUDA stream async calls on GPU1-3 while GPU0 is tied up in the HTTP wait:

```python
def run_micro_sleeptime(self) -> None:
    """Fire micro sleep-time kernels on idle GPUs in parallel with HTTP wait."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(self._crystallize_rules),    # GPU1
            executor.submit(self._reinforce_routes),     # GPU2
            executor.submit(self._classify_objects),     # GPU3
        ]
        # Don't wait here — let them run while HTTP call is in flight.
        # Results written to shared VRAM are visible to GPU0 automatically.
        # We join at the start of seed_frame() for the NEXT step.
        self._pending_futures = futures

def _drain_pending_futures(self) -> None:
    """Called at top of seed_frame() to ensure last micro sleep completed."""
    if self._pending_futures:
        for future in self._pending_futures:
            try:
                future.result(timeout=0.5)
            except Exception:
                pass  # Never block inference for sleep-time failures
        self._pending_futures = []
```

On a single-GPU machine (local dev), the ThreadPoolExecutor still works — the three
functions run sequentially on the same GPU. The architecture is identical; only the
wall-clock overlap changes.

### Multi-GPU VRAM allocation (Kaggle 4×L4 only)

When `torch.cuda.device_count() >= 4`:

```python
INFERENCE_GPU = 0       # execute_task() always runs here
CRYSTAL_GPU = 1         # gre_graph_crystallizer rule extraction
SHADOW_GPU = 2          # trm shadow copy reinforcement
OBJECT_GPU = 3          # ARC3_OBJECT behavior classification
```

When < 4 GPUs: all on GPU0, sequential (same correctness, less speed).
Auto-detect at ARC3EpisodeGalaxy init time; no code path changes in the game loop.

### Between-level consolidation window

When `levels_delta > 0` (level just completed), we have a natural pause before the
next level starts. Use this window for a DEEPER sleep cycle:

```python
if episode_ctx.get("strategy_hint") == "advance_level_pattern_confirmed":
    # Longer consolidation: scan ALL episode frames, not just last 10
    self.episode_galaxy.run_deep_consolidation(gpu_ids=[1, 2, 3])
    # Prune weak rules, merge strong rules into main Galaxy
    # This takes ~500ms but level transitions allow it
```

---

## Part 5 — General 2D Game Mechanics Knowledge Pre-Seeding

Before the first `execute_task()` call, seed the default Galaxy with 2D game mechanics
knowledge. This runs once at `K3DARC3Agent.__init__()` or at game boot.

Create `benchmarks/arc3_game_mechanics_seeder.py`:

```python
"""
Seed the Reality Galaxy and Grammar Galaxy with general 2D game mechanics knowledge.
This gives the TRM prior knowledge about how 2D grid games work before the first frame.
"""

GAME_MECHANICS_STARS = [
    # Reality Galaxy — game concept knowledge
    {
        "type": "CONCEPT",
        "galaxy": "REALITY",
        "name": "movement_budget",
        "description": "A limited resource (yellow bar) that depletes with each move. When exhausted, the level fails or resets. Conserve moves — every step costs budget.",
        "keywords": ["budget", "moves", "yellow_bar", "limited_steps"],
    },
    {
        "type": "CONCEPT",
        "galaxy": "REALITY",
        "name": "lives_system",
        "description": "A finite stock of attempts (red squares). Losing all lives ends the game. Dying wastes a life — avoid hazards. Strategic reset can save lives vs. being trapped.",
        "keywords": ["lives", "red_squares", "attempts", "death", "game_over"],
    },
    {
        "type": "CONCEPT",
        "galaxy": "REALITY",
        "name": "level_goal",
        "description": "Each level has a completion condition, usually reaching a specific cell or collecting an object. Completing a level advances to the next and grants reward.",
        "keywords": ["goal", "target", "level_complete", "win_condition"],
    },
    {
        "type": "CONCEPT",
        "galaxy": "REALITY",
        "name": "strategic_reset",
        "description": "Resetting the current level restores position but consumes a life. Reset is optimal when trapped with insufficient budget to reach the goal. Prefer reset over budget exhaustion.",
        "keywords": ["reset", "restart", "ACTION7", "undo", "life_trade"],
    },
    {
        "type": "CONCEPT",
        "galaxy": "REALITY",
        "name": "exploration_vs_exploitation",
        "description": "Early in a level: explore to build a map of object behaviors. Later: exploit known safe paths. Exploration costs budget; exploitation conserves it.",
        "keywords": ["explore", "exploit", "trial_error", "mapping"],
    },
    {
        "type": "CONCEPT",
        "galaxy": "REALITY",
        "name": "arc_color_semantics",
        "description": "ARC uses colors 0-15. Color 0 (black) is typically background. Rare colors often mark special objects (avatar, goal, key). Most common non-black color is often terrain.",
        "keywords": ["color", "arc", "background", "foreground", "palette"],
    },
    # Grammar Galaxy — 2D game transformation rules (RPN-compatible)
    {
        "type": "RULE",
        "galaxy": "GRAMMAR",
        "name": "move_into_obstacle",
        "condition": "agent_action_is_directional AND adjacent_cell_is_solid",
        "outcome": "no_position_change",
        "confidence": 0.95,
        "keywords": ["blocked", "obstacle", "wall", "solid"],
    },
    {
        "type": "RULE",
        "galaxy": "GRAMMAR",
        "name": "move_into_empty",
        "condition": "agent_action_is_directional AND adjacent_cell_is_walkable",
        "outcome": "position_changes_by_action_vector",
        "confidence": 0.95,
        "keywords": ["move", "walk", "empty", "walkable"],
    },
    {
        "type": "RULE",
        "galaxy": "GRAMMAR",
        "name": "reach_goal",
        "condition": "agent_position_equals_goal_position",
        "outcome": "level_complete_reward_plus_one",
        "confidence": 0.99,
        "keywords": ["goal", "win", "level_complete", "reward"],
    },
    {
        "type": "RULE",
        "galaxy": "GRAMMAR",
        "name": "contact_hazard",
        "condition": "agent_moves_into_hazard_cell",
        "outcome": "death_lives_minus_one",
        "confidence": 0.9,
        "keywords": ["hazard", "death", "kill", "danger"],
    },
    {
        "type": "RULE",
        "galaxy": "GRAMMAR",
        "name": "loop_detection",
        "condition": "same_grid_state_visited_three_times",
        "outcome": "current_path_is_dead_end_try_different_action",
        "confidence": 0.85,
        "keywords": ["loop", "stuck", "repeat", "dead_end"],
    },
]

def seed_game_mechanics(knowledgeverse) -> int:
    """Seed game mechanics stars into Knowledgeverse. Returns count seeded."""
    stars = [dict(s) for s in GAME_MECHANICS_STARS]
    knowledgeverse.seed_stars(stars)
    return len(stars)
```

Call `seed_game_mechanics(kv)` once in `K3DARC3Agent.__init__()` before any frame is
processed. This is one-time cost (~1ms). The TRM will have prior game knowledge
before seeing the first ARC-3 frame.

---

## Part 6 — End-of-Episode House Consolidation

Called in `K3DAgent.run_level()` just before `agent.close()`:

```python
if self.episode_galaxy is not None:
    self.episode_galaxy.consolidate_to_house(
        knowledgeverse=self._knowledgeverse,
        final_score=reward_total,
    )
```

What `consolidate_to_house()` does:

1. **Extract strong rules**: ARC3_RULE stars with confidence ≥ 0.5 AND evidence_count ≥ 3
2. **Persist to Grammar Galaxy** (disk): these rules are now permanent knowledge
3. **Extract winning trace** (if levels_completed > 0):
   - Walk frame_history backward from last reward > 0
   - Find minimal action sequence that completed the level
   - Create `ARC3_WINNING_TRACE` star in Reality Galaxy
4. **Create GAME_ROOM star** in Reality Galaxy:
   ```python
   {
       "type": "ARC3_GAME_ROOM",
       "galaxy": "REALITY",
       "game_id": game_id,
       "sessions_played": ...,  # increment across runs
       "best_score": max(final_score, previous_best),
       "rules_discovered": count_of_persisted_rules,
       "winning_traces_count": ...,
   }
   ```
5. **Prune episode Galaxy**: clear VRAM of ARC3_FRAME and ARC3_OUTCOME stars
   (they are ephemeral — only the rules and traces survive)
6. **Merge GPU shadow copies** into main Knowledgeverse and call `kv.persist()`

On Kaggle, with no internet, this consolidation happens entirely in VRAM until step 6,
which writes to `/kaggle/working/k3d_runtime/` (local disk, not network).

---

## Part 7 — Updated K3DARC3Agent Wiring

In `benchmarks/arc_agi_3.py`, update `K3DARC3Agent`:

```python
class K3DARC3Agent:
    def __init__(self, max_actions: int, knowledgeverse) -> None:
        self._kv = knowledgeverse
        self._tablet = HeadlessTabletMPC(knowledgeverse=knowledgeverse)
        self._episode_galaxy = ARC3EpisodeGalaxy(game_id="unknown", knowledgeverse=knowledgeverse)
        # Seed 2D game mechanics knowledge once
        from benchmarks.arc3_game_mechanics_seeder import seed_game_mechanics
        seed_game_mechanics(knowledgeverse)

    def choose_action(self, frame, *, goal_frame=None, task_data=None,
                      available_actions=None, game_id="ls20",
                      levels_completed=0, episode_context=None) -> dict:
        """Sovereign: build WINE envelope with episode context, execute_task() on GPU."""
        from knowledge3d.tablet.wine.game2d_wine import arc3_game_envelope
        envelope = arc3_game_envelope(
            frame=frame,
            goal=goal_frame or [[]],
            step_count=self._step_count,
            game_id=game_id,
            levels_completed=levels_completed,
            episode_context=episode_context or {},
        )
        result = self._tablet.execute_task(envelope)
        action_code = int(result.get("action_code", 0) or 0)
        action_name = ACTION_NAMES[action_code % len(ACTION_NAMES)]
        return {"action": action_name, "action_code": action_code}

    def learn_from_outcome(self, *, levels_completed: int, frame,
                           action: str, prev_frame, reward: float,
                           lives_delta: int, levels_delta: int) -> None:
        """Sovereign: seed outcome into episode Galaxy, run micro sleep-time."""
        self._episode_galaxy.seed_outcome(
            step_count=self._step_count,
            action=action,
            prev_grid=prev_frame,
            next_grid=frame,
            reward=reward,
            lives_delta=lives_delta,
            levels_delta=levels_delta,
        )
        self._episode_galaxy.run_micro_sleeptime()
        self._step_count += 1

    def get_episode_context(self) -> dict:
        return self._episode_galaxy.get_episode_context(last_n=10)
```

And in `K3DAgent.run_level()`, update the `learn_from_outcome()` call to pass the full
outcome data (action, prev_frame, reward, lives_delta, levels_delta) — the current call
only passes `levels_completed` and `frame`.

---

## Part 8 — ARC-3 Kaggle Notebook Integration

In `notebooks/arc_agi_2_kaggle_submission.py` (the ARC-3 equivalent, create
`notebooks/arc_agi_3_kaggle_agent.py` if needed), the multi-GPU setup:

```python
# Detect Kaggle multi-GPU environment
import torch
gpu_count = torch.cuda.device_count()
if gpu_count >= 4:
    print(f"Kaggle multi-GPU: {gpu_count}×L4 detected — parallel sleep-time enabled")
    # ARC3EpisodeGalaxy will auto-detect and use GPU1-3 for micro sleep-time
else:
    print(f"Single GPU: {gpu_count} device(s) — sequential micro sleep-time")

# Boot K3D on GPU0
kv = Knowledgeverse(storage_root=STORAGE_ROOT)

# Run ARC-3 agent with living memory
agent = K3DAgent("ls20", max_steps=500, allow_remote_compat=True, knowledgeverse=kv)
result = agent.run_level()
print(json.dumps(result, indent=2))
```

---

## What NOT to Do

- Do NOT use Python to reason about which action is better — that's gre_arc_reasoner's job
- Do NOT skip `seed_frame()` before `execute_task()` — episode context must be current
- Do NOT write to disk during the game loop — VRAM only until `consolidate_to_house()`
- Do NOT add `--no-remote-compat` to the game runner — it will break on Kaggle where the official SDK may not be available
- Do NOT run multi-GPU code when `torch.cuda.device_count() < 2` — auto-degrade cleanly
- Do NOT seed ARC3_FRAME stars with Python-computed "what action to take" — that's sovereignty violation

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ARC3_LIVING_MEMORY_REPORT_2026-04-09.md` with:

1. `arc3_episode_galaxy.py` created (yes/no, line count)
2. `arc3_game_mechanics_seeder.py` created (yes/no, star count seeded)
3. `arc3_game_envelope()` extended with `episode_context` (yes/no, file + line)
4. `K3DAgent.run_level()` wired with `seed_outcome()` + `run_micro_sleeptime()` (yes/no)
5. `K3DARC3Agent.learn_from_outcome()` updated signature (yes/no)
6. Multi-GPU auto-detection: `torch.cuda.device_count()` check in ARC3EpisodeGalaxy (yes/no)
7. `consolidate_to_house()` implemented (yes/no)
8. Tests: `tests/test_arc3_living_memory.py` — at minimum: episode Galaxy seeds frames,
   micro sleep-time runs without error, get_episode_context() returns correct shape (pass/fail)
9. Smoke run: `bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 10`
   — report: did episode_context appear in the WINE envelope? (check via log or debug print)
10. `echosys_ingest` tmux session still alive (tmux ls)
