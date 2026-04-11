# ARC3 BETWEEN-ATTEMPT Consolidation Protocol Review

## Architectural Mapping: 4-Layer Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 4: META-RULES                          │
│  "When stuck for N frames, try orthogonal exploration"         │
│  "Consolidate avatar identity before path rules"                │
│  ═══════════════════════════════════════════════════════════════ │
│                    LAYER 3: RULES                               │
│  "Color 1 blocks movement", "Position (3,7) is impassable"      │
│  "Moving into color 2 causes death"                            │
│  ═══════════════════════════════════════════════════════════════ │
│                    LAYER 2: MEANING                             │
│  "Color 6 = avatar", "Color 1 = wall", "Color 4 = goal"        │
│  "I am the entity that moves when I issue ACTION1-ACTION4"     │
│  ═══════════════════════════════════════════════════════════════ │
│                    LAYER 1: FORM                                │
│  Grid [[0,1,6],[0,0,4]], centroid (0.0, 2.0), frame #47      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. What Must PERSIST Between Attempts (No Reload)

### Layer 2 — Meaning (Permanent Galaxy Stars)

These are `MeaningCentricStar` entries that, once established, become fixtures of the constellation for the current level-context:

```python
# PERSIST: Color → Role Semantics (the most critical knowledge)
color_role_stars = {
    "avatar": MeaningCentricStar(
        concept="avatar_identity",
        anchor="color_6_is_avatar",      # Learned via action-correlation
        confidence=1.0,                    # Binary once confirmed
        provenance="action_delta_correlation_between_frames_3_and_4"
    ),
    "wall": MeaningCentricStar(
        concept="impassable_boundary",
        anchor="color_1_blocks_movement",
        confidence=0.8,                    # Increases with each blocked event
        provenance="action2_at_2_3_resulted_in_no_position_change"
    ),
    "goal": MeaningCentricStar(
        concept="level_termination_positive",
        anchor="color_4_triggers_level_complete",
        confidence=1.0,
        provenance="reached_color_4_at_5_7_episode_terminated_reward"
    ),
    "hazard": MeaningCentricStar(
        concept="level_termination_negative",
        anchor="color_2_triggers_death",
        confidence=1.0,
        provenance="touched_color_2_at_3_1_avatar_removed"
    )
}
```

**Rationale**: The code's `_COLOR_SEMANTIC` dictionary provides static defaults, but ARC3 levels remap colors. The agent must *discover* that "in *this* level, color 6 is my avatar." This is the most foundational knowledge — without it, no rule can be formed. Once discovered via action-correlation (the cell that moves when I move), it must be a **permanent Galaxy star** that survives attempt boundaries.

### Layer 3 — Rules (Consolidated Path/Constraint Knowledge)

```python
# PERSIST: Movement constraints — learned rules about topology
movement_rules = {
    "blocked_positions": {
        # (row, col, direction): outcome — survives across attempts
        (2, 3, "south"): RuleStar(
            condition="avatar_at_2_3_move_south",
            outcome="blocked",
            valence="collision_barrier_impassable",
            evidence_count=2,              # Accumulated across attempts
            confidence=0.9
        ),
    },
    "level_layout": {
        "door_positions": [(5, 7), (9, 2)],    # Discovered exits
        "goal_position": (11, 15),             # Terminal location
        "hazard_zones": [(3, 1), (4, 1)],     # Death coordinates
    },
    "movement_economics": {
        "optimal_path_cost": 42,               # Minimum steps observed
        "observed_costs": [42, 55, 63],        # Historical attempt costs
        "budget_remaining_estimate": 30,       # For current attempt
    }
}
```

**Critical distinction**: The *rule* "position (2,3) is blocked going south" persists. The *event* "at frame 47 I tried to go south from (2,3)" is purged. The rule is the distillate; the event is the raw material.

### Layer 4 — Meta-Rules (Cross-Level Strategies)

```python
# PERSIST: Learning strategies that transcend single attempts
meta_rules = {
    "exploration_policy": "if_stuck_for_3_frames_try_unvisited_direction",
    "avatar_identification_protocol": "compare_grids_pre_post_action_to_find_moving_cell",
    "hazard_avoidance_escalation": "if_death_at_X_retry_with_2_cell_margin",
}
```

---

## 2. What Must Be PURGED Between Attempts

### Layer 1 — Form (All Frame-Specific State)

```python
class AttemptState:
    """
    All fields below are EPHEMERAL — they die with the failed attempt.
    They are the raw sensory stream, not the distilled knowledge.
    """
    
    # PURGE: Frame-specific grid snapshots
    _last_grid: list[list[int]] = None          # Raw observation
    _prev_grid: list[list[int]] = None          # Previous observation
    _centroid_position: tuple[float, float] = None  # Last known avatar location
    
    # PURGE: Action history from the failed trajectory
    _recent_actions: deque[str] = deque(maxlen=32)  # Stale action trace
    _step_count: int = 0                          # Reset to 0 on new attempt
    
    # PURGE: Temporary signal accumulators
    _stuck_counter: int = 0                      # "Haven't moved in N frames" counter
    _drift_detected: bool = False                 # Transient navigation flag
    _repeated_action_count: dict[str, int] = {}   # "Tried ACTION2 four times in a row"
    
    # PURGE: Attempt-specific outcome
    _death_position: tuple[int, int] | None = None  # Where avatar died THIS attempt
    _last_blocked_direction: str | None = None       # Last blocked direction THIS attempt
```

### The Purge Protocol

```python
def purge_between_attempts(self) -> None:
    """
    Called when episode ends (death, timeout, level_complete).
    Clears Layer 1 form-data while preserving Layer 2-4 stars.
    """
    # ─── PURGE all Layer 1 ephemeral state ───
    self._last_grid = None
    self._prev_grid = None
    self._centroid_position = None
    
    # ─── PURGE stale action history ───
    self._recent_actions.clear()
    self._step_count = 0
    
    # ─── PURGE transient signals ───
    self._stuck_counter = 0
    self._drift_detected = False
    self._repeated_action_count.clear()
    
    # ─── PURGE attempt-specific outcomes ───
    self._death_position = None
    self._last_blocked_direction = None
    
    # ─── DO NOT PURGE (these are Layer 2-3) ───
    # self._color_role_stars        ← PERSISTS
    # self._movement_rules          ← PERSISTS
    # self._level_layout            ← PERSISTS
    # self._movement_economics      ← PERSISTS
    # self._meta_rules              ← PERSISTS
```

**The invariant**: If knowledge was extracted *from* the failed attempt (e.g., "color 2 at (3,1) causes death"), it is already a Rule star before the purge. The purge cleans the *stream*, not the *distillate*.

---

## 3. Micro-Sleeptime on GPU During Gameplay

### The Timing Opportunity

```python
async def step_with_consolidation(self, action: str) -> dict:
    """
    Game loop step with GPU-native micro-consolidation
    during the API round-trip latency.
    """
    # ─── PHASE 1: Dispatch action to game API ───
    action_future = self.game_api.step_async(action)
    
    # ─── PHASE 2: GPU micro-consolidation WHILE WAITING ───
    # This runs on GPU kernels, NOT in Python threads.
    # The CPU is blocked on the API response anyway.
    with self._gpu_stream_guard():  # Separate CUDA stream
        self._run_micro_consolidation_sync()
    
    # ─── PHASE 3: Receive response ───
    response = await action_future
    
    # ─── PHASE 4: Process response, generate briefs ───
    self._process_frame(response)
    
    return response
```

### GPU Kernel Dispatch Schedule

```
┌─────────────────────────────────────────────────────────────┐
│                  GAME STEP TIMELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  t=0ms   ACTION DISPATCHED → Game API                        │
│          │                                                   │
│          ▼                                                   │
│  t=1ms   ┌─────────────────────────────────────────┐        │
│          │  GPU STREAM 1: Micro-Consolidation       │        │
│          │                                          │        │
│          │  1. sleep_time_micro.ptx                 │        │
│          │     • Strengthen high-confidence rules    │        │
│          │     • Weaken contradicted hypotheses      │        │
│          │     • ~2-5ms on GPU                      │        │
│          │                                          │        │
│          │  2. sleep_cluster_refiner.ptx             │        │
│          │     • Merge overlapping position rules    │        │
│          │     • Prune low-evidence clusters         │        │
│          │     • ~1-3ms on GPU                      │        │
│          │                                          │        │
│          │  3. galaxy_memory_updater.cu              │        │
│          │     • Update star scores (recent events)  │        │
│          │     • Decay stale rule confidence          │        │
│          │     • ~1-2ms on GPU                      │        │
│          │                                          │        │
│          │  4. lora_gpu.cu                           │        │
│          │     • Contrastive update on specialist     │        │
│          │     • Shadow copy comparison              │        │
│          │     • ~3-5ms on GPU                      │        │
│          └─────────────────────────────────────────┘        │
│          │                                                   │
│          ▼                                                   │
│  t=15ms  API RESPONSE RECEIVED                              │
│          │                                                   │
│          ▼                                                   │
│  t=16ms  FRAME PROCESSING (CPU+GPU inference)              │
│          • Compare grids → detect avatar movement           │
│          • Classify outcome (moved/blocked/death/goal)      │
│          • Generate briefs for next consolidation cycle     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Kernel Specifications

```python
class GPUConsolidationScheduler:
    """
    Manages micro-consolidation kernel dispatch on GPU
    during gameplay API round-trips.
    """
    
    def _run_micro_consolidation_sync(self) -> None:
        """
        Launches GPU kernels for between-step consolidation.
        Runs SYNCHRONOUSLY on a separate CUDA stream — 
        the CPU waits for the API anyway, so we block here too,
        but the GPU does the actual work.
        """
        # ─── Kernel 1: sleep_time_micro.ptx ───
        # Purpose: Rapid pass over recent briefs
        #   • Strengthen stars with accumulating evidence
        #   • Weaken stars with contradicting evidence
        #   • Merge near-duplicate entries
        self._launch_ptx_kernel(
            "sleep_time_micro.ptx",
            briefs=self._pending_briefs_buffer,  # GPU-resident
            galaxy_stars=self._galaxy_star_tensor,  # GPU-resident
            passes=1,  # Single micro-pass (full passes are for full SleepTime)
            stream=self._consolidation_stream
        )
        
        # ─── Kernel 2: sleep_cluster_refiner.ptx ───
        # Purpose: Topological refinement of rule clusters
        #   • Merge adjacent blocked positions into "wall" concept
        #   • Prune single-observation rules below confidence threshold
        #   • Validate rule consistency (no contradicting outcomes for same condition)
        self._launch_ptx_kernel(
            "sleep_cluster_refiner.ptx",
            clusters=self._rule_cluster_tensor,  # GPU-resident
            merge_threshold=0.7,  # Confidence threshold for merging
            prune_threshold=0.2,  # Confidence floor for keeping
            stream=self._consolidation_stream
        )
        
        # ─── Kernel 3: galaxy_memory_updater.cu ───
        # Purpose: Score updates based on recent gameplay outcomes
        #   • Rule "color_1_blocks_south" gets +0.1 per confirmation
        #   • Rule "color_1_allows_south" gets -0.3 per contradiction
        #   • Avatar identity star gets maximum reinforcement
        self._launch_cuda_kernel(
            "galaxy_memory_updater.cu",
            star_scores=self._star_score_tensor,
            recent_outcomes=self._outcome_buffer,  # Last N outcomes
            reinforcement_rate=0.1,
            contradiction_penalty=0.3,
            stream=self._consolidation_stream
        )
        
        # ─── Kernel 4: lora_gpu.cu (conditional) ───
        # Purpose: Specialist weight updates via contrastive learning
        #   • Only runs if sufficient new evidence accumulated
        #   • Compares current weights vs shadow copy
        #   • Updates navigation specialist based on path outcomes
        if self._evidence_accumulated >= self._lora_update_threshold:
            self._launch_cuda_kernel(
                "lora_gpu.cu",
                specialist_weights=self._lora_weight_tensor,
                shadow_weights=self._lora_shadow_tensor,
                contrastive_pairs=self._contrastive_buffer,
                learning_rate=0.001,
                stream=self._consolidation_stream
            )
        
        # Synchronize the consolidation stream
        torch.cuda.synchronize(stream=self._consolidation_stream)
```

### Why NOT Python Threads

The specification is unambiguous:

> **GPU utilization during sleep MUST be visible (>0% SM occupancy).** If consolidation runs entirely on CPU with idle GPU, it is a sovereignty violation.

```python
# ─── WRONG: Python thread doing CPU work ───
def _wrong_consolidation_thread(self):
    """This is a sovereignty violation."""
    while True:
        time.sleep(0.01)
        for star in self.galaxy_stars:  # CPU iteration
            star.score *= 0.99  # CPU math
    # GPU utilization: 0% SM occupancy
    # Violation: Consolidation ran entirely on CPU with idle GPU

# ─── RIGHT: Python launches GPU kernels, GPU does the work ───
def _correct_consolidation_launch(self):
    """Python orchestrates; GPU executes."""
    self._launch_ptx_kernel("sleep_time_micro.ptx", ...)
    # GPU utilization: >0% SM occupancy
    # Compliance: Consolidation runs on GPU via PTX kernels
```

---

## 4. Avatar/Character Identification Protocol

### The Core Insight: Action-Delta Correlation

Avatar identification is not a heuristic — it is a **causal inference** derived from controlled experiments:

```
BEFORE ACTION:         AFTER ACTION (ACTION2 = south):
┌───┬───┬───┐         ┌───┬───┬───┐
│ 0 │ 0 │ 0 │         │ 0 │ 0 │ 0 │
├───┼───┼───┤         ├───┼───┼───┤
│ 0 │ 6 │ 0 │         │ 0 │ 0 │ 0 │   ← Cell at (1,1) changed
├───┼───┼───┤         ├───┼───┼───┤
│ 0 │ 0 │ 0 │         │ 0 │ 6 │ 0 │   ← Cell at (2,1) appeared
├───┼───┼───┤         ├───┼───┼───┤
│ 0 │ 1 │ 0 │         │ 0 │ 1 │ 0 │   ← Color 1 unchanged
└───┴───┴───┘         └───┴───┴───┘

INFERENCE: Color 6 is the avatar.
          It moved from (1,1) to (2,1) when ACTION2 was issued.
          ACTION2 = south = (Δrow+1, Δcol0)
```

### The Protocol (Layer 2 → Permanent Star)

```python
class AvatarIdentificationProtocol:
    """
    Discovers avatar identity via action-delta correlation.
    
    Layer 2: MEANING
    Status: FOUNDATIONAL — all subsequent reasoning depends on this.
    Persistence: Permanent Galaxy star once confirmed.
    """
    
    def __init__(self):
        self._candidate_avatar_colors: dict[int, float] = {}  # color → confidence
        self._confirmed_avatar_color: int | None = None
        self._confirmation_frames: int = 0
        self._required_confirmations: int = 3  # Must see 3 consistent movements
    
    def process_frame_transition(
        self,
        prev_grid: list[list[int]],
        next_grid: list[list[int]],
        action: str
    ) -> int | None:
        """
        Core algorithm: Find the cell that moved in the direction of the action.
        
        Returns: Confirmed avatar color, or None if not yet confirmed.
        """
        if self._confirmed_avatar_color is not None:
            return self._confirmed_avatar_color
        
        delta = _movement_delta(action)
        if delta is None:
            return None  # Non-movement action, no information
        
        # Find cells that disappeared from prev_grid
        disappeared: dict[int, list[tuple[int, int]]] = {}
        appeared: dict[int, list[tuple[int, int]]] = {}
        
        rows, cols = len(prev_grid), len(prev_grid[0])
        for r in range(rows):
            for c in range(cols):
                prev_val = prev_grid[r][c]
                next_val = next_grid[r][c]
                if prev_val != next_val:
                    # Cell changed
                    if next_val == _background_color(prev_grid):
                        # Disappeared (was prev_val, now background)
                        disappeared.setdefault(prev_val, []).append((r, c))
                    if prev_val == _background_color(next_grid):
                        # Appeared (was background, now next_val)
                        appeared.setdefault(next_val, []).append((r, c))
        
        # Check if any disappeared→appeared pair matches the action delta
        for color in set(disappeared.keys()) & set(appeared.keys()):
            for (dr, dc) in disappeared[color]:
                for (ar, ac) in appeared[color]:
                    expected_r = dr + delta[0]
                    expected_c = dc + delta[1]
                    if (ar, ac) == (expected_r, expected_c):
                        # MATCH: This color moved in the action's direction!
                        self._candidate_avatar_colors[color] = \
                            self._candidate_avatar_colors.get(color, 0) + 0.5
                        
                        # Check confirmation threshold
                        if self._candidate_avatar_colors[color] >= self._required_confirmations:
                            self._confirmed_avatar_color = color
                            self._elevate_to_permanent_star(color)
                            return color
        
        # Also check: moved but no new position (wall collision)
        # If only disappeared and no appeared, avatar tried to move but was blocked
        for color in disappeared:
            if color not in appeared:
                self._candidate_avatar_colors[color] = \
                    self._candidate_avatar_colors.get(color, 0) + 0.1
                # Lower confidence — we saw it vanish but not reappear
                # Could be death OR blocked movement
        
        return None
    
    def _elevate_to_permanent_star(self, avatar_color: int) -> None:
        """
        Promote avatar identity to Layer 2 MeaningCentricStar.
        This is a FOUNDATIONAL star — all navigation rules reference it.
        """
        star = MeaningCentricStar(
            concept="avatar_identity",
            anchor=f"color_{avatar_color}_is_avatar",
            confidence=1.0,  # Binary once confirmed via 3+ consistent observations
            provenance=f"action_delta_correlation_{self._confirmation_frames}_frames",
            persistence="permanent",  # Survives all attempt boundaries
            referenced_by=[  # All subsequent rules reference this star
                "movement_rules",
                "hazard_avoidance_rules",
                "goal_proximity_rules",
            ]
        )
        self.galaxy.add_star(star)
```

### Identification State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│            AVATAR IDENTIFICATION STATE MACHINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [UNIDENTIFIED]────action-delta correlation────▶[CANDIDATE]      │
│       │                                               │          │
│       │                                          confirmation     │
│       │                                          count < 3        │
│       │                                               │          │
│       │                                               ▼          │
│       │                                        [CANDIDATE]      │
│       │                                          │    │          │
│       │                                    +1 confirm  │          │
│       │                                          │    │          │
│       │                              ┌───────────┘    │ -1 contra │
│       │                              │                │          │
│       │                              ▼                ▼          │
│       │                        [CONFIRMED]    [DEMOTED]         │
│       │                         color=N         │               │
│       │                              │           │               │
│       │                    ┌─────────┘           │               │
│       │                    ▼                     │               │
│       │           PERMANENT GALAXY STAR          │               │
│       │           Layer 2: Meaning               │               │
│       │           Never purged                    │