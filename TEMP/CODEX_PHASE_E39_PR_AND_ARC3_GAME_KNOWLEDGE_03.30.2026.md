# Codex — Phase E.39: PR + ARC3 Game Knowledge Ingestion

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** IMMEDIATE — two tasks: (1) PR to sync, (2) game knowledge ingestion

---

## Task 1: Create PR and Sync with Remote

### What to Commit

All Phase E work since the last push. This is a significant milestone:
- First live ARC3 level completion (score 3.57 on ARC3 account)
- Full benchmark infrastructure (MMLU 30%, ARC3 local 100%)
- Boot optimization (5+ min → ~40s)
- Sovereign embedding pipeline
- Multiple new PTX kernels and bridges
- WINE architectural direction

### PR Structure

Create a single PR from `main` to remote `main` with all staged + untracked
work. Group the commit message by area:

**Suggested commit message:**
```
feat: Phase E — live ARC3 level completion + sovereign benchmark pipeline

ARC3:
- First live ARC3 level completion on ls20-9607627b (score 3.57)
- Local ARC3 benchmark: 20/20 (100%)
- K3DARC3Agent with live server boundary (start screen, movement, levels)
- arc3_local.py: deterministic grid-navigation benchmark

Benchmarks:
- Full benchmark runner: MMLU 30%, GSM8K 0%, LHE 5%, ARC2 5%, ARC3 100%
- Stratified sampling across easy/mid/hard difficulty thirds
- Streaming JSONL per suite with progress callbacks
- Hardware-adaptive parallelization spec (E.37)

Sovereignty:
- RPNEmbeddingEngine: trigram embedding (Float32Vector, no numpy)
- Cached flat GPU buffer (.npy) + semantic CSR graph (.npz)
- knn_graph_build.cu: sovereign GPU KNN (replaces numpy matmul)
- sleep_time_micro.cu/ptx: micro sleep-time consolidation kernel
- gpu_task_dispatch.cu/ptx: GPU task dispatch kernel
- arc3_frame_encoder.cu/ptx: ARC3 frame encoding kernel

Infrastructure:
- Boot: 5+ min → ~40s via cache chain
- Galaxy VRAM table + foundational galaxy builder
- Persistent brain + sovereign text embedder
- WINE tablet proceduralization spec (E.35)
- 4-way reading strategy spec (E.38)
```

### Steps

1. Stage all relevant files (exclude any secrets, .env, large binaries)
2. Create the commit
3. Push to remote
4. Create PR via `gh pr create`

**Files to EXCLUDE from commit:**
- Any files under `/K3D/Knowledge3D.local/` (local runtime data)
- Any `.env` or secrets files
- Any `.npz`, `.npy`, `.pkl` cache files
- Any `__pycache__` directories

**Files to INCLUDE:**
- All `TEMP/*.md` specs (these ARE the architectural record)
- All `benchmarks/` changes
- All `knowledge3d/` changes
- All `scripts/` changes
- All `tests/` changes
- All `docs/` changes
- `CODEX.md`, `ATTRIBUTIONS.md` updates
- `.vscode/extensions.json` (if intentional)

---

## Task 2: ARC3 Game Knowledge — House Memory + Boot Ingestion

### Daniel's Insight

> "there's no instructions on the game! and the game evolves, the next level
> includes a yellow recharge movements block and then a coloured block to
> change colors (often needs to step twice to achieve the proper color) and
> a shape changing block — so it's also game logic, meaning we need knowledge
> on game logic saved at the house memory and tied to the actual actions"

The ARC3 games have NO instruction manual. The agent must LEARN game mechanics
through observation and persist that knowledge for reuse. This is exactly what
the House (permanent memory) is for.

### What Must Be Persisted (Game Mechanic Knowledge)

These are MEANING stars — universal game concepts, NOT benchmark-specific:

#### Core Mechanics (From Level 1)

```
Star: "spatial_navigation_grid"
  meaning: "Movement on a discrete 2D grid via cardinal directions"
  surface_forms: ["grid movement", "tile navigation", "discrete steps"]
  symlinks: → Drawing.RECT, Grammar.cardinal_directions

Star: "switch_actuator"
  meaning: "Walking over a switch changes the state of a linked object"
  surface_forms: ["switch", "trigger", "actuator", "pressure plate"]
  properties: {trigger: "walk_over", effect: "toggle_target_state"}
  symlinks: → Grammar.cause_effect, Reality.mechanical_switch

Star: "lock_key_pattern_match"
  meaning: "A locked passage opens when the current pattern matches the target"
  surface_forms: ["key fits lock", "pattern match unlock", "shape key"]
  properties: {condition: "pattern_equals_target", effect: "passage_opens"}
  symlinks: → Grammar.conditional_gate, Math.equality_test

Star: "level_progression"
  meaning: "Completing objectives advances to the next challenge stage"
  surface_forms: ["next level", "stage clear", "level complete"]
  properties: {trigger: "objective_complete", effect: "advance_stage"}
  symlinks: → Grammar.sequence_progression
```

#### Level 2+ Mechanics (From Daniel's Observation)

```
Star: "movement_recharge_block"
  meaning: "Stepping on a yellow block refills the movement budget"
  surface_forms: ["recharge", "refuel", "energy pickup", "stamina restore"]
  color_signature: "yellow"
  properties: {trigger: "walk_over", effect: "restore_movement_points"}
  symlinks: → Reality.energy_conservation, Grammar.resource_replenishment

Star: "color_transform_block"
  meaning: "Stepping on a colored block changes the entity's color property"
  surface_forms: ["color changer", "paint block", "dye station"]
  properties: {
    trigger: "walk_over",
    effect: "change_color",
    note: "may need multiple steps to reach target color (cyclic)"
  }
  symlinks: → Drawing.COLOR, Grammar.cyclic_state_machine

Star: "shape_transform_block"
  meaning: "Stepping on a block changes the entity's shape property"
  surface_forms: ["shape changer", "morph block", "transform station"]
  properties: {
    trigger: "walk_over",
    effect: "change_shape",
    note: "shape must match door target to unlock passage"
  }
  symlinks: → Drawing.geometric_transform, Grammar.state_transition
```

#### Meta-Knowledge (Game Logic Patterns)

```
Star: "no_instruction_discovery"
  meaning: "Game rules are not stated — must be inferred by observing
           cause and effect of actions on game state"
  surface_forms: ["learn by doing", "trial and error", "implicit rules"]
  symlinks: → Grammar.inductive_reasoning, Reality.empirical_observation

Star: "visual_state_encoding"
  meaning: "Game state is encoded visually — colors, shapes, and spatial
           positions represent logical properties"
  surface_forms: ["visual logic", "color means state", "shape means type"]
  symlinks: → Drawing.color_semantics, Grammar.visual_encoding

Star: "multi_step_state_transform"
  meaning: "Reaching a target state may require multiple transformation steps
           (e.g., step on color block twice for correct color)"
  surface_forms: ["double step", "cyclic transform", "iterative approach"]
  symlinks: → Grammar.iteration, Math.modular_arithmetic
```

### How to Store (House JSONL)

These stars go into the House as permanent knowledge — loaded once at boot,
never recomputed. They are MEANING stars (universal concepts), not
benchmark-specific entries.

**Storage location:** A House JSONL file alongside existing galaxy data:

```
/K3D/Knowledge3D.local/house/game_mechanics.jsonl
```

Each line is a star entry with the standard format used by the galaxy loader.

**Boot ingestion:** At Knowledgeverse init, load `game_mechanics.jsonl`
alongside other House JSONL files. These stars become part of the permanent
Galaxy, symlinked to Drawing, Grammar, Reality, and Math galaxies.

### How to Grow (Observation → Knowledge)

After each ARC3 game session, the sleep-time consolidation should:

1. **Analyze the action log**: Which actions caused state changes?
2. **Extract new mechanics**: Did the agent encounter a new block type?
3. **Create new stars**: Persist discovered mechanics as House JSONL entries
4. **Symlink to existing knowledge**: Connect to relevant Galaxy entries

This is the SOAR "chunking" principle from the Adaptive Reasoning Budget spec:
slow deliberative reasoning → crystallized cached knowledge.

For NOW (this sprint), manually create the stars listed above from Daniel's
observations. The automated observation→knowledge pipeline comes later.

---

## Execution Order

1. **PR first** — commit all Phase E work, push, create PR
2. **Game knowledge JSONL** — create `game_mechanics.jsonl` with the stars above
3. **Boot ingestion** — wire the JSONL into Knowledgeverse init
4. **Verify** — confirm the stars appear in Galaxy after boot
5. **Run LS20 again** — with game knowledge loaded, attempt levels 2+

---

## Success Criteria

### PR
- [ ] All Phase E files committed and pushed
- [ ] PR created with clear summary
- [ ] CI passes (if applicable)
- [ ] PR URL reported

### Game Knowledge
- [ ] `game_mechanics.jsonl` created with 10+ game mechanic stars
- [ ] Stars use meaning-first naming (not "arc3_switch" but "switch_actuator")
- [ ] Symlinks to existing Galaxy entries (Drawing, Grammar, Reality, Math)
- [ ] Loaded at boot into Galaxy (verified by entry count increase)
- [ ] Knowledge persists across restarts (House JSONL on disk)
- [ ] State saved after each session (sleep-time consolidation)
