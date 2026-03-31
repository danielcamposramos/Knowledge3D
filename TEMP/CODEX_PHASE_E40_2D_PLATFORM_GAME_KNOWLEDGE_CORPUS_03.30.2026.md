# Codex — Phase E.40: 2D Platform Game Knowledge Corpus

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** HIGH — broadens ARC3 capability from one game to ALL 2D platform games

---

## Daniel's Direction

> "more than just this specific game logic, what if we import all we can find
> on platform 2d games and proceduralize it as knowledge, also symlinking game
> controls to actions?"

The current game_mechanics.jsonl has 10 stars for ARC3-specific mechanics.
This is too narrow. ARC3 games are 2D platform/puzzle games — the agent should
know EVERYTHING about 2D game design, mechanics, controls, and patterns.
Universal knowledge that helps with ANY 2D game, not just LS20.

---

## Scope: What to Research and Proceduralize

### 1. Movement & Physics

Universal 2D movement patterns the agent should know:

- **Cardinal movement** (4-direction grid): up/down/left/right on discrete grid
- **8-direction movement**: including diagonals
- **Continuous movement**: pixel-level with velocity/acceleration
- **Gravity**: downward pull, jumping arcs, falling
- **Jumping mechanics**: single jump, double jump, wall jump, variable height
- **Sliding/ice physics**: momentum continuation after input stops
- **Conveyor belts**: forced directional movement on specific tiles
- **Ladders/climbing**: vertical movement on climbable surfaces
- **Swimming/water**: altered physics in liquid areas
- **Teleporters/portals**: instant position change between linked points
- **Moving platforms**: tiles that carry the character
- **Wrap-around**: exiting one edge enters from the opposite

### 2. Interactive Objects / Blocks

Common 2D game objects and their behaviors:

- **Switches/buttons**: toggle state of linked objects (pressure plates, levers)
- **Doors/gates**: block passage until condition met (key, switch, kill all enemies)
- **Keys**: collectible items that unlock specific doors (color-coded)
- **Pushable blocks/crates**: objects the character can push to solve puzzles
- **Breakable blocks**: destroyed by specific action (jump from below, attack)
- **Collectibles**: items that increase score/count (coins, gems, stars)
- **Power-ups**: temporary or permanent ability modifiers
- **Checkpoints**: save progress position within a level
- **Spikes/hazards**: instant damage or death on contact
- **Moving enemies/obstacles**: patrol patterns, follow AI
- **Springs/bounce pads**: launch character upward
- **Fans/wind**: push character in a direction
- **Timers**: countdown that triggers events
- **Recharge/energy pickups**: restore a resource (movement points, health)
- **Color changers**: modify character property (may be cyclic)
- **Shape changers**: modify character form (to match lock requirements)
- **One-way platforms**: passable from one direction only
- **Crumbling platforms**: disappear after being stood on briefly

### 3. Puzzle Patterns

Common 2D puzzle structures:

- **Lock and key**: find key → open door → access new area
- **Pattern matching**: match a shape/color/sequence to unlock
- **Sokoban**: push blocks onto marked positions
- **Light reflection**: redirect beams using mirrors
- **Pipe/circuit completion**: connect endpoints with path segments
- **Sequence/order puzzles**: activate switches in correct order
- **Timing puzzles**: perform actions within time windows
- **State toggle chains**: flipping one switch affects multiple objects
- **Maze navigation**: find path through complex walkable structure
- **Multi-character**: control multiple entities to cooperate
- **Inventory management**: carry limited items, use at correct locations
- **Weight/pressure**: heavier objects trigger different switches
- **Color mixing**: combine colors to create target color

### 4. Controls → Actions Mapping

Symlink game controls to the ARC3 action space:

```
ACTION1 (Up)    → Star: "control_move_up"
  symlinks: → jump, climb_up, swim_up, menu_scroll_up, look_up
  meaning: "Move the controlled entity upward in the game space"

ACTION2 (Down)  → Star: "control_move_down"
  symlinks: → crouch, descend, swim_down, menu_scroll_down, look_down
  meaning: "Move the controlled entity downward in the game space"

ACTION3 (Left)  → Star: "control_move_left"
  symlinks: → walk_left, run_left, face_left, retreat
  meaning: "Move the controlled entity leftward in the game space"

ACTION4 (Right) → Star: "control_move_right"
  symlinks: → walk_right, run_right, face_right, advance
  meaning: "Move the controlled entity rightward in the game space"

ACTION5 (Perform) → Star: "control_interact"
  symlinks: → use_item, open_door, talk, pick_up, activate, attack
  meaning: "Interact with the object or entity at current position"

ACTION6 (Click)  → Star: "control_point_select"
  symlinks: → click_target, select_position, aim, place_object
  meaning: "Select or interact with a specific (x,y) position"

ACTION7 (Undo)  → Star: "control_undo"
  symlinks: → rewind, take_back, reset_last_move
  meaning: "Reverse the most recent action"

RESET (0) → Star: "control_restart"
  symlinks: → restart_level, reset_state, begin_again
  meaning: "Reset to initial state of current level or game"
```

### 5. Visual Encoding Patterns

How 2D games encode information visually:

- **Color = type/state**: red = danger, green = safe, yellow = collectible, blue = water
- **Flashing/blinking**: temporary state, urgency, interactable
- **Size = importance**: larger objects are more significant
- **Border/outline**: interactive vs decorative distinction
- **Animation**: movement implies behavior (patrolling enemy, flowing water)
- **Icons/symbols**: universal game symbols (heart = health, star = score, key = unlock)
- **Contrast**: important objects stand out from background
- **Spatial grouping**: related objects placed near each other

### 6. Level Design Patterns

Common structural patterns in 2D game levels:

- **Linear progression**: start left, exit right (or bottom to top)
- **Hub and spoke**: central area with branching paths
- **Backtracking**: must revisit earlier areas with new abilities
- **Rising difficulty**: each level introduces one new mechanic
- **Tutorial by design**: first encounter with mechanic is safe/forgiving
- **Checkpoint spacing**: save points before difficult sections
- **Secret areas**: hidden paths rewarding exploration
- **Boss rooms**: larger space for special challenge
- **Resource gating**: limited resource (moves, keys) forces efficiency

### 7. Game State Machine Patterns

How game state transitions work:

- **Title → Start → Gameplay → Level Complete → Next Level → Win**
- **Gameplay → Death → Retry (from checkpoint or level start)**
- **Gameplay → Pause → Resume**
- **Level Complete → Score Screen → Next Level**
- **Cutscene → Gameplay** (click/key to advance)

---

## How to Research

Use web search to find comprehensive resources on:

1. "2D platformer game mechanics taxonomy"
2. "2D puzzle game design patterns"
3. "game design vocabulary" / "game mechanics glossary"
4. "platform game controls standard"
5. "tile-based game object types"
6. "sokoban puzzle mechanics"
7. "2D game visual encoding conventions"
8. "level design principles 2D games"

Also examine the ARC3 game engine source at
`/K3D/K3D_llama_cpp/datasets/ARC-AGI-3-Agents/` for specific game mechanics
the ARC3 games use. Check if there are game definition files that reveal
what block types exist.

---

## How to Proceduralize

### Star Structure

Each mechanic becomes a meaning-first star in the same format as E.39's
game_mechanics.jsonl, with all 4 layers:

**Layer 1 (Form):** `visual_rpn` — how the mechanic LOOKS
**Layer 2 (Meaning):** `meaning_rpn` — what it IS (semantic definition)
**Layer 3 (Rules):** `behavior_rpn` — how it BEHAVES (cause → effect)
**Layer 4 (Meta-Rules):** `meta_refs` — when to APPLY this knowledge

### Symlinks

Every star must be cross-referenced:
- To Galaxy entries (Drawing, Grammar, Reality, Math, Tool)
- To action controls (ACTION1-7 mapped to game inputs)
- To related mechanics (switch → door, key → lock, etc.)
- Bidirectional (per project convention)

### Storage

Extend the existing House JSONL files:
- `/K3D/Knowledge3D.local/house/game_mechanics.jsonl` — core mechanics (expand from 10 to 100+ stars)
- `/K3D/Knowledge3D.local/house/Reality.jsonl` — physics and state entries
- `/K3D/Knowledge3D.local/house/Grammar.jsonl` — behavioral rules
- `/K3D/Knowledge3D.local/house/Tool.jsonl` — meta-rules and strategies

### Naming Convention

MEANING-FIRST, never benchmark-named:
- ✅ `switch_actuator`, `gravity_downward_pull`, `collectible_score_increment`
- ❌ `arc3_switch`, `ls20_yellow_block`, `game_recharge`

---

## Deliverables

1. **Research**: Compile comprehensive 2D game mechanics inventory (web search + ARC3 engine analysis)
2. **Stars**: 100+ meaning-first stars covering all categories above
3. **Symlinks**: Full cross-referencing including controls → actions
4. **House JSONL**: Populate all 4 JSONL files
5. **Boot verification**: All stars load at init, discoverable by navigator
6. **Tests**: Extend test_game_mechanics_boot.py to verify new star count
7. **Commit + push**: Single commit with all new knowledge

---

## Success Criteria

- [ ] 100+ game mechanic stars in House JSONL
- [ ] All 7 categories covered (movement, objects, puzzles, controls, visual, level design, state machine)
- [ ] ACTION1-7 symlinked to game control meanings
- [ ] Stars load at boot (verified by test)
- [ ] Star count increase from 130,928 to 131,000+
- [ ] No benchmark-specific naming
- [ ] Committed and pushed to remote
