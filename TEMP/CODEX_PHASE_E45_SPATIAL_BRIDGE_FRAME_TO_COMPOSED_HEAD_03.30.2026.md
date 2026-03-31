# Codex — Phase E.45: Spatial Bridge — Frame Grid to Composed Head Pipeline

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — this is THE gap. Without this, no level ever solves by reasoning.

---

## The Exact Gap

I traced the complete data flow from frame to action. Here's where it breaks:

```
TODAY'S FLOW (broken):

  Frame (64×64 grid of ints 0-15)
    ↓
  _frame_to_query_text()           ← BOTTLENECK: spatial grid → text tokens
    "visual transformation task grid 64x64
     object at row 30 col 20
     action move left switch target visible"
    ↓
  _embed_query_gpu()               ← text → 16-dim embedding vector
    ↓
  _select_composed_head_candidate  ← Morton+LED-A*+Frustum search Galaxy by EMBEDDING
    ↓
  Galaxy match                     ← finds Drawing galaxy entry about grid transforms
    ↓
  _derive_action_from_result()     ← tries output_grid delta, falls through
    ↓
  default: 0 (Move Up)            ← NO REASONING HAPPENED

THE GAP: The 64×64 spatial grid is converted to TEXT, then to a 16-dim
embedding. By the time it reaches the composed head pipeline, ALL spatial
structure is lost. Morton Octree indexes Galaxy entries in embedding space,
NOT walkable cells on the game board. LED-A* searches Galaxy neighborhoods,
NOT paths through the game level.
```

The composed head pipeline IS spatial reasoning — but it's reasoning about
Galaxy topology (which knowledge stars are near each other in meaning space),
not about the game board topology (which cells are walkable and connected).

---

## What SHOULD Happen

```
CORRECT FLOW:

  Frame (64×64 grid of ints 0-15)
    ↓
  Frame → Spatial Graph on GPU     ← NEW: grid becomes walkable graph in VRAM
    - Color segmentation: walkable (grey) vs wall (dark) vs objects
    - Object detection: avatar, targets, mechanics
    - Adjacency: which walkable cells connect to which
    ↓
  Morton Encode walkable cells     ← EXISTING kernel, new INPUT
    - Each walkable cell → Morton code
    - Spatial index for O(1) neighbor lookup
    ↓
  LED-A* pathfind: avatar → target ← EXISTING kernel, new INPUT
    - Start: avatar's Morton code
    - Goal: target's Morton code (door, cross, recharge)
    - Cost: 1 per step (Manhattan on walkable graph)
    - Result: sequence of Morton codes = path
    ↓
  Path → Action sequence           ← NEW: decode Morton path to ACTION1-4
    - Consecutive cell deltas → Up/Down/Left/Right
    - First action in sequence = immediate next move
    ↓
  Knowledge context (Galaxy)       ← EXISTING: game mechanic stars inform STRATEGY
    - "movement_recharge_block" star → route through recharge if budget low
    - "strategic_reset" star → when to give up current attempt
    - "lock_key_pattern_match" star → white cross before door
    ↓
  TRM decides                      ← EXISTING: chooses which objective to pathfind to
    - Level just started? → find avatar + nearest mechanic target
    - Budget low? → pathfind to nearest recharge block
    - Shape mismatch? → pathfind to white cross first, then door
    - Path blocked? → try alternate route
```

The key insight: **Morton Octree and LED-A* must operate on TWO spatial
domains simultaneously:**

1. **Galaxy space** (existing): navigate knowledge stars to find the right
   STRATEGY (which target to go for, what order, when to reset)
2. **Frame space** (new): navigate the game board to find the actual PATH
   (how to reach the chosen target)

Both use the SAME kernels (Morton, LED-A*). The difference is the input
data — Galaxy entries in VRAM vs. walkable cells in VRAM.

---

## What Already Exists (Kernel Inventory)

| Kernel | Current Use | ARC3 Frame Use |
|--------|-------------|----------------|
| Morton Octree | Index Galaxy entries by embedding coords | Index walkable cells by (x,y) position |
| LED-A* | Navigate Galaxy graph (star→star) | Pathfind on walkable graph (cell→cell) |
| Frustum Cull | Focus on relevant Galaxy neighborhood | Focus on region around avatar |
| Dynamic LOD | Detail level for Galaxy entries | Could zoom: overview first, then detail |
| Nine-Chain Swarm | Parallel reasoning over Galaxy candidates | Parallel pathfinding to multiple targets |
| Halting Gate | Check if Galaxy reasoning converged | Check if any path reaches objective |
| Drawing bridge | `grid_to_surface()` / GPU grid transforms | Frame → GPU surface already works |

The critical fact: `_execute_arc_transform_gpu()` already calls
`bridge.grid_to_surface(input_grid)` to upload the grid to VRAM.
The frame IS already going to the GPU — but only for static transforms
(rotate, mirror, color remap), not for spatial analysis.

---

## The Spatial Bridge: What Codex Must Build

### Component 1: Frame → Walkable Graph (GPU)

Take the 64×64 grid (already on GPU via Drawing bridge) and produce:

1. **Color segmentation**: Classify each cell:
   - Walkable (light grey, color 5 or similar) → included in graph
   - Wall/background (dark, color 0) → excluded
   - Avatar (colors 0+1 composite, ~6 cells) → start node
   - Target types: door (bordered room, color 9), cross (white, color 15),
     recharge (yellow, colors 4/8), reference box (bottom-left, ignore)
   - Status bar (bottom 3 rows) → exclude from walkable area

   This could be a new PTX kernel or use the existing Drawing bridge's
   surface operations with a threshold pass.

2. **Connected component labeling**: On the walkable mask, identify which
   cells connect to which. A simple flood-fill or union-find on GPU.
   This gives the walkable graph.

3. **Object centroids**: For each identified object type, compute its
   centroid position. This is already partially done in Python
   (`_avatar_centroid`, `_select_mechanic_target`) — move it to GPU.

### Component 2: Morton Encoding of Walkable Cells

The existing Morton Octree kernel takes 3D coordinates and produces
Morton codes. For the 2D game grid:

- Input: (x, y) coordinates of walkable cells
- Output: Morton codes (interleave x,y bits → 12-bit code for 64×64)
- The Morton index gives O(1) spatial neighbor lookup

This is a SUBSET of what the Morton kernel already does (2D instead of 3D).

### Component 3: LED-A* on the Game Board

The existing LED-A* kernel does graph pathfinding. Feed it:

- **Start node**: avatar's Morton code
- **Goal node**: target's Morton code
- **Edge weights**: 1 per walkable-to-walkable step
- **Heuristic**: Manhattan distance (optimal for grid)
- **Result**: ordered list of Morton codes from start to goal

For multi-objective (recharge → cross → door), run LED-A* multiple times
or use Nine-Chain Swarm to pathfind in parallel to all targets.

### Component 4: Path → Action Decoder

Convert the Morton-code path to ACTION1-4:

```
For consecutive pairs (cell_i, cell_i+1) in path:
  delta_row = row(cell_i+1) - row(cell_i)
  delta_col = col(cell_i+1) - col(cell_i)
  if delta_row == -1: ACTION1 (Up)
  if delta_row == +1: ACTION2 (Down)
  if delta_col == -1: ACTION3 (Left)
  if delta_col == +1: ACTION4 (Right)
```

Return the FIRST action in the sequence. The next frame will confirm
progress, and the agent re-plans from the new position.

### Component 5: Strategy Selection via Galaxy (Existing)

The Galaxy path still runs — but now it answers WHAT to pathfind to,
not HOW to move:

- TRM navigates Galaxy → "I should go to the white cross first because
  my shape doesn't match the door" (from `lock_key_pattern_match` star)
- LED-A* pathfinds on the frame → "The shortest path from my position
  to the white cross goes Right×3, Up×5, Left×2"
- Combine: first action = Right

This is the dual-domain design: Galaxy = strategy, Frame = tactics.

---

## Integration Point: Where in the Pipeline

The spatial bridge should integrate at the `execute_task` level. When
`task_type == "ARC_TASK"` and the task has an `input_grid`:

```
BEFORE the existing flow runs:
  1. Upload input_grid to GPU (Drawing bridge: grid_to_surface)
  2. Run color segmentation → walkable mask
  3. Find avatar position and target positions
  4. Morton-encode walkable cells
  5. LED-A* from avatar to best target
  6. Decode path → first ACTION

IF pathfinding succeeds (path exists):
  → Return action directly (confidence = path quality metric)

IF pathfinding fails (no path, or objects not found):
  → Fall through to existing embedding-based Galaxy search
  → This is the exploration/fallback path
```

This way, the existing flow is preserved as a fallback, but the spatial
bridge handles the common case of navigating a visible game board.

---

## Color Mapping: Must Be Discovered from Frames

The exact color indices used by LS20 are unknown. Codex must:

1. Extract frames from the log at different game states
2. Map which color values correspond to:
   - Walkable floor (expected: light grey, maybe color 5)
   - Walls/background (expected: dark, color 0)
   - Avatar (colors 0+1 composite confirmed by existing code)
   - Status bar elements
   - Game mechanics (cross, recharge, door)

```python
# Quick analysis script:
import json
from collections import Counter
with open('/K3D/Knowledge3D.local/logs/arc3_live_20260330_220553.jsonl') as f:
    row = json.loads(f.readline())
    frame = row['frame']
    # Overall color distribution
    colors = Counter(cell for r in frame for cell in r)
    print("Color distribution:", colors.most_common())
    # Bottom rows (status bar)
    for i in range(len(frame)-5, len(frame)):
        row_colors = Counter(frame[i])
        print(f"Row {i}: {row_colors.most_common()}")
    # Find avatar (0+1 cluster)
    for r in range(len(frame)):
        for c in range(len(frame[r])):
            if frame[r][c] == 1:  # orange component
                print(f"Color 1 at ({r},{c})")
```

This analysis should be the FIRST thing Codex does — the color mapping
determines everything downstream.

---

## Sovereignty Compliance

All components must be sovereign:

- Frame → walkable graph: PTX kernel or Drawing bridge (GPU)
- Morton encoding: existing sovereign Morton kernel (GPU)
- LED-A* pathfinding: existing sovereign LED-A* kernel (GPU)
- Path → action: trivial conversion (can be on GPU or minimal Python)
- Strategy selection: existing Galaxy navigation (GPU)

The ONLY Python involved is the glue that calls the bridge functions
and passes the result back as an action dict. This is I/O code, not
reasoning — consistent with the ~200 lines target.

---

## What This Changes

| Before E.45 | After E.45 |
|-------------|------------|
| Frame → text → embedding → Galaxy search | Frame → spatial graph → pathfinding |
| Morton indexes Galaxy entries | Morton indexes BOTH Galaxy AND walkable cells |
| LED-A* navigates knowledge stars | LED-A* navigates BOTH knowledge AND game board |
| Agent chases nearest color cluster | Agent pathfinds to strategic target |
| Confidence = 0.000 (no real reasoning) | Confidence = path quality (real metric) |
| Level 1 needs 60+ actions, fails | Level 1 pathfinds in ~13 optimal actions |
| Level 2 impossible (no spatial reasoning) | Level 2 navigable (same pathfinding) |

---

## Success Criteria

- [ ] Color mapping extracted from real LS20 frames
- [ ] Frame → walkable mask works on GPU (Drawing bridge or new kernel)
- [ ] Avatar and target positions identified from frame
- [ ] Morton encoding of walkable cells (2D subset of existing kernel)
- [ ] LED-A* pathfinding from avatar to target on walkable graph
- [ ] Path decoded to ACTION1-4 sequence
- [ ] Level 1 solved by pathfinding (not script, not heuristic)
- [ ] Level 2 shows real navigation toward objective
- [ ] Confidence > 0 when pathfinding succeeds
- [ ] Falls through to Galaxy search when pathfinding fails
- [ ] All spatial operations sovereign (GPU/PTX)

---

## Architectural Note

This is the moment the composed head pipeline starts doing what it was
DESIGNED for: spatial reasoning on spatial data. Until now, the pipeline
has been reasoning about text-derived embeddings — powerful for knowledge
retrieval, but not spatial navigation. E.45 gives the pipeline its first
ACTUAL spatial input.

The beauty: the same Morton + LED-A* kernels that navigate the Galaxy
(knowledge space) now also navigate the game board (physical space).
This is exactly the House/Galaxy duality — the House IS spatial reality,
the Galaxy IS internal cognition. The same spatial reasoning serves both.

Daniel: "The TRM should be able to use knowledge to adapt by its own on
moving forward." E.45 gives the TRM spatial EYES (frame perception) and
spatial LEGS (pathfinding). The Galaxy gives it the BRAIN (strategy).
All sovereign, all GPU.
