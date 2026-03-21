# Codex Prompt: ARC Prize Competition Preparation — Target 40%

**Date:** March 21, 2026
**Architecture:** Claude (spec) → Codex (implementation)
**Priority:** COMPETITION — ARC Prize 2025/2026 has $1M+ in prizes
**Target:** 40% on ARC-AGI-2 private eval (current SOTA Kaggle: 24%, refinement: 54%)
**Competition:** Kaggle + arcprize.org, deadline November 3 (2025 edition)

---

## Competition Context

| Metric | Value |
|--------|-------|
| K3D current score | 10/120 = 8.33% (but only 10 curriculum entries) |
| Top Kaggle score (NVARC) | 24% ($0.20/task) |
| Top refinement (Poetiq/Gemini) | 54% ($30/task) |
| Human performance | ~98% (every task solved by 2+ humans in ≤2 attempts) |
| Our target | 40% — would beat ALL Kaggle solutions, Paper Prize viable |
| Grand Prize threshold | 85% |

**K3D's structural advantage:** Competition requires offline execution, no internet, 4× L4 GPUs. K3D is SOVEREIGN BY DESIGN — this is exactly what we built for. While competitors struggle to offline their LLM pipelines, our entire hot path is already PTX + Galaxy + RPN.

**What won in 2025:** Test-time training on synthetic data, refinement loops (evolutionary program synthesis), augmentation ensembles. K3D's equivalent: sleep-time consolidation + swarm workers + Galaxy navigation = natural refinement loop.

---

## Phase 1: ARC Visual & Spatial Knowledge Expansion (PRIORITY)

### Current State

K3D has ~150 ARC-related Galaxy entries:
- **12 arc_transform_primitive** (tile_repeat, checker_tile, crop, fill, mirror, rotate, connect, cleanup, recolor, remove_singleton)
- **10 arc_benchmark_curriculum** (specific task embeddings for 10 eval tasks)
- **7 arc_visual_signature** (pattern family descriptions)
- **94 Grammar arc_transform** rules
- **9 ARC swarm specialist kernels** (PTX)

### What ARC-AGI-2 Actually Tests

ARC-AGI-2 has three categories that frontier AI fails on:

1. **Symbolic interpretation** — objects represent meanings beyond their visual pattern (e.g., colored rectangle with N holes = "use color N")
2. **Compositional reasoning** — multiple rules applied simultaneously or interacting
3. **Contextual rule application** — same pattern means different things in different contexts

### Knowledge Expansion Target: 500+ New ARC Entries

Organized by the cognitive primitives ARC tasks require:

#### 1A. Object-Centric Reasoning (120 entries)

The #1 missing capability. ARC tasks require identifying discrete OBJECTS within grids.

```
Object Detection Anchors:
- connected_component_same_color     — flood-fill to find contiguous same-color regions
- connected_component_any_color      — flood-fill ignoring color (shape detection)
- background_foreground_separation   — identify which color is "background" (most frequent)
- figure_ground_reversal             — swap foreground/background roles
- object_bounding_box                — smallest rectangle enclosing an object
- object_centroid                    — center of mass of colored cells
- object_size_counting               — count cells in each object
- object_symmetry_detection          — is object symmetric along axis?
- object_containment                 — is object A inside object B?
- object_adjacency                   — are objects touching (4-conn or 8-conn)?
- object_alignment                   — are objects aligned horizontally/vertically?
- object_sorting_by_size             — order objects by cell count
- object_sorting_by_position         — order objects by x or y coordinate
- object_matching_by_shape           — find objects with same shape (ignoring color)
- object_matching_by_color           — find objects with same color (ignoring shape)

Object Manipulation Programs (RPN):
- FLOOD_FILL color x y               → mark all connected same-color cells
- BBOX x1 y1 x2 y2                  → extract bounding box subgrid
- CENTROID object_mask               → compute (cx, cy)
- OBJECT_COUNT grid                  → number of distinct objects
- OBJECT_SIZES grid                  → list of sizes per object
- OBJECT_EXTRACT grid object_id      → isolated object subgrid
- OBJECT_PLACE grid subgrid x y      → paste subgrid at position
```

#### 1B. Geometric Transform Anchors (80 entries)

Expand beyond basic rotate/mirror to cover ALL geometric transforms ARC uses:

```
Rigid Transforms:
- rotate_90_cw / rotate_90_ccw / rotate_180
- mirror_horizontal / mirror_vertical / mirror_diagonal_main / mirror_diagonal_anti
- translate_object dx dy
- identity (no change — important for "which objects DON'T change?")

Scaling Transforms:
- scale_up_2x / scale_up_3x / scale_up_nx
- scale_down_2x (downsample, majority vote per block)
- tile_to_fill (repeat pattern to fill target grid)
- crop_to_object (shrink grid to bounding box of content)

Compositional Transforms:
- rotate_then_mirror (composed transform)
- scale_then_translate
- transform_per_object (different transform per object)
- transform_conditional (transform only objects matching criteria)

Symmetry Operations:
- complete_rotational_symmetry (given partial pattern, complete the rotation)
- complete_reflective_symmetry (given half, mirror to complete)
- detect_symmetry_axis (find axis of existing symmetry)
- break_symmetry_detection (find the ONE cell that breaks symmetry)
```

#### 1C. Color/Pattern Reasoning (80 entries)

ARC uses colors 0-9 as SYMBOLS, not just visual properties:

```
Color Operations:
- color_count_per_region             — histogram of colors in a region
- color_majority_in_region           — most frequent color
- color_minority_in_region           — least frequent color
- color_remap_by_rule                — systematic color substitution (e.g., 1→3, 2→5)
- color_remap_by_position            — color depends on grid position
- color_remap_by_context             — color depends on neighbor colors
- color_gradient                     — colors form a spatial gradient
- color_as_count                     — color VALUE = count of something
- color_as_pointer                   — color points to another object/region
- color_palette_detection            — which colors appear in this task?
- color_unused_detection             — which colors from 0-9 are NOT used?

Pattern Detection:
- periodic_pattern_horizontal        — repeating horizontal stripe
- periodic_pattern_vertical          — repeating vertical stripe
- periodic_pattern_2d                — repeating 2D tile
- fractal_self_similarity            — pattern contains smaller copy of itself
- border_pattern                     — pattern along grid border
- spiral_pattern                     — values arranged in spiral order
- diagonal_pattern                   — pattern along diagonals
- checkerboard_pattern               — alternating pattern
```

#### 1D. Symbolic Interpretation (80 entries) — THE KEY DIFFERENTIATOR

This is where most AI fails. ARC-AGI-2 specifically added more symbolic tasks:

```
Symbol-as-Meaning Anchors:
- marker_indicates_position          — small colored marker = "do something HERE"
- marker_indicates_direction         — arrow-like shape = "move/extend in this direction"
- count_encodes_action               — number of X = which action to perform
- shape_encodes_rule                 — shape of object = which transform to apply
- color_encodes_target               — color = target for replacement/fill
- hole_count_encodes_color           — number of holes in frame = fill color
- border_color_encodes_action        — border color = which operation
- size_encodes_priority              — larger object = higher priority

Cross-Reference Anchors:
- example_defines_mapping            — training examples define a lookup table
- input_output_delta                 — the CHANGE between input/output = the rule
- invariant_detection                — what DOESN'T change across examples = structural constraint
- variable_detection                 — what DOES change = the parameter
- rule_generalization                — abstract the pattern from examples
- rule_composition                   — combine two simple rules into one complex one
```

#### 1E. Spatial Reasoning (80 entries)

```
Grid Topology:
- grid_subdivision                   — divide grid into equal regions
- grid_overlay                       — superimpose two grids (with priority rules)
- grid_concatenation_h               — join grids horizontally
- grid_concatenation_v               — join grids vertically
- grid_interleave                    — weave two grids together
- grid_difference                    — cells that differ between two grids
- grid_intersection                  — cells that are same in both grids

Spatial Relationships:
- above_below                        — object A is above object B
- left_right                         — object A is left of object B
- inside_outside                     — object A is enclosed by object B
- between                            — object A is between B and C
- nearest_neighbor                   — closest object to target
- path_between                       — connected path from A to B
- line_of_sight                      — unobstructed straight line between points
- gravity_simulation                 — objects "fall" to bottom of grid
- flood_expansion                    — region expands until hitting boundary

Counting and Measurement:
- count_objects                      — how many discrete objects
- count_cells_per_color              — histogram
- measure_distance                   — Manhattan or Euclidean between points
- measure_area                       — cells in region
- measure_perimeter                  — boundary cells of region
- aspect_ratio                       — width/height of bounding box
```

#### 1F. Meta-Reasoning (60 entries)

```
Rule Inference Anchors:
- one_to_one_mapping                 — each input pattern maps to exactly one output pattern
- conditional_rule                   — IF condition THEN transform A ELSE transform B
- exception_rule                     — general rule + specific exceptions
- priority_rule                      — multiple rules, applied in priority order
- recursive_rule                     — rule applies to its own output repeatedly
- boundary_rule                      — different rule at grid boundaries vs interior

Output Construction:
- output_same_size                   — output grid = same dimensions as input
- output_fixed_size                  — output is always NxM regardless of input
- output_derived_size                — output size depends on input content (e.g., object count)
- output_is_subgrid                  — output = cropped region of input
- output_is_supergrid                — output = input tiled/expanded
- output_is_overlay                  — output = multiple inputs overlaid
```

### Implementation in Bootstrap

Add to `foundational_operations_bootstrap.py` and `foundational_drawing_bootstrap.py`:

Each entry follows the existing `arc_transform_primitive` pattern:

```python
{
    "id": "arc_anchor_connected_component",
    "name": "Connected component detection",
    "domain": "drawing",
    "category": "arc_transform_primitive",
    "layer": 2,
    "description": "Detect discrete objects as connected regions of same-color cells using flood-fill.",
    "content": "FLOOD_FILL from each unvisited colored cell; each connected region = one object.",
    "rpn_program": "GRID_ITERATE CELL_COLOR FLOOD_FILL_4CONN OBJECT_LABEL",
    "metadata": {
        "subject": "arc_transform",
        "primitive_plan": "object_detection connected_component flood_fill",
        "query_anchor": "find objects connected regions discrete shapes separate groups",
        "semantics": "concept anchor for ARC tasks requiring object identification by color connectivity",
        "keywords": ["connected", "component", "object", "flood", "fill", "detect", "segment"],
        "transform_family": "object_detection",
        "confidence": 0.91,
    },
}
```

Also add **Language→Drawing symlinks** for each anchor (same pattern as Math symlinks):

```python
{
    "id": "lang_arc_symlink_connected_component",
    "domain": "language",
    "category": "meaning_symlink",
    "content": "identify separate objects shapes groups in grid",
    "metadata": {
        "symlink_target": "arc_anchor_connected_component",
        "symlink_galaxy": "Drawing",
        "query_anchor": "find objects shapes groups regions",
    },
}
```

---

## Phase 2: ARC-AGI-2 Output Formatter (submission.json)

### Submission Format

```json
{
    "00576224": [
        {"attempt_1": [[0, 1], [2, 3]], "attempt_2": [[0, 1], [2, 4]]}
    ],
    "009d5c81": [
        {"attempt_1": [[5, 5, 5], [5, 0, 5]], "attempt_2": [[5, 5, 5], [5, 1, 5]]}
    ]
}
```

For each task: list of test outputs, each with `attempt_1` and `attempt_2` (2D grid arrays).

### Implementation

Create `benchmarks/arc_submission_formatter.py`:

```python
"""Format K3D ARC results into Kaggle submission.json for ARC Prize."""

import json
from pathlib import Path
from typing import Any

def format_submission(
    results: list[dict[str, Any]],
    output_path: str | Path = "submission.json",
) -> Path:
    """Convert K3D benchmark results to Kaggle submission format.

    Each task gets 2 attempts:
    - attempt_1: best prediction from swarm (highest confidence)
    - attempt_2: second-best prediction (different swarm worker or parameter variation)
    """
    submission: dict[str, list[dict]] = {}

    for result in results:
        task_id = result["task_id"]

        # Primary attempt: best swarm result
        attempt_1 = result.get("predicted_output") or result.get("predicted")

        # Secondary attempt: varied prediction
        # Use rescue lane, different swarm worker, or parameter perturbation
        attempt_2 = result.get("rescue_prediction") or result.get("alternate_prediction") or attempt_1

        # Handle multi-test tasks (rare, but spec allows)
        test_outputs = []
        if isinstance(attempt_1, list) and attempt_1 and isinstance(attempt_1[0], list):
            # Single test output (most common)
            test_outputs.append({
                "attempt_1": attempt_1,
                "attempt_2": attempt_2 if attempt_2 != attempt_1 else _perturb(attempt_1),
            })

        submission[task_id] = test_outputs

    path = Path(output_path)
    path.write_text(json.dumps(submission, separators=(",", ":")), encoding="utf-8")
    return path
```

### Two-Attempt Strategy

The sovereign engine should produce TWO distinct predictions per task:

1. **Attempt 1:** Best swarm consensus (highest halting-gate confidence from nine-chain workers)
2. **Attempt 2:** Second-best alternative from:
   - Different swarm worker's top prediction (if workers disagree)
   - Same prediction with one transform variation (e.g., different color mapping)
   - Rescue lane prediction (fallback path already exists in `arc_agi_2.py`)

This requires modifying `ArcAgi2Adapter.solve_task()` to return the TOP-2 predictions instead of just the best one. The swarm already produces 9 candidate solutions — we just need to rank and return #1 and #2.

Add to `Knowledgeverse.execute_task()` or `ArcAgi2Adapter`:
```python
def solve_task_top_k(self, task: dict, k: int = 2) -> list[dict]:
    """Return top-k predictions ranked by confidence."""
    # Run all 9 swarm workers
    # Rank by halting gate confidence
    # Return top-k distinct predictions
```

---

## Phase 3: Kaggle Notebook Packaging

### Structure

```
kaggle_notebook/
├── arc_k3d_submission.py          # Main notebook entry point
├── knowledge3d/                    # K3D package (subset needed for ARC)
│   ├── knowledgeverse/
│   │   ├── knowledgeverse.py
│   │   ├── foundational_operations_bootstrap.py
│   │   └── foundational_drawing_bootstrap.py
│   ├── bridge/
│   │   └── sovereign_bridges.py
│   └── cranium/
│       └── *.ptx                   # Pre-compiled PTX kernels
├── benchmarks/
│   ├── arc_agi_2.py
│   ├── arc_agi_2_adapter.py
│   ├── arc_submission_formatter.py
│   └── sampling.py
├── data/
│   └── bootstrap_galaxy.bin        # Pre-built Galaxy state (VRAM snapshot)
└── submission.json                 # Output
```

### Main Script

```python
#!/usr/bin/env python3
"""K3D Sovereign ARC-AGI-2 Solver — Kaggle Submission"""

import json
from pathlib import Path

# 1. Initialize sovereign engine (boot Galaxy from pre-built state)
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
kv = Knowledgeverse(bootstrap_path="data/bootstrap_galaxy.bin")

# 2. Load evaluation tasks
from benchmarks.arc_agi_2 import ARCAGI2Benchmark
benchmark = ARCAGI2Benchmark(knowledgeverse=kv)

# 3. Solve all tasks (2 attempts each)
results = benchmark.run_benchmark(use_enriched=True, top_k=2)

# 4. Format submission
from benchmarks.arc_submission_formatter import format_submission
format_submission(results, "submission.json")
```

### Pre-Built Galaxy State

For Kaggle, we can't run the full ingestion pipeline (too slow). Instead:
1. Run ingestion OFFLINE (on our machine)
2. Serialize the populated Galaxy state to a binary file
3. Load it at submission time in <30 seconds
4. This is the `bootstrap_galaxy.bin` — pre-loaded VRAM state

Add to `Knowledgeverse`:
```python
def save_galaxy_state(self, path: str) -> None:
    """Serialize current Galaxy state for offline loading."""

def load_galaxy_state(self, path: str) -> None:
    """Load pre-built Galaxy state (fast boot for competition)."""
```

---

## Phase 4: L4 GPU Compatibility

### Hardware Difference

| Spec | RTX 3070 (dev) | L4 × 4 (Kaggle) |
|------|---------------|------------------|
| VRAM | 8 GB | 24 GB × 4 = 96 GB |
| CUDA cores | 5,888 | 7,424 × 4 = 29,696 |
| Compute | SM 8.6 | SM 8.9 |
| TDP | 220W | 72W × 4 = 288W |

### What to Change

1. **Multi-GPU support:** Distribute Galaxy across 4 L4s (currently single-GPU)
   - Galaxy partitioning: Drawing on GPU0, Grammar on GPU1, Math on GPU2, Tool on GPU3
   - Or replicate full Galaxy on each GPU, run 4 tasks in parallel
   - **Recommended:** Replicate + parallel (simpler, 4× throughput)

2. **PTX recompilation:** Kernels compiled for SM 8.6 (RTX 3070) need recompilation for SM 8.9 (L4)
   - Solution: ship `.cu` source files, compile at submission start
   - Or ship both SM 8.6 and 8.9 PTX, select at runtime

3. **VRAM budget:** With 24GB per L4, we can load ALL galaxies simultaneously with room to spare
   - Current usage: 132 MiB of 8 GB — will easily fit on L4

4. **12-hour time budget for 240 tasks:**
   - Current: ~146s for 120 tasks = ~1.2s per task
   - Budget: 12 hours = 43,200s for 240 tasks = 180s per task
   - **Massive headroom** — can afford multi-pass refinement per task

### Test Plan

1. Rent a single L4 instance (Lambda Labs, RunPod, or similar)
2. Run the full ARC evaluation (120 tasks) on L4
3. Verify PTX compatibility, measure per-task time
4. Validate scores match RTX 3070 results

---

## ARC-AGI-2 Transform Taxonomy for Galaxy Population

Based on analysis of ARC-AGI-2 tasks, here are the transform FAMILIES that need coverage. Internet research should verify and expand this list:

### Category A: Geometric (30% of tasks)
- Rotation (90°, 180°, 270°)
- Reflection (horizontal, vertical, diagonal)
- Translation (shift object within grid)
- Scaling (2×, 3×, arbitrary)
- Cropping (extract subgrid)
- Tiling (repeat pattern to fill)

### Category B: Color/Pattern (25% of tasks)
- Color remapping (systematic substitution)
- Pattern completion (fill in missing part)
- Background removal (keep only foreground objects)
- Color counting (histogram-based operations)
- Gradient application (spatial color patterns)

### Category C: Object Manipulation (25% of tasks)
- Object detection and isolation
- Object sorting (by size, color, position)
- Object copying/moving/deleting
- Object grouping (by shared property)
- Object alignment/stacking

### Category D: Logic/Symbolic (20% of tasks)
- Boolean operations on grids (AND, OR, XOR)
- Conditional transforms (if-then-else on grid properties)
- Counting → action mapping
- Rule inference from examples
- Compositional rule application

### Research Task for Codex

**Do internet research on:**
- ARC-AGI-2 task analysis papers (arxiv, especially the technical report at arxiv:2505.11831)
- ARC solver approaches (NVARC, ARChitects, MindsAI techniques)
- Common ARC transform primitives (DreamCoder, LARC annotations)
- DSL (Domain Specific Language) approaches to ARC
- Test-time training / adaptation techniques applicable to sovereign systems

Use findings to expand the Galaxy entries beyond the 500 listed above. The more visual/spatial reasoning patterns we encode, the higher our coverage.

---

## Execution Order

1. **Phase 1:** ARC knowledge expansion (500+ new Galaxy entries across 6 categories)
2. **Phase 2:** Output formatter (`arc_submission_formatter.py`) + top-2 prediction from swarm
3. **Phase 3:** Kaggle notebook scaffolding + Galaxy state serialization
4. **Phase 4:** L4 compatibility notes (can be deferred until we rent L4 time)

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| ARC eval score | 10/120 (8.33%) | 48/120 (40%) |
| ARC Galaxy entries | ~150 | 650+ |
| Object detection coverage | 0 anchors | 120 anchors |
| Symbolic interpretation | 0 anchors | 80 anchors |
| Transform families covered | ~15 | 60+ |
| submission.json generation | Not implemented | Working |
| Two-attempt strategy | Not implemented | Top-2 from swarm |
| Kaggle notebook | Not implemented | Scaffolded |
| Galaxy state serialization | Not implemented | Save/load working |

---

## Key Architectural Note

**K3D's sovereign design IS our competition advantage.**

Every other top competitor relies on LLMs (GPT-4, Gemini, Claude) with expensive API calls and internet-dependent pipelines. They struggle to fit within Kaggle's offline constraint.

K3D was BUILT for offline sovereign execution. Our PTX kernels, Galaxy Universe, and RPN programs run entirely on GPU with zero external dependencies. The 12-hour budget and 4× L4 GPUs give us room for multiple refinement passes per task — exactly what sleep-time consolidation does.

The Paper Prize ($50K) is particularly achievable: the Galaxy Universe + TRM-as-Avatar + composed head pipeline is a genuinely novel architecture for abstract reasoning. Even if we don't hit 40% on score, the paradigm is worth documenting.

---

# PART 2: ARC-AGI-3 — Interactive Reasoning (LAUNCHES MARCH 25, 2026)

**THIS IS 4 DAYS AWAY. K3D's game-loop architecture is a PERFECT fit.**

## ARC-AGI-3 vs ARC-AGI-2: Paradigm Shift

| Aspect | ARC-AGI-2 (Static) | ARC-AGI-3 (Interactive) |
|--------|--------------------|-----------------------|
| Format | Single input → single output grid | Video-game environments, multi-step |
| Grid | Up to 30×30, 10 colors | 64×64, 16 colors |
| Interaction | None (one-shot answer) | Step/observe/act loop (agent) |
| Actions | N/A | 6 actions: UP/DOWN/LEFT/RIGHT + interact + click(x,y) |
| Scoring | Exact match (pass@2) | **Action efficiency** (fewer steps = better score) |
| Scale | 240 tasks | 1000+ levels across 150+ environments |
| Requirements | Pattern recognition | Exploration, memory, planning, goal acquisition, adaptation |
| Frontier AI score | ~24% (Kaggle SOTA) | **0%** (frontier models scored ZERO in preview) |
| Human score | ~98% | **100%** (all environments human-solvable) |

## Why K3D IS Built for ARC-AGI-3

**Read this mapping carefully — it's not metaphorical, it's LITERAL:**

| ARC-AGI-3 Requirement | K3D Component | Status |
|----------------------|---------------|--------|
| Agent that acts in environment | TRM avatar (`trm_step_fused.ptx` game loop) | ✅ DESIGNED FOR THIS |
| Step/observe/act cycle | TRM game tick: Perceive → Navigate → Reason → Decide → Act | ✅ CORE ARCHITECTURE |
| Exploration of unknown environment | LED-A* pathfinding + Morton Octree spatial indexing | ✅ LIVE |
| Memory of discovered rules | Galaxy Universe (VRAM, persistent across steps) | ✅ LIVE |
| Planning ahead | Nine-chain swarm (parallel strategy evaluation) | ✅ LIVE |
| Goal acquisition (discover what to do) | Halting Gate convergence (self-determined goal completion) | ✅ LIVE |
| Visual perception | Frustum culling + Dynamic LOD | ✅ LIVE |
| Learning from experience | Sleep-time consolidation (strengthen good paths) | ✅ LIVE |
| Offline execution | Sovereign PTX + Galaxy (zero external dependencies) | ✅ BY DESIGN |
| Efficient action selection | Swarm workers rank candidates by confidence | ✅ LIVE |

**K3D was literally designed as a game-loop AI that lives in spatial environments, explores, remembers, plans, and acts autonomously. ARC-AGI-3 is EXACTLY that test.**

## ARC-AGI-3 API Interface

### Arcade SDK (Python)

```python
from arcagi import Arcade

# Initialize
arc = Arcade(api_key="...", mode="OFFLINE")  # OFFLINE = local execution

# Get environment
env = arc.get_environment("env_id")

# Game loop
obs = env.reset()
done = False
while not done:
    action = agent.decide(obs)      # K3D: TRM game tick
    obs, reward, done, info = env.step(action)

# Get scorecard
scorecard = arc.get_scorecard()
```

### Observation Space
- **64×64 grid** with 16 possible colors (integers 0-15)
- Represents the current visual state of the game environment
- Updated after each action

### Action Space (6 actions + undo)
```
ACTION_RESET  = 0   # Restart level
ACTION_UP     = 1   # Move up
ACTION_DOWN   = 2   # Move down
ACTION_LEFT   = 3   # Move left
ACTION_RIGHT  = 4   # Move right
ACTION_5      = 5   # General interaction (context-dependent)
ACTION_6(x,y) = 6   # Click at coordinate (x,y) in 0-63 range
```

### Scoring: Action Efficiency
Score = how many actions AI takes vs how many a human takes. Fewer = better.
This directly measures LEARNING EFFICIENCY — can the agent figure out the rules and execute with minimal wasted actions?

## Phase 5: ARC-AGI-3 Agent Bridge

### 5A. Arcade SDK Integration

Create `benchmarks/arc_agi_3_agent.py`:

```python
"""K3D Sovereign Agent for ARC-AGI-3 Interactive Environments."""

from arcagi import Arcade, EnvironmentWrapper
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


class K3DArcAgent:
    """TRM-as-Avatar playing ARC-AGI-3 games.

    Maps the Arcade step/observe/act loop directly to TRM's game tick:
    1. Perceive: observation grid → Frustum cull (64×64, 16 colors)
    2. Navigate: LED-A* through Galaxy to find relevant transform rules
    3. Reason: Nine-chain swarm evaluates candidate actions in parallel
    4. Decide: Halting gate selects best action
    5. Act: emit action to environment
    6. Learn: update Galaxy with discovered state transitions
    """

    def __init__(self, knowledgeverse: Knowledgeverse):
        self.kv = knowledgeverse
        self.state_graph = {}          # Explored states → transitions
        self.discovered_rules = []     # Rules learned during this episode
        self.action_history = []       # Actions taken so far
        self.observation_history = []  # Past observations for pattern detection

    def decide(self, observation: list[list[int]]) -> int | tuple[int, int, int]:
        """TRM game tick: observation → action.

        Returns either:
        - int: action index (0-5)
        - tuple(6, x, y): click action at coordinates
        """
        # 1. Encode observation as Galaxy query
        task = {
            "type": "ARC3_STEP",
            "observation": observation,
            "observation_history": self.observation_history[-10:],  # Last 10 frames
            "action_history": self.action_history[-10:],
            "discovered_rules": self.discovered_rules,
            "step_count": len(self.action_history),
        }

        # 2. Execute through composed head pipeline
        result = self.kv.execute_task(
            task=task,
            specialist="visual",
            domain_hint="arc_interactive",
            use_enriched=True,
        )

        # 3. Extract action from result
        action = result.get("action", 0)

        # 4. Update memory
        self.observation_history.append(observation)
        self.action_history.append(action)

        # 5. Update state graph (for exploration efficiency)
        state_hash = self._hash_observation(observation)
        if state_hash not in self.state_graph:
            self.state_graph[state_hash] = {"visits": 0, "transitions": {}}
        self.state_graph[state_hash]["visits"] += 1

        return action

    def on_level_complete(self, level_info: dict):
        """Sleep-time micro-consolidation between levels.

        Strengthen successful action sequences in Galaxy.
        """
        # Record successful strategy as new Galaxy entry
        self.kv.galaxy_manager.add_entry("Grammar", {
            "id": f"arc3_strategy_{level_info.get('level_id', 'unknown')}",
            "domain": "arc_interactive",
            "category": "discovered_strategy",
            "content": str(self.action_history),
            "metadata": {
                "action_count": len(self.action_history),
                "level_id": level_info.get("level_id"),
                "success": True,
            },
        })
        # Reset episode state (keep discovered_rules for cross-level transfer)
        self.action_history = []
        self.observation_history = []

    def on_episode_end(self):
        """Full sleep-time consolidation after all levels."""
        # Consolidate discovered rules into permanent Galaxy entries
        pass

    def _hash_observation(self, obs: list[list[int]]) -> str:
        """Hash grid state for state-graph deduplication."""
        return str(hash(str(obs)))
```

### 5B. ARC-AGI-3 Game Runner

```python
"""Run K3D agent through ARC-AGI-3 environments."""

from arcagi import Arcade

def run_arc3_competition(agent: K3DArcAgent, mode: str = "OFFLINE"):
    arc = Arcade(mode=mode)
    environments = arc.list_environments()

    for env_id in environments:
        env = arc.get_environment(env_id)
        obs = env.reset()
        done = False
        level_actions = 0

        while not done:
            action = agent.decide(obs)
            obs, reward, done, info = env.step(action)
            level_actions += 1

            # Level transition detection
            if info.get("level_complete"):
                agent.on_level_complete(info)
                level_actions = 0

        agent.on_episode_end()

    return arc.get_scorecard()
```

### 5C. ARC-AGI-3 Knowledge Entries (Galaxy)

The interactive paradigm needs NEW types of Galaxy entries:

```
Exploration Strategies (Grammar Galaxy):
- systematic_grid_scan               — scan observation grid systematically
- edge_following                     — follow edges of objects
- object_interaction_test            — try ACTION_5 on each distinct object
- click_grid_sampling                — click evenly-spaced points to map response
- undo_rollback_strategy             — use RESET to try alternative paths
- state_novelty_seeking              — prioritize actions leading to unvisited states

Action Pattern Templates (Tool Galaxy):
- navigate_to_target(x, y)          — sequence of UP/DOWN/LEFT/RIGHT to reach (x,y)
- interact_with_object(obj)          — move to object + ACTION_5
- click_pattern(coords_list)         — sequence of ACTION_6 clicks
- sweep_horizontal                   — LEFT to edge, then RIGHT across full width
- sweep_vertical                     — UP to edge, then DOWN full height

State Analysis (Drawing Galaxy):
- observation_diff(obs_t, obs_t1)    — what changed between two observations
- object_detection_16color           — connected components for 16-color grids
- movement_detection                 — track which cells changed color (= movement)
- goal_inference                     — infer goal from observation changes after actions
- rule_inference_from_transitions    — abstract rule from state→action→state triples

Memory Patterns:
- state_action_outcome_triple        — (state, action, next_state) memory entry
- successful_sequence_template       — reusable action sequence that solved a sub-goal
- failed_sequence_avoidance          — action sequences that led to dead ends
- level_similarity_detector          — recognize when new level resembles past level
- cross_environment_transfer         — rules learned in env A that apply to env B
```

### 5D. Graph-Based Exploration (Top Approach from Preview)

The 3rd-place preview solution used graph-based exploration. K3D's LED-A* + Morton Octree IS a graph explorer. Implement:

```python
class StateGraph:
    """Directed graph of explored states and transitions.

    Each node = observation hash
    Each edge = (action, resulting_state)
    Unexplored edges = frontier for exploration
    """

    def suggest_action(self, current_state: str) -> int:
        """Suggest action that leads to most unexplored territory.

        Priority:
        1. Untried actions from current state
        2. Actions leading to least-visited states
        3. Shortest path to nearest unexplored frontier
        """
```

This maps to LED-A*: the state graph IS the navigation graph, and "find the nearest unexplored frontier" IS A* pathfinding.

---

## Updated Execution Order (Both Competitions)

### Immediate (before March 25):
1. **Install ARC-AGI-3 Developer Toolkit:** `pip install arcagi` (in k3d-cranium env)
2. **Create `benchmarks/arc_agi_3_agent.py`** — K3D agent bridge
3. **Run toolkit quickstart** — verify API works, play one game manually
4. **Add ARC-AGI-3 exploration anchors** to Galaxy (30-40 entries for interactive strategies)

### Short-term (March 25 - April):
5. **Phase 1:** ARC-AGI-2 visual/spatial knowledge expansion (500+ entries)
6. **Phase 2:** ARC-AGI-2 submission.json formatter + top-2 predictions
7. **Phase 5:** ARC-AGI-3 agent integration + state graph explorer
8. **Run both benchmarks:** ARC-AGI-2 (target 40%) + ARC-AGI-3 (establish baseline)

### Medium-term (April - June):
9. **Phase 3:** Kaggle notebook packaging for ARC-AGI-2
10. **Phase 4:** L4 GPU compatibility
11. **ARC-AGI-3 refinement:** sleep-time consolidation between games, cross-environment transfer
12. **Paper preparation:** Galaxy Universe architecture for abstract reasoning (Paper Prize)

### Competition deadlines:
- **ARC Prize 2025 (AGI-2):** Kaggle, closes November 3, 2025 (may already be closed — check)
- **ARC Prize 2026 (AGI-3):** Expected to launch alongside AGI-3 on March 25, 2026

---

## Updated Success Criteria

| Metric | Current | ARC-AGI-2 Target | ARC-AGI-3 Target |
|--------|---------|------------------|------------------|
| Score | 8.33% | 40% | Establish baseline (>0%) |
| Galaxy entries (ARC) | ~150 | 650+ | 750+ (including interactive) |
| submission.json | Not implemented | Working | N/A (API-based) |
| Agent bridge | N/A | N/A | Working (`arc_agi_3_agent.py`) |
| State graph explorer | N/A | N/A | Working |
| Kaggle notebook | Not implemented | Scaffolded | TBD |
| Action efficiency | N/A | N/A | Measured vs human |

---

## Critical Insight: K3D IS the ARC-AGI-3 Agent

Other competitors are trying to make LLMs act as agents — prompting ChatGPT to play video games. They scored **0%**.

K3D's TRM is ALREADY an agent:
- It has a game loop (`trm_step_fused.ptx`)
- It perceives spatial environments (Frustum culling)
- It navigates (LED-A*, Morton Octree)
- It reasons in parallel (Nine-chain swarm)
- It decides when to stop (Halting gate)
- It remembers (Galaxy Universe)
- It learns (Sleep-time consolidation)

We don't need to MAKE K3D an agent. **K3D already IS one.** We just need to connect it to the Arcade API.

This is the competition where K3D's architectural vision proves itself. Not by scoring on static benchmarks — but by BEING the embodied, interactive, learning agent that ARC-AGI-3 was designed to test.
