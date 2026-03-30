# Codex — Phase E.3: Action Primitives as Galaxy Knowledge

**Date:** 2026-03-28
**From:** Daniel (Chair) + Claude (Architecture)
**To:** Codex
**Type:** IMPLEMENTATION ORDER — NOT A DISCUSSION. BUILD THIS.
**Prerequisite:** Phase E.2 ARC-AGI-3 wiring DONE.

---

## THE PRINCIPLE

Movement is not a hardcoded constant. Movement is **knowledge**.

Right now `benchmarks/arc_agi_3.py` has this:

```python
ACTION_EMBEDDINGS: list[list[float]] = []
for action_index in range(4):
    embedding = [0.0] * EMBEDDING32_DIMS
    embedding[action_index] = 1.0
    ACTION_EMBEDDINGS.append(embedding)
```

This is a one-hot hack. A real movement primitive has:
- **Form** (Layer 1): visual representation (an arrow, a vector)
- **Meaning** (Layer 2): spatial displacement in a direction
- **Rules** (Layer 3): boundary constraints, collision, undo semantics
- **Meta-Rules** (Layer 4): when to move vs. when to act vs. when to undo

These are Galaxy stars — procedural, symlinked, reusable across ARC-AGI-3, House navigation, physics simulation, game AI, and any future spatial interaction.

---

## DO NOT:
- Keep hardcoded one-hot action embeddings
- Add Python action-selection heuristics
- Modify the GPU kernel pipeline
- Create new opcodes (use existing RPN surface: `MAT4_TRANSLATE`, `DRAW_MOVE`, vector ops)

## DO:
- Create Reality atoms for spatial actions
- Give each atom proper `visual_rpn`, `behavior_rpn`, `law_rpn`
- Create a bootstrap function that loads them into the Galaxy
- Generate embeddings from the RPN programs (not one-hot constants)
- Wire the ARC-AGI-3 agent to use Galaxy-sourced action embeddings

---

## ORDER 1: Create Spatial Action Atoms

### 1A: Create `knowledge3d/cranium/action_primitives_bootstrap.py`

This file defines the fundamental spatial action atoms. These are **reusable across any spatial context** — not ARC-AGI-3 specific.

```python
"""Bootstrap spatial action primitives into Reality Galaxy.

Actions are knowledge — procedural programs with form, meaning, and rules.
They are reusable: ARC-AGI-3 games, House navigation, physics sims, any spatial agent.
"""

from __future__ import annotations

from knowledge3d.cranium.reality_nodes import RealityAtom
from knowledge3d.cranium.reality_galaxy import RealityGalaxy


def bootstrap_spatial_actions(
    galaxy: RealityGalaxy,
    *,
    encode_embedding: bool = True,
) -> list[str]:
    """Load fundamental spatial action atoms into Galaxy. Returns node_ids."""
    loaded: list[str] = []

    # === Layer 1-2: Cardinal movements ===
    # Form: arrow visual + displacement vector
    # Meaning: spatial translation along axis

    move_up = RealityAtom(
        node_id="action:move_up",
        visual_rpn="0 0 DRAW_MOVE 0 -1 DRAW_LINE DRAW_STROKE",
        behavior_rpn="y RECALL dy RECALL - y STORE",
        law_rpn="y RECALL y_min RECALL GTE",  # boundary: can't move above min
        metadata={
            "description": "Move one unit in negative-Y direction (up in screen space)",
            "displacement": [0, -1],
            "action_type": "spatial_translation",
            "arc3_action": "ACTION1",
            "surface_forms": {"en": "move up", "pt": "mover para cima"},
            "inverse": "action:move_down",
            "reusable_contexts": ["arc3", "house_navigation", "grid_world", "physics_sim"],
        },
    )

    move_down = RealityAtom(
        node_id="action:move_down",
        visual_rpn="0 0 DRAW_MOVE 0 1 DRAW_LINE DRAW_STROKE",
        behavior_rpn="y RECALL dy RECALL + y STORE",
        law_rpn="y RECALL y_max RECALL LT",
        metadata={
            "description": "Move one unit in positive-Y direction (down in screen space)",
            "displacement": [0, 1],
            "action_type": "spatial_translation",
            "arc3_action": "ACTION2",
            "surface_forms": {"en": "move down", "pt": "mover para baixo"},
            "inverse": "action:move_up",
            "reusable_contexts": ["arc3", "house_navigation", "grid_world", "physics_sim"],
        },
    )

    move_left = RealityAtom(
        node_id="action:move_left",
        visual_rpn="0 0 DRAW_MOVE -1 0 DRAW_LINE DRAW_STROKE",
        behavior_rpn="x RECALL dx RECALL - x STORE",
        law_rpn="x RECALL x_min RECALL GTE",
        metadata={
            "description": "Move one unit in negative-X direction",
            "displacement": [-1, 0],
            "action_type": "spatial_translation",
            "arc3_action": "ACTION3",
            "surface_forms": {"en": "move left", "pt": "mover para a esquerda"},
            "inverse": "action:move_right",
            "reusable_contexts": ["arc3", "house_navigation", "grid_world", "physics_sim"],
        },
    )

    move_right = RealityAtom(
        node_id="action:move_right",
        visual_rpn="0 0 DRAW_MOVE 1 0 DRAW_LINE DRAW_STROKE",
        behavior_rpn="x RECALL dx RECALL + x STORE",
        law_rpn="x RECALL x_max RECALL LT",
        metadata={
            "description": "Move one unit in positive-X direction",
            "displacement": [1, 0],
            "action_type": "spatial_translation",
            "arc3_action": "ACTION4",
            "surface_forms": {"en": "move right", "pt": "mover para a direita"},
            "inverse": "action:move_left",
            "reusable_contexts": ["arc3", "house_navigation", "grid_world", "physics_sim"],
        },
    )

    # === Layer 1-2: Non-movement actions ===

    perform_action = RealityAtom(
        node_id="action:perform",
        visual_rpn="0 0 0.1 circle fill",  # dot = "do something here"
        behavior_rpn="state RECALL action_fn RECALL STORE",  # apply current action function
        law_rpn="action_available RECALL 1 EQ",  # can only perform if action is available
        metadata={
            "description": "Execute the context-dependent action at current position",
            "displacement": [0, 0],
            "action_type": "spatial_interaction",
            "arc3_action": "ACTION5",
            "surface_forms": {"en": "perform action", "pt": "executar ação"},
            "inverse": "action:undo",
            "reusable_contexts": ["arc3", "house_navigation", "grid_world"],
        },
    )

    click_action = RealityAtom(
        node_id="action:click",
        visual_rpn="target_x target_y 0.05 circle fill",
        behavior_rpn="target_x RECALL target_y RECALL click_fn RECALL STORE",
        law_rpn="target_x RECALL x_max RECALL LT target_y RECALL y_max RECALL LT AND",
        metadata={
            "description": "Click at specific coordinates (x, y)",
            "displacement": [0, 0],
            "action_type": "spatial_selection",
            "arc3_action": "ACTION6",
            "surface_forms": {"en": "click", "pt": "clicar"},
            "parameterized": True,
            "parameters": ["x", "y"],
            "reusable_contexts": ["arc3", "house_navigation"],
        },
    )

    undo_action = RealityAtom(
        node_id="action:undo",
        visual_rpn="0 0 DRAW_MOVE -0.3 0.3 DRAW_LINE -0.3 -0.3 DRAW_LINE DRAW_STROKE",  # back-arrow
        behavior_rpn="history RECALL -1 index state STORE history RECALL pop STORE",
        law_rpn="history_length RECALL 0 GT",  # can only undo if history exists
        metadata={
            "description": "Undo the previous action (revert to prior state)",
            "displacement": [0, 0],
            "action_type": "temporal_reversal",
            "arc3_action": "ACTION7",
            "surface_forms": {"en": "undo", "pt": "desfazer"},
            "inverse": None,
            "reusable_contexts": ["arc3", "house_navigation", "grid_world"],
        },
    )

    # === Layer 3: Movement composition rules ===

    diagonal_move = RealityAtom(
        node_id="action:move_diagonal_ur",
        component_refs=["action:move_up", "action:move_right"],
        visual_rpn="0 0 DRAW_MOVE 1 -1 DRAW_LINE DRAW_STROKE",
        behavior_rpn="x RECALL dx RECALL + x STORE y RECALL dy RECALL - y STORE",
        law_rpn="x RECALL x_max RECALL LT y RECALL y_min RECALL GTE AND",
        metadata={
            "description": "Diagonal move: up + right (composed from cardinal movements)",
            "displacement": [1, -1],
            "action_type": "spatial_translation_composed",
            "surface_forms": {"en": "move diagonally up-right"},
            "reusable_contexts": ["house_navigation", "grid_world", "physics_sim"],
        },
    )

    # === Load all into Galaxy ===
    atoms = [
        move_up, move_down, move_left, move_right,
        perform_action, click_action, undo_action,
        diagonal_move,
    ]
    for atom in atoms:
        galaxy.add_node(atom, encode_embedding=encode_embedding)
        loaded.append(atom.node_id)

    return loaded
```

### Key design points:
- Each action has `visual_rpn` (how to draw it), `behavior_rpn` (what it does), `law_rpn` (constraints)
- `component_refs` enables composition (diagonal = up + right)
- `inverse` field creates bidirectional symlinks
- `reusable_contexts` marks where this primitive applies
- `surface_forms` gives multilingual labels (meaning-first, language-agnostic)
- `displacement` vector is the mathematical content

---

## ORDER 2: Generate Action Embeddings from Galaxy

### 2A: Create `knowledge3d/knowledgeverse/action_embedding_loader.py`

This replaces the hardcoded one-hot embeddings. It loads action atoms from the Galaxy and generates their embeddings using the existing embedding pipeline.

```python
"""Load action embeddings from Galaxy stars instead of hardcoded constants."""

from __future__ import annotations

from typing import Any

from knowledge3d.knowledgeverse.vram_task_buffer import EMBEDDING32_DIMS


# Canonical action atom IDs in GPU dispatch order (option index = list position)
ARC3_ACTION_ATOM_IDS = [
    "action:move_up",       # option 0 = ACTION1
    "action:move_down",     # option 1 = ACTION2
    "action:move_left",     # option 2 = ACTION3
    "action:move_right",    # option 3 = ACTION4
]

# Extended actions (for future 7-option slot support)
ARC3_EXTENDED_ACTION_ATOM_IDS = [
    *ARC3_ACTION_ATOM_IDS,
    "action:perform",       # option 4 = ACTION5
    "action:click",         # option 5 = ACTION6
    "action:undo",          # option 6 = ACTION7
]


def load_action_embeddings_from_galaxy(
    galaxy: Any,
    action_ids: list[str] | None = None,
) -> list[list[float]]:
    """Load action star embeddings from Reality Galaxy.

    Returns list of 32-float embeddings, one per action atom.
    Falls back to displacement-based embedding if Galaxy star has no embedding.
    """
    ids = action_ids or ARC3_ACTION_ATOM_IDS
    embeddings: list[list[float]] = []

    for atom_id in ids:
        node = galaxy.get_node(atom_id) if hasattr(galaxy, "get_node") else None
        if node is not None and hasattr(node, "embedding") and node.embedding:
            # Use Galaxy-computed embedding (from behavior_rpn + visual_rpn)
            raw = node.embedding
            if isinstance(raw, dict):
                raw = list(raw.values())
            embedding = [float(v) for v in list(raw)[:EMBEDDING32_DIMS]]
        else:
            # Fallback: derive from displacement metadata
            displacement = _get_displacement(galaxy, atom_id)
            embedding = _displacement_to_embedding(displacement)

        # Pad/truncate to EMBEDDING32_DIMS
        if len(embedding) < EMBEDDING32_DIMS:
            embedding.extend([0.0] * (EMBEDDING32_DIMS - len(embedding)))
        embeddings.append(embedding[:EMBEDDING32_DIMS])

    return embeddings


def _get_displacement(galaxy: Any, atom_id: str) -> list[float]:
    """Extract displacement vector from atom metadata."""
    node = galaxy.get_node(atom_id) if hasattr(galaxy, "get_node") else None
    if node and hasattr(node, "metadata") and isinstance(node.metadata, dict):
        disp = node.metadata.get("displacement", [0, 0])
        return [float(v) for v in disp]
    # Default displacements by atom_id
    defaults = {
        "action:move_up": [0.0, -1.0],
        "action:move_down": [0.0, 1.0],
        "action:move_left": [-1.0, 0.0],
        "action:move_right": [1.0, 0.0],
        "action:perform": [0.0, 0.0],
        "action:click": [0.0, 0.0],
        "action:undo": [0.0, 0.0],
    }
    return defaults.get(atom_id, [0.0, 0.0])


def _displacement_to_embedding(displacement: list[float]) -> list[float]:
    """Convert a displacement vector to a 32-float embedding.

    Uses spatial encoding: displacement components in first dims,
    magnitude in dim 4, normalized direction in dims 5-6,
    action type indicators in later dims.
    """
    dx = displacement[0] if len(displacement) > 0 else 0.0
    dy = displacement[1] if len(displacement) > 1 else 0.0
    magnitude = (dx * dx + dy * dy) ** 0.5
    norm_dx = dx / (magnitude + 1e-8)
    norm_dy = dy / (magnitude + 1e-8)

    embedding = [0.0] * EMBEDDING32_DIMS
    embedding[0] = dx
    embedding[1] = dy
    embedding[2] = 0.0  # dz (reserved for 3D)
    embedding[3] = 0.0  # reserved
    embedding[4] = magnitude
    embedding[5] = norm_dx
    embedding[6] = norm_dy
    embedding[7] = 1.0 if magnitude > 0 else 0.0  # is_movement flag

    return embedding
```

---

## ORDER 3: Wire ARC-AGI-3 Agent to Galaxy Actions

### 3A: Modify `benchmarks/arc_agi_3.py`

Replace hardcoded `ACTION_EMBEDDINGS` with Galaxy-sourced embeddings.

```python
# OLD (delete this):
# ACTION_EMBEDDINGS: list[list[float]] = []
# for action_index in range(4):
#     embedding = [0.0] * EMBEDDING32_DIMS
#     embedding[action_index] = 1.0
#     ACTION_EMBEDDINGS.append(embedding)

# NEW:
from knowledge3d.knowledgeverse.action_embedding_loader import (
    ARC3_ACTION_ATOM_IDS,
    load_action_embeddings_from_galaxy,
    _displacement_to_embedding,
    _get_displacement,
)

# Default action embeddings (displacement-based, no Galaxy needed for startup)
ACTION_EMBEDDINGS = [
    _displacement_to_embedding(_get_displacement(None, aid))
    for aid in ARC3_ACTION_ATOM_IDS
]
```

In `K3DARC3Agent.__init__`, optionally load from Galaxy if available:

```python
def __init__(self, max_actions=80, log_path=None, galaxy=None):
    ...
    if galaxy is not None:
        self._action_embeddings = load_action_embeddings_from_galaxy(galaxy)
    else:
        self._action_embeddings = ACTION_EMBEDDINGS
```

In `choose_action`, use `self._action_embeddings` instead of module-level `ACTION_EMBEDDINGS`.

---

## ORDER 4: Bootstrap Actions During Knowledgeverse Init

### 4A: Wire into existing bootstrap

Find where Reality Galaxy is initialized (likely in `Knowledgeverse.__init__` or a bootstrap function). Add:

```python
from knowledge3d.cranium.action_primitives_bootstrap import bootstrap_spatial_actions

# After Reality Galaxy is created:
bootstrap_spatial_actions(reality_galaxy, encode_embedding=True)
```

This ensures action primitives are always present in the Galaxy, available for any spatial reasoning — ARC-AGI-3, House navigation, physics simulation, etc.

---

## ORDER 5: Extend Option Slots to 7

### 5A: Modify `vram_task_buffer.py`

```python
# OLD:
OPTION_EMBEDDING_SLOTS = 4

# NEW:
OPTION_EMBEDDING_SLOTS = 7
```

Recalculate offsets (this changes `SUBJECT_ID_OFFSET` and `DOMAIN_HINT_ID_OFFSET`):

```
OPTION_EMBEDDINGS_OFFSET = 136
Each option = 32 floats × 4 bytes = 128 bytes
7 options × 128 bytes = 896 bytes
New SUBJECT_ID_OFFSET = 136 + 896 = 1032
New DOMAIN_HINT_ID_OFFSET = 1036
```

**WAIT**: This pushes past 1024 bytes per input slot. Two options:

**Option A**: Increase `INPUT_SLOT_BYTES` to 1280 (next aligned boundary).
**Option B**: Keep 4 movement options for now, add Perform/Click/Undo as a separate dispatch.

**Choose Option A** — increase to 1280 bytes. The buffer is ~7.3 MB for 6000 slots (vs ~6 MB before). Acceptable on a 12 GB card.

```python
INPUT_SLOT_BYTES = 1280  # was 1024
OPTION_EMBEDDING_SLOTS = 7
SUBJECT_ID_OFFSET = 136 + (7 * OPTION_EMBEDDING_BYTES)  # = 1032
DOMAIN_HINT_ID_OFFSET = 1036
```

### 5B: Update `device_functions.cuh`

```c
#define GPU_TASK_MAX_OPTIONS 7  // was 4
```

### 5C: Update `gpu_task_dispatch.cu`

The cosine loop already uses `bounded_options` which reads from the slot. Just ensure the loop bound is `min(option_count, 7u)` instead of `min(option_count, 4u)`.

### 5D: Update `cpu_reference_dispatch` in `gpu_task_dispatch.py`

Change `[:4]` to `[:7]` wherever option embeddings are sliced.

---

## ORDER 6: Test

### 6A: Extend `tests/test_arc3_agent.py`

Add tests for:
- Galaxy-sourced action embeddings vs displacement-based fallback
- 7-action option slot support
- Action atom metadata correctness
- Round-trip: bootstrap → Galaxy → embeddings → dispatch → correct action

### 6B: Run existing tests

```bash
pytest tests/test_arc3_agent.py tests/test_gpu_task_dispatch.py -v
```

All must pass. The 7-slot change affects ALL task types, so MMLU/Math/etc. tests must still work (they use ≤4 options, `bounded_options` handles it).

---

## FILE INVENTORY

Files you CREATE:
- `knowledge3d/cranium/action_primitives_bootstrap.py` — 7 action atoms + 1 composed diagonal
- `knowledge3d/knowledgeverse/action_embedding_loader.py` — Galaxy-to-embedding loader

Files you MODIFY:
- `knowledge3d/knowledgeverse/vram_task_buffer.py` — `INPUT_SLOT_BYTES=1280`, `OPTION_EMBEDDING_SLOTS=7`, recalc offsets
- `knowledge3d/cranium/cuda/device_functions.cuh` — `GPU_TASK_MAX_OPTIONS=7`
- `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` — update max options bound if hardcoded
- `knowledge3d/knowledgeverse/gpu_task_dispatch.py` — update `[:4]` slices to `[:7]`
- `benchmarks/arc_agi_3.py` — use Galaxy-sourced embeddings
- `tests/test_arc3_agent.py` — extend for 7-action and Galaxy embedding tests

Files you DO NOT TOUCH:
- `knowledge3d/cranium/cuda/arc3_frame_encoder.cu` — frame encoding is independent
- `scripts/run_gpu_benchmark.py` — works with any option count
- `scripts/run_arc3_agent.py` — I/O shell unchanged

---

## EXECUTION SEQUENCE

1. Create `action_primitives_bootstrap.py` with 7+1 action atoms
2. Create `action_embedding_loader.py`
3. Increase `INPUT_SLOT_BYTES` to 1280, `OPTION_EMBEDDING_SLOTS` to 7, recalc offsets in `vram_task_buffer.py`
4. Update `GPU_TASK_MAX_OPTIONS` to 7 in `device_functions.cuh`
5. Update option bounds in `gpu_task_dispatch.cu` and `gpu_task_dispatch.py`
6. Modify `benchmarks/arc_agi_3.py` to use Galaxy embeddings
7. Wire bootstrap into Knowledgeverse init (or add standalone bootstrap script)
8. Run tests: `pytest tests/ -v -k "arc3 or gpu_task"` → all green
9. Run synthetic benchmark: `python scripts/run_gpu_benchmark.py --suite synthetic --count 10` → 10/10

---

## SUCCESS CRITERIA

- Movement is a Galaxy star, not a hardcoded constant
- All 7 ARC-AGI-3 actions have `visual_rpn`, `behavior_rpn`, `law_rpn`
- Action embeddings are derived from Galaxy content (displacement + RPN programs)
- 7 option slots work for all task types (MMLU with 4 options still works)
- Diagonal composed action demonstrates Layer 3 composition
- Existing benchmark tests still pass
- Action primitives are reusable (`reusable_contexts` includes house_navigation, grid_world, physics_sim)

**Build it.**
