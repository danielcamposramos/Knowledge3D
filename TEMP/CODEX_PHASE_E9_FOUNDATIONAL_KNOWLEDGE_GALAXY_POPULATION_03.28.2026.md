# Codex — Phase E.9: Foundational Knowledge — Populate the Galaxy

**Date:** 2026-03-28
**From:** Daniel (Chair) + Claude (Architecture)
**To:** Codex
**Type:** IMPLEMENTATION ORDER
**Prerequisite:** Phase E.8 done. Galaxy VRAM table exists. Kernel reads from it, follows component_refs, sleep-time updates learnable stars. **BUT: the Galaxy has only 7 action stars + a few component references. It's nearly empty. Daniel: "start focusing on knowledge build inside the knowledgeverse as symlinked as possible (with maths and such — because procedural drawing is all maths)"**

---

## THE VISION

Daniel's foundational insight: **procedural drawing IS mathematics**. A Bézier curve is a polynomial. A rotation is trigonometry. A translation is vector addition. A line is a parametric equation. Drawing primitives and math symbols are the SAME KNOWLEDGE viewed from two angles — form and meaning.

The Galaxy VRAM table must contain a rich, cross-linked knowledge foundation where:
- Drawing primitives reference the math they embody
- Math symbols reference the visual forms they look like
- Action primitives reference the geometric concepts they perform
- Grammar rules reference the symbols and operations they transform
- Everything has meaning-centric surface forms (language-agnostic center)

**Per the specs:**
- **Foundational Knowledge Spec §1.1**: 4-layer architecture (Form → Meaning → Rules → Meta-Rules). Lower layers are CANONICAL — upper layers reference via symlinks, not duplication.
- **Dual Client Contract §1.6**: Save Information Principle. DON'T duplicate what exists. Use references.
- **MeaningCentricStar**: Already has `component_refs`, `visual_refs`, `grammar_refs`, `reality_refs`, `surface_forms` — the full symlink apparatus.
- **Foundational Drawing Bootstrap**: 48 entries already exist with symlink metadata to math_galaxy and character_galaxy. NOT loaded into VRAM yet.
- **Math Symbols Registry**: 1,121 symbols. Pure character lists — no RPN programs, no embeddings, no cross-references yet.

---

## THE PLAN: 3 Knowledge Populations

### Population A: Drawing-Math Foundation (~40 stars)
Core drawing primitives that ARE math, loaded with bidirectional cross-references.

### Population B: Math Operations + Symbols (~30 stars)
Essential math operations with visual_rpn (how the symbol looks) and behavior_rpn (what it computes).

### Population C: Spatial Reasoning Atoms (~20 stars)
Geometric concepts that underpin both ARC-AGI-3 and House navigation.

**Total: ~90 stars loaded into Galaxy VRAM table** (well within 256 max_stars). Each star is deeply cross-linked to others via component_refs.

---

## DO NOT:
- Import numpy, scipy, sympy, or any external math library
- Create new Python orchestration layers
- Change the kernel or sleep-time code (E.8 kernel is stable)
- Change the VRAM table layout (160-byte star records, unchanged)
- Cap or filter knowledge (per Daniel: "NEVER cap or limit knowledge loading")

## DO:
- Extend `build_arc3_galaxy_table()` into a richer `build_foundational_galaxy_table()`
- Load drawing primitives from existing `foundational_drawing_bootstrap.py`
- Create math operation stars with RPN programs AND visual glyphs
- Create spatial reasoning stars (symmetry, translation, rotation, reflection)
- Wire ALL cross-references as component_refs (Galaxy table indices)
- Give every star meaning-centric embedding (language-agnostic, semantically rich)
- Keep the first 7 slots as ARC3 action stars (backward compatible)

---

## ORDER 1: Create `knowledge3d/knowledgeverse/foundational_galaxy_builder.py`

This module builds the complete foundational Galaxy star table. It replaces `build_arc3_galaxy_table()` as the primary Galaxy loader.

### 1A: Architecture

```python
"""Build the foundational Galaxy — drawing, math, spatial, and action knowledge.

Per Foundational Knowledge Spec §1.1: 4-layer architecture with symlinks.
Per Dual Client Contract §1.6: Save Information Principle.

Drawing IS math. A Bézier curve is a polynomial. A rotation is trigonometry.
Every drawing primitive references the math it embodies.
Every math symbol references the visual form it looks like.
This IS the meaning-centric design.
"""
```

### 1B: Star Categories

Use `star_type` to classify:
- 0 = action (existing 7 ARC3 actions)
- 1 = drawing (visual primitives)
- 2 = character (glyphs, digits)
- 3 = grammar (transformation rules)
- 4 = reality (physics, spatial)
- 5 = math (operations, symbols)
- 6 = spatial_concept (geometric abstractions)

### 1C: Galaxy IDs

Hash galaxy names to uint32:
- `_fnv1a32("reality")` → existing reality galaxy
- `_fnv1a32("drawing")` → drawing galaxy
- `_fnv1a32("math")` → math galaxy
- `_fnv1a32("grammar")` → grammar galaxy
- `_fnv1a32("spatial")` → spatial concept galaxy

---

## ORDER 2: Population A — Drawing-Math Foundation Stars

Load these from the existing `foundational_drawing_bootstrap.py` data, enriched with cross-references.

**Key insight: each drawing primitive's RPN program IS its math definition.** The embedding should encode both the visual and mathematical meaning.

### 2A: Vector Operation Stars (10 stars)

From `_vector_primitives()` in `foundational_drawing_bootstrap.py`:

| Star | Name | RPN Program | Component Refs |
|------|------|-------------|----------------|
| vec2_add | 2D Vector Add | V1_X V2_X ADD V1_Y V2_Y ADD | [add_op, point_star] |
| vec2_sub | 2D Vector Subtract | V1_X V2_X SUB V1_Y V2_Y SUB | [sub_op, point_star] |
| vec2_dot | 2D Dot Product | V1_X V2_X MUL V1_Y V2_Y MUL ADD | [mul_op, add_op] |
| vec2_length | 2D Vector Length | VX VX MUL VY VY MUL ADD SQRT | [mul_op, add_op, sqrt_op] |
| vec2_normalize | 2D Vector Normalize | VX VY VEC2_NORM_DIV | [vec2_length, div_op] |
| vec3_add | 3D Vector Add | V1_X V2_X ADD ... V1_Z V2_Z ADD | [vec2_add, add_op] |
| vec3_cross | 3D Cross Product | A_Y B_Z MUL A_Z B_Y MUL SUB ... | [mul_op, sub_op] |
| vec3_normalize | 3D Vector Normalize | VX VY VZ VEC3_NORM_DIV | [vec2_normalize] |
| mat4_translate | Translation Transform | TX TY TZ MAT4_TRANSLATE | [vec3_add, move_up_action] |
| mat4_scale | Scale Transform | SX SY SZ MAT4_SCALE | [mul_op] |

**Embedding construction for drawing stars:**
Hash the RPN tokens (existing `_fnv1a32` method), PLUS encode the category semantically:
- dims 0-1: displacement/direction (from metadata)
- dim 2: category signature (vector_ops=0.8, curves=0.6, matrix_ops=0.4, etc.)
- dim 3: complexity (token count normalized)
- dims 4-7: cross-modal flags (symlink targets: math=1.0, character=0.8, etc.)
- dims 8-31: FNV-1a hash buckets from RPN tokens (existing method)

### 2B: Curve Stars (5 stars)

From `_bezier_primitives()`:

| Star | Name | RPN | Component Refs |
|------|------|-----|----------------|
| quadratic_bezier | Quadratic Bézier | T 1.0 SWAP SUB DUP MUL P0 MUL ... | [mul_op, add_op, pow_op] |
| cubic_bezier | Cubic Bézier | T 1.0 SWAP SUB DUP DUP MUL MUL P0 MUL ... | [quadratic_bezier, mul_op] |
| bezier_tangent | Bézier Tangent (derivative!) | CUBIC_BEZIER_DERIVATIVE | [cubic_bezier, derivative_op] |
| bezier_arc_length | Bézier Arc Length | CURVE_SAMPLE_32 POLYLINE_LENGTH | [vec2_length, add_op] |
| line_segment | Line Segment | P0 P1 LERP | [vec2_sub, mul_op] |

**Critical symlink:** `bezier_tangent` references `derivative_op` from Population B. Drawing-math cross-reference.

### 2C: Transform Stars (5 stars)

| Star | Name | RPN | Component Refs |
|------|------|-----|----------------|
| translate_2d | 2D Translation | TX TY VEC2_ADD | [vec2_add, move_up_action] |
| rotate_2d | 2D Rotation | THETA DEG2RAD ROT2D_MAT | [sin_op, cos_op] |
| scale_2d | 2D Scale | SX SY MAT2_SCALE | [mul_op] |
| reflect_x | Reflect over X axis | Y -1 MUL Y STORE | [mul_op, negate_op] |
| reflect_y | Reflect over Y axis | X -1 MUL X STORE | [mul_op, negate_op] |

**Critical symlinks:** `translate_2d` references `move_up_action` (action star index 0) — movement IS translation. `rotate_2d` references `sin_op` and `cos_op` from Population B.

---

## ORDER 3: Population B — Math Operation Stars

These stars represent mathematical operations. Each has:
- `visual_rpn`: how the symbol looks (glyph Bézier or Unicode reference)
- `behavior_rpn`: what it computes (RPN program)
- `surface_forms` encoded in embedding dims

### 3A: Arithmetic Operations (8 stars)

| Star | Symbol | Visual RPN | Behavior RPN | Surface Forms | Component Refs |
|------|--------|-----------|-------------|---------------|----------------|
| add_op | + | DRAW_CROSS_PLUS | A B ADD | en:add, pt:somar | [] |
| sub_op | − | DRAW_HLINE | A B SUB | en:subtract, pt:subtrair | [add_op] |
| mul_op | × | DRAW_CROSS_X | A B MUL | en:multiply, pt:multiplicar | [add_op] |
| div_op | ÷ | DRAW_FRACTION_BAR | A B DIV | en:divide, pt:dividir | [mul_op] |
| pow_op | ^ | DRAW_SUPERSCRIPT | BASE EXP POW | en:power, pt:potência | [mul_op] |
| sqrt_op | √ | DRAW_RADICAL | X SQRT | en:square root, pt:raiz quadrada | [pow_op] |
| negate_op | − | DRAW_HLINE | X -1 MUL | en:negate, pt:negar | [mul_op] |
| abs_op | \| | DRAW_VLINE_PAIR | X ABS | en:absolute value, pt:valor absoluto | [negate_op] |

### 3B: Trigonometric Operations (4 stars)

| Star | Symbol | Behavior RPN | Component Refs |
|------|--------|-------------|----------------|
| sin_op | sin | THETA SIN | [rotate_2d] |
| cos_op | cos | THETA COS | [rotate_2d] |
| pi_const | π | 3.14159265 | [sin_op, cos_op] |
| circle_const | ○ | R 2 MUL PI MUL (circumference) | [pi_const, mul_op] |

**Critical symlinks:** `sin_op` references `rotate_2d` from Population A — sin/cos ARE rotation. `circle_const` references `pi_const` AND `mul_op`.

### 3C: Calculus Concepts (4 stars)

| Star | Symbol | Behavior RPN | Component Refs |
|------|--------|-------------|----------------|
| derivative_op | ∂ | DX F_NEXT F_PREV SUB SWAP DIV | [sub_op, div_op] |
| integral_op | ∫ | DX F_SUM MUL | [add_op, mul_op] |
| sum_op | ∑ | N 0 DO I RECALL F_EVAL ADD LOOP | [add_op] |
| delta_op | ∆ | F_NEW F_OLD SUB | [sub_op] |

**Critical symlinks:** `derivative_op` references `sub_op` AND `div_op` — the derivative IS subtraction divided by infinitesimal. `bezier_tangent` (Population A) will reference this. Drawing↔Math circular link.

### 3D: Embedding Construction for Math Stars

```python
def _math_star_embedding(star_def: dict) -> list[float]:
    """Build semantically rich 32-float embedding for a math star.

    Per Foundational Knowledge Spec §1.2-1.4: embeddings encode
    BOTH visual form AND computational meaning.
    """
    embedding = [0.0] * 32

    # Dims 0-1: operational signature
    # (from behavior_rpn: what kind of operation is this?)
    behavior = str(star_def.get("behavior_rpn", ""))
    if "ADD" in behavior or "SUM" in behavior:
        embedding[0] = 0.8
    if "MUL" in behavior or "POW" in behavior:
        embedding[0] = -0.8
    if "SUB" in behavior or "DIV" in behavior:
        embedding[1] = -0.7
    if "SIN" in behavior or "COS" in behavior:
        embedding[1] = 0.9

    # Dim 2: star_type signature
    type_sigs = {5: 0.9, 1: 0.6, 4: 0.3, 3: -0.5, 6: -0.9}
    embedding[2] = type_sigs.get(star_def.get("star_type", 0), 0.0)

    # Dim 3: complexity (token count normalized)
    tokens = behavior.split()
    embedding[3] = min(1.0, len(tokens) / 10.0)

    # Dims 4-7: cross-modal flags
    embedding[4] = 1.0 if star_def.get("visual_rpn") else 0.0
    embedding[5] = 1.0 if star_def.get("component_refs") else 0.0
    embedding[6] = len(star_def.get("component_refs", [])) / 4.0
    embedding[7] = 1.0 if star_def.get("surface_forms") else 0.0

    # Dims 8-31: FNV-1a hash buckets from ALL RPN tokens
    # (visual + behavior + any referenced RPN)
    all_rpn = " ".join([
        str(star_def.get("visual_rpn", "")),
        behavior,
    ]).strip()
    for token in all_rpn.split():
        if token:
            bucket = 8 + (_fnv1a32(token) % 24)
            embedding[bucket] += 0.25

    return embedding
```

---

## ORDER 4: Population C — Spatial Reasoning Atoms

These are the geometric CONCEPTS that underpin spatial reasoning in ARC-AGI-3. They bridge action primitives and mathematical operations.

### 4A: Geometric Concept Stars (10 stars)

| Star | Concept | Meaning RPN | Component Refs |
|------|---------|------------|----------------|
| symmetry_x | X-axis symmetry | GRID FLIP_X COMPARE | [reflect_x] |
| symmetry_y | Y-axis symmetry | GRID FLIP_Y COMPARE | [reflect_y] |
| symmetry_rotate | Rotational symmetry | GRID 90 ROTATE COMPARE | [rotate_2d] |
| translation_concept | Spatial translation | OBJECT DX DY TRANSLATE | [translate_2d, vec2_add] |
| pattern_repeat | Pattern repetition | TILE OFFSET COPY | [translation_concept, add_op] |
| adjacency | Spatial adjacency | CELL NEIGHBOR_4 CHECK | [vec2_sub, abs_op] |
| containment | Spatial containment | INNER OUTER BOUNDS_CHECK | [vec2_sub, abs_op] |
| color_fill | Region fill | SEED COLOR FLOOD_FILL | [adjacency] |
| grid_cell | Grid cell concept | ROW COL CELL_INDEX | [mul_op, add_op] |
| boundary | Boundary detection | CELL NEIGHBOR_4 DIFF_COUNT | [adjacency, sub_op] |

### 4B: ARC-Specific Reasoning Stars (5 stars)

| Star | Concept | Meaning RPN | Component Refs |
|------|---------|------------|----------------|
| object_detect | Detect contiguous object | GRID CONNECTED_COMPONENTS | [adjacency, color_fill] |
| transform_detect | Detect transformation | FRAME_A FRAME_B DIFF CLASSIFY | [delta_op, symmetry_x] |
| goal_infer | Infer goal from examples | EXAMPLES PATTERN_EXTRACT GENERALIZE | [pattern_repeat, transform_detect] |
| action_evaluate | Evaluate action outcome | STATE ACTION APPLY SCORE | [delta_op, translation_concept] |
| explore_strategy | Exploration vs exploitation | HISTORY DIVERSITY_SCORE THRESHOLD COMPARE | [sum_op, div_op] |

**Critical symlinks:** `transform_detect` references `delta_op` (from Population B, calculus!) and `symmetry_x` (from Population C). ARC-AGI reasoning IS calculus applied to grids.

---

## ORDER 5: Wire All Cross-References

After all stars are created, resolve component_refs as Galaxy table indices. This is the critical step — it makes the symlink pattern REAL for the kernel.

```python
def build_foundational_galaxy_table(
    galaxy: Any | None = None,
) -> list[dict[str, Any]]:
    """Build the complete foundational Galaxy.

    Per Foundational Knowledge Spec §1.1: 4-layer cross-linked architecture.

    Star layout:
      [0-6]:   ARC3 action stars (backward compatible)
      [7-26]:  Drawing-Math foundation (Population A)
      [27-48]: Math operations + symbols (Population B)
      [49-68]: Spatial reasoning atoms (Population C)
      [69+]:   Component stars (walk_to, reach, etc.)
    """
    stars = []
    id_to_index = {}

    # Phase 1: ARC3 actions (indices 0-6, same as build_arc3_galaxy_table)
    # ... existing code ...

    # Phase 2: Drawing-Math foundation
    for drawing_entry in _drawing_math_stars():
        idx = len(stars)
        id_to_index[drawing_entry["id"]] = idx
        stars.append(drawing_entry)

    # Phase 3: Math operations
    for math_entry in _math_operation_stars():
        idx = len(stars)
        id_to_index[math_entry["id"]] = idx
        stars.append(math_entry)

    # Phase 4: Spatial reasoning atoms
    for spatial_entry in _spatial_reasoning_stars():
        idx = len(stars)
        id_to_index[spatial_entry["id"]] = idx
        stars.append(spatial_entry)

    # Phase 5: Resolve all component_refs to table indices
    for star in stars:
        ref_ids = star.pop("_ref_ids", [])
        star["component_refs"] = [
            id_to_index[ref_id]
            for ref_id in ref_ids
            if ref_id in id_to_index
        ][:4]

    return stars
```

---

## ORDER 6: Update Agent to Use Foundational Galaxy

### 6A: Update `arc_agi_3.py`

Replace `build_arc3_galaxy_table()` with `build_foundational_galaxy_table()`:

```python
from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table

class K3DARC3Agent:
    def __init__(self, ...):
        # ...
        self.galaxy_table = GalaxyVRAMTable(max_stars=256)
        galaxy_stars = build_foundational_galaxy_table(galaxy=self.reality_galaxy)
        self.galaxy_table.load_stars(galaxy_stars)
```

The first 7 stars are still ARC3 actions. The kernel reads these for action scoring. But now the kernel ALSO has 80+ additional stars to compose richer embeddings from when following component_refs.

### 6B: Update `galaxy_vram_table.py`

Keep `build_arc3_galaxy_table()` for backward compatibility but mark it deprecated. Add import/re-export of `build_foundational_galaxy_table`.

---

## ORDER 7: Tests

### 7A: Existing tests pass
```bash
pytest tests/test_arc3_agent.py tests/test_gpu_task_dispatch.py tests/test_vram_task_buffer.py tests/test_galaxy_vram_table.py -v
```

### 7B: New test — foundational Galaxy population
Add tests to `tests/test_galaxy_vram_table.py`:
1. `test_foundational_galaxy_population_count`:
   - Call `build_foundational_galaxy_table()`
   - Assert star_count >= 60 (at least 7 actions + 20 drawing + 16 math + 15 spatial + component stars)
   - Assert star_count <= 256 (within table capacity)

2. `test_foundational_cross_references`:
   - Build table → for each star with component_refs, verify each ref index < star_count
   - Verify no STAR_NULL_REF in active refs
   - Verify at least 20 stars have non-empty component_refs (symlink coverage)

3. `test_drawing_math_symlink`:
   - Find `bezier_tangent` star → verify it has component_ref to `derivative_op`
   - Find `translate_2d` star → verify it has component_ref to an action star (index < 7)
   - These test the drawing↔math cross-link pattern

4. `test_spatial_reasoning_refs`:
   - Find `symmetry_x` → verify component_ref to `reflect_x`
   - Find `transform_detect` → verify component_ref to `delta_op`
   - These test the spatial↔math cross-link pattern

5. `test_embedding_semantic_differentiation`:
   - Build all star embeddings
   - Compute cosine similarity between add_op and sub_op → should be moderate (related but different)
   - Compute cosine similarity between add_op and sin_op → should be low (different domains)
   - Compute cosine similarity between vec2_add and add_op → should be high (same underlying operation)

### 7C: Benchmark regression
```bash
# Run existing offline benchmark — must still hold
pytest tests/test_phase_e_runners.py -v
```

---

## FILE INVENTORY

Files you CREATE:
- `knowledge3d/knowledgeverse/foundational_galaxy_builder.py` — complete foundational Galaxy builder

Files you MODIFY:
- `knowledge3d/knowledgeverse/galaxy_vram_table.py` — re-export `build_foundational_galaxy_table`
- `benchmarks/arc_agi_3.py` — use `build_foundational_galaxy_table` instead of `build_arc3_galaxy_table`
- `tests/test_galaxy_vram_table.py` — add foundational population tests

Files you DO NOT TOUCH:
- `knowledge3d/cranium/cuda/*` — kernel unchanged, reads same VRAM table format
- `knowledge3d/knowledgeverse/persistent_brain.py`
- `knowledge3d/knowledgeverse/sleep_time_micro.py`
- `knowledge3d/knowledgeverse/foundational_drawing_bootstrap.py` — read from it, don't modify
- `knowledge3d/cranium/math_symbols_registry.py` — reference it, don't modify
- `knowledge3d/cranium/action_primitives_bootstrap.py` — still builds RealityAtoms, unchanged

---

## EXECUTION SEQUENCE

1. Create `foundational_galaxy_builder.py` with all 3 populations
2. Wire cross-references (component_refs as table indices)
3. Build embeddings for all stars (semantic, hash-based, cross-modal)
4. Update `arc_agi_3.py` to use foundational Galaxy
5. Update `galaxy_vram_table.py` to re-export
6. Add tests → all green
7. Run offline benchmark → Synthetic 10/10, MMLU >= 30%, ARC3 20/20
8. Report: total star count, cross-reference count, embedding quality metrics

---

## SUCCESS CRITERIA

- **Star count**: >= 60, <= 256
- **Cross-reference density**: >= 20 stars have non-empty component_refs
- **Drawing↔Math links**: at least 5 drawing stars reference math operation stars
- **Math↔Drawing links**: at least 3 math stars reference drawing/visual stars
- **Spatial↔Math links**: at least 5 spatial stars reference math operation stars
- **Action↔Spatial links**: at least 3 action stars reference spatial concept stars (via transitive component_refs)
- **Embedding differentiation**: cosine(related_stars) > cosine(unrelated_stars)
- **Backward compatible**: ARC3 actions still at indices 0-6, all benchmarks hold
- **No Python hot path changes**: kernel reads same VRAM table format

## WHAT THIS MEANS

When the kernel reasons about "move_up" (star 0), it follows component_refs to:
- `translate_2d` → which references `vec2_add` → which references `add_op`
- The swarm now has 4 levels of semantic context: action → spatial concept → drawing primitive → math operation

When sleep-time strengthens "move_up" after a successful ARC action, the nudge propagates: the Galaxy table entry at index 0 shifts toward the reasoning state that ALSO encoded translate_2d + vec2_add + add_op context.

Over many ARC games, the action stars evolve embeddings that encode WHICH GEOMETRIC OPERATIONS work in which situations. The Galaxy learns. The meaning-centric design EVOLVES through sleep cycles.

**This is knowledge, not code. Build the Galaxy.**
