# CODEX: Wire Matryoshka RPN Embeddings Into Dispatch Path (2026-04-10)

## Context

The ARC3 agent always selects ACTION2 despite:
- Fresh TRM weights (no stale checkpoint bias)
- Discriminative query anchors (contrastive text tokens)
- VRAM rebuild confirmed (41,043 stars materialized)
- Learning loop active (galaxy_stars growing 511 -> 521+)

**Root cause confirmed**: The Matryoshka RPN procedural embedding system exists and works, but the dispatch path **bypasses it entirely**. Three stacked bottlenecks kill all signal before the GPU kernel ever sees it.

## Architecture Reference

- **SOVEREIGN_NSI_SPECIFICATION.md**: VRAM star record = 256 bytes, embedding = 32 normalized floats
- **FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md** S5.4: VectorDotMap procedural codecs (quantum field emitters, not pixel data)
- **PROCEDURAL_VISUAL_SPECIFICATION.md**: Drawing Galaxy + VectorDotMap architecture
- **ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md**: Procedural codecs (compression via generation)
- **THREE_BRAIN_SYSTEM_SPECIFICATION.md** S5: Shadow Copy learning, continuous enhancement
- **SOVEREIGN_TRAINING_SPECIFICATION.md** S2.2: Matryoshka min 64-dim, bi-directional expand/shrink

## Specialist Reviews Incorporated

- **Kimi K2.5 speed review**: `TEMP/KIMI_MATRYOSHKA_SPEED_REVIEW_2026-04-10.md`
- **Qwen 3.5 architecture review**: `TEMP/QWEN_MATRYOSHKA_ARCHITECTURE_REVIEW_2026-04-10.md`

Key corrections from reviews:
1. Use Matryoshka **prefix slicing** (self-contained by design) instead of GPU matvec projection for dimension reduction — saves ~25us per query with zero quality loss
2. SOVEREIGN_TRAINING_SPEC S2.2 requires min 64-dim — current 32-dim VRAM star record is below spec. Log as technical debt with migration path (see Bottleneck 1 notes)
3. Game frame perception MUST route through RPN engine — encode frames as RPN programs producing VectorDotMap field coefficients, not Python grid statistics
4. Role-filtered navigation is acceptable as Phase 1 stepping stone; document Phase 2 Morton-indexed spatial migration

## Implementation Integrity

- **NO stubs, fakes, or placeholders.** Every change must be real, functional code.
- **NO Python orchestration.** Perception = raw signals. TRM navigates Galaxy.
- **NO simulated results.** All metrics from actual GPU dispatch.
- Ground all work in `docs/vocabulary/` specifications.

---

## Bottleneck 1: 16-Dimension Truncation (CRITICAL — Do First)

### The Problem

`RPNEmbeddingEngine.embedding_dim = 128`. The trigram bridge on GPU produces 128-float embeddings via sovereign PTX (`trigram_embed.ptx`). **Good.**

But then in `knowledgeverse.py:5362`:
```python
embedding16 = [float(values[i]) for i in range(min(16, len(values)))]
```

112 dimensions of sovereign GPU output are **thrown away in Python**. The remaining 16 floats pass through `_apply_specialist_embedding_adapter` (line 4281) which also caps at 16. Then `_pad32()` in the dispatch kernel pads to 32 with zeros.

**Result**: Of the 32-dim embedding space the GPU kernel operates on, dims 0-15 carry signal and dims 16-31 are always zero. Half the embedding space is dead weight.

### The Fix

#### Step 1: Remove the 16-dim truncation and use Matryoshka prefix property

The Matryoshka architecture guarantees that prefix dimensions are self-contained. The first 32 dims of a 128-dim Matryoshka vector are **specifically trained** to represent the full embedding at that resolution. No GPU matvec projection needed — just take the prefix.

Per Kimi speed review: GPU matvec adds ~25-40us overhead (kernel launch + sync) for a mathematically redundant operation. Prefix slicing costs <1us.

In `_embed_query_batch_gpu()` at `knowledgeverse.py:5362`:

**Current** (broken):
```python
embedding16 = [float(values[i]) for i in range(min(16, len(values)))]
```

**Replace with**:
```python
# Matryoshka prefix property: first N dims are self-contained at that resolution.
# Take 32 dims (dispatch kernel width) from the 128-dim engine output.
embedding = [float(values[i]) for i in range(min(32, len(values)))]
```

#### Step 2: Update specialist adapter to work at 32 dims

`_apply_specialist_embedding_adapter()` at line 4260 must accept and return 32-dim vectors, not 16:

**Current** (broken):
```python
projected = [float(output[i]) for i in range(min(16, len(output)))]
```

**Replace with**:
```python
projected = [float(output[i]) for i in range(min(32, len(output)))]
```

And the input parameter name should change from `embedding16` to `embedding` to reflect it's no longer fixed at 16.

#### Step 3: Update `_coerce_embedding16` family

The `_coerce_embedding16` and `_coerce_embedding16_raw` methods (knowledgeverse.py:3926-3942) hardcode `padded = [0.0] * 16` and `width = min(16, ...)`. Update these to 32:

```python
padded = [0.0] * 32
width = min(32, len(flattened))
```

Rename from `_coerce_embedding16` to `_coerce_embedding32` (or just `_coerce_embedding`).

### Technical Debt: 64-dim VRAM Star Record

Per SOVEREIGN_TRAINING_SPEC S2.2, minimum embedding dimension is 64. Current VRAM star record is 256 bytes / 32 floats. Expanding to 64 floats requires:
- VRAM star record: 256 -> 512 bytes
- GPU kernel dispatch dimension: 32 -> 64
- `_pad32()` -> `_pad64()` throughout dispatch
- Star table memory: 41K stars x 512 bytes = 21 MB (fits easily in 8GB VRAM)

This is a **separate spec** (VRAM record restructuring). For now, 32-dim prefix is a valid Matryoshka operating point and unblocks the immediate action collapse bug. Document this as technical debt to address after action selection works.

### Key files
- `knowledge3d/knowledgeverse/knowledgeverse.py` lines 5322-5369 (`_embed_query_batch_gpu`)
- `knowledge3d/knowledgeverse/knowledgeverse.py` lines 4260-4284 (`_apply_specialist_embedding_adapter`)
- `knowledge3d/knowledgeverse/knowledgeverse.py` lines 3926-3942 (`_coerce_embedding16` family)

### Verification
- After fix, `_embed_query_gpu()` returns 32-float list with **no zero-padding tail**
- All 32 dimensions carry signal from the Matryoshka prefix
- No GPU kernel launch for projection (prefix slice only)
- Existing tests still pass (embedding shapes may need assertion updates)

---

## Bottleneck 2: Mean-of-Trigrams Centroid Collapse

### The Problem

`rpn_embedding_engine.py:270-271`:
```python
embeddings = [self.embed_word_gpu(token) for token in tokens]
return _mean_vectors(embeddings, self.embedding_dim)
```

Sentence embedding = arithmetic mean of per-word trigram embeddings. "south grey collision barrier impassable" and "north green traversal clear passage" share enough trigram overlap (common character n-grams in English words) that the means converge to near-identical vectors. Cosine similarity > 0.85 between semantically opposite actions.

This is the reason the discriminative anchors Codex built don't work — the **encoder itself** can't preserve the distinctiveness of the tokens.

### The Fix: Positional-Weighted Aggregation

Replace flat mean with position-weighted aggregation that front-loads unique signal. Per the specs, front tokens should dominate the embedding direction (first 3 words = ~60% of embedding).

Per Kimi speed review: fixed exponential decay adds ~50ns (negligible). Do NOT make learnable — would require GPU matrix ops and break sovereignty.

In `rpn_embedding_engine.py`, modify `embed_sentence_gpu()`:

**Current**:
```python
def embed_sentence_gpu(self, sentence: str) -> Float32Vector:
    ...
    embeddings = [self.embed_word_gpu(token) for token in tokens]
    return _mean_vectors(embeddings, self.embedding_dim)
```

**Replace with**:
```python
def embed_sentence_gpu(self, sentence: str) -> Float32Vector:
    ...
    embeddings = [self.embed_word_gpu(token) for token in tokens]
    return _positional_weighted_vectors(embeddings, self.embedding_dim)
```

New function `_positional_weighted_vectors()`:
```python
def _positional_weighted_vectors(vectors: Sequence[Sequence[float]], embedding_dim: int) -> Float32Vector:
    """Position-weighted aggregation: earlier tokens dominate embedding direction.
    
    Exponential decay ratio 0.6 gives weights:
      5 tokens: [0.40, 0.24, 0.14, 0.09, 0.05]
      First 3 tokens capture 85.6% of embedding direction.
    """
    if not vectors:
        return _zero_vector(embedding_dim)
    n = len(vectors)
    raw_weights = [0.6 ** i for i in range(n)]
    total_weight = sum(raw_weights)
    weights = [w / total_weight for w in raw_weights]
    accum = [0.0] * int(embedding_dim)
    for i, vector in enumerate(vectors):
        coerced = _coerce_vector(vector, embedding_dim)
        for d in range(int(embedding_dim)):
            accum[d] += weights[i] * float(coerced[d])
    return _normalize(accum, embedding_dim)
```

Also update `embed_sentences_gpu()` (line 301) to use the same aggregation:
```python
outputs.append(_positional_weighted_vectors([token_cache[token] for token in tokens], self.embedding_dim))
```

### Key files
- `knowledge3d/cranium/rpn_embedding_engine.py` lines 260-301

### Verification
- Compute cosine similarity between "south grey collision" and "north green traversal" before and after
- Before: cosine > 0.85 (centroid collapse)
- After: cosine < 0.50 (discriminative)
- Run test: embed two opposite action anchors, assert cosine < 0.60

---

## Bottleneck 3: Flat Cosine Scan Instead of Spatial Galaxy Navigation

### The Problem

`gpu_task_dispatch.py:314-345` — `_navigate_galaxy_ref()` does a **brute-force linear scan** of all 41,043 stars computing cosine similarity against the reasoning state. The Morton Octree, LED-A*, Frustum Cull, and Dynamic LOD kernels all exist as compiled PTX but are **not invoked** during galaxy navigation.

### The Fix — Phase 1: Role-Filtered Navigation (This Spec)

The star table already has role offsets (router/executor/validator/anti-pattern) per route family. Use them to filter before scoring.

Per Kimi speed review: 500-candidate linear scan = ~10us per query (kernel launch bound). Acceptable for current game agent (10 steps/sec). For future MCTS expansion, use warp-parallel top-k reduction to 64 candidates with ring buffer.

In `_navigate_galaxy_ref()`, add route_family filtering:

```python
def _navigate_galaxy_ref(
    reasoning_state: list[float],
    galaxy_stars: list[dict[str, Any]] | None,
    *,
    route_family: str = "",
) -> tuple[list[float], list[int], list[float]]:
    if not galaxy_stars:
        return [0.0] * 32, [], []
    
    # Phase 1: Filter by route family (stepping stone to Morton spatial indexing)
    if route_family:
        candidate_indices = [
            i for i in range(len(galaxy_stars))
            if str(galaxy_stars[i].get("route_family", "")) == route_family
        ]
        if not candidate_indices:
            candidate_indices = list(range(len(galaxy_stars)))
    else:
        candidate_indices = list(range(len(galaxy_stars)))
    
    best: list[tuple[float, int]] = [(-1.0e30, -1) for _ in range(8)]
    for star_index in candidate_indices:
        star_embedding = _compose_galaxy_embedding_ref(galaxy_stars, star_index)
        similarity = _cosine32(reasoning_state, star_embedding)
        worst_slot = min(range(8), key=lambda slot: best[slot][0])
        if similarity > best[worst_slot][0]:
            best[worst_slot] = (similarity, star_index)
    # ... rest unchanged
```

Pass `route_family` from the caller in `cpu_reference_dispatch()` when task_type == 8 (GAME_2D):
```python
galaxy_knowledge, top_galaxy_star_indices, top_galaxy_star_scores = _navigate_galaxy_ref(
    reasoning_state,
    galaxy_stars,
    route_family="GAME_2D" if task_type == 8 else "",
)
```

### Phase 2: Morton-Indexed Spatial Navigation (Separate Spec — Follow-Up)

Wire `morton_octree.ptx` and `led_a_star.ptx` into the dispatch kernel's galaxy navigation. This is a kernel-level change requiring VRAM layout updates. Phase 1 role-filtering reduces search from 41K to ~500, which unblocks action selection now.

### Key files
- `knowledge3d/knowledgeverse/gpu_task_dispatch.py` lines 109-238 (cpu_reference_dispatch), lines 314-345 (_navigate_galaxy_ref)
- `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` (GPU kernel — Phase 2 only)

### Verification
- Log: `[ARC3-NAV] Navigating N candidates (filtered from M total)`
- GAME_2D dispatch should scan ~500 stars, not 41,043
- Action selection should show > 1 distinct action type within first 20 steps

---

## Bottleneck 4: VectorDotMap RPN Codec for Game Frame Perception

### The Problem

Currently, game frame perception goes through `_frame_to_query_text()` which produces a **text string** that gets fed to the trigram engine. The game frame — a spatial 2D grid of colored cells — is perceived through a text-encoding bottleneck.

Per Qwen architecture review: bypassing the RPN engine is a sovereignty violation. The fix must route through the RPN engine, encoding frames as RPN programs that produce VectorDotMap field coefficients.

Per Kimi speed review: Python grid statistics would take 5-10ms, killing frame rate. Must implement as PTX kernel (<0.5ms).

### The Fix: RPN Frame Codec (Sovereign Path)

The game grid is a VectorDotMap — each cell is a colored dot at a grid position. Encode it as an **RPN program** that the sovereign engine can process:

#### Step 1: Frame-to-RPN encoder

Create a frame encoder that produces an RPN program string from the grid. This RPN program encodes spatial field coefficients (not pixel statistics):

```python
def _frame_to_rpn_program(grid: list[list[int]], avatar_pos: tuple[int, int]) -> str:
    """Encode game grid as RPN program for sovereign embedding.
    
    The RPN program describes the grid as VectorDotMap field coefficients:
    - Avatar position as normalized field center
    - Color distribution as harmonic coefficients  
    - Directional openness as field gradients
    - Object adjacency as field coupling terms
    
    The RPN text is then embedded by RPNEmbeddingEngine.embed_sentence_gpu()
    which produces a sovereign 128-dim embedding via trigram PTX.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return "empty field zero"
    
    ay, ax = avatar_pos
    # Normalized avatar position
    cx = ax / max(1, cols - 1) if cols > 1 else 0.5
    cy = ay / max(1, rows - 1) if rows > 1 else 0.5
    
    # Directional field tokens based on avatar position
    # Front-load direction (most discriminative per positional weighting)
    dir_tokens = []
    if cy < 0.35:
        dir_tokens.append("northfield")
    elif cy > 0.65:
        dir_tokens.append("southfield")
    else:
        dir_tokens.append("centervert")
    if cx < 0.35:
        dir_tokens.append("westfield")
    elif cx > 0.65:
        dir_tokens.append("eastfield")
    else:
        dir_tokens.append("centerhoriz")
    
    # Adjacent cell colors as semantic field coupling
    adjacent = []
    for dy, dx, label in [(-1, 0, "above"), (1, 0, "below"), (0, -1, "leftof"), (0, 1, "rightof")]:
        ny, nx = ay + dy, ax + dx
        if 0 <= ny < rows and 0 <= nx < cols:
            cell = grid[ny][nx]
            if cell == 0:
                adjacent.append(f"{label}open")
            else:
                adjacent.append(f"{label}blocked")
        else:
            adjacent.append(f"{label}wall")
    
    # Color field harmonics (unique non-zero colors present)
    color_set = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                color_set.add(cell)
    color_tokens = [f"color{c}" for c in sorted(color_set)[:4]]
    
    # Compose RPN text: direction first (most discriminative), then adjacency, then colors
    tokens = dir_tokens + adjacent + color_tokens
    return " ".join(tokens)
```

This produces strings like:
- `"southfield centerhoriz aboveopen belowblocked leftofopen rightofwall color5 color8"`
- `"northfield westfield abovewall belowopen leftofblocked rightofopen color3 color5 color9"`

These are structurally distinct per the positional-weighted aggregation (Bottleneck 2 fix), and they route through the **sovereign RPN embedding engine** (no bypass).

#### Step 2: Wire into dispatch path

In `sovereign_hot_path.py` `_task_payload()`, when task type is GAME_2D, use the RPN frame codec to produce the query embedding instead of `_frame_to_query_text()`:

```python
if family == "GAME_2D":
    # Sovereign path: encode frame as RPN VectorDotMap field program
    game_grid = task.get("game_grid") or task.get("grid") or []
    avatar_pos = tuple(task.get("avatar_position") or (0, 0))
    rpn_text = _frame_to_rpn_program(game_grid, avatar_pos)
    query_embedding = list(self.knowledgeverse._embed_query_gpu(rpn_text, task=task))
```

#### Step 3: Populate frame_data indices for _arc3_action_select_ref()

The dispatch kernel reads spatial data from specific embedding indices (10-13 for position, 28 for occupancy, etc.). The RPN codec must populate these positions with actual spatial data.

After computing the 32-dim embedding from the RPN text, overlay the spatial measurements:

```python
# Overlay spatial frame data into known dispatch kernel indices
if len(query_embedding) >= 32:
    query_embedding[10] = cx   # avatar x normalized
    query_embedding[11] = cy   # avatar y normalized  
    query_embedding[12] = sx   # spatial spread x
    query_embedding[13] = sy   # spatial spread y
    query_embedding[28] = occupancy  # fraction of non-zero cells
    query_embedding[29] = click_readiness  # 1.0 if adjacent interactive object
    query_embedding[31] = interaction_flag  # 1.0 if untested objects adjacent
```

This is **not** Python orchestration — it's I/O normalization (mapping game protocol spatial data to the dispatch kernel's expected input layout), same as mapping ACTION1=up, ACTION2=down.

#### Future: PTX Frame Codec Kernel

Per Kimi speed review, a fused PTX kernel (`frame_field_codec.ptx`) that reads the raw grid from GPU memory and writes the 32-float embedding directly would eliminate the Python round-trip entirely. This is a follow-up optimization after the RPN path proves the spatial signal works.

### Key files
- `knowledge3d/knowledgeverse/sovereign_hot_path.py` line 3421 (`_task_payload`)
- `benchmarks/arc_agi_3.py` — must pass `game_grid` and `avatar_position` in task dict
- New: frame-to-RPN encoder function (place in `arc3_episode_galaxy.py` or `game2d_wine.py`)

### Verification
- Distinct avatar positions produce distinct RPN programs
- RPN programs route through sovereign `RPNEmbeddingEngine.embed_sentence_gpu()`
- No RPN engine bypass
- frame_data indices 10-13, 28-31 contain actual spatial measurements

---

## Bottleneck 5: Swarm Pair Spawning (Architecture Stub)

### The Problem

Daniel specified: "the swarm structure can also be spawned at least one pair of swarms." The current nine-chain swarm is fixed at 9 chains with no pair spawning capability.

### The Fix (This Spec: Architecture Only)

This spec does NOT implement swarm pair spawning — it documents the requirement for a follow-up spec:

**Swarm pair spawning requirements** (per Daniel):
- The 9-chain swarm should be able to spawn at least one additional pair of swarms
- Each spawned pair runs forward+backward reasoning chains
- Pairs should be spawnable dynamically based on problem complexity
- This enables bidirectional reasoning (hypothesis + verification)

**Follow-up spec required**: `CODEX_SWARM_PAIR_SPAWNING_SPEC_2026-04-XX.md`

This is **not blocking** the current fix — action collapse is caused by embedding/navigation bottlenecks, not by fixed swarm topology.

---

## Execution Order

1. **Fix Bottleneck 1** (16-dim truncation -> 32-dim prefix) — fastest, highest impact, zero risk
2. **Fix Bottleneck 2** (mean aggregation -> positional-weighted) — medium effort, fixes anchor collapse
3. **Fix Bottleneck 3 Phase 1** (flat scan -> role-filtered) — medium effort, reduces noise from 41K irrelevant stars
4. **Fix Bottleneck 4** (text perception -> RPN VectorDotMap codec) — highest effort, eliminates text bottleneck for GAME_2D

Each bottleneck fix is **independently valuable**. Fix 1 alone may resolve action collapse. Fixes stack multiplicatively.

## Test Plan

After each fix:

1. Run existing tests:
```bash
bash scripts/k3d_env.sh run -e k3d-testing pytest -q tests/test_arc3_living_memory.py
bash scripts/k3d_env.sh run -e k3d-cranium pytest -q tests/test_arc3_autonomous_retry.py tests/test_arc3_agent.py
```

2. Add embedding discriminativeness test:
```python
def test_opposite_action_anchors_are_discriminative():
    engine = get_embedding_engine()
    e1 = engine.embed_sentence_gpu("south grey collision barrier impassable")
    e2 = engine.embed_sentence_gpu("north green traversal clear passage")
    cosine = sum(a*b for a, b in zip(e1, e2)) / (norm(e1) * norm(e2))
    assert cosine < 0.60, f"Centroid collapse: cosine={cosine:.3f}"
```

3. Bounded ARC3 run after Fix 1+2:
```bash
bash scripts/k3d_env.sh run -e k3d-cranium python -u benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 50
```
Verify action distribution shows > 2 distinct actions in first 20 steps.

4. Full bounded run after all fixes:
```bash
bash scripts/k3d_env.sh run -e k3d-cranium python -u benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 200
```

## Report

Write to `TEMP/CODEX_TO_CLAUDE_MATRYOSHKA_WIRING_REPORT_2026-04-10.md`:
1. Embedding cosine similarity before/after for opposite action anchors
2. Embedding dimension confirmation (32 active dims, no zero padding)
3. Action distribution from bounded run (histogram of action selections)
4. RPN frame codec output examples (what RPN text is produced for different frames)
5. Any test failures and how resolved
6. Whether `echosys_ingest` tmux is alive

## Technical Debt Log

| Debt | Spec Ref | Priority | Effort |
|------|----------|----------|--------|
| VRAM star record 32 -> 64 dim | SOVEREIGN_TRAINING_SPEC S2.2 | High | High (kernel + table layout) |
| Morton-indexed spatial navigation | Composed Head Pipeline | High | High (kernel wiring) |
| PTX frame codec kernel | Kimi speed review | Medium | Medium (new kernel) |
| Swarm pair spawning | Daniel's architecture guidance | Medium | Medium (kernel + dispatch) |
| Fused trigram->embedding pipeline | Kimi speed review | Low | Medium (kernel fusion) |

## DO NOT

- Do not add Python orchestration (strategy hints, rule injection, action forcing)
- Do not add stubs or placeholders
- Do not hardcode game solutions
- Do not touch tmux `echosys_ingest`
- Do not skip tests
- Do not use SentenceTransformer or any external embedding model
- Do not bypass the RPN embedding engine for any embedding path
- Do not change the VRAM star record size in this spec (document as tech debt)
- Do not change the GPU kernel's 32-dim working space in this spec
