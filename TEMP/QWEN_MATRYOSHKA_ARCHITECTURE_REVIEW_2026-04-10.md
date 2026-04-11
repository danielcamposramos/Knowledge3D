# ARCHITECTURAL CORRECTNESS REVIEW
## Codex: Wire Matryoshka RPN Embeddings Into Dispatch Path

**Review Date**: 2026-04-10  
**Reviewer**: Sovereignty Compliance Audit  
**Status**: ⚠️ **MULTIPLE SOVEREIGNTY VIOLATIONS DETECTED**

---

## Executive Summary

| Concern | Status | Severity | Spec Reference |
|---------|--------|----------|----------------|
| 1. Matryoshka 128→32 projection | ❌ VIOLATION | Critical | SOVEREIGN_TRAINING_SPEC.md S2.2 |
| 2. Procedural frame RPN bypass | ❌ VIOLATION | Critical | SOVEREIGN_NSI_SPEC.md |
| 3. VectorDotMap vs Python statistics | ❌ VIOLATION | High | FOUNDATIONAL_KNOWLEDGE_SPEC.md S5.4 |
| 4. Role-filtered navigation | ⚠️ DEAD-END | Medium | Composed Head Pipeline |
| 5. Nine-chain swarm pairing | ⚠️ MISSING | Medium | Daniel's swarm guidance |
| 6. VectorDotMap game perception | ❌ VIOLATION | High | PROCEDURAL_VISUAL_SPEC.md |

**Overall Assessment**: This Codex fixes real bottlenecks but introduces **4 sovereignty violations** and **2 architectural dead-ends**. Cannot merge without corrections.

---

## 1. Matryoshka Dimension Projection (128→32) ❌ SOVEREIGNTY VIOLATION

### The Violation

**Codex Spec**:
```python
# Projects 128-dim → 32-dim for dispatch
projected = matryoshka.base.project_vector(full_embedding, 32)
```

**SOVEREIGN_TRAINING_SPECIFICATION.md S2.2** (referenced in Codex architecture):
> "Matryoshka min 64-dim, bi-directional expand/shrink"

**VRAM Star Record Constraint**:
- Current: 256 bytes = 32 floats
- Required for sovereignty: 512 bytes = 64 floats

### Why This Matters

The 32-dim target violates the **minimum embedding dimension** established in sovereign training. This creates:

1. **Information loss**: 128→32 loses 75% of embedding space vs 128→64 (50% loss)
2. **Training/inference mismatch**: Model trained on 64-dim minimum, inference at 32-dim
3. **Future expansion blocked**: Cannot expand to 64+ without VRAM record restructuring

### Required Fix

**Option A (Recommended)**: Expand VRAM star record to 64-dim
```python
# SOVEREIGN_NSI_SPECIFICATION.md update
VRAM_STAR_RECORD_SIZE = 512  # bytes (64 floats × 8 bytes)
EMBEDDING_DIM = 64           # minimum per S2.2
```

**Option B (Temporary)**: Document as technical debt with migration path
```markdown
## Technical Debt: VRAM Star Record Expansion
- Current: 32-dim (256 bytes)
- Target: 64-dim (512 bytes) per SOVEREIGN_TRAINING_SPEC.md S2.2
- Migration: Phase 2026-Q3 VRAM restructuring
```

### Verdict
**❌ CANNOT MERGE** without either expanding VRAM records or documenting explicit sovereignty debt with migration timeline.

---

## 2. Procedural Frame Embedding Bypass ❌ SOVEREIGNTY VIOLATION

### The Violation

**Codex Spec (Bottleneck 4)**:
> "The procedural frame embedding bypasses the RPN embedding engine entirely for GAME_2D queries"

**SOVEREIGN_NSI_SPECIFICATION.md**:
> "The RPN engine IS the sovereign embedding path"

### Why This Matters

1. **Dual embedding paths** create sovereignty fragmentation
2. GAME_2D queries become second-class citizens (no RPN program representation)
3. Breaks the principle: **all perception flows through RPN**

### Required Fix

Encode procedural frame data **AS an RPN program** that the engine can embed:

```python
# Instead of bypassing RPN engine:
frame_embedding = compute_python_statistics(game_frame)

# Encode as RPN program:
rpn_program = encode_frame_as_rpn(game_frame)  # VectorDotMap field coefficients
frame_embedding = rpn_engine.embed(rpn_program)  # Sovereign path
```

Per **PROCEDURAL_VISUAL_SPECIFICATION.md Section 2.2**:
> "Store field coefficients, not pixel data"

### Verdict
**❌ CANNOT MERGE** without routing GAME_2D through RPN embedding engine with procedural encoding.

---

## 3. VectorDotMap vs Python Grid Statistics ❌ SOVEREIGNTY VIOLATION

### The Violation

**Codex Spec**:
> "VectorDotMap integration: the spec mentions VectorDotMap for game frame perception but the actual fix is manual Python grid statistics"

**FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md S5.4**:
> "Store field equations, not pixel data"

**PROCEDURAL_VISUAL_SPECIFICATION.md Section 2.2**:
```rpn
# VectorDotMap: Store field equation
FIELD_COEF 0.7 0.3 0.5 0.2    # 4 coefficients = 16 bytes
DENSITY 0.8                   # Dot density parameter
EMIT_FIELD                    # Generate dots at render time
```

### Why This Matters

| Approach | Storage | Sovereignty | Scalability |
|----------|---------|-------------|-------------|
| Python statistics | O(n) pixel data | ❌ Violates S5.4 | Fixed resolution |
| VectorDotMap fields | O(1) coefficients | ✅ Compliant | Infinite LOD |

Manual Python statistics = **pixel-data thinking** in a procedural architecture.

### Required Fix

Encode game grid as **quantum field coefficients**:

```python
# Instead of:
grid_stats = {
    'mean_intensity': np.mean(frame),
    'variance': np.var(frame),
    'edge_density': compute_edges(frame)
}

# Use VectorDotMap field encoding:
field_coeffs = encode_grid_as_quantum_field(frame)  # 4-16 coefficients
star_record.field_equation = field_coeffs           # Store equations, not stats
```

### Verdict
**❌ CANNOT MERGE** without replacing Python statistics with VectorDotMap field coefficients.

---

## 4. Role-Filtered Navigation ⚠️ ARCHITECTURAL DEAD-END

### The Concern

**Codex Spec**:
> "Adds route_family filtering to _navigate_galaxy_ref()"

**Composed Head Pipeline** (Morton → LED-A* → Frustum → LOD → Swarm → Halting):
- Navigation uses **spatial indexing** (Morton codes, LED-A*)
- String-based role filtering is O(n) linear search

### Why This Matters

1. **Scalability**: String filtering doesn't scale beyond ~10K stars
2. **Sovereignty**: Spatial indexing is the sovereign navigation path
3. **Consistency**: Breaks the composed head pipeline contract

### Assessment

This appears to be a **stepping stone** (quick fix for immediate bug) but risks becoming **technical debt** if not migrated to spatial indexing.

### Required Fix

```python
# Instead of string-based filtering:
if star.route_family == query_role:  # O(n) linear search

# Use Morton code spatial indexing:
morton_range = compute_morton_range(query_position, radius)
candidate_stars = spatial_index.query(morton_range)  # O(log n)
```

### Verdict
**⚠️ CONDITIONAL MERGE** acceptable if documented as Phase 1 with Phase 2 spatial indexing migration scheduled (max 2 sprints).

---

## 5. Nine-Chain Swarm Pairing ⚠️ MISSING REQUIREMENT

### The Concern

**Codex Spec**:
> "The nine-chain swarm is fixed at 9 chains"

**Daniel's Guidance**:
> "The swarm structure can also be spawned at least one pair of swarms"

### Why This Matters

1. **Swarm pairing** enables bidirectional reasoning (forward/backward chains)
2. **Fixed 9-chain** limits emergent swarm topology
3. **Sovereignty**: Swarm structure should be dynamic, not hardcoded

### Assessment

This is a **missing requirement**, not a violation. The Codex doesn't address swarm pairing but doesn't explicitly prevent it either.

### Required Fix

Add swarm pair spawning capability:

```python
# In swarm initialization:
def spawn_swarm_pair(self, query):
    forward_swarm = self._spawn_chain(direction='forward')
    backward_swarm = self._spawn_chain(direction='backward')
    return SwarmPair(forward_swarm, backward_swarm)
```

### Verdict
**⚠️ MERGE WITH TODO** - Add GitHub issue for swarm pair spawning (Phase 2026-Q2).

---

## 6. VectorDotMap Game Perception ❌ SOVEREIGNTY VIOLATION

### The Violation

**Codex Spec**:
> "Does this spec properly leverage VectorDotMap for game perception, or does it just compute Python statistics?"

**PROCEDURAL_VISUAL_SPECIFICATION.md Section 1.2**:
| Traditional | K3D Procedural |
|-------------|----------------|
| Store 25MB pixel grid | Store 2KB RPN program |
| Fixed resolution | Infinite LOD |
| No semantics | Full semantic links |

**Current Implementation**: Python statistics (pixel-data thinking)  
**Required Implementation**: VectorDotMap field coefficients (procedural thinking)

### Why This Matters

This is the **same violation as #3** but specifically for game perception. The Codex mentions VectorDotMap but doesn't implement it.

### Required Fix

Per **PROCEDURAL_VISUAL_SPECIFICATION.md Section 2.2 Layer -1**:

```rpn
# Game frame as VectorDotMap field
FIELD_COEF c0 c1 c2 c3          # Quantum field coefficients
DENSITY_FIELD density_map       # Variable density
EMIT_REGION x y w h             # Emit dots in game grid region
```

```python
# Python implementation:
class GameFrameEncoder:
    def encode(self, frame):
        coeffs = self._extract_field_coefficients(frame)  # Not statistics
        return VectorDotMap(field_coeffs=coeffs)
```

### Verdict
**❌ CANNOT MERGE** without proper VectorDotMap integration (same as #3).

---

## Summary of Required Changes

| # | Change | Priority | Effort | Blocker |
|---|--------|----------|--------|---------|
| 1 | Expand VRAM star record 32→64 dim | Critical | High | Yes |
| 2 | Route GAME_2D through RPN engine | Critical | Medium | Yes |
| 3 | Replace Python stats with VectorDotMap fields | Critical | Medium | Yes |
| 4 | Migrate role-filter to spatial indexing | Medium | Low | No (document debt) |
| 5 | Add swarm pair spawning | Medium | Medium | No (create issue) |
| 6 | Proper VectorDotMap game perception | Critical | Medium | Yes |

---

## Recommendation

**DO NOT MERGE** in current state. Four sovereignty violations (#1, #2, #3, #6) must be resolved before integration.

**Minimum Viable Fix**:
1. Expand VRAM star record to 64-dim (512 bytes)
2. Route all embeddings through RPN engine (no bypasses)
3. Replace Python statistics with VectorDotMap field coefficients
4. Document role-filter as technical debt with migration timeline
5. Create GitHub issue for swarm pair spawning

**Estimated Effort**: 3-5 days for critical fixes + 1 day documentation

---

## Sovereignty Compliance Checklist

- [ ] SOVEREIGN_TRAINING_SPEC.md S2.2: Min 64-dim embeddings ✅/❌
- [ ] SOVEREIGN_NSI_SPEC.md: RPN engine is sovereign embedding path ✅/❌
- [ ] FOUNDATIONAL_KNOWLEDGE_SPEC.md S5.4: Store field equations, not pixel data ✅/❌
- [ ] PROCEDURAL_VISUAL_SPEC.md: VectorDotMap field coefficients ✅/❌
- [ ] Composed Head Pipeline: Spatial indexing for navigation ✅/❌
- [ ] Swarm Architecture: Dynamic pairing support ✅/❌

**Current Compliance**: 0/6 ❌  
**Required for Merge**: 6/6 ✅