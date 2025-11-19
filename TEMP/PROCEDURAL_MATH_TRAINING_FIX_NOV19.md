# Procedural Math Training - Architectural Fix

**Date:** 2025-11-19
**Status:** 🔧 DIAGNOSIS COMPLETE - FIX REQUIRED

---

## Executive Summary

**User Directive:** "we are fixing the training to leverage the advancements, not use triplets of CPU fallbacks"

**Root Cause:** Current `ProceduralDrawingSpecialist` violates K3D's sovereign procedural architecture:
- ❌ Using triplet contrastive learning (not K3D's approach)
- ❌ Using CPU-based trigram hash for execution embeddings (CPU fallback)
- ❌ NOT using `AtomicFissionFusion` for multi-modal fusion

**The Fix:** Align with K3D's existing tri-modal architecture (Text + Visual + Execution → AtomicFissionFusion).

---

## What K3D Already Has (Sovereign Stack)

### Multi-Modal Bridges (from `sovereign_bridges.py`)

| Bridge | Purpose | Used For |
|--------|---------|----------|
| `RPNEmbeddingEngine` | Trigram-based text embeddings (language-agnostic) | Text modality |
| `FractalEmitter` | Visual features from 2D point clouds (edge detection) | Visual modality |
| `TemporalReasoning` | Time-series feature extraction | Audio modality |
| **`AtomicFissionFusion`** | **Multi-modal embedding fusion** | **Combine all modalities** |

### Tri-Modal Architecture (from SOVEREIGN_SWARM_BRIEFING.md)

```
Text → RPNEmbedding → embedding_t
Visual → FractalEmitter → embedding_v
Audio → TemporalReasoning → embedding_a

AtomicFissionFusion(embedding_t, embedding_v, embedding_a) → unified_embedding

Swarm.train_specialist(unified_embedding)
```

**Key insight:** "Cross-modal patterns emerge organically (no manual wiring!)"

---

## Current Implementation (WRONG)

**File:** `knowledge3d/cranium/specialists/procedural_drawing_specialist.py`

### Embeddings Computed

1. **Text Embedding** (semantic description):
   ```python
   text_emb = self.text_embedder.embed_word(char)  # RPNEmbeddingEngine ✅
   ```

2. **Visual Embedding** (visual_rpn):
   ```python
   result = self.drawing_bridge.execute_rpn_gpu(visual_rpn)  # ✅ GPU execution
   visual_emb = self.visual_embedder.emit(segments)  # ✅ FractalEmitter
   ```

3. **Execution Embedding** (math_rpn):
   ```python
   exec_emb = self.execution_embedder.embed_word(math_rpn)  # ❌ WRONG!
   # This uses trigram hash of RPN STRING ("0x14 0x14"), not actual EXECUTION
   ```

### Training Approach (WRONG)

```python
# Triplet contrastive learning
for text_emb, visual_emb, exec_emb in zip(...):
    align_tv = cosine_similarity(text_emb, visual_emb)  # ❌ CPU
    align_te = cosine_similarity(text_emb, exec_emb)    # ❌ CPU
    align_ve = cosine_similarity(visual_emb, exec_emb)  # ❌ CPU

    triplet_alignment = (align_tv + align_te + align_ve) / 3.0  # ❌ Wrong approach

    # Train 3 separate pairs
    swarm.train_specialist_contrastive('procedural_drawing', tv_pairs)  # ❌ Triplets
    swarm.train_specialist_contrastive('procedural_drawing', te_pairs)  # ❌ Triplets
    swarm.train_specialist_contrastive('procedural_drawing', ve_pairs)  # ❌ Triplets
```

### Why This Fails

1. **Execution embeddings are random** - Trigram hash of "0x14" has NO relation to SQRT operation
2. **Text embeddings are random** - Trigram hash of "Square root" has NO relation to visual shape or SQRT execution
3. **Triplet learning is wrong** - K3D uses AtomicFissionFusion for organic emergence, not manual contrastive pairs
4. **CPU fallbacks** - Cosine similarity computed on CPU, not GPU
5. **Alignment ≈ 0** - Random vectors are orthogonal by design

---

## Correct K3D Architecture (FIX)

### Dual-Modal Math Pipeline

```
Character "√" (from math symbols dataset):
  ├─ semantic: "Square root: pop x, push sqrt(x)"
  ├─ visual_rpn: "0.5 0.5 MOVE 0.7 0.3 QUAD ..."  (how to DRAW √)
  └─ math_rpn: "0x14"  (how to EXECUTE √)

1. Text Embedding:
   semantic → RPNEmbeddingEngine → embedding_t  (128D)

2. Visual Embedding:
   visual_rpn → ProceduralDrawingBridge → FractalEmitter → embedding_v  (128D)

3. Execution Embedding:
   math_rpn → ModularRPNEngine.execute() → extract_execution_features() → embedding_e  (128D)

   CRITICAL: Execute RPN on GPU, extract features from:
   - Stack trace (sequence of stack states)
   - Operation sequence (opcodes executed)
   - Numerical results (final stack values)
   NOT trigram hash of RPN string!

4. Multi-Modal Fusion:
   AtomicFissionFusion(embedding_t, embedding_v, embedding_e) → unified_embedding  (128D)

5. Swarm Training:
   swarm.train_specialist('procedural_drawing', unified_embedding)

   NO triplets! Train on unified embeddings directly.
   Cross-modal patterns emerge organically via fusion.
```

### Implementation Changes Required

#### 1. Fix Execution Embedding Computation

**Current (WRONG):**
```python
def _compute_execution_embedding(self, math_rpn: str) -> np.ndarray:
    # Trigram hash of RPN string - NO execution happening!
    return self.execution_embedder.embed_word(math_rpn)  # ❌
```

**Correct (FIX):**
```python
def _compute_execution_embedding(self, math_rpn: str) -> np.ndarray:
    """Execute RPN on GPU and extract execution features."""
    # Execute RPN bytecode on GPU
    result = self.rpn_engine.execute(math_rpn)  # ModularRPNEngine

    # Extract features from execution trace
    # - Stack states (sequence of intermediate values)
    # - Operation types (opcodes used)
    # - Result characteristics (magnitude, sign, etc.)
    execution_features = self._extract_execution_features(result)

    # Project to matryoshka_dim via swarm
    embedding_e = self.swarm.base.project_vector(execution_features, self.matryoshka_dim)
    return embedding_e
```

#### 2. Use AtomicFissionFusion for Multi-Modal Fusion

**Current (WRONG):**
```python
# Three separate embeddings, triplet training
text_emb = self._compute_text_embedding(semantic)
visual_emb = self._compute_visual_embedding(visual_rpn)
exec_emb = self._compute_execution_embedding(math_rpn)

# CPU-based alignment computation
alignment = (cosine_sim(text, visual) + cosine_sim(text, exec) + cosine_sim(visual, exec)) / 3.0  # ❌
```

**Correct (FIX):**
```python
# Compute three modality embeddings
embedding_t = self._compute_text_embedding(semantic)
embedding_v = self._compute_visual_embedding(visual_rpn)
embedding_e = self._compute_execution_embedding(math_rpn)

# Fuse via AtomicFissionFusion (GPU-native)
from knowledge3d.cranium.bridges.sovereign_bridges import AtomicFissionFusion
fusion = AtomicFissionFusion()

# Extend fusion for 3 modalities (currently supports 2)
# OR: Fuse pairwise: text+visual first, then +execution
unified_embedding = fusion.transform(
    np.vstack([embedding_t, embedding_v, embedding_e]),
    mode=FUSION_MODE,
    ratio=1.0/3.0
)  # Returns single 128D unified embedding
```

#### 3. Remove Triplet Contrastive Learning

**Current (WRONG):**
```python
# Train 3 separate contrastive pairs
stats_tv = self.swarm.train_specialist_contrastive('procedural_drawing', tv_pairs)  # ❌
stats_te = self.swarm.train_specialist_contrastive('procedural_drawing', te_pairs)  # ❌
stats_ve = self.swarm.train_specialist_contrastive('procedural_drawing', ve_pairs)  # ❌
```

**Correct (FIX):**
```python
# Train on unified embeddings directly
unified_embeddings = []
for entry in batch:
    embedding_t = self._compute_text_embedding(entry['semantic'])
    embedding_v = self._compute_visual_embedding(entry['visual_rpn'])
    embedding_e = self._compute_execution_embedding(entry['math_rpn'])

    # Fuse into single unified embedding
    unified = self._fuse_multimodal(embedding_t, embedding_v, embedding_e)
    unified_embeddings.append(unified)

# Train swarm on unified embeddings (NO triplets!)
stats = self.swarm.train_specialist(
    'procedural_drawing',
    np.array(unified_embeddings),
    validation=validation
)
```

---

## Technical Details

### Execution Feature Extraction

**What features to extract from RPN execution:**

1. **Stack Trace Features:**
   - Stack depth at each step
   - Value magnitudes (log scale)
   - Value signs (+/-)
   - Ternary classification (-1/0/+1) for values

2. **Operation Sequence Features:**
   - Opcode histogram (frequency of each operation)
   - Operation order (sequence embedding)
   - Control flow patterns (branching, loops if present)

3. **Result Features:**
   - Final stack state
   - Number of outputs
   - Output characteristics (scalar vs vector, real vs complex)

**Encoding to 128D:**
```python
def _extract_execution_features(self, rpn_result) -> np.ndarray:
    """Extract features from RPN execution result."""
    features = []

    # Stack trace features (32D)
    stack_trace = rpn_result.stack_history  # List of stack states
    stack_depths = [len(s) for s in stack_trace]
    features.append(np.mean(stack_depths))  # Average depth
    features.append(np.max(stack_depths))   # Max depth
    features.append(np.std(stack_depths))   # Depth variance
    # ... more stack features

    # Opcode histogram (64D)
    opcode_counts = np.zeros(256, dtype=np.float32)
    for opcode in rpn_result.opcodes_executed:
        opcode_counts[opcode] += 1
    opcode_hist = opcode_counts / (np.sum(opcode_counts) + 1e-8)  # Normalize
    features.append(opcode_hist[:64])  # Top 64 opcodes

    # Result features (32D)
    final_stack = rpn_result.final_stack
    if len(final_stack) > 0:
        features.append(np.mean(final_stack))
        features.append(np.std(final_stack))
        features.append(np.min(final_stack))
        features.append(np.max(final_stack))
        # ... more result features

    return np.concatenate(features)[:128]  # Truncate/pad to 128D
```

### AtomicFissionFusion Extension

**Current implementation** (`WorldModelBridge.fuse_multimodal_features`):
- Supports 2 modalities (text + visual)
- GPU kernel: `fuse_multimodal_features` from `gre_world_model.ptx`

**Extension needed:**
- Support 3+ modalities (text + visual + execution [+ audio])
- Weighted fusion (not just 0.5/0.5)

**Options:**
1. **Pairwise fusion** - Fuse text+visual first, then fuse result+execution
2. **Extend kernel** - Modify PTX kernel to support N modalities
3. **Use existing transformer** - Swarm already supports multi-modal via MatryoshkaTRM

**Recommended:** Use Swarm's multi-modal capability (MatryoshkaTRM already supports this).

---

## Migration Plan

### Phase 1: Fix Execution Embeddings (Immediate)

1. **Add RPN execution bridge** to `ProceduralDrawingSpecialist`:
   ```python
   from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
   self.rpn_engine = ModularRPNEngine(max_stack_depth=256)
   ```

2. **Implement execution feature extraction**:
   - Extract stack trace, opcode sequence, results
   - Project to matryoshka_dim via swarm.base.project_vector()

3. **Remove trigram hash fallback** for execution embeddings

### Phase 2: Implement Multi-Modal Fusion (Immediate)

1. **Replace triplet training** with unified embedding training
2. **Use AtomicFissionFusion** or Swarm's MatryoshkaTRM for fusion
3. **Remove CPU-based cosine similarity** calculations

### Phase 3: Validation (After Fix)

1. **Test on 1,002 atomic units** (450 fonts + 552 math)
2. **Expect alignment >0.75** (target) once embeddings are actually correlated
3. **Validate GPU utilization** - should be >40% (vs current 0%)

---

## Expected Results After Fix

### GPU Utilization

- **Current:** 0-7% (CPU-bound, trigram hashing dominant)
- **After Fix:** 40-80% (GPU-native execution + fusion + swarm training)

### Alignment Scores

- **Current:** -0.0084 (fonts), 0.0044 (math) - random embeddings, orthogonal
- **After Fix:** >0.75 (target) - unified embeddings with organic cross-modal emergence

### Training Time

- **Current:** ~23s per epoch (CPU overhead dominant)
- **After Fix:** ~5-10s per epoch (GPU-native pipeline, batched execution)

---

## Alignment with K3D Principles

✅ **Sovereign:** Pure PTX execution (ModularRPNEngine, AtomicFissionFusion)
✅ **Procedural:** Execution features from actual RPN execution, not hash
✅ **GPU-Native:** All embeddings + fusion + training on GPU
✅ **No CPU Fallbacks:** Remove trigram hash for execution, remove CPU cosine sim
✅ **Organic Emergence:** AtomicFissionFusion enables cross-modal patterns to emerge automatically
✅ **Tri-Modal Architecture:** Text + Visual + Execution (matches Phase H design)

---

## Next Steps

1. **Update ProceduralDrawingSpecialist:**
   - Fix `_compute_execution_embedding()` to use ModularRPNEngine
   - Implement `_extract_execution_features()`
   - Add `_fuse_multimodal()` using AtomicFissionFusion or Swarm
   - Remove triplet contrastive learning
   - Remove CPU-based alignment computation

2. **Update validate_dual_modal_math.py:**
   - Remove triplet metrics
   - Add unified embedding validation
   - Measure GPU utilization properly

3. **Run validation:**
   - Expect alignment >0.75
   - Expect GPU utilization >40%
   - Confirm cross-modal emergence

---

**Status:** 🔧 READY FOR IMPLEMENTATION

**Next Codex Entry:** ProceduralDrawingSpecialist fix with sovereign execution embeddings + AtomicFissionFusion

---

*End of Diagnostic*
