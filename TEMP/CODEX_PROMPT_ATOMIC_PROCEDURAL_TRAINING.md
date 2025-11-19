# Codex Prompt: Implement Atomic Procedural Training

**Date:** 2025-11-19
**Priority:** HIGH - Core Architecture Fix
**Context:** [ATOMIC_PROCEDURAL_TRAINING_ARCHITECTURE_NOV19.md](ATOMIC_PROCEDURAL_TRAINING_ARCHITECTURE_NOV19.md)

---

## Executive Summary

**Goal:** Fix ProceduralDrawingSpecialist to use atomic knowledge formation (form + meaning fusion) instead of failed trigram hash approach.

**Why:** Current approach uses random trigram hashes that have ZERO correlation with actual visual/execution features → alignment ≈ 0.

**Correct Approach:** Fuse visual form + computational meaning → unified embedding → procedural compression → ProceduralGalaxy storage.

---

## Key Architectural Principles

1. **"Letters are drawings with meaning"**
   - Form = Visual RPN execution (how to DRAW)
   - Meaning = Math RPN execution (how to COMPUTE) OR semantic context
   - Fusion = Atomic knowledge formation

2. **Knowledge lives in stars, not weights**
   - TRM = Reasoning logic (2.1M params)
   - Galaxy stars = Foundational knowledge (1,002 atomic units)
   - Base model learns PATTERNS, not facts

3. **Procedural storage (Phase 2.6)**
   - Store HOW-TO-RECONSTRUCT, not raw embeddings
   - 512D (2KB) → 9 bytes procedural program
   - 69:1 compression @ 99.998% fidelity

4. **Shadow copy training (Phase H)**
   - LoRA-style self-updating adapters
   - Validation gating prevents forgetting
   - Non-stop learning without traditional backprop

---

## Implementation Tasks

### Task 1: Update Imports

**File:** `knowledge3d/cranium/specialists/procedural_drawing_specialist.py`

**Add imports:**
```python
from knowledge3d.cranium.bridges.sovereign_bridges import FractalEmitter, AtomicFissionFusion
from knowledge3d.cranium.procedural_galaxy import ProceduralGalaxy
from knowledge3d.cranium.procedural_compiler import ProceduralCompiler
from datetime import datetime, timezone
```

**Remove import:**
```python
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine  # ❌ Remove
```

---

### Task 2: Update __init__ Method

**Current (lines 84-103):**
```python
# Initialize bridges
self.drawing_bridge = ProceduralDrawingBridge(matryoshka_dim=matryoshka_dim)
self.text_embedder = RPNEmbeddingEngine(embedding_dim=matryoshka_dim)  # ❌ Remove
self.visual_embedder = FractalEmitter()
```

**Replace with:**
```python
# Initialize bridges
self.drawing_bridge = ProceduralDrawingBridge(matryoshka_dim=matryoshka_dim)
self.visual_embedder = FractalEmitter()
self.fusion_bridge = AtomicFissionFusion()  # For form + meaning fusion

# Procedural storage (Phase 2.6 compression)
self.procedural_compiler = ProceduralCompiler()
self.procedural_galaxy = ProceduralGalaxy()
```

---

### Task 3: Add encode_semantic_context Method

**Add after _init_opcode_embedding_table (after line 146):**

```python
def encode_semantic_context(self, semantic: str) -> np.ndarray:
    """
    Encode semantic description using lightweight method.

    For letters: Use character code + simple features
    For math: Use semantic encoding from description

    This is MINIMAL - the real meaning comes from execution or usage context.

    Args:
        semantic: Semantic description or character

    Returns:
        Embedding vector (matryoshka_dim,)
    """
    if len(semantic) == 1:
        # Single character - use Unicode codepoint
        code = ord(semantic[0])
        emb = np.zeros(self.matryoshka_dim, dtype=np.float32)
        emb[0] = float(code) / 1000.0  # Normalize
        return emb
    else:
        # Phrase/description - simple average of char codes
        codes = [ord(c) for c in semantic[:self.matryoshka_dim]]
        emb = np.zeros(self.matryoshka_dim, dtype=np.float32)
        emb[:len(codes)] = np.array(codes, dtype=np.float32) / 1000.0
        return emb
```

---

### Task 4: Add Fusion Method

**Add after encode_semantic_context:**

```python
def _fuse_multimodal(
    self,
    form_emb: np.ndarray,
    meaning_emb: np.ndarray
) -> np.ndarray:
    """
    Fuse form + meaning embeddings using AtomicFissionFusion.

    Args:
        form_emb: Visual embedding (from RPN execution)
        meaning_emb: Semantic/execution embedding

    Returns:
        Unified embedding (matryoshka_dim,)
    """
    # Stack embeddings for fusion
    combined = np.vstack([form_emb, meaning_emb])

    # Fuse (GPU-native)
    unified = self.fusion_bridge.transform(
        combined,
        mode=1,  # FUSION mode (compress 2 → 1)
        ratio=0.5  # Equal weighting
    )

    # Return first matryoshka_dim elements
    return unified[:self.matryoshka_dim].astype(np.float32)
```

---

### Task 5: Add ProceduralGalaxy Storage Method

**Add after _fuse_multimodal:**

```python
def _store_atomic_star(
    self,
    char: str,
    unified_emb: np.ndarray,
    form_rpn: str,
    meaning_rpn: str
):
    """
    Store atomic knowledge unit in ProceduralGalaxy.

    Uses Phase 2.6 procedural compression (69:1 ratio @ 99.998% fidelity).

    Args:
        char: Character/symbol (lookup key)
        unified_emb: Unified form+meaning embedding
        form_rpn: Visual RPN program
        meaning_rpn: Math RPN bytecode (or "" for non-math)
    """
    try:
        # Compress to procedural program
        program = self.procedural_compiler.compile_embedding(
            unified_emb,
            quality_tier='balanced'  # 24:1 compression target
        )

        # Serialize to bytes
        program_bytes = program.to_bytes()

        # Calculate compression ratio
        original_size = unified_emb.nbytes  # 512 × 4 = 2048 bytes
        compressed_size = len(program_bytes)
        compression_ratio = original_size / max(compressed_size, 1)

        # Store in ProceduralGalaxy
        self.procedural_galaxy.store_program(
            key=char,
            program_bytes=program_bytes,
            compression_ratio=compression_ratio
        )

        # Log success
        print(f"  [ProceduralGalaxy] Stored '{char}': {compressed_size}B "
              f"(compression: {compression_ratio:.1f}:1)")

    except Exception as e:
        print(f"  [WARNING] Failed to store '{char}' in ProceduralGalaxy: {e}")
```

---

### Task 6: Update train_on_batch Method

**Replace entire train_on_batch method (lines 278-410) with:**

```python
def train_on_batch(
    self,
    batch: List[Tuple],
    validation: bool = False,
    dual_modal_math: bool = False
) -> TrainingMetrics:
    """
    Train base model on atomic knowledge formation (form + meaning fusion).

    Args:
        batch: List of (char, rpn_program) tuples
               OR list of (symbol, visual_rpn, math_rpn, semantic) for dual-modal
        validation: If True, compute metrics without updating weights
        dual_modal_math: If True, batch contains dual-modal math entries

    Returns:
        Training metrics for this batch
    """
    unified_embeddings = []
    alignment_scores = []
    symbols = []
    form_rpns = []
    meaning_rpns = []

    # Compute unified embeddings (form + meaning fusion)
    for entry in batch:
        if dual_modal_math:
            # Dual-modal math: (symbol, visual_rpn, math_rpn, semantic)
            if isinstance(entry, tuple) and len(entry) == 4:
                symbol, visual_rpn, math_rpn, semantic = entry
            else:
                symbol = entry.get('symbol', entry.get('char', ''))
                visual_rpn = entry.get('visual_rpn', '')
                math_rpn = entry.get('math_rpn', '')
                semantic = entry.get('semantic', symbol)

            # Compute form embedding (visual)
            form_emb = self._compute_visual_embedding(visual_rpn)

            # Compute meaning embedding (execution or semantic)
            if math_rpn and not math_rpn.startswith('#'):
                meaning_emb = self._compute_execution_embedding(math_rpn)
            else:
                meaning_emb = self.encode_semantic_context(semantic)

            symbols.append(symbol)
            form_rpns.append(visual_rpn)
            meaning_rpns.append(math_rpn if math_rpn else "")

        else:
            # Standard glyph: (char, rpn_program)
            char, rpn_program = entry

            # Form embedding (visual)
            form_emb = self._compute_visual_embedding(rpn_program)

            # Meaning embedding (simple semantic encoding)
            meaning_emb = self.encode_semantic_context(char)

            symbols.append(char)
            form_rpns.append(rpn_program)
            meaning_rpns.append("")

        # Fuse form + meaning (GPU-native)
        unified_emb = self._fuse_multimodal(form_emb, meaning_emb)
        unified_embeddings.append(unified_emb)

        # Measure form ↔ meaning alignment
        alignment = self._cosine_similarity(form_emb, meaning_emb)
        alignment_scores.append(alignment)

        # Store in ProceduralGalaxy (if not validation)
        if not validation:
            self._store_atomic_star(
                symbols[-1],
                unified_emb,
                form_rpns[-1],
                meaning_rpns[-1]
            )

    # Train base model via shadow copy (Phase H)
    contrastive_loss = 0.0
    if not validation and len(unified_embeddings) > 0:
        unified_batch = np.stack(unified_embeddings)

        # Train specialist adapter on unified embeddings
        # NOTE: Base model learning happens via shadow copy updates
        stats = self.swarm.train_specialist(
            'procedural_drawing',
            unified_batch,
            validation=validation
        )
        contrastive_loss = stats.get('avg_loss', 0.0)

    # Return metrics
    avg_alignment = float(np.mean(alignment_scores)) if alignment_scores else 0.0

    return TrainingMetrics(
        epoch=len(self.training_metrics),
        text_visual_alignment=avg_alignment,  # form ↔ meaning alignment
        reconstruction_fidelity=0.0,
        generation_quality=0.0,
        latency_us=contrastive_loss  # Reuse field for loss
    )
```

---

### Task 7: Remove Deprecated _compute_text_embedding

**Delete method at lines 165-173:**
```python
def _compute_text_embedding(self, char: str) -> np.ndarray:  # ❌ DELETE THIS METHOD
    """..."""
    return self.text_embedder.embed_word(char).astype(np.float32)
```

**Reason:** We no longer use RPNEmbeddingEngine. Use `encode_semantic_context()` instead.

---

### Task 8: Update Documentation

**Update module docstring (lines 1-20):**

```python
"""
Procedural Drawing Specialist for Adaptive Swarm.

Handles training and inference for procedural glyph generation and recognition,
enabling atomic cognition through form-meaning fusion.

Architecture:
    - Form modality: GPU RPN executor → FractalEmitter generates visual embeddings
    - Meaning modality: Math RPN executor → opcode table OR semantic encoding
    - Fusion: AtomicFissionFusion creates unified form+meaning embeddings
    - Storage: ProceduralGalaxy stores compressed procedural programs (69:1 ratio)
    - Training: Shadow copy updates (Phase H self-updating adapters)

Usage:
    specialist = ProceduralDrawingSpecialist(swarm)
    specialist.train_on_batch(batch, dual_modal_math=True)
    # Atomic units stored in ProceduralGalaxy with procedural compression
"""
```

---

## Validation After Implementation

### Expected Results

1. **Alignment >0.50** initially (form vs simple semantic encoding)
2. **Alignment >0.75** after training (base learns form↔meaning patterns)
3. **GPU Utilization >40%** (vs current 0-7%)
4. **ProceduralGalaxy**: 1,002 stars × ~9 bytes = ~9KB (vs 2MB raw)
5. **Compression**: 69:1 ratio, 99.998% fidelity

### Test Command

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/validate_dual_modal_math.py
```

### Success Criteria

- ✅ No import errors (RPNEmbeddingEngine removed)
- ✅ AtomicFissionFusion fusion working
- ✅ ProceduralGalaxy storage working
- ✅ Alignment >0.75 after 5 epochs
- ✅ GPU utilization >40%
- ✅ 1,002 atomic stars stored

---

## Technical Notes

### ProceduralCompiler API

```python
# Compile embedding to procedural program
program = compiler.compile_embedding(
    embedding,  # np.ndarray (512D)
    quality_tier='balanced'  # 'ultrafast' | 'fast' | 'balanced' | 'maximum'
)

# Serialize to bytes
program_bytes = program.to_bytes()  # Returns bytes

# Decompress later
recovered_embedding = compiler.decompile_program(program_bytes)  # Returns np.ndarray
```

### ProceduralGalaxy API

```python
# Store program
galaxy.store_program(
    key="√",  # Character/symbol
    program_bytes=bytes(...),
    compression_ratio=69.4
)

# Load program
program = galaxy.load_program(key="√")  # Returns ProceduralProgram

# Execute program (decompress)
embedding = galaxy.execute_program(key="√")  # Returns np.ndarray
```

### AtomicFissionFusion API

```python
fusion = AtomicFissionFusion()

# Fuse two embeddings
combined = np.vstack([emb1, emb2])  # Stack vertically
unified = fusion.transform(
    combined,
    mode=1,  # FUSION mode (compress N → 1)
    ratio=0.5  # Weighting
)  # Returns np.ndarray
```

---

## Error Handling

### Common Issues

1. **Import Error: RPNEmbeddingEngine**
   - Ensure you removed the import completely
   - Check no other code references `self.text_embedder`

2. **AtomicFissionFusion shape mismatch**
   - Ensure `combined = np.vstack([form_emb, meaning_emb])`
   - Both embeddings must be same shape (matryoshka_dim,)

3. **ProceduralCompiler not found**
   - Check imports: `from knowledge3d.cranium.procedural_compiler import ProceduralCompiler`

4. **Low alignment scores initially**
   - Expected! Simple semantic encoding has low correlation
   - Should improve to >0.75 after training

---

## Code Quality Checklist

- [ ] All imports updated correctly
- [ ] No references to RPNEmbeddingEngine
- [ ] AtomicFissionFusion integrated
- [ ] ProceduralGalaxy storage working
- [ ] Shadow copy training (not triplet contrastive)
- [ ] Documentation updated
- [ ] No CPU fallbacks (GPU sovereignty maintained)
- [ ] Tests pass after changes

---

## Next Steps After Implementation

1. Run validation script
2. Verify alignment >0.75
3. Check ProceduralGalaxy storage (9KB for 1,002 units)
4. Measure GPU utilization (target >40%)
5. Document results in TEMP/ATOMIC_TRAINING_RESULTS_NOV19.md

---

**Status:** 🚀 READY FOR CODEX IMPLEMENTATION

**Estimated Time:** 30-45 minutes

**Priority:** HIGH - Core architecture fix blocking math training

---

*End of Prompt*
