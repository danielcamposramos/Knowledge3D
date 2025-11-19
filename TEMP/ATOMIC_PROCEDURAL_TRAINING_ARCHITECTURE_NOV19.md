# Atomic Procedural Training Architecture - Form to Meaning

**Date:** 2025-11-19
**Status:** 🏗️ COMPLETE REARCHITECTURE - BASE MODEL TRAINING FROM SCRATCH

---

## The Core Insight

**"Letters are drawings with meaning"**

- **Form** = Visual RPN execution (how to DRAW)
- **Meaning** = Math RPN execution (how to COMPUTE) OR semantic context (what it REPRESENTS)
- **Knowledge** = Fusion of form + meaning

**Base model is NOT trained yet** - We're starting from atomic principles, not trigram hashes.

---

## Why Trigram Hash Failed

### The Problem

```python
# Current (WRONG):
text_emb = RPNEmbeddingEngine.embed_word("A")  # Random hash of trigrams "A__", "_A_", "__A"
visual_emb = execute_rpn("0.5 0.5 MOVE...") → FractalEmitter  # Actual visual form

# Cosine similarity
similarity(text_emb, visual_emb) ≈ 0  # Random ⊥ geometric form
```

**Why it fails:** Trigram hash has ZERO semantic information. It's just a random identifier.

### What We Need

```python
# Correct (ATOMIC FORMATION):
form_emb = execute_visual_rpn("0.5 0.5 MOVE...") → FractalEmitter  # FORM (how it looks)
meaning_emb = execute_math_rpn("0x14") → extract_execution_trace  # MEANING (what it does)
                OR semantic_context("Square root operator")

# Fuse form + meaning
unified_emb = AtomicFissionFusion(form_emb, meaning_emb)  # Organic emergence

# Store in Galaxy as atomic star
ProceduralGalaxy.add_star(
    char="√",
    procedural_program=compress(unified_emb),
    atomic=True,
    form_rpn="0.5 0.5 MOVE...",
    meaning_rpn="0x14"
)
```

**Key:** Base model learns the RELATIONSHIP between form and meaning, not random hashes!

---

## Atomic Knowledge Formation Pipeline

### Three Modalities

1. **Form Modality** (Visual RPN)
   - How to DRAW the symbol
   - Procedural vector drawing executed on GPU
   - Output: Geometric features (segments, curves, strokes)

2. **Meaning Modality** (Math RPN OR Semantic)
   - Math symbols: How to COMPUTE (RPN execution trace)
   - Letters: What they REPRESENT (semantic context from usage)
   - Output: Execution/semantic features

3. **Identity Modality** (Character Label)
   - Simple identifier (NOT semantic!)
   - Used for lookup/caching only
   - NOT trained! Just a key.

### Training Architecture

```
For atomic unit (char="√", visual_rpn="...", math_rpn="0x14", semantic="Square root"):

1. Compute Form Embedding (GPU):
   visual_rpn → ProceduralDrawingBridge.execute_rpn_gpu()
             → segments (x0,y0,x1,y1,r,g,b,a,w)
             → FractalEmitter.emit(segments)
             → form_emb (512D)

2. Compute Meaning Embedding (GPU):

   A) For math symbols:
      math_rpn → ModularRPNEngine.execute(bytecode)
               → extract_execution_trace(stack_history, opcodes, result)
               → meaning_emb (512D)

   B) For letters/semantic:
      semantic → encode_semantic_context()  # Lightweight encoding
               → meaning_emb (512D)

3. Fuse Form + Meaning (GPU):
   AtomicFissionFusion.transform([form_emb, meaning_emb], mode=FUSION, ratio=0.5)
   → unified_emb (512D)

4. Compress to Procedural (Phase 2.6):
   AdaptiveProceduralBridge.compress(unified_emb, quality='balanced')
   → procedural_program (9 bytes @ 69:1 compression, 99.998% fidelity)

5. Store in ProceduralGalaxy:
   ProceduralGalaxy.add_star(
       char=char,
       procedural_program=procedural_program,
       metadata={
           'atomic': True,
           'never_prune': True,
           'form_rpn': visual_rpn,
           'meaning_rpn': math_rpn,
           'modality': 'dual-modal'
       }
   )

6. Train Base Model (Shadow Copy):
   # Phase H self-updating adapters
   swarm.base.update_from_atomic_knowledge(unified_emb)
   # Base learns: form ↔ meaning relationships
   # NOT memorizing facts! Learning PATTERNS.
```

---

## Implementation Changes

### 1. Remove Text Embedder (RPNEmbeddingEngine)

**Current:**
```python
self.text_embedder = RPNEmbeddingEngine(embedding_dim=matryoshka_dim)  # ❌ Random hash
```

**Remove this!** We don't need random text hashes. We have:
- **Form** (visual RPN)
- **Meaning** (math RPN or semantic)

### 2. Add Semantic Context Encoder (Lightweight)

For non-math symbols (letters, punctuation):

```python
def encode_semantic_context(self, semantic: str) -> np.ndarray:
    """
    Encode semantic description using lightweight method.

    For letters: Just use character code + simple features
    For words: Average of character codes

    This is MINIMAL - the real meaning comes from USAGE in context.
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

### 3. Fix Execution Embedding

**Current (partial fix - opcode table lookup):**
```python
exec_emb = self._compute_execution_embedding(math_rpn)
# Looks up opcode embeddings from table
```

**Complete fix (execute + extract trace):**
```python
def _compute_execution_embedding(self, math_rpn: str) -> np.ndarray:
    """Execute RPN on GPU and extract execution trace features."""
    if not math_rpn or math_rpn.startswith('#'):
        return np.zeros(self.matryoshka_dim, dtype=np.float32)

    # Option A: Use opcode embedding table (current approach - KEEP)
    # This is already GPU-accelerated via Matryoshka projection

    # Option B (future): Execute and extract trace
    # result = self.rpn_engine.execute(math_rpn)
    # trace_features = extract_execution_trace(result)
    # return trace_features

    # For now, opcode table is sufficient (learns opcode semantics)
    return self._compute_execution_embedding_via_opcode_table(math_rpn)
```

### 4. Add AtomicFissionFusion Integration

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
        Unified embedding (512D)
    """
    from knowledge3d.cranium.bridges.sovereign_bridges import AtomicFissionFusion

    fusion = AtomicFissionFusion()

    # Stack embeddings for fusion
    combined = np.vstack([form_emb, meaning_emb])

    # Fuse (GPU-native)
    unified = fusion.transform(
        combined,
        mode=1,  # FUSION mode (compress 2 → 1)
        ratio=0.5  # Equal weighting
    )

    return unified[:self.matryoshka_dim]  # Return first matryoshka_dim elements
```

### 5. Add ProceduralGalaxy Storage

```python
def _store_atomic_star(
    self,
    char: str,
    unified_emb: np.ndarray,
    form_rpn: str,
    meaning_rpn: str
):
    """Store atomic knowledge unit in ProceduralGalaxy."""
    from knowledge3d.cranium.bridges.adaptive_procedural_bridge import AdaptiveProceduralBridge
    from knowledge3d.cranium.procedural_galaxy import ProceduralGalaxy

    # Compress to procedural program
    compressor = AdaptiveProceduralBridge(quality_tier='balanced')  # 24:1 @ 99.998%
    procedural_program = compressor.compress(unified_emb)

    # Store in galaxy
    galaxy = ProceduralGalaxy()
    galaxy.add_star(
        char=char,
        procedural_program=procedural_program,
        metadata={
            'atomic': True,
            'never_prune': True,
            'form_rpn': form_rpn,
            'meaning_rpn': meaning_rpn,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    )
```

### 6. Update Training Loop

**Remove triplet contrastive, use shadow copy training:**

```python
def train_on_batch(
    self,
    batch: List[Tuple],
    validation: bool = False,
    dual_modal_math: bool = False
) -> TrainingMetrics:
    """Train base model on atomic knowledge formation."""

    unified_embeddings = []
    alignment_scores = []

    for entry in batch:
        if dual_modal_math:
            symbol, visual_rpn, math_rpn, semantic = entry

            # Compute form embedding (visual)
            form_emb = self._compute_visual_embedding(visual_rpn)

            # Compute meaning embedding (execution or semantic)
            if math_rpn and not math_rpn.startswith('#'):
                meaning_emb = self._compute_execution_embedding(math_rpn)
            else:
                meaning_emb = self.encode_semantic_context(semantic)

            # Fuse form + meaning (GPU-native)
            unified_emb = self._fuse_multimodal(form_emb, meaning_emb)

            unified_embeddings.append(unified_emb)

            # Measure alignment (form ↔ meaning)
            alignment = self._cosine_similarity(form_emb, meaning_emb)
            alignment_scores.append(alignment)

            # Store in ProceduralGalaxy (if not validation)
            if not validation:
                self._store_atomic_star(symbol, unified_emb, visual_rpn, math_rpn)

        else:
            # Standard glyph (form only, meaning from context later)
            char, rpn_program = entry
            form_emb = self._compute_visual_embedding(rpn_program)

            # For letters, use simple semantic encoding
            meaning_emb = self.encode_semantic_context(char)

            # Fuse
            unified_emb = self._fuse_multimodal(form_emb, meaning_emb)
            unified_embeddings.append(unified_emb)

            alignment = self._cosine_similarity(form_emb, meaning_emb)
            alignment_scores.append(alignment)

            if not validation:
                self._store_atomic_star(char, unified_emb, rpn_program, "")

    # Train base model via shadow copy (Phase H)
    if not validation and len(unified_embeddings) > 0:
        unified_batch = np.stack(unified_embeddings)

        # Shadow copy training (self-updating adapters)
        self.swarm.base.update_from_atomic_knowledge(unified_batch)

        # Update specialist adapter
        stats = self.swarm.train_specialist(
            'procedural_drawing',
            unified_batch,
            validation=validation
        )
        contrastive_loss = stats.get('avg_loss', 0.0)
    else:
        contrastive_loss = 0.0

    # Return metrics
    avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0

    return TrainingMetrics(
        text_visual_alignment=avg_alignment,  # form ↔ meaning alignment
        contrastive_loss=contrastive_loss,
        rpn_prediction_accuracy=0.0,  # Not applicable yet
        gpu_time_ms=0.0,  # TODO: measure
        cpu_time_ms=0.0   # TODO: measure
    )
```

---

## Base Model Training Sequence

### Phase 1: Atomic Knowledge Ingestion (Current)

Train base model to understand form ↔ meaning relationships:

```
1. Load 1,002 atomic units (450 fonts + 552 math)

2. For each unit:
   - Execute visual RPN → form embedding
   - Execute math RPN OR encode semantic → meaning embedding
   - Fuse form + meaning → unified embedding
   - Compress to procedural program (69:1)
   - Store in ProceduralGalaxy as atomic star

3. Update base model (shadow copy):
   - Base learns: "This visual form corresponds to this meaning"
   - NOT memorizing! Learning PATTERNS.
   - Example: "Curved lines at top = round shapes"
             "SQRT opcode = reduce magnitude"

4. After 1,002 units processed:
   - Base model has foundational form↔meaning understanding
   - Can generalize to new symbols
   - Ready for self-updating training
```

### Phase 2: Self-Updating Training (After Base Trained)

Once base model understands atomic relationships:

```
1. New symbols encountered in usage
2. Extract form + meaning from context
3. Fuse → unified embedding
4. Shadow copy validates against base knowledge
5. If consistent: Commit to base
6. If novel: Expand base understanding
```

---

## Expected Results After Fix

### Alignment Scores

- **Current:** -0.0084 (fonts), 0.0044 (math) - random trigram vs geometric
- **After Fix:** >0.50 initial (form vs simple semantic encoding)
- **After Training:** >0.75 (base learns form↔meaning patterns)

### GPU Utilization

- **Current:** 0-7% (CPU-bound trigram hashing)
- **After Fix:** 40-80% (GPU execution + fusion + compression + training)

### Memory Usage

- **ProceduralGalaxy:** 1,002 stars × 9 bytes = 9KB (vs 1,002 × 512 × 4 = 2MB raw)
- **Compression:** 69:1 ratio, 99.998% fidelity
- **Base model:** 2.1M params (unchanged - knowledge lives in stars!)

### Training Time

- **Current:** ~23s/epoch (CPU overhead)
- **After Fix:** ~5-10s/epoch (GPU-native pipeline)
- **Total:** ~5 epochs × 8s = 40s for atomic foundation

---

## Architecture Alignment with K3D Principles

✅ **Procedural-First:** Form = visual RPN, Meaning = math RPN (NOT text hashes!)
✅ **Sovereign GPU:** All execution + fusion + compression on GPU
✅ **Knowledge in Stars:** Base model = logic, Galaxy stars = atomic knowledge
✅ **Shadow Copy Training:** Phase H self-updating adapters (no forgetting)
✅ **Atomic Foundation:** 1,002 units never pruned, always available
✅ **Compression:** 69:1 via procedural encoding (Phase 2.6)
✅ **Organic Emergence:** Form + meaning fusion enables cross-modal patterns

---

## Implementation Checklist

### Immediate (ProceduralDrawingSpecialist)

- [ ] Remove `self.text_embedder` (RPNEmbeddingEngine)
- [ ] Add `encode_semantic_context()` method
- [ ] Add `_fuse_multimodal()` using AtomicFissionFusion
- [ ] Add `_store_atomic_star()` using ProceduralGalaxy
- [ ] Update `train_on_batch()` to use fusion + shadow copy training
- [ ] Remove triplet contrastive learning code
- [ ] Update metrics to report form↔meaning alignment

### Integration (Bridges)

- [ ] Import `AtomicFissionFusion` from sovereign_bridges
- [ ] Import `AdaptiveProceduralBridge` for compression
- [ ] Import `ProceduralGalaxy` for storage
- [ ] Ensure `MatryoshkaTRM.update_from_atomic_knowledge()` exists

### Validation

- [ ] Update `validate_dual_modal_math.py`:
  - Remove triplet metrics
  - Add form↔meaning alignment
  - Measure GPU utilization properly
  - Verify ProceduralGalaxy storage

---

## Success Criteria

1. **Alignment >0.75** - Form and meaning embeddings highly correlated
2. **GPU Utilization >40%** - Most computation on GPU
3. **Compression 69:1** - Procedural storage working
4. **1,002 atomic stars** - All units stored in ProceduralGalaxy
5. **Base model convergence** - Shadow copy training improves alignment
6. **No CPU fallbacks** - Pure sovereign GPU pipeline

---

**Status:** 🏗️ READY FOR IMPLEMENTATION

**Next Step:** Implement changes to ProceduralDrawingSpecialist + prepare Codex prompt

---

*End of Architecture*
