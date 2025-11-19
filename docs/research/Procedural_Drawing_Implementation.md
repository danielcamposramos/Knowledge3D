# Procedural Drawing Implementation Guide

**Version:** 1.0
**Date:** 2025-11-18
**Status:** Stage 2 In Progress (Device-Side RPN Evaluation)

---

## Overview

Implementation of the Procedural Vector Drawing research vision ([Procedural_Vector_Drawing.md](Procedural_Vector_Drawing.md)) for atomic cognition through visual-text alignment.

**Core Principle:** Store *how to reconstruct* (RPN programs), not raw data (pixels).

---

## Architecture Stages

### Stage 1: Offline Dataset Generation ✅ **COMPLETE**

**Status:** Production-ready
**Components:**
- `knowledge3d/ingestion/fonts/font_to_rpn_dataset.py` - Font → RPN conversion
- `knowledge3d/cranium/procedural_fonts.py` - Glyph extraction + RPN encoding
- `knowledge3d/ingestion/fonts/rpn_dataset_loader.py` - Dataset loading utilities

**Capabilities:**
- Parse TTF/OTF fonts via fontTools (CPU-based, offline only)
- Extract Bézier curves as normalized segments
- Convert to RPN programs (MOVE, LINE, SET_COLOR, STROKE_WIDTH)
- Compile RPN to bytecode for GPU consumption
- Export JSONL (human-readable) + NPZ (packed bytecode)
- Style inference (weight/italic → stroke width/color)

**Dataset Format:**
```jsonl
{"font": "DejaVuSans.ttf", "char": "A", "rpn": "1.0 1.0 1.0 1.0 SET_COLOR 1.0 STROKE_WIDTH 0.0 0.5 MOVE ..."}
```

**Usage:**
```bash
bash scripts/k3d_env.sh run python3 -m knowledge3d.ingestion.fonts.font_to_rpn_dataset \
  --fonts /usr/share/fonts/truetype \
  --out /K3D/Knowledge3D.local/datasets/font_rpn_168k.jsonl \
  --emit-bytecode-npz /K3D/Knowledge3D.local/datasets/font_rpn_168k_bytecode.npz
```

**Output:** 168K+ glyph RPN programs from system fonts

---

### Stage 2: Device-Side RPN Evaluation ⏳ **IN PROGRESS**

**Status:** Partial implementation (basic opcodes only)
**Components:**
- `knowledge3d/cranium/kernels/rpn_executor.cu` - PTX RPN VM ✅ **DONE**
- `knowledge3d/cranium/ptx/rpn_executor.ptx` - Compiled kernel ✅ **DONE**
- `knowledge3d/cranium/bridges/procedural_drawing_bridge.py` - Bridge integration ✅ **DONE**
- `tests/test_rpn_executor_gpu.py` - GPU test coverage ✅ **DONE**

**Current Opcodes:** MOVE, LINE, SET_COLOR, SET_LINE_WIDTH
**Pending Opcodes:** QUAD, CUBIC, ARC, STROKE, FILL

**Performance:**
- Latency guard: 26,000 µs (26 ms) - realistic for 3060 target
- Target: <10 µs per RPN program (pending full optimization)

**Next Steps (Codex):**
1. Extend `rpn_executor.cu` with QUAD/CUBIC/ARC opcodes
2. Add STROKE/FILL rasterization integration
3. Optimize stack operations for <10µs target
4. Full test coverage for all opcodes

---

### Stage 3: Training Integration ✅ **READY**

**Status:** Code complete, pending first training run
**Components:**
- `knowledge3d/cranium/specialists/procedural_drawing_specialist.py` - Specialist ✅ **DONE (Claude)**
- `knowledge3d/cranium/ternary_utils.py` - Ternary classification ✅ **DONE (Claude)**
- `scripts/train_adaptive_swarm.py` - Training mode integrated ✅ **DONE (Claude)**

**Training Loop:**
```python
# For each (character, rpn_bytecode) pair:
1. Text embedding:   "A" → RPNEmbeddingEngine → 128-dim vector
2. Visual embedding: RPN bytecode → GPU execute → FractalEmitter → 128-dim vector
3. Cross-modal loss: cosine_distance(text_emb, visual_emb)
4. Update specialist: Minimize distance (pull text ≈ visual together)
```

**Usage:**
```bash
bash scripts/k3d_env.sh run python3 scripts/train_adaptive_swarm.py \
  --mode procedural_drawing \
  --rpn-dataset /K3D/Knowledge3D.local/datasets/font_rpn_168k.jsonl \
  --epochs 10 \
  --matryoshka-dim 512 \
  --batch-size 32
```

**Expected Results:**
- Text-visual alignment: >0.85 cosine similarity for same character
- Cross-font generalization: Train on Arial, validate on DejaVu (>80% accuracy)
- Latency: <100 µs inference (text → visual or visual → text)

**Pending (Codex):**
1. First training run validation
2. SSIM reconstruction fidelity metric
3. Generation quality evaluation (visual → RPN decoder)

---

### Stage 4: Full Sovereignty 🔮 **FUTURE**

**Status:** Deferred (not a blocker for training)
**Scope:**
- PTX TTF parser - parse font binary tables directly on GPU
- Zero fontTools dependency at runtime
- <50 µs font loading latency

**Why Deferred:**
- Stage 1 CPU parsing is acceptable for **offline dataset generation**
- Sovereignty requirement applies to **runtime inference**, not preprocessing
- Stages 2-3 prove the pipeline before committing to complex PTX parser

**When to Implement:** After Stages 2-3 fully validated and training loop proven

---

## Ternary Enhancements (Setun-Inspired)

**Philosophy:** Balanced ternary (-1, 0, +1) for efficient GPU decisions

**Classifications:**
```python
# Font weight: -1 (light), 0 (normal), +1 (bold)
weight_ternary = classify_font_weight(550)  # → 0

# Stroke complexity: -1 (simple), 0 (medium), +1 (complex)
complexity = classify_stroke_complexity(segment_count=15)  # → 0

# Routing decision: -1 (reject), 0 (uncertain), +1 (accept)
decision = ternary_route(confidence=0.85)  # → +1
```

**Applications:**
- **Matryoshka selection:** -1 → 64-dim, 0 → 512-dim, +1 → 2048-dim
- **Style routing:** Apply ternary weight to stroke width scaling
- **Swarm routing:** Ternary confidence gates for specialist dispatch

**Pending (Codex):**
1. Device-side ternary gates in `rpn_executor.cu`
2. Ternary metadata in GlyphDescriptor
3. Integration with router-specialist bootstrap

---

## Success Metrics

### Stage 2 (RPN Executor)
- ✅ Latency <10 µs per RPN program execution
- ⏳ 100% opcode coverage (MOVE, LINE, QUAD, CUBIC, ARC, STROKE, FILL)
- ⏳ Pixel-perfect match with host reference (99.9% SSIM)

### Stage 3 (Training)
- ⏳ Text-visual alignment: cosine similarity >0.85 for same character
- ⏳ Cross-font generalization: Train on Arial, validate on DejaVu (>80% accuracy)
- ⏳ Generative quality: Generated glyphs pass human evaluation (>85% recognizable)

### Stage 4 (Sovereignty - Future)
- ⬜ Zero fontTools imports at runtime
- ⬜ PTX parser handles 95%+ of system fonts without errors
- ⬜ <50 µs font loading latency

---

## Connection to Atomic Cognition

**Learning Path:**
1. **Drawing primitives** (Stage 2) → Model learns curves, lines, arcs
2. **Letter recognition** (Stage 3) → Model learns visual-text alignment ("A" text ≈ "A" visual)
3. **Word composition** (Phase G+) → Model learns spatial grammar ("cat" = c+a+t arranged)
4. **Phrase synthesis** (Phase J) → Model generates complex procedural scenes

**This Enables:**
- **Sovereign OCR** - Visual → text via learned embeddings (no Tesseract)
- **Procedural Text Rendering** - Text → visual via RPN generation
- **Cross-Modal Grounding** - "A" text ≈ "A" visual ≈ /eɪ/ audio
- **W3C Standards** - Procedural Compression, Dual-Texture Fonts

---

## File Manifest

### Kernels
- `knowledge3d/cranium/kernels/rpn_executor.cu` - RPN VM (basic ops)
- `knowledge3d/cranium/kernels/procedural_glyph_rasterizer.cu` - Rasterizer

### Bridges
- `knowledge3d/cranium/bridges/procedural_drawing_bridge.py` - Host orchestration
- `knowledge3d/cranium/bridges/procedural_glyph_bridge.py` - Rasterizer bridge

### Specialists
- `knowledge3d/cranium/specialists/procedural_drawing_specialist.py` - Training specialist
- `knowledge3d/cranium/ternary_utils.py` - Ternary classification utilities

### Ingestion
- `knowledge3d/ingestion/fonts/font_to_rpn_dataset.py` - Dataset builder
- `knowledge3d/ingestion/fonts/rpn_dataset_loader.py` - Dataset loader
- `knowledge3d/cranium/procedural_fonts.py` - Font utilities

### Training
- `scripts/train_adaptive_swarm.py` - Training script (procedural_drawing mode)

### Tests
- `tests/test_rpn_executor_gpu.py` - GPU RPN tests
- `tests/test_procedural_drawing_bridge.py` - Bridge tests
- `tests/test_procedural_fonts.py` - Font utilities tests
- `tests/test_procedural_drawing_performance.py` - Performance benchmarks

---

## Development Workflow

### 1. Generate RPN Dataset (Once)
```bash
bash scripts/k3d_env.sh run python3 -m knowledge3d.ingestion.fonts.font_to_rpn_dataset \
  --fonts /usr/share/fonts/truetype \
  --out /K3D/Knowledge3D.local/datasets/font_rpn_168k.jsonl \
  --emit-bytecode-npz /K3D/Knowledge3D.local/datasets/font_rpn_168k_bytecode.npz
```

### 2. Extend RPN Executor (Codex - Next)
```bash
# Edit knowledge3d/cranium/kernels/rpn_executor.cu
# Add QUAD/CUBIC/ARC opcodes

# Recompile PTX
nvcc -ptx -arch=sm_86 --ptxas-options=-v \
  knowledge3d/cranium/kernels/rpn_executor.cu \
  -o knowledge3d/cranium/ptx/rpn_executor.ptx

# Test
pytest tests/test_rpn_executor_gpu.py -xvs
```

### 3. Run Training (After Stage 2 complete)
```bash
bash scripts/k3d_env.sh run python3 scripts/train_adaptive_swarm.py \
  --mode procedural_drawing \
  --rpn-dataset /K3D/Knowledge3D.local/datasets/font_rpn_168k.jsonl \
  --epochs 10 \
  --matryoshka-dim 512 \
  --batch-size 32
```

### 4. Validate Results
- Check text-visual alignment metrics
- Validate on unseen fonts
- Test generative capability (text → RPN)

---

## Known Issues

1. **Arc approximation** - Still uses host math; device-side version pending
2. **Font metadata caching** - Could be more efficient for large datasets
3. **RPN decoder** - Visual → RPN reverse path not yet implemented
4. **SSIM computation** - Reconstruction fidelity metric placeholder

---

## References

- Research vision: [docs/research/Procedural_Vector_Drawing.md](Procedural_Vector_Drawing.md)
- Swarm architecture: [TEMP/PHASE_H_COMPLETE.md](../../TEMP/PHASE_H_COMPLETE.md)
- Tri-modal completion: [TEMP/PHASE_H_TRIMODAL_COMPLETION.md](../../TEMP/PHASE_H_TRIMODAL_COMPLETION.md)
- W3C standards: [docs/vocabulary/](../vocabulary/)

---

**Last Updated:** 2025-11-18 by Claude (Swarm Partner)
