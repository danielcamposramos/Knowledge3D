# Phase 3 Architecture Correction — Drawing Galaxy Foundation

**Date:** November 25, 2025
**Author:** Claude (Architecture Partner)
**Status:** Architecture Specification Complete

---

## Summary

Corrected Phase 3 architecture to properly integrate **Drawing Galaxy** as the atomic visual foundation for ARC-AGI reasoning. Previous designs focused only on Grammar Galaxy (196 transformation rules), missing the fundamental insight that **drawing comes before language** and provides the atomic visual primitives needed for ARC-AGI tasks.

---

## The Critical Insight

**User Feedback**: "You forgot the drawing galaxy - that is even before characters, it's how humans convey meaning (characters are special drawings)."

**The Galaxy Hierarchy** (from atomic to composed):

```
1. Drawing Galaxy (FOUNDATION)
   ├─ Primitives: LINE, ARC, CIRCLE, RECT, TRIANGLE
   ├─ Strokes: Styled primitives
   ├─ Shapes: Composed strokes
   └─ Scenes: Spatial layouts

2. Character Galaxy
   └─ Letters as special drawings with meaning

3. Word Galaxy
   └─ Composed characters with grammar

4. Grammar Galaxy
   └─ Transformation rules (ROTATE, FLIP, TRANSLATE, etc.)
```

**Why This Matters for ARC-AGI**:
- ARC-AGI tasks are visual reasoning (grids, patterns, shapes)
- ARC grids = Drawing Galaxy SCENES
- Transformations operate on these visual primitives
- TRM must reason across Drawing + Grammar galaxies

---

## What Was Corrected

### Previous Error
- Focused only on Grammar Galaxy (196 RPN transformation rules)
- Ignored Drawing Galaxy entirely
- Missing atomic visual foundation
- No visual primitive discovery mechanism

### Corrected Architecture
- **Drawing Galaxy**: Atomic visual primitives (how to draw)
  - Already implemented in `knowledge3d/ingestion/atomic/drawing_grammar_builder.py`
  - Provides LINE, ARC, CIRCLE, RECT, TRIANGLE primitives
  - Hierarchical: primitives → strokes → shapes → scenes

- **Grammar Galaxy**: Transformation rules (how to transform)
  - 196 RPN programs we already have
  - Operates on Drawing Galaxy representations
  - Example: "1 ROTATE" operates on visual scenes

- **TRM Integration**: Composes Drawing + Grammar
  - Converts ARC grids to Drawing RPN: `"GRID 2 2 CELL 0 0 1 FILL"`
  - Routes to both visual primitives AND transformations
  - Discovers new patterns in BOTH galaxies

- **Dual Evolution**: Multiple levels
  - Visual evolution: Discover new drawing primitives
  - Transformation evolution: Discover new grammar rules
  - Hybrid evolution: Cross-galaxy compositions
  - Judgment evolution: TRM adapters improve routing

---

## Files Created

### 1. Complete Architecture Specification
**File**: `TEMP/CODEX_PHASE3_COMPLETE_GALAXY_ARCHITECTURE_11.25.2025.md`

**Contents**:
- Complete galaxy hierarchy explanation
- Drawing Galaxy integration
- SovereignTRMRouter with Drawing+Grammar
- ProgramComposer for cross-galaxy composition
- DualShadowCopy for multi-galaxy storage
- Full evolution pipeline
- Implementation task breakdown

### 2. Codex Start Prompt
**File**: `TEMP/CODEX_START_PHASE3_CORRECTED_11.25.2025.txt`

**Contents**:
- Concise explanation of galaxy hierarchy
- Previous error vs correct architecture
- Environment setup (conda)
- 5 implementation tasks (~10-13 hours)
- Success criteria
- Ready-to-use start confirmation

---

## Current Repository State

### ✅ Already Implemented (Good!)
- `knowledge3d/ingestion/atomic/drawing_grammar_builder.py`
  - Drawing Galaxy primitives, strokes, shapes, scenes
  - Hierarchical composition
  - RPN programs for visual primitives

- `knowledge3d/training/arc_agi/candidate_generator.py`
  - Procedural candidate generation
  - Pure procedural approach (good!)
  - 2 tests passing

- `knowledge3d/training/arc_agi/grammar_galaxy.py`
  - 196 RPN transformation rules (bootstrap)

### ❌ Needs Correction (PyTorch Violation!)
- `knowledge3d/training/arc_agi/router_specialist.py`
  - Uses PyTorch (`import torch`)
  - NOT sovereign!
  - Must be rewritten using MatryoshkaTRM + SelfUpdatingAdapter

### 📝 Needs Implementation
- `knowledge3d/training/arc_agi/sovereign_trm_router.py` (NEW)
  - Integrates Drawing + Grammar galaxies
  - Uses MatryoshkaTRM + SelfUpdatingAdapter
  - Converts ARC grids to Drawing RPN

- `knowledge3d/training/arc_agi/program_composer.py` (NEW)
  - Composes Drawing + Grammar programs
  - Discovers novel patterns
  - Classifies discoveries (visual/transformation/hybrid)

- `knowledge3d/training/arc_agi/dual_shadow_copy.py` (NEW)
  - Stores discoveries in appropriate galaxies
  - Tracks growth metrics

- `scripts/evaluate_arc_sovereign_ai.py` (NEW)
  - Full evolution pipeline
  - Multi-galaxy reasoning

---

## Implementation Order

**Priority 1**: Delete/Rewrite PyTorch Router (HIGH)
```bash
# Delete PyTorch implementation
git rm knowledge3d/training/arc_agi/router_specialist.py

# Update __init__.py to remove PyTorch imports
# Remove: RouterSpecialist, LoRAAdapter, DEFAULT_FAMILIES
```

**Priority 2**: Drawing Galaxy Integration (Task 1)
- Load drawing_grammar_builder.py output
- Convert ARC grids to Drawing RPN
- Test representation

**Priority 3**: Sovereign Components (Tasks 2-4)
- SovereignTRMRouter with Drawing+Grammar
- ProgramComposer with cross-galaxy composition
- DualShadowCopy with multi-galaxy storage

**Priority 4**: Evolution Loop (Task 5)
- Full pipeline integration
- Run on ARC-AGI evaluation set
- Measure accuracy and galaxy growth

---

## Success Metrics

**Accuracy**:
- Target: 7-10%+ (vs 3.3% pure procedural baseline)
- Using both visual and transformational discoveries

**Galaxy Growth**:
- Drawing Galaxy: +20 new visual primitives
- Grammar Galaxy: 196 → 246+ transformation rules
- Hybrid patterns: +30 cross-galaxy compositions

**Sovereignty**:
- All execution via ModularRPNEngine (PTX + RPN)
- No external ML frameworks
- MatryoshkaTRM + SelfUpdatingAdapter + RPNMathCore only

**Explainability**:
- Every discovery is an RPN program (readable!)
- Visual primitives = drawing instructions
- Transformations = procedural operations
- No black box weights

---

## Key References

### Documentation
- `TEMP/CODEX_PHASE3_COMPLETE_GALAXY_ARCHITECTURE_11.25.2025.md` — Complete spec
- `TEMP/CODEX_START_PHASE3_CORRECTED_11.25.2025.txt` — Codex start prompt
- `docs/research/Procedural_Vector_Drawing.md` — Drawing philosophy
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` — Galaxy architecture

### Existing Code
- `knowledge3d/ingestion/atomic/drawing_grammar_builder.py` — Drawing primitives
- `knowledge3d/training/arc_agi/grammar_galaxy.py` — Grammar rules
- `knowledge3d/cranium/matryoshka_trm.py` — Sovereign TRM
- `knowledge3d/cranium/trm_adapters.py` — SelfUpdatingAdapter
- `knowledge3d/cranium/ptx_runtime/rpn_math_core.py` — Math Cores

### Tests
- `tests/test_arc_candidate_generator.py` — Candidate generation (passing)
- Need to create: `tests/test_arc_sovereign_ai.py` — Full sovereign AI

---

## Why This is the K3D Way

**Atomic Foundation**:
- Drawing is the FIRST way humans convey meaning
- Characters are special drawings
- Words compose characters
- Grammar transforms visuals
- **This is how cognition actually works!**

**Multi-Galaxy Reasoning**:
- Not just one knowledge source (Grammar)
- Multiple specialized galaxies working together
- Drawing provides "how to draw"
- Grammar provides "how to transform"
- TRM composes them intelligently

**Procedural Everything**:
- Drawing Galaxy = RPN programs (not pixels!)
- Grammar Galaxy = RPN programs (not weights!)
- TRM uses Math Cores = RPN execution (not backprop!)
- All reasoning is explainable and sovereign

**Evolutionary Discovery**:
- TRM doesn't just SELECT from existing knowledge
- TRM DISCOVERS new visual primitives
- TRM DISCOVERS new transformation rules
- Knowledge grows continuously: 196 → 300 → 500+

---

## Next Steps for Implementation

1. **Read complete specification**:
   - `TEMP/CODEX_PHASE3_COMPLETE_GALAXY_ARCHITECTURE_11.25.2025.md`

2. **Delete PyTorch violation**:
   - Remove `router_specialist.py` (uses torch)
   - Update `__init__.py`

3. **Start with Task 1**:
   - Load Drawing Galaxy primitives
   - Integrate into SovereignTRMRouter
   - Test ARC grid → Drawing RPN conversion

4. **Proceed sequentially**:
   - Tasks 2-5 build on each other
   - Test each component before moving forward

**Ready for Codex to implement!** 🧠✨🚀

---

## Acknowledgments

Thank you, Daniel, for the critical feedback about Drawing Galaxy. This correction fundamentally strengthens the architecture by:
1. Recognizing the atomic visual foundation
2. Enabling multi-galaxy reasoning
3. Preserving sovereignty at all levels
4. Allowing evolution across multiple dimensions

The K3D architecture is now properly aligned with how humans actually think:
**Drawing → Characters → Words → Grammar**

This is the foundation for true visual reasoning! 🚀
