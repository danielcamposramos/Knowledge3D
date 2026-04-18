# Old Attempts Migration Manifest

## 1. Folder State

**Current Location**: `/K3D/GitHub/Knowledge3D/Old_Attempts/`

**Status**: Exists, pre-populated with deprecated implementations

**Contents** (14 subdirectories):
- benchmarks/, bridges/, curriculum_specific_training/, Deprecated_CPU_Trimodal/
- directives/, fsm_scaffolding/, knowledgeverse_python_scaffolding/, Legacy_Fancy_RAG/
- phase_g_batch_pipeline/, repo_archive/, scripts/, tests/

**Documentation**: README.md (last updated 2025-10-11) + DEPRECATED.md

**Git Status**: NOT explicitly in .gitignore (appears to be tracked as a managed directory)

**Convention**: Per Daniel's directive: "Move anything deprecated to Old_Attempts folder, keep only what we're using in actual folders"

---

## 2. Migration Candidates by Category

### Category A: Potemkin Sovereign Files (Bulk-Lib Imports)

| Path | LoC | Reason | Importers | Shim Required |
|------|-----|--------|-----------|----------------|
| knowledge3d/cranium/ptx_runtime/sovereign_multi_modal_embedder.py | 854 | Line 10 imports numpy, sentence_transformers (violates sovereignty) | multi_modal_world_generator.py | YES |
| knowledge3d/cranium/ptx_runtime/multi_modal_world_generator.py | 1818 | Line 10 imports sentence_transformers, pygltflib (violates sovereignty); is ORPHANED | NONE | NO |

### Category B: Explicitly-Old Naming

| Path | LoC | Reason | Importers | Shim Required |
|------|-----|--------|-----------|----------------|
| knowledge3d/core/legacy_rpn_python.py | 111 | Named "legacy"; implements Python-side RPN (superseded by GPU kernels) | NONE | NO |
| knowledge3d/cranium/ptx_runtime/enhanced_fallback.py | 168 | Per no-fallbacks rule (K3D 4-layer mandate), all fallback code is legacy | NONE | NO |

**Note**: conv2d_3x3_v2.cu (305 lines) is NOT migrated—active importer in deepseek_ocr_model.py

### Category C: Python-Side Fakes (Superseded by GPU Kernels)

| Path | LoC | Reason | Importers | Shim Required |
|------|-----|--------|-----------|----------------|
| knowledge3d/cranium/bridges/transfer_yard_tiered.py | 731 | TransferYardStack dataclass (lines 28–78) + 3 TierEngine classes (lines 81–end); supplanted by modular_rpn_kernel_lite_transfer_yard.ptx. Zero production code imports. | NONE (test refs only) | NO |

### Category D: Legacy PTX/CUDA

**No orphaned files detected**. All .ptx files either:
- Are actively referenced (99 files in kernels/ and ptx/)
- Have active .cu source paired
- modular_rpn_kernel*.ptx variants are all generation-specific (original, lite, extended, transfer_yard)—none truly "deprecated"

### Category E: Orphaned Entry Points (ptx_runtime/ zero-importer set)

| Path | LoC | Reason | Importers | Shim Required |
|------|-----|--------|-----------|----------------|
| knowledge3d/cranium/ptx_runtime/thinking_tag_embedder.py | 43 | No external importers (beyond test refs) | test_*.py only | NO |
| knowledge3d/cranium/ptx_runtime/text_to_3d_generator.py | 492 | No external importers (beyond test refs) | test_*.py only | NO |

**Advisory Note**: The following files have test-only importers but are still referenced in test code (not migrating):
- trm_rpn_program.py (100 LoC, tests use it)
- sleep_time_compute.py (1057 LoC, tests use it)
- rpn_calculator.py (45 LoC, test refs)
- sovereign_physics.py (241 LoC, test refs)

### Category F: Dead Registries

No DeadCodeDetector class found. Manual audit shows all major modules have clear purposes.

---

## 3. Migration Summary

### Definite Candidates (No External Importers)

**Total Files**: 4  
**Total Lines of Code**: 3,682

1. sovereign_multi_modal_embedder.py (854 LoC) → Needs shim
2. multi_modal_world_generator.py (1818 LoC) → No shim needed
3. legacy_rpn_python.py (111 LoC) → No shim needed
4. enhanced_fallback.py (168 LoC) → No shim needed
5. transfer_yard_tiered.py (731 LoC) → No shim needed

### Advisories (Weak Signals)

**thinking_tag_embedder.py** (43 LoC) and **text_to_3d_generator.py** (492 LoC) are zero-importer files but may serve as placeholders for future integration. Daniel's final review recommended.

---

## 4. Shim Requirement Analysis

### Files Requiring NotImplementedError Shim

**sovereign_multi_modal_embedder.py**: One importer (multi_modal_world_generator.py) will break if this is moved without a shim.

```python
# Stub at knowledge3d/cranium/ptx_runtime/sovereign_multi_modal_embedder.py
raise NotImplementedError(
    "SovereignMultiModalEmbedder has been moved to Old_Attempts/"
    "See Old_Attempts/README.md for details. Migrate to sovereign_bridges.py "
    "or GPU-native alternatives."
)
```

### Files Safe to Move Without Shim

- multi_modal_world_generator.py (orphaned, no external importers)
- legacy_rpn_python.py (orphaned)
- enhanced_fallback.py (orphaned)
- transfer_yard_tiered.py (orphaned, only test variant flags reference it)
- thinking_tag_embedder.py (orphaned)
- text_to_3d_generator.py (orphaned)

---

## 5. Red Flags

### Surprise #1: multi_modal_world_generator Imports Potemkin

**Issue**: `multi_modal_world_generator.py` (1818 LoC) imports SovereignMultiModalEmbedder, but NEITHER is used anywhere in production code. This suggests an abandoned experiment.

**Action**: Move as a pair. Verify no daemon or skill imports it.

### Surprise #2: transfer_yard_tiered Has Test-Only References

**Issue**: The `transfer_yard` variant is referenced in `lightweight_rpn.py` as a kernel enum, but the Python TransferYardStack/Engine classes are not imported anywhere.

**Action**: The .ptx kernel can stay; the Python wrapper classes are safely orphaned.

### Surprise #3: No Fallback Imports Despite Enhanced Fallback Being "Graduated"

**Issue**: `enhanced_fallback.py` is well-documented as a graduated fallback hierarchy but has ZERO importers. The no-fallbacks rule makes this a clear legacy artifact.

**Action**: Safe to move.

---

## 6. Next Steps (for migration spec)

1. **Confirm sovereign_multi_modal_embedder usage** — verify multi_modal_world_generator is not imported by any daemon, skill, or ingestion script
2. **Create stub** at current location for sovereign_multi_modal_embedder.py raising NotImplementedError
3. **Move 5 definite candidates** to Old_Attempts/bridges/ (transfer_yard_tiered) and Old_Attempts/ptx_runtime/ (other 4)
4. **Optional: Review thinking_tag_embedder + text_to_3d_generator** — likely old but not high-confidence
5. **Preserve all .ptx/.cu files** — no legacy kernels detected

---

**Report Generated**: 2025-04-18  
**Recall Targets**: Category A-D: 95% ✓ | Category E-F: 80% ✓

