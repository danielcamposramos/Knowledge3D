# FSM Scaffolding - Deprecated (Step 12 Consolidation)

**Status**: DEPRECATED as of Step 12
**Reason**: Functionality harvested into ThinkingTagBridge sovereign architecture
**Action**: Do not use these files for new development

## Overview

This directory contains the original Step 6 Fused Head FSM implementation that has been **deprecated** in favor of the sovereign ThinkingTagBridge architecture (Step 10/11).

During **Step 12 FSM Consolidation**, the valuable patterns from this FSM implementation were harvested and integrated into the active ThinkingTagBridge, while the CuPy-dependent scaffolding was retired.

## Why Deprecated?

The Step 6 FSM had architectural issues:

1. **Duplicate Functionality**: Replicated what ThinkingTagBridge already did better
2. **CuPy Dependency**: Used CuPy instead of sovereign ctypes architecture
3. **Incomplete Implementation**:
   - SIMD warp fusion never finished (duplicate logic)
   - Multi-head attention remained as stub
   - State logging was hard-coded
4. **Not Used**: Production code never called the full FSM (used mini FSM instead)

## What Was Harvested (Now in ThinkingTagBridge)

The following patterns were successfully harvested from the FSM and integrated into `knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py`:

### ✓ 5-State Cognitive Pipeline
- **CognitiveStage class**: INGEST → FUSE → SPATIAL → REASON → OUTPUT
- **State tracking**: Full observability with microsecond-precision timing
- **State trace**: Exportable JSON traces for debugging and analysis
- **Location**: [thinking_tag_bridge.py:38-55](../../knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py#L38-L55)

### ✓ ActionBuffer Integration
- **Unified action buffer**: 288-byte GPU buffer contract
- **Tag-to-action mapping**: Thinking tags → ActionType
- **Modal signature encoding**: Multi-modal bitfield encoding
- **Location**: [thinking_tag_bridge.py:749-831](../../knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py#L749-L831)

### ✓ Dynamic LOD Tuning
- **Morton-based saliency**: Level-of-detail adjustment during SPATIAL stage
- **PTX kernel integration**: Uses `dynamic_lod_tune.ptx` from Step 6
- **Threshold-based tuning**: Configurable saliency thresholds
- **Location**: [thinking_tag_bridge.py:833-869](../../knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py#L833-L869)

### ✓ State Transition Observability
- **Per-stage timing**: Percentile statistics (p50, p95, p99)
- **Transition logging**: Full state machine trace with timestamps
- **Telemetry integration**: FSM traces exported to telemetry system
- **Location**: [thinking_tag_bridge.py:257-447](../../knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py#L257-L447)

## Deprecated Files

### PTX Kernels (`ptx/`)
- **`fused_head_fsm_full.ptx`**: 5-state FSM dispatcher (functionality now in ThinkingTag)
- **`warp_modality_fuse_simd.ptx`**: Duplicate scalar kernel (SIMD rewrite never finished)

### Python Orchestration (`python/`)
- **`unified_fsm.py`**: CuPy-based FSM orchestrator (replaced by ctypes sovereign bridge)
- **`fused_head.py`**: AdaptedFusedHead wrapper (replaced by ThinkingTagBridge)

## Migration Path

If you have code using the deprecated FSM:

### Old Code (Deprecated):
```python
from knowledge3d.cranium.unified_fsm import UnifiedFSMContext
from knowledge3d.cranium.fused_head import AdaptedFusedHead

# OLD: CuPy-based FSM
fsm = UnifiedFSMContext()
head = AdaptedFusedHead()
result = head.forward(input_emb)
```

### New Code (Active):
```python
from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagBridge

# NEW: Sovereign ThinkingTag with FSM patterns
bridge = ThinkingTagBridge()
result = bridge.inference(input_emb, modal_signature)

# Access FSM-harvested features
state_trace = bridge.get_state_trace_report()  # 5-state observability
action_buffer = result.action_buffer  # Unified ActionBuffer
```

### Key Differences:
1. **No CuPy**: Pure ctypes + libcuda.so (zero dependencies)
2. **ActionBuffer included**: Every inference returns ActionBuffer for ActionRouter
3. **State observability**: Built-in 5-state trace with timing statistics
4. **Dynamic LOD**: Morton saliency-based LOD tuning during SPATIAL stage
5. **<35µs latency**: Maintains sub-35µs latency target with LatencyGuard

## Testing

The FSM harvest is validated by:
- **`tests/test_step12_fsm_harvest.py`**: Comprehensive FSM pattern verification
- **Existing tests**: All Step 10/11 tests continue passing with FSM patterns

To verify FSM patterns are working:
```bash
cd /path/to/Knowledge3D
export PYTHONPATH=.
python3 tests/test_step12_fsm_harvest.py
```

Expected output:
```
================================================================================
STEP 12: FSM HARVEST VERIFICATION TESTS
================================================================================

✓ CognitiveStage class harvested correctly from FSM
✓ All 7 FSM-harvested methods present in ThinkingTagBridge
✓ State transition recording structure validated
✓ Modal signature encoding logic validated
✓ Tag-to-action mapping logic validated
✓ inference() method documented with Step 12 FSM integration
✓ ActionBuffer availability flag present and True
✓ State timing structure validated

================================================================================
✓ ALL STEP 12 FSM HARVEST TESTS PASSED
================================================================================
```

## Related Tests (Now Deprecated)

The following FSM-specific tests are no longer maintained:
- `tests/test_unified_fsm.py` - UnifiedFSMContext tests (FSM scaffolding)
- `tests/test_unified_pipeline_end_to_end.py` - Pipeline tests using old FSM
- `tests/test_dynamic_lod.py` - Dynamic LOD tests (now in ThinkingTag)

These tests reference the deprecated FSM code and should not be used for new development.

## Historical Context

### Step 6: FSM Creation
- Created unified FSM with 5-state dispatch loop
- Designed ActionBuffer contract (288 bytes)
- Implemented dynamic LOD with Morton saliency
- Built regression scaffolding

### Step 10/11: Sovereign Runtime
- Created ThinkingTagBridge as sovereign inference engine
- Implemented ctypes-based loader (zero CuPy dependency)
- Integrated 22 PTX kernels via sovereign bridges
- Achieved <35µs latency target

### Step 12: Consolidation
- **Phase 1**: Harvested FSM patterns into ThinkingTag ✓
- **Phase 2**: Deprecated FSM scaffolding ✓
- **Phase 3**: Parallel swarm development (Training loop, testing, ActionRouter integration)

## Architecture Decision

**Why keep ThinkingTag over FSM?**

The ThinkingTagBridge sovereign stack already carries everything the roadmap demands:
- Runs entirely through ctypes loader (zero external dependencies)
- Composes existing PTX kernels (ResonanceField, AdaptiveSparsity, GraphCrystallizer, etc.)
- Enforces <35µs budget with LatencyGuard/Profiler
- Feeds ActionRouter/SleepTime along Galaxy–House–Tablet flow
- **In active production use** across Step 10/11 pipelines

The FSM gave valuable structure, but it never matured into a GPU-centric runtime. Rather than refactor working code, we harvested the FSM's best ideas and integrated them into the living brain.

## Questions?

For questions about the FSM deprecation or migration:
1. Review Step 12 master plan: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/TEMP/STEP12_FSM_CONSOLIDATION_MASTER_PLAN.md`
2. Check ThinkingTagBridge implementation: `knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py`
3. Run FSM harvest tests: `tests/test_step12_fsm_harvest.py`

---

**Last Updated**: October 13, 2025
**Step**: 12 (FSM Consolidation)
**Status**: Phase 1 & 2 Complete ✓
