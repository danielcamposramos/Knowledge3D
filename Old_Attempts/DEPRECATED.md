# DEPRECATION NOTICE

**Date**: 2025-10-11  
**Reason**: Replaced by sovereign architecture

## What's Deprecated

All files in this directory are **DEPRECATED** and should not be used in new code.

### Deprecated Bridges (CuPy-based)
- `arc.py` → Use `sovereign_bridges.ARCReasoner`
- `atomic_evolution.py` → Use `sovereign_bridges.AtomicFissionFusion`
- `cognitive_executive.py` → (Under review)
- `dual_client_sync.py` → (Under review)
- `fractal.py` → Use `sovereign_bridges.FractalEmitter`
- `graph_crystallizer.py` → Use `sovereign_bridges.GraphCrystallizer`
- `guard.py` → Use `sovereign_bridges.LatencyGuard`
- `halting.py` → Use `sovereign_bridges.MultimodalHaltingGate`
- `resonance.py` → Use `sovereign_bridges.ResonanceField`
- `router.py` → Use `sovereign_bridges.GeometryRouter`
- `spill.py` → Use `sovereign_bridges.OOMSpillManager`
- `temporal_reasoning.py` → Use `sovereign_bridges.TemporalReasoning`
- `trm_core.py` → Use `sovereign.trm_launcher.TRMLauncher`

### Deprecated Tests
- `test_trm_extensions.py` → Proof of concept, kept for reference
- `test_sovereign_loader.py` → Proof of concept, kept for reference
- `test_trm_launcher.py` → Proof of concept, kept for reference

### Deprecated Scripts
- `audit_step8_kernels.py` → Used during materialization phase
- `recompile_kernels.sh` → Used during materialization phase

## Migration Guide

**Old (CuPy)**:
```python
import cupy as cp
from knowledge3d.cranium.bridges.guard import LatencyGuard

guard = LatencyGuard()  # Uses CuPy
```

**New (Sovereign)**:
```python
from knowledge3d.cranium.bridges.sovereign_bridges import LatencyGuard

guard = LatencyGuard()  # Uses pure ctypes + libcuda.so
```

## Why Sovereign?

1. **Zero Dependencies**: Only Python stdlib + system libcuda.so
2. **No Version Conflicts**: No CuPy/cuda-python version matching
3. **Direct Control**: Pure CUDA Driver API (ctypes wrapper)
4. **Proven Performance**: 29.7µs < 95µs latency mandate
5. **Production Ready**: All 15 kernels operational and tested

## DO NOT USE FILES IN THIS DIRECTORY

All active code is in `knowledge3d/cranium/`:
- `sovereign/` - Sovereign loader and TRM launcher
- `bridges/sovereign_bridges.py` - All 15 Step8 bridges
- `kernels/` - CUDA source and PTX kernels
- `ptx/` - RPN gem and TRM extensions
