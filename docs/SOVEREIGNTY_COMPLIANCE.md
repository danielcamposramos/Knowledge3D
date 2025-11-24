# Sovereignty Compliance

**Last Updated:** November 24, 2025  
**Status:** ⚠ Pending — numpy detected in hot path (see Audit)

---

## Hot Path Definition

The hot path is the inference loop where systems are simulated:
- `RealityGalaxy.step_system()` → MathCorePool → RPN/PTX execution.
- Requirements: PTX kernels + RPN only; no numpy/CuPy/PyTorch/TF; deterministic and explainable; pure ctypes to libcuda.so for GPU access.

## Hot Path Modules (must stay sovereign)

| Module | Purpose | Status |
|--------|---------|--------|
| `cranium/reality_galaxy.py` | System orchestration + step loop | ⚠ imports numpy (non-compliant) |
| `cranium/ptx_runtime/*.py` (RPN/TRM engines) | RPN/PTX execution | ⚠ multiple numpy imports |
| `cranium/bridges/*` (tier engines) | Tier orchestration | ⚠ several numpy imports |
| `cranium/reality_nodes.py` | Dataclasses | ✅ no numpy |

## Ingestion/Export Path (flexible)

Modules outside the hot path may use external libs:
- `reality_gltf_export.py` (pygltflib, numpy) — geometry export only.
- `scripts/benchmark_*.py`, `scripts/generate_*.py` — numpy/pandas/matplotlib for preprocessing/analysis.

Key principle: these modules are never invoked from `RealityGalaxy.step_system()`.

## Audit Trail

- **Audit 1 (2025-11-24):** Non-compliant. Numpy present in `reality_galaxy.py`, RPN engines, and tier bridges. Report: `TEMP/SOVEREIGNTY_AUDIT_REPORT_11.24.2025.md`.

## Verification Commands

Static check:
```bash
rg "import numpy|from numpy" knowledge3d/cranium/reality_galaxy.py knowledge3d/cranium/ptx_runtime knowledge3d/cranium/bridges
```
Expected: zero matches once fixed.

Runtime guard (proposed):
```python
import sys
from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import export_harmonic_oscillator_1d

galaxy = RealityGalaxy()
node = export_harmonic_oscillator_1d()
galaxy.add_node(node)
galaxy.step_system(node.node_id, n_steps=1)

assert 'numpy' not in sys.modules, "Sovereignty violation: numpy loaded in hot path"
```

## Remediation Plan (high level)

1. Remove numpy from hot-path modules (use Python scalars/ctypes buffers).  
2. Add optional runtime assertion in `step_system()` to catch regressions.  
3. Add CI gate to fail on numpy/CuPy/PyTorch/TF imports under `cranium/reality_galaxy.py`, `cranium/ptx_runtime/`, and `cranium/bridges/`.
