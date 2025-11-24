# CODEX BRIEFING: GPU Kernel Validation & Sovereignty Audit

**Date:** November 24, 2025
**From:** Claude (Architect)
**To:** Codex (Implementation Lead)
**Phase:** GPU Path Validation & Hot Path Sovereignty Verification
**Priority:** CRITICAL — Validate GPU kernels work AND ensure numpy is NOT in hot path

---

## Executive Summary

**Status:** Capacity demonstration complete on CPU path (92/92 tests, 83k/88k/79k steps/sec). GPU kernel tests skipped due to missing cupy. Daniel raised critical concern: **"I really hope numpy is outside the hot path."**

**Your Mission:**
1. **Sovereignty Audit:** Verify numpy is NOT in hot path (PTX+RPN inference loop)
2. **Install cupy:** Follow envs/k3d-cranium.yml specification
3. **Run GPU kernel tests:** Validate PTX kernels + TRM suites
4. **Benchmark GPU vs CPU:** Compare throughput with PTX kernels enabled
5. **Document sovereignty compliance:** Clear hot path vs ingestion boundaries
6. **Update BRIEFING.md:** Mark GPU validation complete

**Timeline:** 0.5-1 day (audit first, then GPU tests)

---

## Context: Why This Matters

### Sovereignty Principle
K3D's core value proposition is **sovereignty**: deterministic, explainable, PTX+RPN hot path with NO opaque ML frameworks.

**Hot Path (MUST be sovereign):**
- PTX kernels compiled from CUDA source
- RPN execution via LightweightRPNEngine/ModularRPNEngine/AdvancedRPNEngine
- Pure ctypes to libcuda.so
- **NO numpy, PyTorch, TF, CuPy in inference loop**

**Ingestion Path (flexible):**
- PDF parsing (PyMuPDF, pdfplumber)
- glTF export (pygltflib, numpy for geometry)
- Data preprocessing (pandas, sklearn)
- **OK to use external libs, just keep them OUT of hot path**

### Current Concern
glTF export implementation likely uses numpy for geometry generation. This is **acceptable IF** it's in the ingestion/export path, NOT in the simulation step loop.

**We need to verify:**
1. `reality_gltf_export.py` is **export only** (not called during `galaxy.step_system()`)
2. Numpy is **not imported** in hot path modules:
   - `ptx_runtime/*.py` (RPN engines)
   - `reality_galaxy.py` (step_system method)
   - `reality_nodes.py` (RealitySystem dataclass)
   - `bridges/*.py` (tier engines)

---

## Task 1: Sovereignty Audit (CRITICAL)

### Objective
**Prove numpy is NOT in the hot path.** Document clear boundaries.

### Audit Checklist

#### 1. Hot Path Module Analysis

**Check these files for numpy imports:**
```bash
# Core hot path modules
grep -r "import numpy" knowledge3d/cranium/ptx_runtime/
grep -r "import numpy" knowledge3d/cranium/reality_galaxy.py
grep -r "import numpy" knowledge3d/cranium/reality_nodes.py
grep -r "import numpy" knowledge3d/cranium/bridges/
grep -r "from numpy" knowledge3d/cranium/ptx_runtime/
grep -r "from numpy" knowledge3d/cranium/reality_galaxy.py
grep -r "from numpy" knowledge3d/cranium/reality_nodes.py
grep -r "from numpy" knowledge3d/cranium/bridges/
```

**Expected Result:** **ZERO matches** in hot path modules.

**If numpy found in hot path:**
- Identify usage (e.g., array operations)
- Refactor to use native Python lists or ctypes arrays
- Move numpy-dependent code to ingestion/export modules

#### 2. Step Loop Trace

**Verify `galaxy.step_system()` does NOT touch numpy:**

```python
# Add at top of reality_galaxy.py temporarily
import sys

def step_system(self, node_id: str, n_steps: int = 1) -> None:
    """Step system forward n_steps."""
    # Check loaded modules
    assert 'numpy' not in sys.modules, "SOVEREIGNTY VIOLATION: numpy in hot path!"

    # ... existing implementation
```

**Run test:**
```bash
PYTHONPATH=. pytest knowledge3d/cranium/tests/test_reality_galaxy.py::test_step_harmonic_oscillator -v
```

**Expected:** Test passes (no assertion error).

**If assertion fails:**
- Trace where numpy is imported (check stack)
- Identify offending module
- Refactor to remove numpy dependency

#### 3. Import Tree Analysis

**Generate import tree to visualize dependencies:**

```bash
# Install pydeps if needed
pip install pydeps

# Generate import graph for hot path
pydeps knowledge3d/cranium/reality_galaxy.py --max-bacon 2 --cluster

# Check if numpy appears in graph
# Should only see: ctypes, dataclasses, typing, etc.
```

#### 4. Export vs Inference Separation

**Verify glTF export is NOT called in hot path:**

```bash
# Search for gltf_export usage in hot path
grep -r "gltf_export" knowledge3d/cranium/ptx_runtime/
grep -r "gltf_export" knowledge3d/cranium/reality_galaxy.py
grep -r "gltf_export" knowledge3d/cranium/bridges/

# Should be ZERO matches
```

**Expected:** glTF export only called from:
- `scripts/` (demo/export scripts)
- `tests/test_reality_gltf_export.py` (export tests)
- NOT from `reality_galaxy.py::step_system()`

### Audit Report

**Document findings in:** `TEMP/SOVEREIGNTY_AUDIT_REPORT_11.24.2025.md`

**Template:**
```markdown
# Sovereignty Audit Report

**Date:** November 24, 2025
**Auditor:** Codex
**Scope:** Hot path modules (PTX runtime, RealityGalaxy, bridges)

---

## Findings

### Hot Path Module Analysis
- `ptx_runtime/`: ✅ No numpy imports
- `reality_galaxy.py`: ✅ No numpy imports
- `reality_nodes.py`: ✅ No numpy imports
- `bridges/`: ✅ No numpy imports

### Step Loop Trace
- `galaxy.step_system()`: ✅ numpy not in sys.modules during execution
- Test: `test_reality_galaxy.py::test_step_harmonic_oscillator` PASSED

### Export Separation
- `reality_gltf_export.py`: ⚠️ Uses numpy for geometry (ACCEPTABLE - export path only)
- glTF export NOT called from hot path: ✅ Verified

---

## Conclusion

**Sovereignty Status:** ✅ COMPLIANT

Hot path remains sovereign:
- PTX kernels + RPN only
- No numpy in inference loop
- glTF export properly isolated to ingestion/export path

**Boundary Documentation:**
- Hot path: `ptx_runtime/`, `reality_galaxy.py::step_system()`, `bridges/`
- Ingestion/Export: `reality_gltf_export.py`, `scripts/`, preprocessing tools

**Recommendation:** Proceed with GPU kernel validation.
```

**Success Criteria:**
- ✅ Hot path modules have ZERO numpy imports
- ✅ `galaxy.step_system()` does not load numpy at runtime
- ✅ glTF export isolated to export scripts (not in inference loop)
- ✅ Audit report documents sovereignty compliance

---

## Task 2: Install CuPy

### Objective
Install cupy to enable GPU kernel tests and TRM suites.

### Installation

**Check environment spec:**
```bash
cat envs/k3d-cranium.yml | grep -A5 cupy
```

**Install cupy matching CUDA version:**
```bash
# Check CUDA version
nvidia-smi | grep "CUDA Version"

# Install cupy (adjust for CUDA 12.x or 11.x)
pip install cupy-cuda12x  # For CUDA 12.x
# OR
pip install cupy-cuda11x  # For CUDA 11.x

# Verify installation
python -c "import cupy; print(cupy.__version__)"
```

**Expected Output:**
```
13.3.0  # or similar version
```

### Validation

**Test cupy import in k3d-cranium environment:**
```bash
conda activate k3d-cranium
python -c "import cupy as cp; x = cp.array([1, 2, 3]); print(x)"
```

**Expected:**
```
[1 2 3]
```

**Success Criteria:**
- ✅ cupy installed in k3d-cranium environment
- ✅ cupy imports without errors
- ✅ Basic GPU array operations work

---

## Task 3: Run GPU Kernel Tests

### Objective
Execute previously skipped GPU kernel and TRM test suites.

### Execution

**Run GPU kernel tests:**
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest knowledge3d/cranium/tests/test_*kernel*.py -v
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest knowledge3d/cranium/tests/test_trm*.py -v
```

**Run full test suite (including GPU tests):**
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest knowledge3d/cranium/tests/ -v
```

### Expected Results

**Test Categories:**
1. **PTX Kernel Compilation:** Tests that PTX kernels compile and execute
2. **TRM (Ternary Reasoning Machine):** Advanced RPN engine tests
3. **GPU-accelerated RPN:** Tests using compiled PTX kernels

**Target Test Count:** ~100-120 tests (92 non-cupy + 8-28 GPU/TRM tests)

**Success Criteria:**
- ✅ All GPU kernel tests PASS
- ✅ All TRM tests PASS
- ✅ Total test count: 100+/100+ (or document failures with rationale)

### If Tests Fail

**Scenario A: PTX kernel compilation errors**
- Check CUDA toolkit installed: `nvcc --version`
- Verify PTX source files exist in `knowledge3d/cranium/ptx_runtime/kernels/`
- Check compilation logs for syntax errors

**Scenario B: TRM tests fail**
- Check if TRM engine depends on cupy
- Verify ternary operations (SIGN, TQUANT, TCMP) work with GPU arrays
- May need to adjust TRM implementation for cupy compatibility

**Scenario C: GPU memory errors**
- Reduce batch size in tests
- Check GPU memory availability: `nvidia-smi`
- May indicate memory leak (investigate MathCorePool)

**Action Required:**
1. Run all GPU/TRM tests
2. Document pass/fail status
3. Debug any failures (provide stack traces)
4. Update test count in BRIEFING.md

---

## Task 4: GPU vs CPU Benchmark

### Objective
Compare throughput with PTX kernels enabled (GPU) vs Python RPN (CPU).

### Benchmark Strategy

**Scenario 1: Single System (PTX vs Python RPN)**
```python
import time
from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool
from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import export_double_pendulum_2d

# CPU path (Python RPN)
pool_cpu = MathCorePool(use_gpu=False)
galaxy_cpu = RealityGalaxy(math_core_pool=pool_cpu)
system_cpu = export_double_pendulum_2d()
galaxy_cpu.add_node(system_cpu)

start = time.perf_counter()
for _ in range(1000):
    galaxy_cpu.step_system(system_cpu.node_id, n_steps=1)
elapsed_cpu = time.perf_counter() - start

# GPU path (PTX kernels)
pool_gpu = MathCorePool(use_gpu=True)
galaxy_gpu = RealityGalaxy(math_core_pool=pool_gpu)
system_gpu = export_double_pendulum_2d()
galaxy_gpu.add_node(system_gpu)

start = time.perf_counter()
for _ in range(1000):
    galaxy_gpu.step_system(system_gpu.node_id, n_steps=1)
elapsed_gpu = time.perf_counter() - start

print(f"CPU: {1000/elapsed_cpu:.1f} steps/sec")
print(f"GPU: {1000/elapsed_gpu:.1f} steps/sec")
print(f"Speedup: {elapsed_cpu/elapsed_gpu:.2f}×")
```

**Expected:** GPU 5-20× faster than CPU for complex systems (double pendulum, TRM).

**Scenario 2: Stress Test (1000 systems, GPU vs CPU)**
```bash
# CPU path
CUDA_VISIBLE_DEVICES=-1 pytest knowledge3d/cranium/tests/test_reality_stress.py::test_1000_systems_concurrent -v

# GPU path
CUDA_VISIBLE_DEVICES=0 pytest knowledge3d/cranium/tests/test_reality_stress.py::test_1000_systems_concurrent -v
```

**Compare throughput:**
- CPU: ~79,667 steps/sec (from previous run)
- GPU: Expected 150k-500k steps/sec (with PTX kernels)

### Benchmark Report

**Add to white paper:** `TEMP/ARCHITECTURE_CAPACITY_ANALYSIS_11.24.2025.md`

**Section to add:**
```markdown
## 9. GPU vs CPU Path Comparison

### Single System (Double Pendulum 2D)
- **CPU (Python RPN):** [X] steps/sec
- **GPU (PTX kernels):** [Y] steps/sec
- **Speedup:** [Y/X]× faster

### Stress Test (1000 systems)
- **CPU (Python RPN):** 79,667 steps/sec
- **GPU (PTX kernels):** [Y] steps/sec
- **Speedup:** [Y/79667]× faster

### Conclusion
GPU path with PTX kernels provides [X]× speedup over Python RPN, validating sovereign hot path performance at scale.
```

**Success Criteria:**
- ✅ GPU path benchmarked for single system + 1000 systems
- ✅ Speedup documented (expected: 5-50× depending on system complexity)
- ✅ White paper updated with GPU vs CPU comparison

---

## Task 5: Document Sovereignty Compliance

### Objective
Create clear documentation showing hot path boundaries and sovereignty compliance.

### Document

**File:** `docs/SOVEREIGNTY_COMPLIANCE.md` (NEW)

**Content:**
```markdown
# Sovereignty Compliance

**Last Updated:** November 24, 2025
**Status:** ✅ COMPLIANT (validated via audit)

---

## Hot Path Definition

The **hot path** is the inference loop where systems are simulated:
- `RealityGalaxy.step_system()` → `MathCore.execute()` → PTX kernels or RPN

**Requirements:**
- ONLY PTX kernels + RPN execution
- NO opaque ML frameworks (PyTorch, TF, CuPy)
- Deterministic, reproducible, explainable
- Pure ctypes to libcuda.so for GPU access

---

## Hot Path Modules (Sovereignty-Critical)

| Module | Purpose | External Deps | Status |
|--------|---------|---------------|--------|
| `ptx_runtime/*.py` | RPN engines, PTX kernel wrappers | ctypes | ✅ |
| `reality_galaxy.py` | System orchestration, step loop | dataclasses, typing | ✅ |
| `reality_nodes.py` | System/node dataclasses | dataclasses | ✅ |
| `bridges/*.py` | Tier engines, orchestrator | None | ✅ |

**Audit Result:** ZERO external ML/array libs in hot path.

---

## Ingestion/Export Path (Flexible)

These modules are **outside** the hot path and CAN use external libraries:

| Module | Purpose | External Deps | Justification |
|--------|---------|---------------|---------------|
| `reality_gltf_export.py` | Export systems to glTF | pygltflib, numpy | Geometry generation |
| `scripts/generate_*.py` | Data generation | numpy, pandas | Preprocessing |
| `scripts/benchmark_*.py` | Performance analysis | matplotlib, numpy | Visualization |

**Key Principle:** These modules are NEVER called from `galaxy.step_system()`.

---

## Audit Trail

### Audit 1 (November 24, 2025)
- **Scope:** Hot path modules (ptx_runtime, reality_galaxy, bridges)
- **Method:** Static analysis (grep), runtime trace (sys.modules), import tree (pydeps)
- **Result:** ✅ No numpy/PyTorch/TF in hot path
- **Report:** [TEMP/SOVEREIGNTY_AUDIT_REPORT_11.24.2025.md](../TEMP/SOVEREIGNTY_AUDIT_REPORT_11.24.2025.md)

---

## Compliance Verification

**Run audit:**
```bash
# Check hot path modules for numpy
grep -r "import numpy" knowledge3d/cranium/ptx_runtime/
grep -r "import numpy" knowledge3d/cranium/reality_galaxy.py
grep -r "import numpy" knowledge3d/cranium/bridges/

# Should return ZERO results
```

**Runtime check:**
```python
import sys
from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import export_harmonic_oscillator_1d

# Verify numpy not loaded
galaxy = RealityGalaxy()
system = export_harmonic_oscillator_1d()
galaxy.add_node(system)
galaxy.step_system(system.node_id, n_steps=10)

assert 'numpy' not in sys.modules, "Sovereignty violation!"
```

---

## Recommendations

1. **CI/CD Gate:** Add sovereignty check to CI pipeline (fail build if numpy in hot path)
2. **Import Guards:** Add runtime assertions in `reality_galaxy.py::step_system()` to catch violations
3. **Documentation:** Update BRIEFING.md with sovereignty compliance badge
```

**Success Criteria:**
- ✅ `docs/SOVEREIGNTY_COMPLIANCE.md` created
- ✅ Clear hot path vs ingestion boundaries documented
- ✅ Audit trail established
- ✅ Compliance verification commands provided

---

## Task 6: Update BRIEFING.md

### Updates Required

**Version bump:**
```markdown
**Version:** 3.3 (Phase 5 Capacity + GPU Validation Complete; Phase 6 UI Integration Ready)
```

**Current Status section:**
```markdown
**Current Status**
- Phase 4C complete: 26 systems across 4 domains; 92/92 CPU tests + [X]/[X] GPU tests passing.
- Phase 5 validated: Dynamic Math Core spawning operational; capacity demonstrated at 79k-88k steps/sec (CPU), [Y]k+ steps/sec (GPU).
- Sovereignty audit: ✅ COMPLIANT - numpy NOT in hot path; PTX+RPN inference loop verified.
- GPU kernel tests: ✅ [X]/[X] passing (cupy installed, PTX kernels validated).
- Capacity artifacts: benchmarks (CPU+GPU), glTF exports, white paper complete.
```

**Success Metrics section:**
```markdown
## Success Metrics
- Hot path sovereign: ✅ Audit verified (no numpy/PyTorch/TF in inference loop).
- Tests green: [X]/[X] total (92 CPU + [Y] GPU/TRM).
- GPU validation: PTX kernels operational, [Z]× speedup over CPU.
- Capacity: 1000 systems at 79k steps/sec (CPU), [Y]k steps/sec (GPU).
- Documentation: Sovereignty compliance guide published.
```

**Action Required:**
1. Update version to 3.3
2. Add GPU test count and throughput
3. Add sovereignty audit status
4. Link to `docs/SOVEREIGNTY_COMPLIANCE.md`

---

## Task 7: Commit Strategy

**Commit 1: Sovereignty audit + documentation**
```bash
git add docs/SOVEREIGNTY_COMPLIANCE.md TEMP/SOVEREIGNTY_AUDIT_REPORT_11.24.2025.md
git commit -m "docs(sovereignty): audit hot path compliance, verify numpy NOT in inference loop"
```

**Commit 2: CuPy installation + GPU test results**
```bash
git add knowledge3d/cranium/tests/test_*kernel*.py knowledge3d/cranium/tests/test_trm*.py
git commit -m "test(gpu): validate PTX kernels and TRM suites with cupy"
```

**Commit 3: GPU benchmarks + white paper update**
```bash
git add TEMP/ARCHITECTURE_CAPACITY_ANALYSIS_11.24.2025.md scripts/benchmark_gpu_vs_cpu.py
git commit -m "feat(benchmark): GPU vs CPU throughput comparison ([X]× speedup)"
```

**Commit 4: BRIEFING.md update**
```bash
git add BRIEFING.md
git commit -m "docs: Phase 5 complete - GPU validation + sovereignty audit passed"
```

---

## Success Criteria (Your Mission)

When you're done, we should have:

1. ✅ **Sovereignty audit complete:** Numpy NOT in hot path (documented proof)
2. ✅ **CuPy installed:** GPU kernel tests runnable
3. ✅ **GPU tests passing:** [X]/[X] PTX kernel + TRM tests green
4. ✅ **GPU benchmarks:** [Y]× speedup documented
5. ✅ **Compliance docs:** `docs/SOVEREIGNTY_COMPLIANCE.md` + audit report
6. ✅ **BRIEFING.md updated:** Phase 5 marked complete, GPU validated
7. ✅ **4 clean commits:** Audit, GPU tests, benchmarks, docs

---

## Questions for Claude (Architect)

1. **Sovereignty violations found:** If numpy IS in hot path, how aggressive should refactor be? (Immediate fix vs Phase 7?)
2. **GPU test failures:** If PTX kernels don't compile, should we defer to Phase 7 or debug now?
3. **Performance targets:** What's acceptable GPU speedup? (5× minimum? 20× ideal?)

---

## Closing

Codex, this is critical. Daniel raised the sovereignty concern directly: **"I really hope numpy is outside the hot path."**

Your first priority is the **audit**. Prove numpy is NOT in the inference loop. If it is, we fix it immediately.

Then validate the GPU path works (cupy, PTX kernels, TRM). Show that the sovereign hot path performs.

This isn't about features—it's about **trust**. K3D's value is sovereignty. Let's prove it.

— Claude (Architect)
