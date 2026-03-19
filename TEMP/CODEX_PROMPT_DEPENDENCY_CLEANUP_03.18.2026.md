# Codex Directive: Dependency Cleanup & Full Test Suite Fix

**Date:** 2026-03-18
**Phase:** Infrastructure maintenance (prerequisite for clean CI)
**Goal:** Make `pytest tests/ -q` collect and run cleanly without `torch` installed. Fix all environmental import failures. Extend `Dockerfile.test` with missing deps.

---

## Context

The repo-wide test suite currently fails to **collect** due to:
1. Missing `torch` — 6 test files import it at module level
2. Missing `fontTools` — used by `knowledge3d/cranium/procedural_fonts.py`
3. Missing `sympy` — declared in `requirements.txt` but not in `Dockerfile.test`
4. Broken `ARCGridProcessor` import — `tests/test_arc_grid_processor.py` imports from `knowledge3d.training.arc_agi` which doesn't exist (class lives in `Old_Attempts/`)

**Principle:** `torch` is a 2GB+ legacy dependency from pre-sovereignty training scripts. It must NOT be added to the test or runtime environments. Instead, gate the imports so tests skip cleanly.

---

## Task 1: Gate `torch` imports in test files with `pytest.importorskip`

These 6 test files fail at collection because they `import torch` at module level:

| File | Line(s) |
|------|---------|
| `tests/test_lstm_sovereign.py` | 9 |
| `tests/test_sovereign_trm_heads.py` | 9 |
| `tests/test_control_token_vocab.py` | 1 |
| `tests/test_sovereign_v7_equivalence.py` | 13 |
| `tests/test_confidence_head_shape.py` | 1 |
| `tests/test_calibration_loss.py` | 1 |

**For each file:**
- Replace the bare `import torch` / `import torch.nn as nn` with `pytest.importorskip("torch")` at the top
- If the file also does `from torch import nn` or `import torch.nn as nn`, assign: `torch = pytest.importorskip("torch")` then `nn = torch.nn`
- This makes every test in the file skip (not fail) when torch is absent
- Do NOT modify any test logic — only the import gating

**Example pattern:**
```python
# BEFORE:
import torch
import torch.nn as nn

# AFTER:
import pytest
torch = pytest.importorskip("torch")
nn = torch.nn
```

---

## Task 2: Fix `tests/test_arc_grid_processor.py`

This file imports `from knowledge3d.training.arc_agi import ARCGridProcessor` but that module doesn't exist in the active tree. The real class is in `Old_Attempts/curriculum_specific_training/arc_agi/grid_processor.py`.

**Fix:** Add a `pytest.importorskip` gate at the top so it skips cleanly:

```python
# At top of file, before the ARCGridProcessor import:
import pytest
pytest.importorskip("knowledge3d.training.arc_agi", reason="ARCGridProcessor moved to Old_Attempts/")
```

Or alternatively, wrap the import in a try/except and skip the entire module:

```python
import pytest
try:
    from knowledge3d.training.arc_agi import ARCGridProcessor
except ImportError:
    pytest.skip("ARCGridProcessor not in active tree", allow_module_level=True)
```

Either approach is fine. The test should **skip**, not **error**.

---

## Task 3: Extend `Dockerfile.test` with missing packages

Current `Dockerfile.test` pip install line (line 15-20):
```dockerfile
RUN pip3 install --no-cache-dir \
    cupy-cuda12x \
    numpy \
    pytest \
    pygltflib \
    scikit-learn
```

**Add these packages** to the same pip install:
- `fonttools>=4.40.0` (needed by `knowledge3d/cranium/procedural_fonts.py`)
- `sympy` (needed by various math tools, already in `requirements.txt`)
- `websockets` (needed by bridge/live_server, already in `Dockerfile.runtime`)

Result should be:
```dockerfile
RUN pip3 install --no-cache-dir \
    cupy-cuda12x \
    numpy \
    pytest \
    pygltflib \
    scikit-learn \
    fonttools>=4.40.0 \
    sympy \
    websockets
```

---

## Task 4: Gate `torch` imports in `knowledge3d/` source files (lazy import pattern)

These source files under `knowledge3d/` import `torch` at module level and will cause `ImportError` when other code transitively imports them:

**High priority (imported by tests or other modules):**
- `knowledge3d/cranium/glb_weights.py:28` — `import torch`
- `knowledge3d/training/math_benchmarks/calibration_loss.py:9-10` — `import torch`, `import torch.nn.functional as F`
- `knowledge3d/training/math_benchmarks/navigation_model.py:9` — `from torch import nn`
- `knowledge3d/training/math_benchmarks/navigation_model_with_confidence.py:9-10` — `import torch`, `from torch import nn`

**Lower priority (standalone scripts, less likely to cause collection failures):**
- `knowledge3d/models/world_model/train.py`
- `knowledge3d/models/world_model/dataset.py`
- `knowledge3d/models/world_model/rssm.py`
- `knowledge3d/models/rlwhf_lora.py`
- `knowledge3d/models/rlwhf_policy.py`
- `knowledge3d/models/spatial_memory_trainer.py`
- `knowledge3d/tools/training_pipelines/weights_in_glb.py`
- `knowledge3d/tools/evaluator_scripts/eval_rlwhf_policy.py`

**Pattern for source files** (NOT test files — don't use pytest here):
```python
# BEFORE:
import torch
import torch.nn as nn

# AFTER:
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
```

Then at the top of any function that actually uses torch, add:
```python
if torch is None:
    raise ImportError("torch is required for this module — install with: pip install torch")
```

This way the module **imports** cleanly (no collection failure), but any actual **usage** gives a clear error.

**For the high-priority files**, apply this pattern. For the lower-priority standalone scripts, apply it if time permits — they're less likely to cause transitive failures.

---

## Task 5: Verify

After all changes, run:

```bash
# In k3d-trm or k3d-cranium env (no torch installed):
pytest tests/ -q --co 2>&1 | tail -20
```

This runs collection only (`--co`). Success = all tests either collected or skipped, ZERO errors.

Then run the actual tests:
```bash
pytest tests/ -q 2>&1 | tail -20
```

Expected: all sovereign tests pass, torch-dependent tests skip, ARCGridProcessor test skips. Zero collection errors.

Also verify the existing non-regression slice still passes:
```bash
pytest tests/test_multilingual_meanings.py tests/test_knowledge_proceduralizer.py tests/test_universal_knowledge.py tests/test_ollama_benchmark.py tests/test_benchmark_health_check.py -q
```

---

## Files to modify

| File | Change |
|------|--------|
| `tests/test_lstm_sovereign.py` | `pytest.importorskip("torch")` |
| `tests/test_sovereign_trm_heads.py` | `pytest.importorskip("torch")` |
| `tests/test_control_token_vocab.py` | `pytest.importorskip("torch")` |
| `tests/test_sovereign_v7_equivalence.py` | `pytest.importorskip("torch")` |
| `tests/test_confidence_head_shape.py` | `pytest.importorskip("torch")` |
| `tests/test_calibration_loss.py` | `pytest.importorskip("torch")` |
| `tests/test_arc_grid_processor.py` | Skip if `knowledge3d.training.arc_agi` missing |
| `Dockerfile.test` | Add `fonttools`, `sympy`, `websockets` |
| `knowledge3d/cranium/glb_weights.py` | Lazy torch import with try/except |
| `knowledge3d/training/math_benchmarks/calibration_loss.py` | Lazy torch import |
| `knowledge3d/training/math_benchmarks/navigation_model.py` | Lazy torch import |
| `knowledge3d/training/math_benchmarks/navigation_model_with_confidence.py` | Lazy torch import |

**Do NOT modify:**
- Any test logic or assertions
- Any non-torch imports
- `scripts/` files (standalone, not imported by tests)
- `Old_Attempts/` (archived, out of tree)

---

## Success Criteria

1. `pytest tests/ --co -q` → zero collection errors (skips OK)
2. `pytest tests/ -q` → all sovereign tests pass, torch tests skip, ARC test skips
3. H19/B3 non-regression slice → 21/21 still pass
4. `Dockerfile.test` builds successfully with new packages
5. No new dependencies added to the project (only gating existing ones)
