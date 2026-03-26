# Codex: EXECUTE Phase 3C — Cut the Remaining 5 Files NOW

**Date:** 2026-03-24
**Context:** You already proved the pattern works — frustum, morton, led_pathfinder, rpn_embedding_engine, trigram_embed_bridge are ALL numpy-free. Tests pass. The pattern is PROVEN. Now apply it 5 more times. No more auditing. EXECUTE.
**Spec grounding:** `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §4.1 — ptx_fallback_rate = 0.0. Every `_np()` call IS a sovereignty violation.

---

## DO THIS IN ORDER. VALIDATE AFTER EACH FILE. DO NOT STOP.

---

### Step 1: sovereign_bridges.py

The ENTIRE numpy debt hangs off ONE function: `_np()` at line 44. Kill it.

**Actions:**
1. DELETE lines 44-47 (`def _np(): ...`)
2. ADD at top: `from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32`
3. ADD at top: `import math`
4. Find-replace pattern — every bridge class does the SAME thing:

   **BEFORE:**
   ```python
   np_mod = _np()
   arr = np_mod.asarray(data, dtype=np_mod.float32)
   arr = np_mod.ascontiguousarray(arr)
   result = np_mod.zeros((N,), dtype=np_mod.float32)
   memcpy_htod(d_arr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes)
   memcpy_dtoh(result.ctypes.data_as(ctypes.c_void_p), d_result, result.nbytes)
   ```

   **AFTER:**
   ```python
   arr = HostTensorF32.from_array_like(data)
   result = HostTensorF32.zeros(N, 1)
   memcpy_htod(d_arr, ctypes.c_void_p(arr.data_ptr), arr.nbytes)
   memcpy_dtoh(ctypes.c_void_p(result.data_ptr), d_result, result.nbytes)
   ```

5. For uint32/uint64 staging (LatencyGuard timestamps, halting gate hashes):
   ```python
   buf = (ctypes.c_uint32 * N)(*values)
   memcpy_htod(d_buf, ctypes.c_void_p(ctypes.addressof(buf)), ctypes.sizeof(buf))
   ```

6. For `np_mod.linspace(a, b, n)`: `[a + (b - a) * i / (n - 1) for i in range(n)]`
7. For `np_mod.sin(x)` / `np_mod.cos(x)`: `math.sin(x)` / `math.cos(x)`
8. For `np_mod.pi`: `math.pi`
9. For `np_mod.tile(arr, (k, 1))`: list comprehension repeating rows
10. For `np_mod.arange(n)`: `list(range(n))` or `[float(i) for i in range(n)]`
11. For `np_mod.stack(arrays)`: `HostTensorF32.from_array_like(nested_list)`

**Return types:** Methods that return `np.ndarray` now return `HostTensorF32` or `list`. This is FINE — callers already go through HostTensorF32-compatible paths after Phase 3B.

**Validate:**
```bash
rg "import numpy|from numpy|_np\(\)" knowledge3d/cranium/bridges/sovereign_bridges.py
# MUST return ZERO
pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py
```

---

### Step 2: nine_chain_specialized_bridge.py

**Actions:**
1. DELETE `import numpy as np` (line 18)
2. ADD: `from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32`
3. Change `SwarmDiagnostics` fields from `np.ndarray` to `HostTensorF32`
4. All staging: same pattern as Step 1

**The bridge creates these arrays for the GPU:**
- Chain states: `(8, 128)` float32 → `HostTensorF32(8, 128)`
- Resonance matrix: `(8, 8)` float32 → `HostTensorF32(8, 8)`
- Chain norms: `(8,)` float32 → `HostTensorF32(8, 1)`
- Weights: `(8,)` float32 → `HostTensorF32(8, 1)`

**Validate:**
```bash
rg "import numpy|from numpy|np\." knowledge3d/cranium/bridges/nine_chain_specialized_bridge.py
# MUST return ZERO
pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py
```

---

### Step 3: router_specialist.py

**Actions:**
1. DELETE `import numpy as np`
2. ADD: `from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32` + `import math`
3. Feature concatenation: `np.concatenate([a, b, c])` → `list(a) + list(b) + list(c)` then `HostTensorF32.from_array_like(flat_list, rows=len(flat_list), cols=1)` if needed
4. Softmax:
   ```python
   max_v = max(values)
   exps = [math.exp(v - max_v) for v in values]
   total = sum(exps)
   result = [e / total for e in exps]
   ```
5. `np.zeros(N)` → `[0.0] * N` or `HostTensorF32.zeros(N, 1)`
6. `np.argmax(arr)` → `max(range(len(arr)), key=lambda i: arr[i])`

**Validate:**
```bash
rg "import numpy|from numpy|np\." knowledge3d/cranium/router_specialist.py
# MUST return ZERO
pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py
```

---

### Step 4: matryoshka_trm.py

**Actions:**
1. DELETE `import numpy as np`
2. ADD: `from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32, RPNMathCore`
3. `np.zeros((D, D))` → `HostTensorF32.zeros(D, D)`
4. `np.random.randn(D, D) * std` → `HostTensorF32.random_normal(D, D, std)`
5. `W[:dim, :dim]` (submatrix slicing) → construct new HostTensorF32 from rows:
   ```python
   sub = HostTensorF32(dim, dim)
   for r in range(dim):
       for c in range(dim):
           sub._buffer[r * dim + c] = self.W._buffer[r * full_dim + c]
   ```
6. Matmul: `np.dot(W, x)` → `RPNMathCore().matmul_host(W, x)`

**Smallest file. Should be fast.**

**Validate:**
```bash
rg "import numpy|from numpy|np\." knowledge3d/cranium/matryoshka_trm.py
# MUST return ZERO
pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py
```

---

### Step 5: query_head_substrate.py

**Actions:**
1. DELETE `import numpy as np`
2. ADD: `from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32` + `import math` + `import bisect`
3. `np.array([...])` → `HostTensorF32.from_array_like([...])` or plain lists
4. `np.argsort(scores)` → `sorted(range(len(scores)), key=lambda i: scores[i])`
5. `np.concatenate` / `np.vstack` → list concat + HostTensorF32 construction
6. `np.searchsorted(thresholds, value)` → `bisect.bisect(thresholds, value)`
7. `np.sum(arr)` → `sum(arr)` (if list) or `sum(arr.flat)` (if HostTensorF32)
8. `np.mean(arr)` → `sum(arr) / len(arr)`
9. `np.max(arr)` → `max(arr)`
10. `np.where(condition)` → `[i for i in range(len(arr)) if condition(arr[i])]`

**This is the BIGGEST file but all its inputs are now sovereign types from the files above.**

**Validate:**
```bash
rg "import numpy|from numpy|np\." knowledge3d/knowledgeverse/query_head_substrate.py
# MUST return ZERO
pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py
```

---

## After ALL 5 — Final Validation

```bash
# Total numpy debt in cranium package
rg "import numpy|from numpy" knowledge3d/cranium/ --type py -c

# Full test pack
pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py tests/test_rpn_sovereignty_phase2.py tests/test_frustum_culling.py tests/test_morton_octree.py tests/test_led_pathfinder.py tests/test_rpn_embeddings.py

# Git hygiene
git diff --check
```

Report the numpy count BEFORE (227) and AFTER for all 5 files. The target is ZERO across these files.

Then launch the warm 35% benchmark with live monitor so we can measure the GPU utilization improvement.

---

## CRITICAL REMINDER

You have ALREADY proven this pattern works 7 times (frustum, morton, led, rpn_embedding, trigram_embed, trm_adapters, adaptive_swarm, rpn_math_core). These 5 files are the SAME pattern. Do not over-think. Do not audit. EXECUTE.

Per `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md`: The TRM IS the avatar. Its brain (Galaxy) lives in VRAM. Every `_np()` call pulls brain activity out of VRAM and into host memory. That is like pulling a game NPC's thoughts out of the GPU and computing them on the CPU. We are ending that TODAY.
