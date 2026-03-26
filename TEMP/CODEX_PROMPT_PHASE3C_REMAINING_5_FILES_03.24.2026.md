# Codex Prompt: Phase 3C Remaining — 5 Files, 227 numpy Uses, FINISH IT

**Date:** 2026-03-24
**Priority:** COMPLETE THIS. Phase 3B done. LED fixed. Embedding engine done. 5 files left.
**Binding specs:**
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §4.1 — ptx_fallback_rate = 0.0
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` — ALL bridge staging sovereign
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` §3 — VRAM-native, no CPU preprocessing

---

## The 5 Remaining Files — Order and Tips

### File 1: sovereign_bridges.py (33 numpy via `_np()` lazy accessor)

**Pattern:** This file has 24 bridge classes. Most use `_np()` (line 44) as a lazy numpy accessor for staging. The `ModularRPNEngine` class (line 1646+) is ALREADY sovereign — pure ctypes, no numpy. The GRE specialist bridges (lines 57-1644) are the debt.

**The Fix:**
1. DELETE `_np()` helper (line 44-47)
2. Add `from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32` at top
3. Every `np_mod = _np()` pattern becomes dead code — remove it

**Per-class migration (they ALL follow the same pattern):**

The UNIVERSAL pattern in every bridge class is:
```python
np_mod = _np()
arr = np_mod.asarray(input, dtype=np_mod.float32)        # staging
arr = np_mod.ascontiguousarray(arr, dtype=np_mod.float32) # ensure contiguous
result = np_mod.zeros((N,), dtype=np_mod.float32)         # output buffer
d_arr = gpu_malloc(arr.nbytes)                            # alloc
memcpy_htod(d_arr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes)  # upload
# ... kernel launch ...
memcpy_dtoh(result.ctypes.data_as(ctypes.c_void_p), d_result, result.nbytes)  # download
```

Replace with:
```python
arr = HostTensorF32.from_array_like(input)                # staging
result = HostTensorF32.zeros(N, 1)                        # output buffer
d_arr = gpu_malloc(arr.nbytes)                            # alloc
memcpy_htod(d_arr, ctypes.c_void_p(arr.data_ptr), arr.nbytes)  # upload
# ... kernel launch (UNCHANGED) ...
memcpy_dtoh(ctypes.c_void_p(result.data_ptr), d_result, result.nbytes)  # download
```

For uint32/uint64 staging (not float32), use `ctypes` arrays directly:
```python
buf = (ctypes.c_uint32 * N)(*values)
memcpy_htod(d_buf, ctypes.c_void_p(ctypes.addressof(buf)), ctypes.sizeof(buf))
```

**Specific classes with non-trivial patterns:**

- **LatencyGuard** (line 57): `timestamps = _np().zeros(2, dtype=_np().uint64)` → `(ctypes.c_uint64 * 2)()`; `flag = _np().zeros(1, dtype=_np().uint32)` → `(ctypes.c_uint32 * 1)()`

- **GalaxyResonanceEngine** (line 272): `resonated` result → `HostTensorF32`. The method currently returns numpy arrays — callers must accept HostTensorF32 or list.

- **FractalEmitter** (line 470): Uses `np_mod.linspace`, `np_mod.sin`, `np_mod.pi` — replace with:
  ```python
  import math
  linspace = [start + (end - start) * i / (n - 1) for i in range(n)]
  sin_values = [math.sin(v) for v in linspace]
  ```

- **AtomicFissionFusion** (line 662): `np_mod.arange`, `np_mod.stack`, `np_mod.sin`, `np_mod.cos` for atom coordinate generation → Python list comprehension with `math.sin`/`math.cos`.

- **CognitiveExecutive** (line 531): `resonance_matrix (8,8)`, `chain_norms (8,)` → `HostTensorF32(8, 8)` and `HostTensorF32(8, 1)`.

- **SleepClusterRefiner** (line 1431) and **SleepGlyphConsolidator** (line 1506): Sleep-time bridges. Same pattern. Replace numpy staging with HostTensorF32.

- **WorldModelBridge** (line 2206): Likely the most complex. Check its interface carefully.

**Return types:** Many bridge methods currently return `np.ndarray`. Change to `HostTensorF32` or `list`. CHECK CALLERS with:
```bash
rg "LatencyGuard|ARCReasoner|GalaxyResonanceEngine|GeometryRouter|FractalEmitter|CognitiveExecutive|ResonanceField|AtomicFissionFusion|DefeasibleResolver|TemporalReasoning|VectorResonator|GraphCrystallizer|SleepClusterRefiner|SleepGlyphConsolidator|MultimodalHaltingGate|GalaxyMemoryUpdater|WorldModelBridge" knowledge3d/ --type py -l
```

---

### File 2: nine_chain_specialized_bridge.py (31 numpy uses)

**Pattern:** The `SwarmDiagnostics` dataclass (line 23-31) currently holds `np.ndarray` fields. The bridge stages 8×128 chain states and 8×8 resonance matrices.

**The Fix:**
1. DELETE `import numpy as np` (line 18)
2. Replace `SwarmDiagnostics` fields with `HostTensorF32` (for float32 matrices) or `list` (for simple vectors):
   ```python
   @dataclass(frozen=True)
   class SwarmDiagnostics:
       resonance_matrix: HostTensorF32   # (8, 8)
       resonance_raw: HostTensorF32      # (8, 8)
       resonance_weights: list           # [float] × 8
       chain_states: HostTensorF32       # (8, 128)
       chain_norms: HostTensorF32        # (8, 1)
   ```

3. All staging: same pattern as sovereign_bridges.py. `np.asarray` → `HostTensorF32.from_array_like`. `np.zeros` → `HostTensorF32.zeros`. `.ctypes.data_as(ctypes.c_void_p)` → `ctypes.c_void_p(tensor.data_ptr)`.

4. CHECK what reads `SwarmDiagnostics` — likely `knowledgeverse.py` Jarvis methods. Those callers need to accept HostTensorF32.

---

### File 3: router_specialist.py (32 numpy uses)

**Pattern:** Router creates feature vectors for routing decisions. Uses `np.concatenate` for feature assembly, `np.zeros` for weight arrays, `np.exp`/`np.sum` for softmax.

**The Fix:**
1. DELETE `import numpy as np`
2. Feature assembly: `np.concatenate` → Python list concatenation, then `HostTensorF32.from_array_like`
3. Softmax: `np.exp(x) / np.sum(np.exp(x))` → Python scalar math:
   ```python
   import math
   max_val = max(values)
   exps = [math.exp(v - max_val) for v in values]
   total = sum(exps)
   softmax = [e / total for e in exps]
   ```
4. Weight arrays: `np.zeros(N)` → `HostTensorF32.zeros(N, 1)` or `[0.0] * N`

---

### File 4: matryoshka_trm.py (16 numpy uses)

**Pattern:** Matryoshka projection multiplies base weights by projection matrices. Uses numpy for matmul and dimension slicing.

**The Fix:**
1. DELETE `import numpy as np`
2. `np.zeros` for weight initialization → `HostTensorF32.zeros`
3. `np.random.randn` for weight init → `HostTensorF32.random_normal`
4. Matmul: `np.dot(W[:dim, :dim], x)` → `RPNMathCore.matmul_host(W_slice, x)` — or just Python scalar dot product for small dims
5. Dimension slicing: `W[:dim, :dim]` → construct a new `HostTensorF32` with the submatrix

**Smallest file. Should be fast.**

---

### File 5: query_head_substrate.py (64 numpy uses)

**Pattern:** This is the orchestrator — it pulls together embeddings, spatial results, bridge outputs, and scores candidates. The numpy uses are: feature array construction, score sorting, LOD level selection, candidate ranking.

**The Fix (DO LAST — all its inputs are now sovereign):**
1. DELETE `import numpy as np`
2. Feature arrays: `np.array([...])` → `HostTensorF32.from_array_like([...])` or plain lists
3. `np.argsort` for candidate ranking → Python `sorted(range(N), key=lambda i: scores[i])`
4. `np.concatenate` / `np.vstack` → list concatenation + `HostTensorF32.from_array_like`
5. LOD thresholds: `np.searchsorted` → Python `bisect.bisect`
6. Score accumulation: `np.sum`, `np.mean`, `np.max` → `sum()`, `sum()/len()`, `max()`

**This file has the most callers. Check ALL callers when changing return types.**

---

## Execution Order — DO THEM SEQUENTIALLY

1. `sovereign_bridges.py` — validate: `rg "import numpy|_np()" sovereign_bridges.py` = ZERO
2. `nine_chain_specialized_bridge.py` — validate: same grep = ZERO
3. `router_specialist.py` — validate: same grep = ZERO
4. `matryoshka_trm.py` — validate: same grep = ZERO
5. `query_head_substrate.py` — validate: same grep = ZERO

After EACH file:
- `pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py`
- `git diff --check`

After ALL FIVE:
- `rg "import numpy|from numpy" knowledge3d/cranium/` — report the TOTAL count. Target: as close to ZERO as possible in the cranium package.
- Run warm 35% benchmark with live monitor.

---

## RULES

1. Do NOT add CPU fallbacks. Fail-fast.
2. Do NOT import numpy. Use `HostTensorF32`, `DeviceTensor`, `ctypes`, `math`.
3. Do NOT change kernel launch signatures or PTX code.
4. CHECK ALL CALLERS when changing return types.
5. Report numpy count BEFORE and AFTER for each file.
6. `_np()` lazy accessor MUST BE DELETED entirely — it is the root of the sovereignty leak in sovereign_bridges.py.
