# Codex Prompt: Phase 3B — Spatial Navigation NumPy Elimination

**Date:** 2026-03-24
**Priority:** CRITICAL — These 3 files fire EVERY QUESTION. 5954 times in a 35% run.
**Binding specs:**
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` — "NPC perception = Frustum culling + LOD." Game NPCs do NOT copy their octree/frustum to host every frame.
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §3 — HOUSE_CONTEXT (2.5GB) and WORLD_VIEW (2GB) are VRAM regions. Spatial structures LIVE there.
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §4.1 — "ptx_fallback_rate MUST be 0.0." CPU fallbacks in spatial code = sovereignty violation.
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` §3 — "No CPU preprocessing." Spatial queries operate on device-resident data.

**Prerequisite:** Phase 3A is complete. Device-side transpose works. Adapter gradient is down to 1 `copy_to_device` per step.

---

## Overview

Three files. Migrate them IN THIS ORDER. Validate EACH before moving to the next. Do NOT half-migrate.

1. `frustum.py` (39 numpy uses) — simplest, self-contained
2. `morton_octree.py` (45 numpy uses) — medium, has device-resident index pattern already
3. `led_pathfinder.py` (62 numpy uses) — hardest, has CPU fallback paths that must go

---

## File 1: frustum.py — Line-by-Line Tips

**Pattern:** This file is ALMOST sovereign already. It loads a PTX kernel, launches it via `loader`, reads results back. The numpy uses are ALL staging — creating arrays to upload and receive results. Replace numpy with `ctypes` arrays and `HostTensorF32` where applicable.

### Imports
- Line 14: `import numpy as np` — DELETE. Replace with `import ctypes` (already imported) and `from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32`.

### `__init__`
- Line 55: `self._zero_template = np.zeros(0, dtype=np.uint8)` — Replace with `self._zero_template_size = 0` and `self._zero_template_ptr = None`. Use a `(ctypes.c_uint8 * N)()` allocated once.
- Line 79: Same pattern — resize the ctypes zero buffer instead of numpy.
- Line 87: Same pattern in `close()`.

### `upload_view_projection`
- Lines 101-113: The view-projection and view matrices are 4x4 float32 = 64 bytes each. Use `HostTensorF32(4, 4)` or directly `(ctypes.c_float * 16)()`.

  ```python
  def upload_view_projection(self, view_proj, view=None):
      vp = HostTensorF32.from_array_like(view_proj, rows=4, cols=4)
      vm = HostTensorF32.from_array_like(view if view is not None else view_proj, rows=4, cols=4)
      loader.memcpy_htod(self._view_proj_ptr, ctypes.c_void_p(vp.data_ptr), vp.nbytes)
      loader.memcpy_htod(self._view_matrix_ptr, ctypes.c_void_p(vm.data_ptr), vm.nbytes)
      self._view_proj_uploaded = True
  ```

### `cull_nodes`
- Lines 127-134: Position and candidate validation. Use `HostTensorF32.from_array_like()` for positions. For candidate_indices (uint32), use `(ctypes.c_uint32 * N)()` directly — `HostTensorF32` is float32 only.
- Lines 141: Empty return — use `(ctypes.c_uint32 * 0)()` or just return an empty list. The CALLER must also accept non-numpy return. Check what calls `cull_nodes` and what type it expects.
- Lines 148-157: `np.ascontiguousarray` staging → `HostTensorF32` for positions, ctypes array for candidates, ctypes zero buffer for flags.
- Lines 151-157: `loader.gpu_malloc` + `loader.memcpy_htod` — already sovereign. Just change the source from numpy `.ctypes.data` to HostTensorF32 `.data_ptr` or ctypes array address.
- Lines 181-184: Download flags and filter. This is the KEY numpy use:
  ```python
  flags_host = np.empty(count, dtype=np.uint8)
  loader.memcpy_dtoh(ctypes.c_void_p(flags_host.ctypes.data), self._flags_ptr, count)
  visible_indices = candidates_contig[flags_host.astype(bool)]
  ```
  Replace with:
  ```python
  flags_buf = (ctypes.c_uint8 * count)()
  loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(flags_buf)), self._flags_ptr, count)
  visible = [cand_buf[i] for i in range(count) if flags_buf[i]]
  result = (ctypes.c_uint32 * len(visible))(*visible)
  ```
  This is the HONEST trade-off: without numpy boolean indexing, you loop in Python. BUT: this loop is O(N) over the cull results, which is tiny (typically hundreds of nodes). The alternative is a GPU compaction kernel (write later). For now, the Python loop is acceptable because it replaces a numpy IMPORT, not a GPU kernel.

### Matrix helpers (lines 244-280)
- `create_perspective_matrix` and `create_view_matrix` use numpy for trig and matrix construction. Replace with `math.radians`, `math.tan`, `math.sqrt`, and `HostTensorF32` for the 4x4 output. These are called ONCE at boot or when the view changes — not per-question.

**IMPORTANT:** The return type of `cull_nodes` currently is `np.ndarray`. Callers will expect this. Check ALL callers of `cull_nodes` and `create_perspective_matrix` / `create_view_matrix`:
- If callers use numpy operations on the result, they need to be adapted too.
- If callers just iterate or index, a ctypes array or list works fine.
- Search: `rg "cull_nodes|cull_from_octree|create_perspective|create_view_matrix" knowledge3d/`

---

## File 2: morton_octree.py — Line-by-Line Tips

**Pattern:** Morton octree already HAS device-resident index pointers (`_d_positions`, `_d_sorted_codes`, `_d_sorted_indices`). The numpy uses are in ENCODING (staging points for PTX kernel) and QUERYING (staging results back). The sorting still uses `np.argsort` — that is the BIGGEST sovereignty violation here.

### Imports
- Line 12: `import numpy as np` — DELETE. Add `HostTensorF32` import.

### `__init__`
- Line 34: `self._gt_program = np.array([0x0000, 0x0001, 0x0028], dtype=np.uint16)` — Replace with `(ctypes.c_uint16 * 3)(0x0000, 0x0001, 0x0028)`.
- Line 35: `self._dummy_vectors = np.zeros((1, 3), dtype=np.float32)` — Replace with `HostTensorF32.zeros(1, 3)` or `(ctypes.c_float * 3)()`.
- Lines 37-40: `self._last_bounds`, `_last_positions`, `_last_sorted_codes`, `_last_sorted_indices` — these are currently numpy arrays cached on host. They are the HOST COPIES of device-resident data. After Phase 3B, the device pointers (`_d_positions`, etc.) are the SOURCE OF TRUTH. The host copies exist only for checkpoint/debug.

### `encode`
- Lines 61-108: The encoding function stages points, launches PTX, reads back codes. Replace numpy staging with ctypes:
  - `np.ascontiguousarray(points, dtype=np.float32)` → `HostTensorF32.from_array_like(points)` (already handles flat list, nested list, or HostTensorF32 input)
  - `pts.min(axis=0)`, `pts.max(axis=0)` → Python loop over `HostTensorF32` rows. OR: upload to device, run a min/max RPN kernel. For now, Python loop is OK — `encode` is called at BOOT, not per-question.
  - `np.zeros(n, dtype=np.uint32)` for codes output → `(ctypes.c_uint32 * n)()`
  - `codes.ctypes.data` → `ctypes.addressof(codes_buf)`

### `sort` (THE BIG ONE)
- Line 120: `np.argsort(codes, kind="mergesort")` — This is a SOVEREIGNTY VIOLATION. Sorting should be a PTX kernel or RPN program. BUT: sorting happens at BUILD TIME (once), not per-question. For Phase 3B, replace `numpy` with Python's `sorted()` with key:
  ```python
  indexed = sorted(range(n), key=lambda i: codes_list[i])
  order = (ctypes.c_uint32 * n)(*indexed)
  values = (ctypes.c_uint32 * n)(*(codes_list[i] for i in indexed))
  ```
  This removes the numpy import. A GPU radix sort kernel is the Phase 4 target — not Phase 3B. Document the debt.

### `query_radius`
- Lines 152-244: This is the PER-QUESTION hot path. The numpy uses here are:
  - Line 171: `np.asarray(query_center)` → `HostTensorF32.from_array_like(query_center, rows=3, cols=1)` or just extract 3 floats.
  - Lines 175-177: `np.zeros(1, dtype=np.uint32)` for count reset → `(ctypes.c_uint32 * 1)(0)`.
  - Lines 196-198: Same pattern for reading back count.
  - Lines 203, 238: `np.zeros(N, dtype=np.uint32)` for results → `(ctypes.c_uint32 * N)()`.
  - ALL memcpy sources: change from `np.ctypes.data` to `ctypes.addressof(buf)`.

### `_encode_query_point`
- Lines 316-328: Uses `self._last_bounds` (numpy arrays) for normalization. After migration, `_last_bounds` should be a pair of `HostTensorF32` or plain Python tuples of 3 floats. The arithmetic here is trivial Python scalar math — no numpy needed.

### `_upload_query_index`
- Lines 330-360: Uploads positions/codes/indices to device. Replace `np.ascontiguousarray` with `HostTensorF32` (for float32 positions) and ctypes arrays (for uint32 codes/indices).

### `_ensure_query_capacity`
- Line 376: `np.dtype(np.uint32).itemsize` → just use `4` (uint32 is 4 bytes). Or `ctypes.sizeof(ctypes.c_uint32)`.

---

## File 3: led_pathfinder.py — Line-by-Line Tips

**Pattern:** This file has EXPLICIT CPU FALLBACKS — `_navigate_csr_cpu`, `cpu_distances`, `return cpu_distances`. These are SOVEREIGNTY VIOLATIONS per KNOWLEDGEVERSE §4.1. The file also uses `heapq` (Python priority queue) and `np.linalg.norm` for distance calculations.

### The Honest Scope Decision

LED-A* has TWO code paths:
1. **GPU path** (`navigate_csr` with `self._astar_kernel`) — uses numpy only for staging. Migration same pattern as frustum/morton.
2. **CPU fallback** (`_navigate_csr_cpu`, `compute_distances` fallback) — full Python/numpy A* implementation.

Per sovereignty spec: **CPU fallbacks MUST GO.** If the GPU kernel fails, we fail-fast and fix the kernel. We do NOT fall back to CPU.

BUT: the CPU A* implementation (`_navigate_csr_cpu`) is currently called when `num_vertices > 4096` (line 232). This is a REAL limitation of the PTX kernel, not a lazy fallback. The fix is either:
- Increase the PTX kernel's vertex limit (kernel change — flag for future)
- Fail-fast with clear error if graph exceeds 4096 vertices
- For Phase 3B: remove numpy from BOTH paths, but keep the Python A* as a TEMPORARY fallback with a loud warning. Mark it as sovereignty debt.

Recommend: Remove numpy, replace with ctypes staging. Keep `_navigate_csr_cpu` as a Python-only (no numpy) implementation for the >4096 vertex case. Add a sovereignty warning log. Flag the PTX kernel vertex limit as Phase 4 debt.

### Imports
- Line 16: `import numpy as np` — DELETE.
- Line 12: `import heapq` — KEEP. `heapq` is stdlib, not a bulk library. It implements the priority queue for the Python A* path.

### `__init__`
- Line 63: `np.array` for RPN program → `(ctypes.c_uint16 * 3)(...)`.
- Line 64: `np.zeros` for dummy vectors → `HostTensorF32.zeros(1, 3)`.

### `compute_distances`
- Lines 70-123: Replace ALL numpy staging with ctypes/HostTensorF32.
- Line 85: `np.linalg.norm(pts - ref, axis=1)` — This is a CPU L2 distance used as the "truth" to validate the GPU result. Remove it entirely. Trust the GPU kernel. If the kernel fails, fail-fast.
- Lines 115-117: `np.allclose(out, cpu_distances)` comparison then fallback — REMOVE. This is the anti-pattern. Trust the GPU result or fail.
- Lines 87-88, 118-119: `if self._dist_kernel is None: return cpu_distances` — Change to `raise RuntimeError`. No fallback.

### `rpn_priority_queue_pop`
- Lines 125-162: Replace `np.asarray` with ctypes arrays or plain Python lists. The RPN comparison itself is sovereign. The numpy is just staging.

### `find_path`
- Lines 164-201: Replace all numpy staging. `np.vstack` → list concatenation or `HostTensorF32`. `np.allclose` → Python `all(abs(a-b) < eps for ...)`.

### `navigate_csr`
- Lines 203-318: Replace numpy staging for CSR arrays with ctypes. The GPU kernel launch is already sovereign.
- Lines 232-241: CPU fallback for >4096 vertices — keep but add warning.
- Lines 290-298, 301-310: RuntimeError fallback and kernel-failure fallback — change to fail-fast. If GPU kernel returns empty path, raise, do NOT fall to CPU.

### `_navigate_csr_cpu`
- Lines 323-366: Replace numpy with plain Python. `row_offsets[node]` → `int(row_offsets_list[node])`. The heapq-based A* works on Python ints/floats. Return a `(ctypes.c_uint32 * len(path))()` or list.

### `_intersects_obstacle` and `_create_detour`
- Lines 368-407: Replace numpy vector math with HostTensorF32 or plain Python scalar math. These are small vectors (3 elements). `np.linalg.norm` → `math.sqrt(sum(x*x for x in v))`. `np.dot` → `sum(a*b for a,b in zip(u,v))`. `np.cross` → explicit 3-element cross product. `np.outer` → explicit loop. `np.clip` → `max(lo, min(hi, x))`.

---

## Return Types: Critical Compatibility Note

ALL THREE FILES currently return `np.ndarray`. Callers depend on this. You MUST check every caller:

```bash
rg "cull_nodes|cull_from_octree|query_radius|build_tree|find_path|navigate_csr|compute_distances|create_perspective|create_view_matrix" knowledge3d/ --type py
```

If callers use numpy operations on the result (e.g., `result[mask]`, `result.shape`, `np.concatenate`), those callers need adaptation too. If callers just iterate or index, a list or ctypes array works.

**Strategy:** Return `list` or `ctypes.Array` from migrated functions. Adapt immediate callers in the same PR. Do NOT adapt callers-of-callers — that is Phase 3C.

---

## Validation Per File

After EACH file:
1. `rg "import numpy|from numpy|np\." <file>` — MUST return ZERO
2. `pytest -q tests/test_frustum_sovereign.py` (or equivalent test for that file)
3. `pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py` — no regression
4. `git diff --check` clean

After ALL THREE:
5. Run focused benchmark smoke (10 questions per suite) to verify spatial pipeline still works end-to-end.

---

## What NOT to Do

1. Do NOT write GPU sorting kernel for `np.argsort` in morton sort. Use Python `sorted()`. GPU radix sort = Phase 4.
2. Do NOT write GPU compaction kernel for frustum boolean filtering. Use Python list comprehension. GPU stream compaction = Phase 4.
3. Do NOT rewrite the Python A* in `_navigate_csr_cpu`. Just remove numpy from it. GPU A* vertex limit increase = Phase 4.
4. Do NOT touch files outside these three. Bridge files (sovereign_bridges.py, nine_chain) = Phase 3C.
5. Do NOT add new dependencies. Only use: `ctypes`, `math`, `struct`, `HostTensorF32`, `loader`.
