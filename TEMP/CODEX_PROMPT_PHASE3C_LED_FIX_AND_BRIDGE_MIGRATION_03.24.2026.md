# Codex Prompt: Phase 3C — LED A* Kernel Fix + Bridge/Engine NumPy Migration

**Date:** 2026-03-24
**Priority:** CRITICAL — Phase 3B landed. Keep moving.
**Binding specs:**
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` — TRM game loop: perceive → navigate → reason → decide → act. ALL stages sovereign.
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §4.1 — ptx_fallback_rate = 0.0. No CPU fallbacks.
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` §3 — VRAM-native workspace. No CPU preprocessing.

---

## Part A: Fix the LED A* "invalid argument" (ROOT CAUSE FOUND)

### The Bug

The `led_astar.ptx` kernel declares **65,548 bytes** of static shared memory:
- `sharedGScore[16384]` (16KB)
- `sharedFScore[16384]` (16KB)
- `sharedParent[16384]` (16KB)
- `frontier[16384]` (16KB)
- `frontierSize` (4B) + `minIdx` (4B) + `minF` (4B) = 12B
- **Total: 65,548 bytes**

The PTX is compiled with `.target sm_75`. On sm_75 (Turing), the maximum STATIC shared memory per block is **64KB** (65,536 bytes). We are 12 bytes over.

On the actual hardware (RTX 3070 = sm_86, Ampere), up to **100KB** is supported — BUT the kernel is compiled for sm_75, so the sm_75 limit applies.

This is why `loader.launch` returns "invalid argument" — the driver rejects the kernel because its static shared memory exceeds the sm_75 limit.

### The Fix (Two Options)

**Option 1 (QUICK — recompile for sm_86):**

The RTX 3070 is sm_86. Recompile the source `.cu` file with `--gpu-architecture=sm_86`. This raises the static shared limit to 100KB. The 65KB kernel fits comfortably.

Find the source:
```bash
find knowledge3d/cranium -name "led_astar.cu" -o -name "led_astar*.cu"
```

Recompile:
```bash
nvcc -ptx -arch=sm_86 led_astar.cu -o knowledge3d/cranium/ptx/led_astar.ptx
```

**Option 2 (BETTER — reduce shared memory):**

The kernel uses 4096 entries per array (16384 bytes / 4 bytes = 4096 uint32). This is why `num_vertices > 4096` triggers the Python fallback. Reducing to 2048 entries would halve shared memory to ~32KB, well within sm_75. BUT: this halves the vertex limit.

The REAL fix is Option 1 + keeping 4096 entries. The RTX 3070 has 100KB shared per SM — we should use it.

### After the Fix

1. Remove the fallback warnings in `led_pathfinder.py` lines 376-389 (empty path → Python fallback) and lines 392-405 (RuntimeError → Python fallback). Replace with fail-fast:
   ```python
   if path_count <= 0:
       raise RuntimeError("led_astar_navigate returned empty path — kernel may need debugging")
   ```

2. Keep the `num_vertices > 4096` guard (line 310-324) — that is a REAL hardware limit of the shared memory arrays, not a fallback. But change the warning to a clear error:
   ```python
   if num_vertices > 4096:
       raise RuntimeError(
           f"CSR graph has {num_vertices} vertices, exceeding the 4096 "
           f"shared-memory limit of led_astar_navigate.ptx"
       )
   ```
   The Python A* is sovereignty debt — flagged, not hidden behind a silent warning.

3. Validate: launch `navigate_csr` with a small graph (< 100 vertices) through the GPU path. It MUST succeed without fallback.

---

## Part B: Bridge + Engine NumPy Migration

### Remaining Debt Map

| File | numpy uses | Role in pipeline |
|------|-----------|-----------------|
| `query_head_substrate.py` | 64 | Candidate scoring, LOD, feature extraction |
| `rpn_embedding_engine.py` | 46 | Query trigram embedding |
| `sovereign_bridges.py` | 33 | Kernel launch staging (all bridge classes) |
| `router_specialist.py` | 32 | Routing decisions |
| `nine_chain_specialized_bridge.py` | 31 | Swarm worker dispatch |
| `matryoshka_trm.py` | 16 | Matryoshka projection |
| **Total** | **222** | |

### Migration Order (by impact × difficulty)

1. **`rpn_embedding_engine.py`** (46 uses) — DO FIRST. This is the ENTRY POINT of every question. Query text → trigram codes → embedding vector. If this is sovereign, the query vector is BORN on device and never needs to leave.

2. **`sovereign_bridges.py`** (33 uses) — DO SECOND. This is the KERNEL LAUNCH LAYER. Every PTX kernel launch goes through here. If bridge staging is sovereign, kernel inputs/outputs stay device-resident between launches.

3. **`nine_chain_specialized_bridge.py`** (31 uses) — DO THIRD. Swarm worker dispatch. Feeds off bridge outputs.

4. **`router_specialist.py`** (32 uses) — DO FOURTH. Routing uses embedding engine + bridge outputs.

5. **`matryoshka_trm.py`** (16 uses) — DO FIFTH. Matryoshka projection. Smallest debt.

6. **`query_head_substrate.py`** (64 uses) — DO LAST. Largest file, but it depends on ALL the above. Easier once the inputs it consumes are already sovereign.

### Strategy (Same Pattern as Phase 3B)

For EACH file:
1. Replace `import numpy as np` with `from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32, DeviceTensor`
2. Replace `np.array` / `np.zeros` / `np.ascontiguousarray` staging with `HostTensorF32` or `ctypes` arrays
3. Replace `np.linalg.norm`, `np.dot`, `np.argsort`, etc. with `RPNMathCore` ops or Python scalar math
4. Replace `np.concatenate`, `np.vstack`, `np.hstack` with Python list ops or `HostTensorF32` construction
5. Check callers of any function whose return type changes from `np.ndarray`
6. Validate: `rg "import numpy|from numpy|np\." <file>` returns ZERO

### Specific Tips per File

#### rpn_embedding_engine.py

This file creates trigram embeddings for query text. The typical numpy pattern:
- `np.zeros(embed_dim)` for accumulator → `HostTensorF32.zeros(embed_dim, 1)` or `(ctypes.c_float * embed_dim)()`
- `np.array(trigram_codes)` for kernel input → ctypes uint16/uint32 array
- Cosine similarity via numpy → `RPNMathCore.vector_norm` + dot product via RPN kernel
- Result concatenation → `HostTensorF32` construction

KEY: The embedding vector should be BORN as a `DeviceTensor` (allocated in VRAM via kernel output). It should NOT be copied to host between the embedding stage and the next stage (spatial navigation). If the next consumer can accept a `DeviceTensor` pointer, the vector never leaves VRAM.

#### sovereign_bridges.py

This is the BRIDGE LAYER — it stages host arrays for kernel launches and reads results back. The pattern:
- Input: Python/numpy array → `memcpy_htod` → kernel → `memcpy_dtoh` → Python/numpy array
- Target: `HostTensorF32` / ctypes staging → `memcpy_htod` → kernel → result stays as `DeviceTensor` (lazy host read)

Many bridge methods will return `DeviceTensor` instead of host arrays. Callers that need host data call `RPNMathCore.copy_to_host()` explicitly. This is the GAME ENGINE PATTERN — results stay in VRAM unless something needs to display them (I/O).

#### nine_chain_specialized_bridge.py

Swarm dispatch. Creates feature arrays for 9 parallel workers. Replace numpy feature construction with `HostTensorF32` rows. If the swarm workers operate on device, their input feature vectors should be `DeviceTensor` pointers.

#### router_specialist.py

Routing decisions. Uses embedding similarity scores. Replace numpy cosine/dot with `RPNMathCore` vector ops. Routing scores should be computed on device.

#### matryoshka_trm.py

Matryoshka projection — projects vectors to different dimension levels. Replace numpy matmul with `RPNMathCore.matmul`. Small file, straightforward.

#### query_head_substrate.py

The largest and most entangled. DO THIS LAST. It orchestrates LOD decisions, candidate scoring, feature extraction. By the time you reach this, all its inputs (embeddings, bridge outputs, router scores) will already be sovereign types. The migration is then mechanical: replace numpy array ops with HostTensorF32/ctypes equivalents.

---

## Validation

After EACH file:
1. `rg "import numpy|from numpy|np\." <file>` — ZERO matches
2. `pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py` — no regression
3. `git diff --check` clean

After ALL files + LED fix:
4. Run warm 35% benchmark with live monitor
5. Target: GPU utilization > 0%, CPU% drops from 146%

---

## RULES

1. Do NOT add CPU fallbacks. If a kernel fails, fail-fast with clear error.
2. Do NOT import numpy in ANY file you touch. Use `HostTensorF32`, `DeviceTensor`, `ctypes`, `math`.
3. Do NOT change PTX kernel signatures. Adapt the Python staging to match.
4. Check ALL callers when changing return types. Adapt immediate callers in same pass.
5. When in doubt, consult `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §4 — sovereignty invariants.
6. Report the numpy count BEFORE and AFTER for each file.

---

## Spec Grounding

| Decision | Spec | Section |
|----------|------|---------|
| Embedding born on device | THREE_BRAIN | "Galaxy = Internal Brain, ALL in VRAM" |
| Bridge results stay in VRAM | KNOWLEDGEVERSE | §3: WORLD_VIEW region (2GB) |
| No CPU preprocessing | SGI | §3 |
| Swarm dispatch on device | THREE_BRAIN | §3.5: Nine-chain internal swarm |
| Routing on device | KNOWLEDGEVERSE | §4.1: ptx_fallback_rate = 0.0 |
| LED A* shared memory | KNOWLEDGEVERSE | §4: fail-fast, no fallbacks |
