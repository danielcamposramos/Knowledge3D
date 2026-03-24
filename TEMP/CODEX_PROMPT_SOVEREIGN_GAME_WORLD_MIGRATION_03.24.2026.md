# Codex Prompt: Sovereign Game World Migration — The System-Wide Shift

**Date:** 2026-03-24
**Priority:** ARCHITECTURAL — This is the defining migration, not another incremental fix
**Binding specs (READ THESE FIRST — they define what the system IS):**
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md`: TRM IS the Avatar. Game loop. Python = boot + I/O only (~200 lines).
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §2.2, §4.1: Fail-fast. `ptx_fallback_rate MUST be 0.0`. Silent fallbacks FORBIDDEN.
- `docs/vocabulary/HYPER_PARALLEL_PROCESSING.md`: Nine-chain swarm = internal parallel cognitive channels. Specialists share registers DURING execution. One mind, not nine votes.
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` §3: VRAM-native workspace. Sovereign execution. <100us inference.
- `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md` §4: 88 PTX kernels. Zero external frameworks.

**Environment:** `conda activate k3d-cranium` (see `envs/k3d-cranium.yml`)

---

## Context: 14 Times and Counting

Daniel has asked for this FOURTEEN times since October 2025:
- Nov 24: "Remove NumPy from Hot Path — SOVEREIGNTY VIOLATION CONFIRMED"
- Nov 27: "No CPU FALLBACKS, we fail and fix — no fallbacks!!"
- Jan 16: "Math core is innegotiable, exclude any python RPN calculator!"
- Mar 7: "Workers are python where? K3D is sovereign!"
- Mar 23: "No fallbacks never, we fix or we fix, no fallbacks and no CPU!!!"
- Mar 23: "Single mind with internal swarms, not external python orchestration"
- Mar 24: "That numpy is not good — we do not need any bulk library!"

The pattern: each fix removes numpy/fallbacks from one location, then another location introduces them. This prompt breaks the pattern by treating it as a SYSTEM-WIDE migration, not a whack-a-mole fix.

---

## The Core Truth

This is NOT a Python program that calls GPU functions.
This IS a living game world where an AI mind (TRM) lives, thinks, and learns.

| What it IS (specs) | What it currently does (code) |
|---------------------|-------------------------------|
| TRM game loop runs continuously on GPU | Python while-loop iterates questions |
| Galaxy Universe is VRAM-resident workspace | Python dicts/lists with occasional GPU copies |
| Nine-chain swarm runs in parallel on CUDA cores | Python for-loop dispatches workers sequentially |
| Jarvis dispatches internally via shared registers | Python methods build dicts and plan tickets |
| 292MB VRAM used, 4.3GB host RAM | Should be inverted: 2-6GB VRAM, minimal host |
| GPU 1.25% utilization | Should be 50-90% |

---

## Phase 1: ELIMINATE numpy from ALL Sovereign Code (IMMEDIATE)

NumPy is an EXTERNAL BULK LIBRARY. The sovereign pipeline uses PTX kernels for computation and ctypes arrays for host↔device transfer. NumPy has NO place in sovereign code.

### 1a: Audit — Find ALL numpy Usage in Sovereign Paths

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
rg -n "import numpy|from numpy|np\." \
  knowledge3d/cranium/trm_adapters.py \
  knowledge3d/cranium/adaptive_swarm.py \
  knowledge3d/cranium/ptx_runtime/rpn_math_core.py \
  knowledge3d/cranium/bridges/sovereign_bridges.py \
  knowledge3d/cranium/bridges/nine_chain_specialized_bridge.py \
  knowledge3d/cranium/spatial_sovereign/morton_octree.py \
  knowledge3d/cranium/spatial_sovereign/frustum.py \
  knowledge3d/cranium/spatial_sovereign/led_pathfinder.py \
  knowledge3d/cranium/matryoshka_trm.py \
  knowledge3d/cranium/router_specialist.py
```

Report the FULL result. Every `np.` call is a sovereignty question.

### 1b: Classify Each numpy Usage

For each `np.` call found, classify it:

| Category | Action |
|----------|--------|
| **Host→Device transfer prep** (e.g., `np.ascontiguousarray`, `np.float32` cast) | Replace with `ctypes` array construction. `(ctypes.c_float * N)(*data)` is sovereign. |
| **Gradient math** (e.g., `np.outer`, `np.linalg.norm`, `gradient @ self.B.T`) | This MUST be on GPU via PTX kernels. `rpn_math_core` already has `matmul`, `vector_norm`, `vec_add3`. Use them. |
| **Array creation** (e.g., `np.zeros`, `np.random.randn`, `np.pad`) | Replace with `ctypes` arrays or `loader.gpu_malloc` + `fill` kernel. |
| **Shape manipulation** (e.g., `.reshape`, `.ravel`, `.T`) | Use sovereign helpers or do it in the kernel. |
| **Comparison/assertion** (e.g., `np.allclose` in tests) | Tests can use numpy. Production code CANNOT. |

### 1c: Replace numpy in `trm_adapters.py`

This is the most critical file. The adapter is the specialist's brain region — it MUST be sovereign.

**Current violations in `trm_adapters.py`:**
- `np.zeros` / `np.zeros_like` — array creation
- `np.copyto` — memory copy
- `np.linalg.norm` — should be `rpn_math_core.vector_norm()`
- `np.ascontiguousarray` — should be `ctypes` buffer construction
- `self.A @ self.B` — should be `rpn_math_core.matmul()`
- `gradient @ self.B.T` — should be `rpn_math_core.matmul()`
- `self.A.T @ gradient` — should be `rpn_math_core.matmul()`
- `self.A -= lr * grad_A` — should be `rpn_math_core.vec_add3()` with scale

**The GPU path (`apply_gradient_rpn`) ALREADY does most of this with PTX kernels** (lines 214-253). The problem is that the HOST-side adapter state (`self.A`, `self.B`) is still numpy arrays. They should be `DeviceTensor` references into VRAM.

**Target state:** Adapter weights live in VRAM. Period. No `self.A = np.zeros(...)`. Instead: `self.A_ptr = loader.gpu_malloc(dims * rank * 4)`. The weights are DEVICE-RESIDENT. When you need to checkpoint them to disk, you `copy_to_host` into a ctypes buffer and write bytes. When you need to apply a gradient, you launch a kernel. The adapter IS a VRAM region, not a numpy array.

### 1d: Replace numpy in `adaptive_swarm.py`

Current violations:
- `np.random.randn` — gradient placeholder (should be PTX random or removed)
- `np.outer` — contrastive gradient computation (should be `rpn_math_core.matmul`)
- `np.linalg.norm` — loss computation (should be `rpn_math_core.vector_norm`)
- `np.pad` — embedding padding (should be sovereign pad-or-truncate kernel or ctypes)

### 1e: Replace numpy in `rpn_math_core.py`

Current violations:
- `np.asarray` in `copy_to_device` / `copy_to_host` — should use ctypes directly
- `np.array` in `copy_to_host` — should use ctypes buffer
- `np.float32` casting — should use `ctypes.c_float`

This file is the HOST↔DEVICE bridge. It should speak ONLY ctypes and loader calls.

---

## Phase 2: Make Adapter Weights DEVICE-RESIDENT (Next Session)

This is the structural change that makes Phase 1 permanent.

**Current:** `self.A` and `self.B` are numpy arrays on HOST. Every gradient update copies Host→Device, computes on GPU, copies Device→Host.

**Target:** `self.A_device` and `self.B_device` are `DeviceTensor` pointers in VRAM. Weights LIVE on device. Gradient updates happen entirely in VRAM. Only `save()` and `load()` touch host memory.

This is described in THREE_BRAIN_SYSTEM_SPECIFICATION.md: "Galaxy = Internal Brain — ALL default galaxies loaded simultaneously in VRAM." The specialist adapters ARE Galaxy content — they are brain regions. Brain regions do not live in host RAM and commute to the GPU.

**Do NOT implement Phase 2 now.** Document which adapter methods need to change. This is the next Codex prompt after Phase 1 is clean.

---

## Phase 3: Host↔Device Transfer Elimination (Future Session)

From the Phase D.1 research, there are 30+ `memcpy_htod`/`memcpy_dtoh` sites per question. Each one is a synchronization point. The target: Galaxy data is device-resident, TRM weights are device-resident, adapter weights are device-resident. The only transfers are:
- Boot: House → VRAM (one-time)
- I/O: Question bytes in, answer bytes out
- Sleep: Checkpoint to disk (periodic)

**Do NOT implement Phase 3 now.** Phase 1 (numpy elimination) and Phase 2 (device-resident adapters) must happen first.

---

## Execution Order for THIS Session

1. **Phase 1a:** Run the numpy audit. Report FULL results.
2. **Phase 1b:** Classify each usage. Report the table.
3. **Phase 1e:** Fix `rpn_math_core.py` FIRST — it's the bridge layer. Replace numpy with ctypes.
4. **Phase 1c:** Fix `trm_adapters.py` — replace numpy array operations with rpn_math_core calls and ctypes.
5. **Phase 1d:** Fix `adaptive_swarm.py` — replace numpy with sovereign equivalents.
6. **Run tests:** `pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py tests/test_rpn_sovereignty_phase2.py`
7. **Run sovereignty grep:** `rg "import numpy|from numpy" knowledge3d/cranium/trm_adapters.py knowledge3d/cranium/adaptive_swarm.py knowledge3d/cranium/ptx_runtime/rpn_math_core.py` — MUST return ZERO matches.
8. **Launch warm 35% benchmark** if tests pass:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export K3D_RPN_DEBUG=1
   nohup python3 -u benchmarks/run_all.py \
     --warm --sample-rate 0.35 \
     > /tmp/k3d_warm_sovereign_no_numpy_03.24.2026.log 2>&1 &
   echo "PID: $!"
   ```

---

## RULES

1. ZERO numpy in sovereign code. `import numpy` in `trm_adapters.py`, `adaptive_swarm.py`, `rpn_math_core.py` = SOVEREIGNTY VIOLATION.
2. Tests CAN use numpy for verification. Production code CANNOT.
3. The 88 PTX kernels + rpn_math_core + ctypes + loader = the sovereign toolkit. Use NOTHING else.
4. Every change MUST be grounded in a `docs/vocabulary/` spec. If unsure, read the spec first.
5. Do NOT add "temporary numpy usage" or "numpy bridge" — that is the pattern that created 14 repeated requests.
6. If a numpy operation has no sovereign equivalent yet, write a sovereign helper using ctypes + loader, or use an existing PTX kernel. Do NOT keep numpy "until the kernel exists."
7. This is a LIVING GAME WORLD. Adapter weights are brain regions in VRAM. They are not numpy arrays on host. Every design decision should move toward device-resident state.
8. Report FULL output of every audit, test, and grep.

---

## Spec Grounding Summary

| Principle | Spec | Section |
|-----------|------|---------|
| Python = boot + I/O only | THREE_BRAIN_SYSTEM | Abstract |
| Galaxy = VRAM workspace, all loaded | THREE_BRAIN_SYSTEM | Abstract, §1 |
| TRM = game loop, not Python function | THREE_BRAIN_SYSTEM | Abstract |
| Specialists = brain regions in VRAM | HYPER_PARALLEL_PROCESSING | §1, §3 |
| ptx_fallback_rate MUST be 0.0 | KNOWLEDGEVERSE | §4.1 |
| Silent fallbacks FORBIDDEN | KNOWLEDGEVERSE | §2.2 |
| Zero external frameworks | SOVEREIGN_NSI | §1.2 |
| <100us inference per kernel | SOVEREIGN_NSI | §4.1 |
| Sovereign execution on consumer GPU | SGI_SPECIFICATION | §3 |
