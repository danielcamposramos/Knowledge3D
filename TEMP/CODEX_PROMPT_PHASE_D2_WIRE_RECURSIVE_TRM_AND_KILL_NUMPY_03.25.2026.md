# Codex: Phase D.2 — Wire Recursive TRM Into Query Path + Kill numpy in Launcher

**Date:** 2026-03-25
**Priority:** IMMEDIATE — D.1 kernel exists but the query path DOES NOT CALL IT. Fix this NOW, then keep going.
**Binding specs:**
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` -- TRM IS the Avatar, RECURSIVE refinement
- `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md` SS4.2 -- 87% converge within 5 iterations, 80.69us for 9 steps
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` SS4.1 -- fail-fast, no silent fallbacks, ptx_fallback_rate = 0.0

**DO NOT STOP between tasks. Execute ALL parts. The instructions are complete.**

---

## THE IMMEDIATE BUG

`knowledgeverse.py` line 963 calls `self._trm.kernel_fused` — but after D.1, `kernel_fused` is `None` because the launcher now loads `kernel_recursive_fused` instead. So right now:

- `_run_single_trm_tick()` either silently returns `{}` or crashes
- The TRM recursive kernel you just wrote is NEVER CALLED during inference
- The query path skips TRM entirely

This must be fixed FIRST.

---

## Part A: Rewire `_run_single_trm_tick` to Use Recursive Kernel

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`
**Function:** `_run_single_trm_tick` (line 956)

**Current (BROKEN — calls None kernel, runs ONE step, no recursion):**
```python
def _run_single_trm_tick(self, query_embedding):
    ...
    launch(
        self._trm.kernel_fused,          # ← None after D.1!
        grid=(1, 1, 1),
        block=(128, 1, 1),               # ← was 128, recursive needs 256
        params=[
            d_q, d_y, d_z, W1, W2, W3, W4,
            d_z_new, d_y_new, d_workspace,   # ← old 10-param signature
        ],
    )
    synchronize()
    y_new_host = self._read_trm_state_vector("d_y_new")
```

**Fix — call `kernel_recursive_fused` with recursion + convergence:**
```python
def _run_single_trm_tick(self, query_embedding):
    ...
    # Allocate small convergence output buffers
    d_steps = gpu_malloc(4)   # 1 x int32
    d_drift = gpu_malloc(4)   # 1 x float32

    try:
        launch(
            self._trm.kernel_recursive_fused,   # ← the REAL recursive kernel
            grid=(1, 1, 1),
            block=(256, 1, 1),                   # ← 256 threads for shared reduction
            params=[
                self._trm_state_buffers["d_q"],
                self._trm_state_buffers["d_y"],  # ← y converges IN PLACE
                self._trm_state_buffers["d_z"],  # ← z converges IN PLACE
                self._trm_weight_buffers["W1"],
                self._trm_weight_buffers["W2"],
                self._trm_weight_buffers["W3"],
                self._trm_weight_buffers["W4"],
                self._trm_state_buffers["d_workspace"],
                ctypes.c_uint64(d_steps.value),
                ctypes.c_uint64(d_drift.value),
                ctypes.c_int32(6),              # max_steps = 6 (Tesla 3/6/9 resonance)
                ctypes.c_float(1e-4),           # epsilon convergence threshold
            ],
        )
        synchronize()

        # Read converged y from d_y (NOT d_y_new — recursive kernel updates in place)
        y_converged = self._read_trm_state_vector("d_y")

        # Read convergence metadata
        steps_host = (ctypes.c_int32 * 1)()
        drift_host = (ctypes.c_float * 1)()
        memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(steps_host)), d_steps, 4)
        memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(drift_host)), d_drift, 4)

        return {
            "y_new_vector_512": y_converged.tolist(),
            "trm_latency_us": latency_us,
            "trm_recursion_steps": int(steps_host[0]),
            "trm_drift": float(drift_host[0]),
        }
    finally:
        gpu_free(d_steps)
        gpu_free(d_drift)
```

**KEY DIFFERENCES from old code:**
1. Uses `kernel_recursive_fused` (NOT `kernel_fused`)
2. Block size 256 (NOT 128) — required for shared memory drift reduction
3. 12 params (NOT 10) — adds `steps_out`, `drift_out`, `max_steps`, `epsilon`
4. Reads converged result from `d_y` (NOT `d_y_new`) — recursive kernel updates y/z in place
5. Returns recursion metadata (steps taken, final drift)

**ALSO:** The workspace buffer must be 4096 floats (16KB) not the old 3072. Check `_initialize_trm_launcher` allocates enough:
```python
# Must be: temp(512) + hidden(1024) + temp2(512) + hidden2(1024) + z_new(512) + y_new(512) = 4096
self._trm_state_buffers["d_workspace"] = gpu_malloc(4096 * 4)
```

---

## Part B: Kill numpy in `trm_launcher.py`

**File:** `knowledge3d/cranium/sovereign/trm_launcher.py`

The file STILL has `import numpy as np` at line 17 and uses numpy in:
- `_encode_pointer_literal()` lines 38-39: `np.array([lo], dtype=np.uint32).view(np.float32)`
- `refine()` lines 155-160: `assert q.dtype == np.float32`, `assert q.shape == (512,)`
- `_refine_ptx()` lines 363-400: `np.zeros`, `np.max`, `np.abs`
- `_refine_rpn()` lines 414-509: same pattern
- `_refine_fused()` lines 553-554: `np.zeros` for readback
- `_init_rpn_buffers()` line 613: `np.asarray`
- `_update_rpn_scalars()` line 620: `np.zeros`

**Fix:**

1. DELETE `import numpy as np`
2. ADD: `from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32`
3. ADD: `import struct`

**Replacements:**

`_encode_pointer_literal`:
```python
def _encode_pointer_literal(ptr, rows: int, cols: int) -> list[float]:
    raw = _ptr_value(ptr)
    lo = raw & 0xFFFFFFFF
    hi = (raw >> 32) & 0xFFFFFFFF
    lo_f = struct.unpack('f', struct.pack('I', lo))[0]
    hi_f = struct.unpack('f', struct.pack('I', hi))[0]
    return [float(rows), float(cols), float(lo_f), float(hi_f)]
```

`refine()` public API — accept HostTensorF32 OR plain lists:
```python
def refine(self, q, y, z, W1, W2, W3, W4, n_steps=6, eps=1e-4):
    # Accept HostTensorF32, lists, or anything with .data_ptr
    ...
```

`_refine_fused()` readback:
```python
y_final = HostTensorF32(512, 1)
z_final = HostTensorF32(512, 1)
memcpy_dtoh(ctypes.c_void_p(y_final.data_ptr), d_y, y_final.nbytes)
memcpy_dtoh(ctypes.c_void_p(z_final.data_ptr), d_z, z_final.nbytes)
```

`_refine_ptx` and `_refine_rpn` — same pattern: replace `np.zeros(512, dtype=np.float32)` with `HostTensorF32(512, 1)`, replace `np.max(np.abs(...))` with Python `max(abs(...))` over flat buffer, replace `.ctypes.data_as(ctypes.c_void_p)` with `ctypes.c_void_p(tensor.data_ptr)`.

For the RPN `_rpn_opcodes` and `_rpn_scalars_host` buffers, use ctypes arrays:
```python
self._rpn_opcodes = (ctypes.c_uint16 * len(op_codes))(*op_codes)
self._rpn_scalars_host = (ctypes.c_float * (len(pointer_layout) * 4))()
```

**Validate:**
```bash
rg "import numpy|from numpy|np\." knowledge3d/cranium/sovereign/trm_launcher.py
# MUST return ZERO

python3 -m compileall knowledge3d/cranium/sovereign/trm_launcher.py
pytest -q tests/test_trm_fused_parity.py
```

---

## Part C: Kill numpy in `trm_engine.py`

**File:** `knowledge3d/cranium/ptx_runtime/trm_engine.py`

This file uses CuPy AND NumPy. It is the OLD TRM engine, now superseded by `trm_launcher.py` for the fused path. But callers still exist in tests.

Two options:
1. If ALL production callers now go through `TRMLauncher` — add a deprecation warning and leave for later
2. If any production code still imports `TRMEngine` — migrate it to HostTensorF32

**Check callers:**
```bash
rg "from.*trm_engine import|import.*trm_engine" knowledge3d/ --type py -l
```

If ONLY test files import it, mark it deprecated. If production code imports it, migrate it.

---

## Part D: Make Fused Backend the ONLY Backend

Right now `TRMLauncher` supports three backends: PTX (legacy), RPN, Fused. The fused recursive kernel IS the architecture. The others are dead weight with numpy dependencies.

1. Make `use_fused=True` the DEFAULT (not opt-in)
2. Mark PTX and RPN backends as deprecated
3. Eventually DELETE them (but not in this pass — just mark them)

In `__init__`:
```python
if use_fused is None:
    self.use_fused = os.getenv("K3D_USE_FUSED_TRM", "1").lower() not in {"0", "false", "no"}
```

Note the default changed from `"0"` to `"1"` — fused is now ON by default.

---

## Part E: Rerun Warm 35% Benchmark

After Parts A-D are validated:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export CUDA_VISIBLE_DEVICES=0
conda activate k3d-cranium

nohup python3 -u benchmarks/run_all.py \
  --warm --sample-rate 0.35 \
  > /tmp/k3d_phaseD2_recursive_trm_warm_35pct_03.25.log 2>&1 &

echo "Phase D.2 benchmark launched. PID: $!"
```

**While it runs:** Start a live monitor (60s sample) focused on GPU utilization.

**EXPECTED:** GPU utilization should be HIGHER than 0.17% because:
- TRM now ACTUALLY runs recursive refinement (up to 6 steps per query) inside ONE kernel
- The kernel does real work (matvec, swiglu, convergence check) without CPU round-trips
- Previously TRM was either silently skipped or ran ONE step

---

## Part F: Report

Write report to `TEMP/CLAUDE_PHASE_D2_RECURSIVE_TRM_REPORT_03.25.2026.md` with:

1. All 5 suite scores + combined
2. GPU utilization (compare with Phase 3C baseline: 0.17%)
3. TRM recursion statistics: average steps to convergence, convergence rate
4. Contrastive/sleep-time outcome
5. numpy count: `rg "import numpy|from numpy" knowledge3d/cranium/sovereign/trm_launcher.py` -- MUST be ZERO
6. numpy count: `rg "import numpy|from numpy" knowledge3d/cranium/ --type py -c`

**Comparison table:**
| Metric | Phase 3C (03.25 AM) | Phase D.2 (this run) |
|--------|---------------------|---------------------|
| GPU util avg | 0.17% | ? |
| GPU util max | 1.00% | ? |
| CPU avg | 113% | ? |
| Combined score | 18.21% | ? |
| TRM steps/query | 0 (skipped) | ? |
| numpy in trm_launcher | YES | ZERO |

---

## EXECUTION ORDER — DO NOT STOP

A -> B -> C -> D -> E -> F

All in sequence. No pauses between. The instructions are HERE.
