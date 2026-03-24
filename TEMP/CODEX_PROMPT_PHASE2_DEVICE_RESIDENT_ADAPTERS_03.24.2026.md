# Codex Prompt: Phase 2 — Device-Resident Adapter Weights

**Date:** 2026-03-24
**Priority:** ARCHITECTURAL — This is the structural change that makes Phase 1 permanent
**Binding specs (READ THESE FIRST):**
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` Abstract: "Galaxy = Internal Brain — ALL default galaxies loaded simultaneously in VRAM." Specialist adapters ARE Galaxy content — brain regions. Brain regions live in VRAM.
- `docs/vocabulary/HYPER_PARALLEL_PROCESSING.md` §1: "The specialist IS a Galaxy neighborhood plus a navigation bias." Adapters are spatial biases, not host arrays.
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §3: TRM_WEIGHTS region (400MB allocated). This is WHERE adapters should live.
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` §3: "VRAM-native workspace." All active computation state resides on device.

**Context:** Phase 1 succeeded — numpy is gone from the three target sovereign files. Contrastive training works. But the live monitor shows **GPU 0% utilization, CPU 146%**. The reason is visible in `apply_gradient_rpn` (trm_adapters.py:199-235): every gradient update copies A, B, A.T, B.T from host to device, computes on GPU, then copies A, B BACK to host. Five `copy_to_device` + two `copy_to_host` PER gradient step. With 1098 chat positives + 3817 negatives = ~4900 gradient steps, that is ~34,300 host↔device memcpy calls in one sleep cycle. This is why GPU is idle — it spends more time waiting for PCIe transfers than computing.

---

## The Problem: Round-Trip Commuting

Current `apply_gradient_rpn` flow (lines 199-235):

```
Host (HostTensorF32)          Device (DeviceTensor)
    self.A ──copy_to_device──> buffers.A
    self.B ──copy_to_device──> buffers.B
    self.B.T ──copy_to_device──> buffers.B_transposed
    self.A.T ──copy_to_device──> buffers.A_transposed
    gradient ──copy_to_device──> buffers.gradient
                                 │
                                 ├─ vector_norm (GPU)
                                 ├─ matmul grad_a (GPU)
                                 ├─ matmul grad_b (GPU)
                                 ├─ scale + add A (GPU)
                                 ├─ scale + add B (GPU)
                                 │
    self.A <──copy_to_host───── buffers.A
    self.B <──copy_to_host───── buffers.B
```

Every gradient step: 5 HtoD + 2 DtoH = 7 memcpy calls. The GPU computation in between is microseconds. The memcpy overhead is milliseconds. The GPU is starved.

## The Target: Weights Live in VRAM

```
Boot:
    self.A_device = DeviceTensor(gpu_malloc(...), dims, rank)
    self.B_device = DeviceTensor(gpu_malloc(...), rank, dims)
    copy_to_device(self.A, self.A_device.ptr)  ← ONE TIME
    copy_to_device(self.B, self.B_device.ptr)  ← ONE TIME

apply_gradient_rpn:
    copy_to_device(gradient, buffers.gradient.ptr)  ← ONLY the new gradient
    transpose(A_device → A_transposed)              ← GPU kernel, no host copy
    transpose(B_device → B_transposed)              ← GPU kernel, no host copy
    matmul(grad_a, gradient, B_transposed)           ← GPU
    matmul(grad_b, A_transposed, gradient)           ← GPU
    scale + add A_device                             ← GPU
    scale + add B_device                             ← GPU
    ← NO copy back to host. Weights stay in VRAM.

save() / checkpoint:
    copy_to_host(A_device.ptr, host_buffer)  ← ONLY when persisting to disk
    copy_to_host(B_device.ptr, host_buffer)
    write to zip
```

Result: 1 HtoD per gradient step (just the gradient), 0 DtoH. That is a 7:1 reduction in memcpy calls. The GPU stays fed.

---

## Implementation Tips

### Tip 1: Add `A_device` and `B_device` to `AdapterDeviceBuffers`

The `AdapterDeviceBuffers` dataclass (trm_adapters.py:81-100) already holds `A`, `B`, `A_transposed`, `B_transposed` as `DeviceTensor` — but these are TEMPORARY buffers that get overwritten every gradient step. The change: make `A` and `B` the PERSISTENT device-resident weights.

Rename the current `buffers.A` / `buffers.B` to clarify they are the LIVE weight pointers, not temporary staging:

```python
@dataclass
class AdapterDeviceBuffers:
    dims: int
    rank: int
    # PERSISTENT device-resident weights (populated once at init, updated in-place by gradient)
    A_weights: DeviceTensor      # [dims, rank] — the actual A weights in VRAM
    B_weights: DeviceTensor      # [rank, dims] — the actual B weights in VRAM
    # TRANSIENT computation buffers (reused each gradient step)
    gradient: DeviceTensor       # [dims, dims]
    grad_a: DeviceTensor         # [dims, rank]
    grad_b: DeviceTensor         # [rank, dims]
    A_transposed: DeviceTensor   # [rank, dims]
    B_transposed: DeviceTensor   # [dims, rank]
    grad_scale: DeviceTensor
    a_scale: DeviceTensor
    b_scale: DeviceTensor
    a_zero: DeviceTensor
    b_zero: DeviceTensor
    # Scale cache
    grad_scale_value: Optional[float] = None
    a_scale_value: Optional[float] = None
    b_scale_value: Optional[float] = None
    # Dirty flag — weights changed since last host sync
    weights_dirty: bool = False
```

### Tip 2: Upload Weights Once at `_ensure_device_buffers`

In `_ensure_device_buffers()`, after allocating all buffers, upload A and B ONE TIME:

```python
RPNMathCore.copy_to_device(self.A, buffers.A_weights.ptr)
RPNMathCore.copy_to_device(self.B, buffers.B_weights.ptr)
```

This is the ONLY upload of A/B weights to device. After this, they live in VRAM.

### Tip 3: Modify `apply_gradient_rpn` to Use Device-Resident Weights

The new flow:
1. Upload ONLY the gradient: `copy_to_device(gradient, buffers.gradient.ptr)`
2. Compute transposes ON DEVICE. If no transpose kernel exists yet, compute A.T and B.T on device using the existing matmul or add a simple `transpose_kernel.ptx`. Alternatively, maintain `A_transposed` and `B_transposed` as device-resident too — update them after each A/B update. This avoids needing a transpose kernel.
3. Run matmul, scale, add — all operating on `buffers.A_weights` / `buffers.B_weights`
4. Mark `buffers.weights_dirty = True`
5. DO NOT copy back to host. Weights stay in VRAM.

### Tip 4: Lazy Host Sync for Checkpoint/Save

Add a method `sync_weights_to_host()` that copies device weights back to the host `HostTensorF32`:

```python
def sync_weights_to_host(self) -> None:
    """Copy device-resident weights to host for checkpoint/save."""
    buffers = self._device_buffers
    if buffers is None or not buffers.weights_dirty:
        return
    RPNMathCore.copy_to_host(buffers.A_weights.ptr, self.A)
    RPNMathCore.copy_to_host(buffers.B_weights.ptr, self.B)
    buffers.weights_dirty = False
```

Call this in `save()` and in any method that needs to read host-side weights (e.g., `get_delta()`).

### Tip 5: Transpose Maintenance

Instead of computing A.T and B.T from host every gradient step, maintain them as device-resident too. After updating A_weights via `vec_add3`, also update A_transposed. Two approaches:

**Option A (simple):** After `vec_add3` updates A_weights, copy A_weights → host, transpose on host, upload A_transposed. Still 2 memcpy per step, but only for the small [dims, rank] matrices, not [dims, dims].

**Option B (sovereign):** Write a `transpose_inplace.ptx` kernel or use the existing `matmul` identity trick: `transpose(M) = M @ I_permuted`. Since `dims` and `rank` are small (128, 16), this is trivial.

**Option C (deferred maintenance):** Only recompute A_transposed/B_transposed at the START of each `apply_gradient_rpn` call from the device-resident A_weights/B_weights. This means one kernel launch per gradient step (transpose), not a memcpy. The transpose kernel operates entirely in VRAM.

Recommend Option C — it's the cleanest. But any option that avoids host roundtrip is acceptable.

### Tip 6: Shadow Weights

`apply_gradient_to_shadow` currently swaps `self.A` / `self.B` with `self.A_shadow` / `self.B_shadow` then calls `apply_gradient_rpn`. With device-resident weights, the shadow should ALSO be device-resident. Add `A_shadow_device` / `B_shadow_device` to buffers. The fork/validate/commit cycle stays on GPU.

### Tip 7: `get_delta()` Must Sync First

`get_delta()` returns `alpha * (A @ B)`. With device-resident weights, either:
- Call `sync_weights_to_host()` first, then compute on host (current behavior, lazy)
- OR compute `A @ B` on device via `matmul` kernel and return the result (sovereign, preferred)

The `matmul_host` helper already exists in `RPNMathCore` — it uploads, computes on GPU, downloads. But with device-resident weights, you can call `matmul` directly on `A_weights` and `B_weights` without any upload.

---

## What NOT to Change

- `save()` / `load()` — these are checkpoint I/O, they SHOULD touch host/disk. Just add `sync_weights_to_host()` call in `save()`.
- `HostTensorF32` — keep it. It's the sovereign host staging type for boot/checkpoint. The point is that ACTIVE weights don't live in it during inference/training.
- `RPNMathCore.copy_to_device` / `copy_to_host` — keep them. They're the sovereign transfer primitives. The point is to call them LESS (once at boot, once at checkpoint), not to remove them.

---

## Validation

1. **Contrastive training still succeeds:** All 4 specialists train: true, checkpoint non-empty
2. **Memcpy count drops:** With `K3D_RPN_DEBUG=1`, count `[loader]` lines per gradient step. Before: 7. After: 1 (just the gradient).
3. **GPU utilization rises:** Not necessarily to 50% yet (Phase 3 addresses that), but above 0%.
4. **No regression:** Benchmark scores >= 18.66% (current baseline)
5. **No numpy:** `rg "import numpy|from numpy" knowledge3d/cranium/trm_adapters.py` returns ZERO

---

## Spec Grounding

| Design Decision | Spec | Section |
|----------------|------|---------|
| Adapters live in VRAM | THREE_BRAIN_SYSTEM | Abstract: "Galaxy = Internal Brain, ALL in VRAM" |
| TRM_WEIGHTS region (400MB) | KNOWLEDGEVERSE | §3: Memory regions |
| Specialist = Galaxy neighborhood + bias | HYPER_PARALLEL_PROCESSING | §1 |
| VRAM-native workspace | SGI_SPECIFICATION | §3 |
| No host↔device round-trips in hot path | KNOWLEDGEVERSE | §4.1: ptx_fallback_rate = 0.0 |
| Checkpoint = I/O, acceptable on host | THREE_BRAIN_SYSTEM | Abstract: "Save game = House persistence" |

---

## Execution Order

1. Modify `AdapterDeviceBuffers` — add persistent weight fields + dirty flag
2. Modify `_ensure_device_buffers()` — upload A/B once at init
3. Modify `apply_gradient_rpn()` — use device-resident weights, upload only gradient
4. Add `sync_weights_to_host()` — lazy host sync for checkpoint
5. Modify `save()` — call `sync_weights_to_host()` first
6. Modify `apply_gradient_to_shadow()` — use device-resident shadow weights
7. Run tests: `pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py`
8. Run sovereignty grep
9. If tests pass — launch warm 35% benchmark with `K3D_RPN_DEBUG=1` and count memcpy lines
