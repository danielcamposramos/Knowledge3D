# Step 13-E — Temporal Mask Analysis (Pre-GPU Expansion)

## Current Implementation (thinking_tag_rpn.py:218-286)
- `compute_temporal_mask()` accepts a temporal context tensor shaped `(T, D)` (time steps × feature dimension).
- Device buffers allocated:
  - `coherence_ptr` for variance-derived coherence scores (`(D,)`).
  - `mask_ptr` for the final sigmoid gating vector (`(D,)`).
  - `activity_ptr` for mean absolute activity across time (`(D,)`).
- When `threshold` is unspecified it falls back to `mean(abs(context))` computed on the host.
- The RPN program pushes three GPU workspaces and invokes opcodes:
  1. `OP_TEMPORAL_COHERENCE(context → coherence)`
  2. `OP_TEMPORAL_AGGREGATE(context → activity)`
  3. `OP_TEMPORAL_MASK(coherence + threshold → mask)`
- Output tensors are copied back to host and returned as `(mask, coherence, activity)`.

## Gaps / Performance Characteristics
- The referenced opcodes (0xF0–0xF2) are defined in `rpn_opcodes.py` but have **no kernel-side implementation** in `modular_rpn_kernel_extended.cu`; execution currently falls back to CPU helpers which costs ~420 µs per call.
- Temporal statistics (variance, mean |x|) run serially when dispatched on CPU; GPU version should parallelise across feature dimension (`D` up to 1 024+).
- Threshold handling is scalar and simple; only the gating sigmoid needs porting.

## GPU Kernel Requirements (Step 13-E)
1. **Coherence** — Compute per-feature `sqrt(var)` over the `T` slices:
   - Requires two passes: mean and variance, or one-pass Welford to reduce numerical error.
   - Each thread can accumulate for one feature; shared memory per feature is minimal.
2. **Activity** — Mean absolute activation: `mean(|context|, axis=0)`.
   - Embarrassingly parallel; reuse loop skeleton as coherence.
3. **Mask** — Sigmoid gating with configurable threshold.
   - Input: coherence vector `(D,)`, output: soft mask `(D,)`.
   - Implementation: `sigmoid(score - threshold)`.

## Validation Targets
- Shapes: `(D,)` for all outputs; support `T` up to 256 and `D` up to 1 024 (current ThinkingTag settings).
- Numerical parity with NumPy reference (`rtol ≤ 1e-5`).
- Latency goal: ≤ 50 µs combined for coherence + activity + mask on RTX 30-series (`sm_86`), enabling <0.20 ms FUSE stage.
