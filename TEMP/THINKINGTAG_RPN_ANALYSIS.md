# ThinkingTag RPN Usage Analysis

**Date:** 2025-10-16  
**Author:** Codex (Phase 1B prep)

---

## Source Locations

- `knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py`
  - `_build_temporal_rpn_program` (lines ~480-520)
  - `_execute_temporal_mlp` (lines ~522-524)
  - `_build_spatial_rpn_program` (lines ~527-537)
  - `_execute_spatial_mlp` (lines ~539-544)
  - RPN executed inside FSM REASON stage (lines ~360-410)

The bridge instantiates `ModularRPNEngine` directly (`self.rpn_engine = ModularRPNEngine()` in `__init__`).

---

## Opcodes Utilised Today

### Imported from `modular_rpn_engine`
- `OP_SPARSE_LOAD` (0x40): loads sparse rows into dense tensors
- `OP_SMAV` (0x41): sparse-matrix/attention-vector product
- `OP_ENTROPY_SUM` (0x42): entropy-weighted aggregation

### Scalar/vector core opcodes (existing Tier-2 surface)
- `0x0A` (`MAX`)
- `0x0B` (`SIGMOID_APPROX`)
- `0x12` (`MUL`)
- `0x06` (`DUP`)

### Custom opcodes emitted as raw bytes
- `0xF0` – annotated “CALL temporal_coherence”
- `0xF1` – “CALL temporal_mask”
- `0xF2` – “CALL crystallize_intermediate”

These `0xFx` opcodes are **not handled** inside the GPU interpreter; the current implementation presumably relies on the legacy Python evaluator to stub in host-side calls. They must be replaced by explicit GPU operations (or run outside of RPN) when adopting the parallel kernel.

---

## FSM Stage & Data Shapes

- **Stage:** REASON (5-state FSM: INGEST → FUSE → SPATIAL → REASON → OUTPUT)
- **Temporal path (`_execute_temporal_mlp`)**
  - Inputs: `x` (input embedding, shape ≈ 512), `sparse_weights` (`{'W1': (256×512), W2: (256×256), W3: (100×256)}`), `context` (temporal deltas, 256)
  - Output: fused logits (≈100)
- **Spatial fallback (`_execute_spatial_mlp`)**
  - Uses same sparse weights but without temporal context

---

## Current Workflow Summary

1. Build RPN bytecode with `RPNProgram` including sparse loads, SMAV, entropy and custom call opcodes.
2. Evaluate via `ModularRPNEngine.eval(program, [x])` (non-parallel interpreter).
3. Post-process with GraphCrystallizer, VectorResonator, etc.

Performance telemetry is collected via `LatencyProfiler`, but raw timing for the RPN portion is not logged in code (budget target <35 µs reported). No explicit measurement of FUSE/REASON stages in repo; benchmarks show sequential interpreter takes significant time.

---

## Integration Candidates

| Option | Description | Pros | Cons |
| --- | --- | --- | --- |
| **A. Tier‑2 Advanced RPN** | Implement custom ThinkingTag opcodes on top of the shared Tier‑2 kernel (`modular_rpn_kernel_extended.cu`). | Reuses existing infrastructure; good for medium tensors (≤1k values). | Need CUDA implementations for `OP_SPARSE_LOAD`, `OP_SMAV`, `OP_ENTROPY_SUM`; must map 0xF* calls to real ops. |
| **B. Tier‑3 Extended RPN** | Port ThinkingTag ops into the Tier‑3 interpreter (same kernel powering TRM). | Proven 47× speedup, competitive with PTX baseline. | Requires extending the Tier‑3 opcode surface; heavier shared memory usage. |
| **C. Hybrid** | Use Tier‑1 for light ops, Tier‑2 for sparse math, Tier‑3 for heavy temporal passes. | Fine-grained optimisation; future-proof. | Additional routing logic; higher complexity for Phase 1B timeline. |

**Recommendation (Phase 1B):** Start with **Tier‑2** implementation (Option A). The data sizes (N≤256, D≤512, T≤256) fit well within the new cooperative kernels and allow re-use of `AdvancedRPNEngine`. Tier‑3 can be targeted in follow-up if we need full TRM parity.

---

## Open Questions / To-Do

1. Translate `0xF0/0xF1/0xF2` semantics:
   - `temporal_coherence`, `temporal_mask`, `crystallize_intermediate` currently invoked via host-side calls (TemporalReasoning / GraphCrystallizer). Decide whether to:
     - Keep them outside the RPN program and run before/after GPU ops, or
     - Implement dedicated CUDA kernels / RPN opcodes.
2. Define precise tensor layouts for `OP_SPARSE_LOAD` & `OP_SMAV`.
   - Need pointer encoding format (RPNProgram `ptr()` writes 64-bit addresses plus rows/cols).
3. Measure baseline latency for `_execute_temporal_mlp` under current sequential interpreter to quantify final speedup.

This document will guide the bridge design and opcode implementation for Phase 1B.

---

## Proposed Integration Plan (Phase 1B)

1. **Specialised Bridge**
   - Implement `ThinkingTagRPNBridge` (Tier‑2 by default) that owns GPU buffers, weight caches, and program builders for temporal/spatial passes.
   - Provide methods:
     - `execute_temporal(input_vec, sparse_weights, context)`
     - `execute_spatial(input_vec, sparse_weights)`
   - Reuse `AdvancedRPNEngine` and new cooperative opcodes.

2. **New GPU Opcodes (Tier‑2)**
   - `OP_MATVEC_F32` (0xA0): dense matrix–vector multiply for arbitrary M×K.
   - `OP_VECTOR_RELU` (0xA1): in-place ReLU on tensor.
   - `OP_VECTOR_MUL_F32` (0xA2): element-wise multiply (dest *= mask).
   - `OP_VECTOR_SIGMOID` (0xA3): logistic sigmoid activation.
   - `OP_ENTROPY_SUM` (0x42) overhaul: compute `-∑ p log(max(p, ε))`.
   - Retain existing helper ops (`OP_MEMCPY_F32`, `OP_FILL_F32`, reductions) for auxiliary flows.

3. **Pointer Encoding**
   - Use `OP_POINTER_LITERAL` with `(rows, cols)` metadata for every tensor pushed on the stack.
   - Pre-allocate intermediate buffers on GPU; reuse across invocations to avoid reallocations.

4. **Temporal Pipeline Mapping**
   - Layer 1: `matvec` → `relu`.
   - Optional: integrate temporal mask by precomputing mask vector (via `TemporalReasoning`) and applying `OP_VECTOR_MUL_F32`.
   - Layer 2: `matvec` → mask multiply → `relu`.
   - Layer 3: `matvec` → `sigmoid`.
   - Entropy: handle via new `OP_ENTROPY_SUM` (push scalar back on stack) and/or fallback CPU verification.

5. **Integration Points**
   - Replace `_execute_temporal_mlp` and `_execute_spatial_mlp` with calls into the new bridge.
   - Keep legacy path under feature flag (`K3D_USE_LEGACY_RPN`) for regression tests.
   - Ensure GraphCrystallizer, Telemetry, and fallback paths continue to operate on the returned numpy arrays.

6. **Testing & Benchmarks**
   - Add GPU tests for new opcodes (matvec, relu, mask, sigmoid, entropy).
   - Extend ThinkingTag unit test to compare legacy vs new bridge outputs (tolerance 1e‑4).
   - Benchmark FUSE stage latency before/after to quantify target 10–20× speedup.
