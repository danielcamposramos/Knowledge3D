# Enhanced GPU-Native RPN Kernel for Cranium Core

This document captures Claude's extended PTX kernel notes (October 2025) and
serves as the implementation guide for the next-generation modular RPN engine.

## Executive Summary

| Strengths | Risk & Mitigation |
|-----------|------------------|
| Direct opcode → hardware execution, no parsing overhead | RPN should be treated as a **compiler target**. Build an infix/DSL front-end that generates RPN automatically. |
| Predictable circular-stack behaviour ideal for warp-level execution | Provide tracing/visual tooling; debugging raw opcode streams is otherwise painful. |
| Fully transparent, auditable compute graph | Expression verbosity (e.g., attention) explodes without a higher-level authoring layer. |

Recommended architecture:

```
Natural Language → Cranium Core → RPN Program → GPU Kernel → Result
```

## Kernel Highlights

- **Entry signature**
  ```ptx
  .visible .entry enhanced_rpn_geometric_kernel(
      .param .u32 param_instance_id,
      .param .u64 param_op_codes,
      .param .u64 param_scalars,
      .param .u64 param_vectors,
      .param .u64 param_matrices,
      .param .u64 param_state,
      .param .u64 param_sync_flags,
      .param .u32 param_token_count
  )
  ```
- Supports 15 instances, each with a 64 element float4 circular stack and error
  register.
- Adds a **matrix literal pool** and **sync flag buffer** for cross-instance
  orchestration.
- Retains auto-pruning semantics: overflow overwrites the oldest entries.

## Opcode Catalogue

### Literals (0–2)
- `LITERAL_SCALAR`, `LITERAL_VECTOR`, `LITERAL_MATRIX` (pushes 16 floats via 4
  successive pushes).

### Arithmetic & Advanced Math (10–39)
- Core arithmetic (`ADD`, `SUB`, `MUL`, `DIV`, `POW`, `NEG`, `ABS`, `FMA`).
- Nonlinear activations (`SIGMOID`, `RELU`, `GELU`, `TANH`, `SOFTMAX`,
  `LAYERNORM`).
- Trigonometric & exponential suite (`SIN`, `COS`, `TAN`, `EXP`, `LOG`, etc.).

### Comparisons (40–49)
- `GT`, `LT`, `EQ`, `MAX`, `MIN` emitting 1.0 / 0.0 sentinels.

### Stack Mechanics (50–59)
- `DUP`, `SWAP`, `DROP`, `OVER`, `ROT`, `CLEAR`.

### Vector Algebra (60–79)
- `DOT`, `CROSS`, `MAG`, `NORM`, `ROTATE`, `SCALE`, `TRANSLATE` with axis-angle
  support.

### Conditional (80–89)
- `IFELSE` selects between the previous two stack entries based on a predicate.

### Matrix Engine (90–99)
- `MATMUL`, `TRANSPOSE`, `DETERMINANT`, `INVERSE` dispatch to 1×1–4×4
  specialisations based on a runtime dimension popped from the stack.

### Statistical Reducers (100–109)
- `SUM`, `MEAN`, `VARIANCE`, `STDDEV` operate over float4 payloads.

### Cross-Instance Semantics (110–113)
- `XREF_PEEK`, `XREF_POP`, `XREF_WAIT`, `BROADCAST` enable inter-instance
  messaging entirely inside GPU global memory.

### Attention & Transformer Ops (120–129)
- `ATTENTION_SCORE`, `QKV_SPLIT`, `ROPE`, `KV_CACHE_UPDATE`.

### Convolutional Blocks (130–139)
- `CONV1D`, `CONV2D_3X3`, `MAXPOOL`, `AVGPOOL`.

### Embedding Utilities (140–149)
- `EMBED_LOOKUP`, `POSITION_ENCODE`, `COSINE_SIMILARITY`, `EMBED_INTERPOLATE`.

### Quantisation (150–159)
- `QUANTIZE_INT8`, `DEQUANTIZE_INT8`, `QUANTIZE_FP16`.

## Error Codes

| Code | Description |
|------|-------------|
| 9001 | Unknown opcode |
| 9002 | Stack underflow |
| 9010 | Invalid cross-instance ID |
| 9011 | Target stack empty |
| 9012 | Cross-instance wait timeout |
| 9013 | Invalid matrix dimension |
| 9014 | Singular matrix (inverse failed) |

## Integration Guidance

1. **Compiler layer first** – build a DSL/infix translator that emits opcode
   streams; hand-written RPN is not maintainable for transformer workloads.
2. **Profiling hooks** – extend the loader to log opcode histograms, stack depth
   usage, and cross-instance latch wait times.
3. **Large matrix extension** – introduce tiled matmul microkernels to cover
   64×64 / 768×768 transforms beyond the current 4×4 range.
4. **Debug tooling** – add debug opcodes (`DEBUG_PRINT`, `DEBUG_TRACE`) and
   visual stack inspectors for Cranium Core sessions.
5. **Compiler targets** – treat the GPU kernel as the canonical backend for
   fused-head reasoning; the Cranium Core emits programs, the House stores
   results as GLB appliances.

## Example Workflow

```
# Instance 0 – vision embedding → broadcast
QKV_SPLIT
ATTENTION_SCORE
BROADCAST

# Instance 3 – fusion head
LITERAL 0
XREF_WAIT
XREF_PEEK    # gather keys
...
SOFTMAX
```

Execute all modalities in parallel; no CPU sync required. Use `XREF_WAIT` and
`BROADCAST` to orchestrate cross-instance data flow.

## Files

- Kernel implementation: `knowledge3d/cranium/ptx/enhanced_rpn_kernel.ptx`
- Design overview: `docs/phase25_ptx_rpn_design.md`
- This document: `docs/ptx/enhanced_rpn_kernel.md`

