# Phase 25 PTX RPN Engine — Design Overview

## Goals
- Replace the current Python-only `RPNCalculator` with a **PTX-resident modular engine**.
- Support **up to 15 concurrent instances** sharing one kernel image but maintaining independent stacks and metadata.
- Provide the operation breadth required for LLM/AI reasoning: arithmetic, exponentials, trig, comparison, stack manipulation, conditionals, vector ops, reduction, and control primitives.
- Guarantee **bounded memory with automatic pruning** (oldest entries dropped once capacity exceeded).
- Keep execution fully inside the fused head: Python only orchestrates kernel calls and marshals buffers.
- Integrate seamlessly with existing PTX loaders (CUDA driver path if available; Torch JIT fallback for CPU development remains acceptable but must mirror semantics).

## High-Level Architecture

### Kernel Entry
```
.visible .entry modular_rpn_geometric_kernel(
    .param .u32 instance_id,
    .param .u64 op_codes_ptr,      // uint16_t codes, length = token_count
    .param .u64 scalars_ptr,       // float32 literals aligned with tokens (operands)
    .param .u64 vectors_ptr,       // float32[3] triples for vector tokens (optional)
    .param .u64 instance_state_ptr,// struct array [15]
    .param .u32 token_count
)
```
- `instance_state` is an array of 15 structs in global memory. Each struct:
  - `float stack[STACK_MAX][4]` (XYZW geometric register). Fourth lane reserved for metadata/flags.
  - `int stack_top` (index of next free slot, wraps mod `STACK_MAX`).
  - `int stack_size` (current count ≤ STACK_MAX).
  - `int last_error` (0 OK; non-zero codes for overflow/invalid op/etc.).
- `STACK_MAX` default 64 entries. When pushing and `stack_size == STACK_MAX`, the oldest (circular buffer head) is overwritten (auto pruning).

### Token Encoding
- Host converts textual tokens into numeric op codes.
- RPN literals: encoded with opcode `OP_LITERAL` + scalar payload in `scalars_ptr` (or `vectors_ptr` for vector literals).
- Operators map to `uint16` codes. Examples:
  - Arithmetic: `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV`, `OP_POW`, `OP_NEG`.
  - Advanced math: `OP_SQRT`, `OP_EXP`, `OP_LOG`, `OP_LOG10`, `OP_SIN`, `OP_COS`, `OP_TAN`, `OP_ASIN`, `OP_ACOS`, `OP_ATAN`, `OP_SINH`, `OP_COSH`, `OP_TANH`.
  - Comparison: `OP_GT`, `OP_GE`, `OP_LT`, `OP_LE`, `OP_EQ`, `OP_NE` (results 1.0/0.0 in X lane).
  - Stack ops: `OP_DUP`, `OP_SWAP`, `OP_DROP`, `OP_OVER`, `OP_ROT`, `OP_CLEAR`.
  - Conditional: `OP_IF_ELSE` (pops condition + two values), `OP_SELECT`, `OP_MAX`, `OP_MIN`.
  - Aggregation: `OP_SUM_N`, `OP_MEAN_N` (consume N from top, where N encoded in W lane or next literal), `OP_DOT`, `OP_CROSS`, `OP_MAG`.
  - Integration helpers: `OP_INT_DELTA`, `OP_DIFF`, `OP_ACCUM` (for calculus-style accumulations).
  - Geometry: `OP_ROTATE`, `OP_SCALE`, `OP_TRANSLATE` operate on vector lanes.
  - Control/metadata: `OP_NOOP`, `OP_ERROR`, `OP_PUSH_STATE`, `OP_POP_STATE` (future use).

### Stack Layout
- Each stack element is a float4 `(x, y, z, w)` describing the geometric payload.
  - Scalar-only operations use X lane (`Y=Z=0`, `W` optionally holds metadata like count).
  - Vector operations use (X,Y,Z).
  - `W` lane reserved for: `count` (for `OP_SUM_N`), `flag`, or `timestamp` if needed.
- Indices stored modulo `STACK_MAX` for auto-pruning. Push writes at `(stack_head + stack_size) % STACK_MAX`; pop decrements `stack_size` and updates head.

### Memory Flow
1. Host prepares contiguous arrays:
   - `op_codes`: `np.uint16` length `token_count`.
   - `scalars`: `np.float32` length `token_count` (unused slots set 0).
   - `vectors`: `np.float32` size `token_count x 3` (unused 0).
2. Host passes pointer to persistent `instance_state` (allocated once, reused across calls).
3. Kernel iterates tokens sequentially, updates stack per opcode, writes back `stack_top`, `stack_size`, `last_error` at end.
4. Kernel writes final top-of-stack element to `instance_state[instance_id].stack[(head + size-1) % STACK_MAX]`. Host reads this value as result.

### Error Handling
- Kernel sets `last_error` and early exits on invalid operations (underflow, domain errors, unknown opcode). Host checks and raises.
- Domain checks (e.g., log of non-positive) produce quiet NaN but also set error code for honesty metrics.

### Host Launcher (`ModularRPNEngine`)
- Lives in `knowledge3d/cranium/phase10/rpn_engine.py`.
- Responsibilities:
  - Load PTX (via existing `NVRTCPTXLoader`, extended to resolve kernels by name).
  - Allocate persistent `instance_state` buffer on device (or CPU fallback via Torch JIT stub mirroring semantics).
  - Convert string expressions -> tokens using curated operator table (populated from math corpus).
  - Provide API: `evaluate(instance_id, expression, mode="scalar"|"vector") -> np.ndarray`
  - Manage fallbacks if GPU unavailable (Torch JIT reimplementation).

### Integration Points
- Replace `RPNCalculator` usages in `meaning_cluster_trainer` and `cognitive_validator` with the new engine (keeping identical interface where possible).
- Ensure diary/logging captures PTX evaluation traces for honesty analysis.

## Curation Pipeline Summary
- New module `knowledge3d/tools/phase25/rpn_corpus_builder.py`:
  - Walks the Advanced Maths JSON directory.
  - Extracts math expressions, definitions, and worked solutions (heuristics: regex for formulas, keywords like "Example", "Exercise").
  - Normalises to structured JSONL: `{ "source", "page", "category", "expression_rpn", "infix", "solution", "notes" }`.
  - Generates token map expansions (e.g., if a book introduces `sinh`, ensure opcode defined).
  - Output stored under `viewer/public/galaxy/working/rpn_corpus/` for training.

## Next Steps
1. Implement RPN corpus builder for curated math sources (`knowledge3d/tools/phase25/rpn_corpus_builder.py`).
2. Ship NVRTC-compiled kernel + launcher (`knowledge3d/cranium/phase10/modular_rpn_engine.py`).
3. Replace legacy calculator with GPU-backed wrapper (`knowledge3d/cranium/phase10/rpn_calculator.py`).
4. Expand validator coverage (scalar + vector checks) to enforce kernel correctness.
5. Wire curated drills into Algorithmic Thinking once corpus is generated.
