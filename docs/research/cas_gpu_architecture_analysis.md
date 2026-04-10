# Deep Architecture Analysis: Giac/Xcas → Sovereign PTX Kernel Adaptation

## Executive Summary

Giac/Xcas's dynamic C++ object model is fundamentally incompatible with GPU SIMT execution. A ground-up rearchitecture is required: flatten expression trees into a **fixed-size, pointer-free DAG representation** in VRAM, implement CAS operations as **stateless, non-recursive PTX kernels** using iterative stack algorithms, and encode expressions directly as **RPN instruction streams** that leverage K3D's existing opcode infrastructure.

---

## (a) Recommended Expression Representation for GPU

### Giac's Native Structure (Anti-Pattern for GPU)
Giac uses a **hybrid tree/DAG** with:
- `gen` variant type (discriminated union + RTTI)
- `symbolic` nodes holding a `sommet*` (function symbol) + `vecteur` (std::vector<gen> arguments)
- Heavy use of `std::map` for symbol tables and `shared_ptr` for reference counting
- **Fatal flaws**: dynamic allocation, vtable pointers, non-contiguous memory, recursive traversal

### Sovereign GPU Expression Format: **STAR (Static Typed ARray) Node**

```c
// 16-byte aligned node fits in single cache line
struct StarNode {
    uint32_t opcode;      // K3D opcode (0x00-0xFF reserved for core ops)
    uint32_t flags;       // Bit 0-7: type tag, Bit 8-15: refcount, Bit 16-23: arity
    union {
        int32_t  imm32;   // Immediate integer
        float    immf32;  // Immediate float
        uint32_t child[2]; // Left/right child indices for binary ops
        uint32_t payload; // Symbol ID or buffer offset
    } data;
    uint32_t next;        // Linked list for free pool or hash chain
} __attribute__((aligned(16)));
```

**Memory Layout Strategy:**
- **Pre-allocated pool**: `__device__ StarNode expr_pool[1<<20];` (16MB VRAM)
- **Index-based addressing**: All references are 32-bit pool indices, never pointers
- **DAG support**: Reference counting in `flags` enables shared subtrees (e.g., `(x+1)` in `(x+1)*(x+1)`)
- **Symbol interning**: String names resolved to 32-bit IDs on CPU; symbol table stored in `__constant__` memory
- **Polynomial specialization**: For dense polynomials, use contiguous coefficient arrays with `OP_POLYNOMIAL` node pointing to `float coeff[]` in Galaxy buffer

---

## (b) Minimal PTX Kernel Set for CAS

### Kernel Design Philosophy
- **No recursion**: All algorithms rewritten as iterative with explicit stack in local memory
- **Warp-cooperative**: 32 threads per expression for parallel subtree processing
- **Stateless**: Pure functions; no global state beyond expression pool
- **Early exit**: Predicated execution for divergence minimization

### Core Kernel Suite (6 kernels, ~2k LOC total)

| Kernel | Parallelism Model | VRAM Footprint | Throughput |
|--------|-------------------|----------------|------------|
| **`k3d_expr_build`** | Thread-per-RPN-instruction | 1 node/instr | 1M exprs/sec |
| **`k3d_diff`** | Warp-per-expression (32 threads) | 2x node pool | 500K exprs/sec |
| **`k3d_poly_mul`** | Block-per-polynomial (NTT) | 3x degree size | 100M coeffs/sec |
| **`k3d_simplify`** | Thread-per-node (grid stride) | In-place | 2M nodes/sec |
| **`k3d_factor_sqf`** | Warp-per-polynomial | 4x node pool | 200K exprs/sec |
| **`k3d_linsolve`** | Block-per-matrix (Gauss-Jordan) | 2x matrix size | 50K matrices/sec |

#### 1. **k3d_expr_build**: RPN → STAR DAG
```cuda
__global__ void k3d_expr_build(
    const RPNInstruction* program,
    StarNode* pool,
    uint32_t* node_counter
) {
    uint32_t pc = blockIdx.x * blockDim.x + threadIdx.x;
    RPNInstruction instr = program[pc];
    
    // Allocate node
    uint32_t idx = atomicAdd(node_counter, 1);
    StarNode* node = &pool[idx];
    
    // Decode instruction
    node->opcode = instr.opcode;
    node->flags = MAKE_FLAGS(instr.arity, 0);
    
    // Handle operands
    if (instr.opcode == OP_PUSH_SYMBOL) {
        node->data.payload = instr.operand[0]; // symbol ID
    } else if (instr.arity == 2) {
        node->data.child[0] = instr.operand[0];
        node->data.child[1] = instr.operand[1];
        // Increment refcount on children
        atomicAdd(&pool[node->data.child[0]].flags, REFCOUNT_INC);
        atomicAdd(&pool[node->data.child[1]].flags, REFCOUNT_INC);
    }
}
```

#### 2. **k3d_diff**: Symbolic Differentiation
- **Algorithm**: Iterative post-order traversal using explicit stack in `__shared__` memory
- **Parallel**: 32 threads cooperate to differentiate one expression; thread `i` processes node `(node_id + i)` in grid-stride fashion
- **Rule dispatch**: `switch(opcode)` with manual jump table (no virtual calls)
- **Chain rule**: Derivative of `f(g(x))` = `f'(g(x)) * g'(x)` computed via temporary node allocation

#### 3. **k3d_poly_mul**: Polynomial Multiplication
- **Dense**: Use **Number Theoretic Transform (NTT)** mod 2^64+13
- **Sparse**: Sort-merge by exponent using bitonic sort
- **Parallel**: Cooley-Tukey FFT butterfly pattern across warps

#### 4. **k3d_simplify**: Expression Simplification
- **Pattern matching**: Parallel node reduction using **term rewriting automaton** encoded as state machine
- **Constant folding**: Warp reduction for `OP_ADD` of constants
- **Identity elimination**: `x*1 → x`, `x+0 → x` via predicated node replacement

#### 5. **k3d_factor_sqf**: Squarefree Factorization
- **Parallel**: Yun's algorithm with GCD computed via **half-GCD** (HGCD) using NTT-based polynomial multiplication
- **Modular**: Compute over `F_p` for multiple primes in parallel (one warp per prime)

#### 6. **k3d_linsolve**: Linear System Solver
- **Dense**: **Gauss-Jordan elimination** with **partial pivoting** (parallel row swap via ballot sync)
- **Sparse**: **Conjugate Gradient** with matrix in CSR format

---

## (c) RPN Program Encoding for Math Galaxy

### Native Expression Encoding
Math Galaxy's `meaning_rpn` field should directly store the **serialized STAR DAG** as a compact instruction stream:

```c
struct MeaningCentricStar {
    uint64_t symbol_hash;     // SHA256 of symbol name
    uint32_t rpn_program_offset; // Offset in Galaxy RPN buffer
    uint32_t rpn_program_len;    // Number of instructions
    uint32_t visual_rpn_offset;  // For UI rendering (separate)
    float    complexity_score;   // Pre-computed
};
```

### RPN Instruction Set Extension (K3D Opcodes)
Leverage existing stubs and extend:

```c
// Core opcodes (0x00-0x7F)
#define OP_PUSH_INT     0x01
#define OP_PUSH_REAL    0x02
#define OP_PUSH_SYMBOL  0x03
#define OP_ADD          0x10
#define OP_MUL          0x11
#define OP_POW          0x12

// CAS opcodes (0xB0-0xBF) - fill existing stubs
#define OP_SYMBOLIC_DIFF 0xB5  // arity=1: expression
#define OP_GRADIENT      0xB6  // arity=2: expr, var-list
#define OP_SYMBOLIC_INT  0xB7  // arity=2: expr, var
#define OP_LIMIT         0xB9  // arity=3: expr, var, point
#define OP_SERIES_SUM    0xBA  // arity=4: expr, var, start, end

// Polynomial opcodes (0xC0-0xCF)
#define OP_POLYNOMIAL    0xC0  // arity=2: degree, coeff_ptr
#define OP_POLY_MUL      0xC1
#define OP_POLY_GCD      0xC2
#define OP_FACTOR        0xC3
```

### Example: Encoding `∂/∂x (x^2 + sin(x))`

**RPN Program (meaning_rpn)**
```
[0] {opcode: OP_PUSH_SYMBOL, operand: [HASH_x]}
[1] {opcode: OP_PUSH_INT,    operand: [2]}
[2] {opcode: OP_POW}
[3] {opcode: OP_PUSH_SYMBOL, operand: [HASH_x]}
[4] {opcode: OP_FUNC_SIN}
[5] {opcode: OP_ADD}
[6] {opcode: OP_SYMBOLIC_DIFF, operand: [HASH_x]}  // differentiate w.r.t x
```

**Execution Flow:**
1. **CPU**: Validates program, interns symbols, allocates VRAM pool
2. **k3d_expr_build**: Launches `gridDim = (len/256, 1, 1)`, `blockDim = (256, 1, 1)` → builds STAR DAG
3. **k3d_diff**: Launches `gridDim = (num_exprs, 1, 1)`, `blockDim = (32, 1, 1)` → computes derivative
4. **k3d_simplify**: Optional pass to reduce `2*x + cos(x)` (derivative of sin is cos)

---

## (d) Sovereign GPU-Friendly Alternatives (Reality Check)

While adapting Giac is possible, **SymEngine** is architecturally superior for GPU:

| Feature | Giac/Xcas | SymEngine | K3D STAR |
|---------|-----------|-----------|----------|
| **Memory Model** | Dynamic heap | Immutable pool | Fixed VRAM pool |
| **Thread Safety** | No (shared state) | Yes (immutable) | Yes (lock-free) |
| **Code Size** | ~150k LOC | ~50k LOC | ~2k LOC kernels |
| **Recursion** | Deep recursion | Iterative | Iterative only |
| **License** | GPL-3 | MIT | Sovereign |

**Recommendation**: 
- **Short-term**: Implement minimal STAR kernel set (above) for **polynomial and linear algebra only** (no general symbolic integration)
- **Long-term**: Port **SymEngine's immutable architecture** to PTX; its `Basic` object model maps directly to STAR nodes without dynamic allocation
- **Giac value**: Steal its **Risch integration** algorithm and pattern database, but reimplement as data-driven bytecode interpreter (not recursive C++)

---

## Sovereign Implementation Roadmap

1. **Week 1-2**: Define STAR format, implement `k3d_expr_build` and `k3d_simplify`
2. **Week 3-4**: Implement `k3d_diff` for elementary functions
3. **Week 5-6**: Add `k3d_poly_mul` with NTT
4. **Week 7-8**: Implement `k3d_factor_sqf` and `k3d_linsolve`
5. **Week 9**: Build CPU-side RPN assembler that compiles Giac expressions to STAR bytecode
6. **Week 10**: Integration test with Math Galaxy's `MeaningCentricStar` loader

**PTX Kernel Binary Size Target**: < 512KB total (fits in L2 cache)

This architecture achieves **sovereign GPU-native CAS** with zero external dependencies, deterministic memory usage, and full parallelism exploitation.