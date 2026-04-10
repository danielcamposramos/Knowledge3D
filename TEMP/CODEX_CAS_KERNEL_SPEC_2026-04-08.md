# Codex Directions: Sovereign CAS Kernel Suite
**Date:** 2026-04-08
**Depends on:** Opcode audit migration (DONE — 47 tests green)
**Priority:** P1 — extends Math Galaxy with sovereign symbolic algebra

---

## Context

The Math Galaxy already holds math symbol stars (∑, ∏, ∫, ∂, ∇ etc.) with `visual_rpn` for rendering.
The `OP_SYMBOLIC_DIFF` (0xB5), `OP_SYMBOLIC_INTEGRATE` (0xB7), `OP_GRADIENT` (0xB6), and related stubs (0xB9-0xBE) exist in the opcode registry but have **no PTX kernel behind them** — they fall through to the default case.

This spec wires the sovereign CAS layer:
- STAR node pool for GPU expression representation (no pointers, no dynamic allocation)
- 4-kernel initial suite: expr_build, diff, poly_mul, simplify
- New polynomial/algebra opcode range: 0x220-0x25F
- Grammar Galaxy transformation rules for CAS rules (chain rule, power rule, trig identities)
- SymEngine Python ingestion path (sovereignty preserved — SymEngine only used at ingestion time)

**Architecture source:** Research filed at `docs/research/cas_gpu_architecture_analysis.md` + `docs/research/cas_rpn_integration_spec.md`.
Deep study recommendation: **SymEngine** (MIT, ~50k LOC, `github.com/symengine/symengine`) — its immutable expression pool architecture maps directly to STAR nodes. **Do NOT** model after Giac (dynamic heap, shared_ptr, deep recursion — GPU-incompatible).

---

## A. STAR Node Format (GPU Expression Representation)

Create `knowledge3d/cranium/kernels/cas_star_node.h`:

```c
#pragma once
#include <stdint.h>

// ──────────────────────────────────────────────────────────────
//  StarNode — 16-byte aligned, zero pointers, pool-indexed
//  All references are uint32_t pool indices (never raw pointers)
// ──────────────────────────────────────────────────────────────
struct StarNode {
    uint32_t opcode;      // K3D opcode — OP_ADD/OP_MUL/OP_SIN/OP_VAR_X/OP_CONST etc.
    uint32_t flags;       // Bits 0-7: arity, Bits 8-15: type_tag, Bits 16-23: refcount
    union {
        float    immf32;       // Immediate float (arity == 0, type_tag == TAG_FLOAT)
        int32_t  immi32;       // Immediate integer (arity == 0, type_tag == TAG_INT)
        uint32_t child[2];     // left/right child pool indices (arity == 2)
        uint32_t payload;      // symbol_id or coeff_buf_offset (arity == 0, type_tag == TAG_SYM)
    } data;
    uint32_t next;             // free-pool chain or hash chain
} __attribute__((aligned(16)));

static_assert(sizeof(StarNode) == 16, "StarNode must be 16 bytes");

// Type tags (flags bits 8-15)
#define TAG_FLOAT   0x01
#define TAG_INT     0x02
#define TAG_SYMBOL  0x03
#define TAG_POLY    0x04    // payload = offset into polynomial coefficient buffer

// Flags macros
#define STAR_FLAGS(arity, type_tag, refcount) \
    (((uint32_t)(arity) & 0xFF) | (((uint32_t)(type_tag) & 0xFF) << 8) | (((uint32_t)(refcount) & 0xFF) << 16))
#define STAR_ARITY(flags)    ((flags) & 0xFF)
#define STAR_TAG(flags)      (((flags) >> 8) & 0xFF)
#define STAR_REFCOUNT(flags) (((flags) >> 16) & 0xFF)

// Pool constants
#define CAS_POOL_SIZE    (1 << 20)   // 1M nodes = 16MB VRAM
#define CAS_COEFF_SIZE   (1 << 18)   // 256K float32 coefficients = 1MB VRAM
#define CAS_NULL_IDX     0xFFFFFFFFu // null/invalid index

// Symbol IDs (interned on CPU, stored in __constant__ memory)
#define SYM_X  0x01
#define SYM_Y  0x02
#define SYM_Z  0x03
#define SYM_W  0x04
#define SYM_PI 0x10
#define SYM_E  0x11
```

Add to `modular_rpn_kernel.cu` (top of file, after existing globals):
```c
// ── CAS expression pool ──────────────────────────────────────
__device__ StarNode  g_cas_pool[CAS_POOL_SIZE];
__device__ float     g_cas_coeffs[CAS_COEFF_SIZE];
__device__ uint32_t  g_cas_pool_top;   // atomic counter for allocation
__device__ uint32_t  g_cas_coeff_top;  // atomic counter for coeff allocation
```

Add Python bindings to `sovereign_bridges.py`:
```python
def bind_cas_pool() -> None:
    """Upload and bind the CAS STAR node pool globals to the modular kernel."""
    # upload 16MB pool + 1MB coeff buffer as zero-initialized device memory
    # store handles in bridge for reuse
    ...
```

---

## B. New CAS Opcode Range: 0x220-0x25F

Add to `rpn_opcodes.py` (new section after `# Entity behavior opcodes`):

```python
# ── Sovereign CAS opcodes (0x220-0x25F) ──────────────────────
# Polynomial algebra layer (feeds the existing 0xB5-0xBE calculus stubs)
OP_POLY_COEFF             = 0x220  # push coefficient onto poly-build stack
OP_POLY_BUILD             = 0x221  # degree → polynomial StarNode from coeff stack
OP_POLY_ADD               = 0x222  # poly_a poly_b → poly_sum
OP_POLY_MUL               = 0x223  # poly_a poly_b → poly_product  (NTT accelerated)
OP_POLY_DIV               = 0x224  # num den → quotient
OP_POLY_REM               = 0x225  # num den → remainder
OP_POLY_GCD               = 0x226  # poly_a poly_b → gcd
OP_POLY_FACTOR            = 0x227  # poly → factor_list (squarefree, Yun's algorithm)
# Simplification and transformation
OP_SIMPLIFY               = 0x228  # expr → simplified (pattern rewriting automaton)
OP_SUBSTITUTE             = 0x229  # expr var_id val → substituted
OP_COLLECT                = 0x22A  # expr var_id → collected terms
OP_RATIONALIZE            = 0x22B  # expr → rationalized
OP_TRIG_SIMPLIFY          = 0x22C  # expr → trig-simplified (sin²+cos²=1, etc.)
OP_LOG_SIMPLIFY           = 0x22D  # expr → log-simplified
# Solving
OP_SOLVE_LINEAR           = 0x22E  # a b → root  (a*x + b = 0)
OP_SOLVE_QUADRATIC        = 0x22F  # a b c → root_1 root_2
OP_LINSOLVE               = 0x230  # matrix_ptr n → solution vector (Gauss-Jordan)
# Pattern and rule application
OP_PATTERN_MATCH          = 0x231  # expr pattern → 1.0 (match) or 0.0 (no match)
OP_RULE_APPLY             = 0x232  # expr rule_galaxy_id → transformed expr
OP_COEFF_EXTRACT          = 0x233  # poly var_id power → coefficient
# Expression construction helpers
OP_CAS_PUSH_SYM           = 0x234  # symbol_id → StarNode on stack
OP_CAS_PUSH_CONST         = 0x235  # float_val → StarNode on stack
OP_CAS_BUILD              = 0x236  # opcode arity → compose StarNode from top of stack
OP_CAS_EVAL               = 0x237  # expr var_id val → float result (numeric eval)
```

These opcodes are all **uint16** — ensure they are emitted via `RPNProgram.u16()` (already supported from the TRM migration in the opcode audit step).

---

## C. CAS Kernel Suite

Create `knowledge3d/cranium/kernels/cas_kernels.cu`.
Include `cas_star_node.h` and `rpn_opcodes.h` (or the relevant opcode constants).

### C1. `k3d_expr_build` — RPN instruction stream → STAR DAG
```c
// Input:  RPN program bytes (uint32 opcodes + operands)
// Output: root pool index of the built expression
// Launch: 1 block × 1 thread (sequential program execution)
//         For batch: one block per expression, expressions independent
__global__ void k3d_expr_build(
    const uint32_t* __restrict__ program,
    uint32_t  program_len,
    uint32_t* out_root_idx        // output: root StarNode index
);
```

Algorithm: iterate over program; for each opcode push a new StarNode; for binary ops pop two children, set child[0]/child[1], push result. Use `atomicAdd(&g_cas_pool_top, 1)` for allocation. No recursion.

### C2. `k3d_diff` — Symbolic differentiation
```c
// Input:  root StarNode index, symbol_id to differentiate with respect to
// Output: root index of derivative expression (new nodes allocated in pool)
// Launch: one warp (32 threads) per expression — threads stride over nodes
__global__ void k3d_diff(
    uint32_t  root_idx,
    uint32_t  var_sym_id,
    uint32_t* out_root_idx
);
```

Algorithm: **iterative post-order traversal** using explicit stack in `__shared__` memory (no recursion). Rule dispatch via `switch(node.opcode)`:

| Node opcode | Derivative rule |
|------------|----------------|
| OP_CONST | → 0 |
| OP_CAS_PUSH_SYM, sym == var | → 1 |
| OP_CAS_PUSH_SYM, sym ≠ var | → 0 |
| OP_ADD | diff(left) + diff(right) |
| OP_SUB | diff(left) - diff(right) |
| OP_MUL | diff(left)*right + left*diff(right) (product rule) |
| OP_DIV | (diff(num)*den - num*diff(den)) / den² |
| OP_POWER (constant exp) | exp * base^(exp-1) * diff(base) |
| OP_SIN | cos(arg) * diff(arg) |
| OP_COS | -sin(arg) * diff(arg) |
| OP_EXP | exp(arg) * diff(arg) |
| OP_LOG | diff(arg) / arg |
| OP_POLY_BUILD | → dispatch to polynomial differentiation (reduces degree by 1) |

After differentiation, call `k3d_simplify` on the result.

### C3. `k3d_poly_mul` — Polynomial multiplication (NTT)
```c
// Input:  two StarNode polynomial roots (TAG_POLY), coefficient buffers
// Output: root of product polynomial
// Launch: block per polynomial pair; use Cooley-Tukey butterfly pattern
__global__ void k3d_poly_mul(
    uint32_t poly_a_idx,
    uint32_t poly_b_idx,
    uint32_t* out_root_idx
);
```

Algorithm: extract coefficient arrays → zero-pad to power-of-2 → NTT mod 2^31-1 → pointwise multiply → inverse NTT → write new StarNode + coeff slice.

### C4. `k3d_simplify` — Pattern-driven simplification
```c
// Input:  root StarNode index
// Output: root index of simplified expression (in-place or new nodes)
// Launch: thread-per-node, grid-stride
__global__ void k3d_simplify(
    uint32_t root_idx,
    uint32_t* out_root_idx
);
```

Patterns (hardcoded as a state machine, order matters):
- Constant folding: `OP_ADD(CONST a, CONST b)` → `CONST(a+b)`
- Identity: `x * 1 → x`, `x + 0 → x`, `x ^ 1 → x`, `x ^ 0 → 1`
- Double negation: `OP_NEGATE(OP_NEGATE(x)) → x`
- Zero: `x * 0 → 0`, `0 / x → 0`
- Trig identity: `OP_SIN²(x) + OP_COS²(x) → 1` (look for pattern in sibling nodes)
- Polynomial merge: adjacent `OP_POLY_ADD` nodes with same variable → merge coefficients

**Grammar Galaxy lookup (DEFERRED — Step 2 of CAS):** Eventually `k3d_simplify` will fetch transformation rules from Galaxy via `OP_GRAMMAR_QUERY` — this lets user-defined identities (Pythagorean theorem, Euler's formula, etc.) be applied. In Step 1, use only the hardcoded patterns above.

---

## D. Grammar Galaxy: Transformation Rules

Add `knowledge3d/cranium/cas_grammar_bootstrap.py`:

```python
"""Bootstrap Grammar Galaxy with CAS transformation rules."""
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar

# Each rule is a MeaningCentricStar with meaning_class="cas_rule"
# pattern_rpn: the expression shape to match (using OP_CAS_PUSH_SYM for wildcards)
# replacement_rpn: the transformed expression

DIFF_RULES = [
    MeaningCentricStar(
        star_id="cas_rule:diff_power",
        meaning_class="cas_rule",
        domain="grammar",
        meaning_rpn="OP_VAR_X OP_CONST n OP_POWER OP_SYMBOLIC_DIFF",
        behavior_rpn="OP_CONST n OP_VAR_X OP_CONST n_minus_1 OP_POWER OP_MUL",
        metadata={"rule_type": "differentiation", "name": "power_rule"},
        confidence=1, polarity=1,
    ),
    MeaningCentricStar(
        star_id="cas_rule:diff_sin",
        meaning_class="cas_rule",
        domain="grammar",
        meaning_rpn="OP_VAR_X OP_SIN OP_SYMBOLIC_DIFF",
        behavior_rpn="OP_VAR_X OP_COS",
        metadata={"rule_type": "differentiation", "name": "diff_sin"},
        confidence=1, polarity=1,
    ),
    MeaningCentricStar(
        star_id="cas_rule:diff_cos",
        meaning_class="cas_rule",
        domain="grammar",
        meaning_rpn="OP_VAR_X OP_COS OP_SYMBOLIC_DIFF",
        behavior_rpn="OP_VAR_X OP_SIN OP_NEGATE",
        metadata={"rule_type": "differentiation", "name": "diff_cos"},
        confidence=1, polarity=1,
    ),
    MeaningCentricStar(
        star_id="cas_rule:diff_exp",
        meaning_class="cas_rule",
        domain="grammar",
        meaning_rpn="OP_VAR_X OP_EXP OP_SYMBOLIC_DIFF",
        behavior_rpn="OP_VAR_X OP_EXP",
        metadata={"rule_type": "differentiation", "name": "diff_exp"},
        confidence=1, polarity=1,
    ),
    MeaningCentricStar(
        star_id="cas_rule:trig_pythagorean",
        meaning_class="cas_rule",
        domain="grammar",
        meaning_rpn="OP_VAR_X OP_SIN OP_CONST 2 OP_POWER OP_VAR_X OP_COS OP_CONST 2 OP_POWER OP_ADD",
        behavior_rpn="OP_CONST 1.0",
        metadata={"rule_type": "simplification", "name": "pythagorean_identity"},
        confidence=1, polarity=1,
    ),
]
```

Wire into ingestion: `ingest_cas_grammar(galaxy_manager)` — scans `DIFF_RULES`, calls `store_meaning_star(...)` into the `Grammar` region.

---

## E. Math Galaxy: Expression Stars

Expression stars are DISTINCT from the existing math symbol stars (∑, ∏, etc.).

New meaning_class: `"expression"` (vs `"math_symbol"` for glyph-only stars).

```python
# Example: the polynomial 3x² + 2x + 1
POLY_3X2_STAR = MeaningCentricStar(
    star_id="expression:poly:3x2_2x_1",
    meaning_class="expression",
    domain="math",
    meaning_rpn=(
        "OP_CONST 3.0 OP_POLY_COEFF "
        "OP_CONST 2.0 OP_POLY_COEFF "
        "OP_CONST 1.0 OP_POLY_COEFF "
        "OP_VAR_X OP_POLY_BUILD"
    ),
    behavior_rpn="OP_SIMPLIFY OP_POLY_FACTOR",
    metadata={
        "polynomial": True,
        "degree": 2,
        "variable": "x",
        "coefficients": [3.0, 2.0, 1.0],
    },
    confidence=1, polarity=1,
)
```

---

## F. Ingestion Path (SymEngine → STAR bytecode → Math Galaxy)

Create `knowledge3d/ingestion/cas_ingestion.py` (ingestion path only, sovereignty preserved):

```python
"""
Ingestion-path CAS utilities.
Uses SymEngine to parse math expressions and emit K3D STAR bytecode.
SymEngine is an ingestion-time tool ONLY — never imported in hot path.
"""

def expression_to_rpn(expr_str: str) -> str:
    """
    Parse a math expression string via SymEngine and return a
    K3D RPN program string encoding the expression as a STAR DAG.
    
    Example:
      expression_to_rpn("x**2 + sin(x)") →
      "OP_VAR_X OP_CONST 2.0 OP_POWER OP_VAR_X OP_SIN OP_ADD"
    """
    try:
        import symengine as se
    except ImportError:
        raise ImportError(
            "symengine is required for CAS ingestion. "
            "Install with: pip install symengine  (in k3d-cranium env)"
        )
    expr = se.sympify(expr_str)
    return _symengine_to_rpn(expr)


def _symengine_to_rpn(expr) -> str:
    """Recursive post-order traversal of SymEngine AST → RPN string."""
    import symengine as se
    if expr.is_Number:
        return f"OP_CONST {float(expr)}"
    if expr.is_Symbol:
        name = str(expr)
        sym_map = {"x": "OP_VAR_X", "y": "OP_VAR_Y", "z": "OP_VAR_Z", "w": "OP_VAR_W"}
        return sym_map.get(name, f"OP_CAS_PUSH_SYM {hash(name) & 0xFFFF}")
    if isinstance(expr, se.Add):
        parts = [_symengine_to_rpn(a) for a in expr.args]
        return " OP_ADD ".join(parts)
    if isinstance(expr, se.Mul):
        parts = [_symengine_to_rpn(a) for a in expr.args]
        return " OP_MUL ".join(parts)
    if isinstance(expr, se.Pow):
        base, exp = expr.args
        return f"{_symengine_to_rpn(base)} {_symengine_to_rpn(exp)} OP_POWER"
    if isinstance(expr, se.sin):
        return f"{_symengine_to_rpn(expr.args[0])} OP_SIN"
    if isinstance(expr, se.cos):
        return f"{_symengine_to_rpn(expr.args[0])} OP_COS"
    if isinstance(expr, se.exp):
        return f"{_symengine_to_rpn(expr.args[0])} OP_EXP"
    if isinstance(expr, se.log):
        return f"{_symengine_to_rpn(expr.args[0])} OP_LOG"
    raise ValueError(f"Unsupported SymEngine node: {type(expr)} {expr}")
```

Do NOT import symengine outside this file. Sovereignty maintained.

---

## G. Wire CAS Opcodes into modular_rpn_kernel.cu

Add cases 0x220-0x237 to the main switch in `modular_rpn_kernel.cu`.
These cases call the CAS device functions (not kernel launches — they are `__device__` functions
called from within the current kernel thread):

```c
case 0x220: {  // OP_POLY_COEFF — push coefficient onto poly-build stack
    float coeff = 0.0f;
    if (!pop_scalar(stack, stack_size, coeff, error_code)) break;
    // Store in coeff scratch space; poly-build collects them
    // STUB: push to poly_coeff_scratch[poly_coeff_top++]
    break;
}
case 0x228: {  // OP_SIMPLIFY — call inline simplify device fn
    // STUB: push 0.0 (identity pass) until k3d_simplify_device() is implemented
    break;
}
case 0x229: {  // OP_SUBSTITUTE — expr var_id val → substituted
    // STUB: push 0.0
    break;
}
// ... remaining cases: all stubs pushing 0.0 initially
```

**Step 1 = stubs + poly_coeff/poly_build.** Full kernel implementations in Step 2.

Do NOT implement `k3d_diff` and `k3d_poly_mul` as inline device functions — they require grid-level parallelism. They stay as separate `__global__` kernel entries invoked from the Python bridge.

---

## H. How Existing Stubs Wire Through

The existing calculus stubs (OP_SYMBOLIC_DIFF=0xB5 etc.) will eventually call `k3d_diff` from the bridge, not from within the modular dispatch. The flow:

```
Python bridge:
  1. Compile meaning_rpn text → bytecode via modular_rpn_engine
  2. If program contains OP_SYMBOLIC_DIFF:
     a. Run modular kernel up to the OP_SYMBOLIC_DIFF instruction
     b. Get expression root index from stack
     c. Launch k3d_diff(root_idx, var_sym_id) as separate kernel
     d. Push resulting derivative root to stack
     e. Continue modular kernel from next instruction
```

For Step 1, the OP_SYMBOLIC_DIFF case in `modular_rpn_kernel.cu` remains a stub. The Python bridge invocation of `k3d_diff` is a later step.

---

## I. TRMLauncher Gap (Noted, Not in Scope)

The opcode audit established that `TRM_MATVEC/SWIGLU` ops (now 0x300-0x304) are NOT modular-dispatch-owned. `TRMLauncher(use_rpn=True)` is therefore conceptually stale.

**Do NOT address this in the CAS step.** The CAS kernels (0x220-0x237) belong to the modular dispatch and are unaffected by the TRM launcher gap. The launcher gap will be resolved when the TRM game loop (Step D in ROADMAP: `trm_step_fused.cu` full dispatch) is wired.

For `rpn_math_core.py` — if it still references the old TRM opcode names, update the imports to new values (0x300-0x304). Do NOT change any logic.

---

## J. File Manifest

New files:
- `knowledge3d/cranium/kernels/cas_star_node.h` — STAR node struct + pool constants
- `knowledge3d/cranium/kernels/cas_kernels.cu` — k3d_expr_build, k3d_diff, k3d_poly_mul, k3d_simplify
- `knowledge3d/cranium/cas_grammar_bootstrap.py` — Grammar Galaxy CAS rules
- `knowledge3d/ingestion/cas_ingestion.py` — SymEngine → RPN (ingestion only)
- `tests/test_sovereign_cas_surface.py` — smoke tests

Modified files:
- `rpn_opcodes.py` — add 0x220-0x237 constants + `__all__`
- `modular_rpn_kernel.cu` — add cases 0x220-0x237 (stubs for now, poly_coeff/poly_build functional)
- `sovereign_bridges.py` — add `bind_cas_pool()`, `launch_k3d_diff()`, `launch_k3d_simplify()`
- `knowledge3d/ingestion/__init__.py` — wire `ingest_cas_grammar()`

---

## K. Smoke Tests

`tests/test_sovereign_cas_surface.py`:

```python
"""Smoke tests for the sovereign CAS surface."""
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op
from knowledge3d.cranium.cas_grammar_bootstrap import DIFF_RULES
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


def test_cas_opcodes_in_dedicated_range():
    cas_ops = [
        op.OP_POLY_COEFF, op.OP_POLY_BUILD, op.OP_POLY_ADD, op.OP_POLY_MUL,
        op.OP_SIMPLIFY, op.OP_SUBSTITUTE, op.OP_SOLVE_LINEAR, op.OP_RULE_APPLY,
    ]
    for v in cas_ops:
        assert 0x220 <= v <= 0x25F, f"CAS op {hex(v)} outside dedicated range 0x220-0x25F"


def test_cas_opcode_no_overlap_with_drawing():
    drawing = set(range(0x200, 0x220))
    cas = set(range(0x220, 0x260))
    assert not drawing & cas


def test_grammar_rules_are_meaning_centric_stars():
    for rule in DIFF_RULES:
        assert isinstance(rule, MeaningCentricStar)
        assert rule.meaning_class == "cas_rule"
        assert rule.domain == "grammar"
        assert rule.meaning_rpn is not None
        assert rule.behavior_rpn is not None


def test_star_node_header_exists():
    import os
    header = "knowledge3d/cranium/kernels/cas_star_node.h"
    assert os.path.exists(header), f"Missing: {header}"


def test_cas_grammar_bootstrap_ingestion():
    """Grammar rules serialize to Galaxy entries without error."""
    for rule in DIFF_RULES:
        entry = rule.to_galaxy_entry()
        assert entry["id"] == rule.star_id
        assert "meaning_star" in entry.get("metadata", {})


def test_cas_ingestion_not_imported_in_hot_path():
    """symengine must not appear in any hot-path module."""
    import ast, pathlib
    hot_path_dirs = [
        "knowledge3d/cranium/ptx_runtime",
        "knowledge3d/cranium/kernels",
        "knowledge3d/cranium/bridges",
        "knowledge3d/knowledgeverse",
    ]
    for d in hot_path_dirs:
        for py in pathlib.Path(d).rglob("*.py"):
            src = py.read_text()
            if "symengine" in src and "cas_ingestion" not in py.name:
                raise AssertionError(f"symengine imported in hot path: {py}")
```

Run gate:
```bash
pytest tests/test_sovereign_cas_surface.py tests/test_opcode_namespace_integrity.py -x -q
```

PTX rebuild: **YES** — `modular_rpn_kernel.cu` will gain new cases.

```bash
nvcc --ptx -arch=sm_86 -O3 --use_fast_math \
  -I knowledge3d/cranium/kernels \
  knowledge3d/cranium/kernels/modular_rpn_kernel.cu \
  -o knowledge3d/cranium/ptx/modular_rpn_kernel.ptx
```

---

## L. Implementation Order

```
CAS-A:  cas_star_node.h  (struct definition only, no build needed)
CAS-B:  rpn_opcodes.py additions (0x220-0x237)
CAS-C:  cas_grammar_bootstrap.py (DIFF_RULES as MeaningCentricStar)
CAS-D:  cas_ingestion.py (SymEngine → RPN, ingestion path only)
CAS-E:  modular_rpn_kernel.cu cases 0x220-0x237 (stubs, poly_coeff/poly_build functional)
CAS-F:  cas_kernels.cu — k3d_expr_build and k3d_diff (device functions, no full kernel launch yet)
CAS-G:  sovereign_bridges.py — bind_cas_pool() + lazy launch stubs
CAS-H:  ingestion/__init__.py — wire ingest_cas_grammar()
CAS-I:  tests/test_sovereign_cas_surface.py
CAS-J:  PTX rebuild
CAS-K:  Run full gate: cas_surface + opcode_integrity + entity + physics + texture
```

After CAS-K passes — report back. Step 2 of CAS (full `k3d_diff` kernel launch from bridge, `OP_SYMBOLIC_DIFF` wired end-to-end) is a separate spec.

---

## What NOT to Do

- **Do NOT** import Giac or Xcas anywhere. SymEngine only, and only in `cas_ingestion.py`.
- **Do NOT** port Giac's C++ object model (dynamic heap, shared_ptr, RTTI). The STAR node pool is the sovereign GPU representation.
- **Do NOT** add TRM cases to the modular dispatch in this step.
- **Do NOT** implement `k3d_diff` as an inline device call inside the modular switch — it needs grid-level launch from the bridge. Step 1 stubs are correct.
- **Do NOT** change `OP_SYMBOLIC_DIFF` (0xB5) or other existing calculus stubs — they stay as stubs, wired in Step 2.
- **Do NOT** use `uint8` for CAS opcode emission — all 0x220+ opcodes must use `RPNProgram.u16()`.
