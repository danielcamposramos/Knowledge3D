# Codex Direction: Sovereign Algebraic System (SAS) Layer
**Date:** 2026-04-08
**Builds on:** `TEMP/CODEX_CAS_KERNEL_SPEC_2026-04-08.md` (Step 1 complete)
**Precondition:** CAS Step 1 is green — `knowledge3d/cranium/kernels/cas_kernels.cu`,
`cas_star_node.h`, `cas_grammar_bootstrap.py`, opcode namespace `0x220-0x237` all in
place and tested.

---

## What "SAS" Means in K3D

CAS (Step 1) gave us the substrate: STAR node pool, k3d_expr_build/diff/simplify/poly_mul,
and the `0x220-0x237` opcode block.

SAS is the semantic layer on top of CAS. Three things that CAS alone cannot do:

| Gap | What's missing | New component |
|-----|---------------|---------------|
| A — Canonical Form | x+1 and 1+x produce different star_ids today. Semantic equality requires structural normalization BEFORE hashing. | `k3d_canonicalize` kernel + Hashcons table |
| B — Semantic Binding | `OP_CAS_PUSH_SYM G` currently pushes an unresolved uint32 symbol_id. At GPU execution time the kernel must look up G's actual value (6.674×10⁻¹¹) from Reality Galaxy without Python. | Boot-time symbol table + `OP_SEMANTIC_RESOLVE` |
| C — Pattern-Match-Replace | Grammar Galaxy CAS rules exist as MeaningCentricStars. The kernel implementation that actually matches a pattern against a live STAR DAG and rewrites it is missing. | `k3d_pattern_match` + `k3d_rule_apply` kernels + defeasible conflict resolution |

All three gaps use the architecture already built. Nothing invented from scratch.

---

## New Opcode Block: SAS Extension `0x238-0x23D`

Add to `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`, **immediately after the CAS block**:

```python
# ── Sovereign Algebraic System (SAS) extension (0x238-0x25F) ─────────────
# Semantic layer on top of the CAS substrate (0x220-0x237)

# Gap A — Canonical Form
OP_CANONICALIZE    = 0x238  # STAR_handle → normalized_handle (in-place + hashcons)
OP_CAS_HASH        = 0x239  # STAR_handle → 8-byte xxHash3 key pushed onto RPN stack

# Gap B — Semantic Binding
OP_SEMANTIC_RESOLVE = 0x23A # symbol_id → float value (from __constant__ symbol table)

# Gap C — Pattern-Match-Replace
OP_RULE_SELECT         = 0x23B  # grammar_query_embedding k → best Grammar rule handle
OP_CONTEXTUAL_REWRITE  = 0x23C  # rule_handle STAR_handle → rewritten_handle
OP_SEMANTIC_EQUIV      = 0x23D  # handle_a handle_b → 1 if canonically equal, else 0
```

Add all six to `__all__`.

---

## New Header: `sas_hashcons.h`

Create `knowledge3d/cranium/kernels/sas_hashcons.h`.

This is the only new header. It defines the GPU Hashcons table that deduplicates
structurally identical STAR subtrees after canonicalization.

```c
#pragma once

#include "cas_star_node.h"
#include <stdint.h>

// ──────────────────────────────────────────────────────────────────────────
//  SAS Hashcons — global linear-probe hash table in device memory
//
//  Key:   (opcode, child0, child1_or_next, flags)  — all uint32_t
//  Value: pool index of the canonical node
//
//  Warp-level dedup: before atomic insert, __match_any_sync detects that
//  another lane already inserted the same key this warp-cycle.
//
//  Table lives in __device__ memory alongside the CAS pool.
// ──────────────────────────────────────────────────────────────────────────

#define SAS_HASHCONS_SIZE  (1u << 20)   // 1M slots, must match or exceed CAS_POOL_SIZE
#define SAS_HASHCONS_EMPTY 0xFFFFFFFFu  // sentinel = empty slot

struct HashconsSlot {
    uint32_t key_opcode;
    uint32_t key_child0;
    uint32_t key_child1;   // child1 / next / poly_meta depending on node type
    uint32_t key_flags;
    uint32_t pool_idx;     // canonical pool index for this key, or EMPTY
    uint32_t _pad;         // keep 24-byte struct aligned to 8
};

// Compute slot index from key.  Uses a simple multiplicative hash.
__device__ __forceinline__ uint32_t hashcons_slot(
    uint32_t opcode, uint32_t child0, uint32_t child1, uint32_t flags)
{
    uint32_t h = opcode * 2654435761u ^ child0 * 2246822519u
               ^ child1 * 3266489917u ^ flags * 668265263u;
    return h & (SAS_HASHCONS_SIZE - 1u);
}

// Linear-probe lookup.  Returns pool_idx of matching entry or SAS_HASHCONS_EMPTY.
__device__ __forceinline__ uint32_t hashcons_lookup(
    const HashconsSlot* __restrict__ table,
    uint32_t opcode, uint32_t child0, uint32_t child1, uint32_t flags)
{
    uint32_t slot = hashcons_slot(opcode, child0, child1, flags);
    for (uint32_t probe = 0; probe < 32u; ++probe) {
        const HashconsSlot& s = table[(slot + probe) & (SAS_HASHCONS_SIZE - 1u)];
        if (s.pool_idx == SAS_HASHCONS_EMPTY) return SAS_HASHCONS_EMPTY;
        if (s.key_opcode == opcode && s.key_child0 == child0
            && s.key_child1 == child1 && s.key_flags == flags)
            return s.pool_idx;
    }
    return SAS_HASHCONS_EMPTY;
}

// Atomic insert.  Returns the winning pool_idx (either ours or a concurrent racer's).
__device__ __forceinline__ uint32_t hashcons_insert(
    HashconsSlot* __restrict__ table,
    uint32_t opcode, uint32_t child0, uint32_t child1, uint32_t flags,
    uint32_t pool_idx)
{
    uint32_t slot = hashcons_slot(opcode, child0, child1, flags);
    for (uint32_t probe = 0; probe < 32u; ++probe) {
        const uint32_t s = (slot + probe) & (SAS_HASHCONS_SIZE - 1u);
        uint32_t old = atomicCAS(&table[s].pool_idx, SAS_HASHCONS_EMPTY, pool_idx);
        if (old == SAS_HASHCONS_EMPTY) {
            // We claimed this slot — write the key fields.
            table[s].key_opcode = opcode;
            table[s].key_child0 = child0;
            table[s].key_child1 = child1;
            table[s].key_flags  = flags;
            return pool_idx;
        }
        // Slot taken — check if it's the same key (another racer won).
        if (table[s].key_opcode == opcode && table[s].key_child0 == child0
            && table[s].key_child1 == child1 && table[s].key_flags == flags)
            return old;
        // Different key — keep probing.
    }
    return pool_idx;  // fallback: treat as non-deduplicated (pool not wasted)
}
```

---

## New Kernel File: `sas_kernels.cu`

Create `knowledge3d/cranium/kernels/sas_kernels.cu`.

This file owns the three SAS kernels: `k3d_canonicalize`, `k3d_pattern_match`,
`k3d_rule_apply`. It shares the CAS pool from `cas_kernels.cu` via `extern "C"` device
symbol declarations (same pattern as modular_rpn_kernel.cu uses for CAS globals).

### File skeleton

```c
#include "cas_star_node.h"
#include "sas_hashcons.h"
#include <cuda_runtime.h>
#include <stdint.h>

// ── Shared device globals from cas_kernels.cu (declared there, referenced here) ──
extern __device__ StarNode g_cas_pool[];
extern __device__ float    g_cas_coeffs[];
extern __device__ uint32_t g_cas_pool_top;

// ── SAS-private device globals ──────────────────────────────────────────────────
extern "C" {
__device__ HashconsSlot g_sas_hashcons[SAS_HASHCONS_SIZE];

// Boot-time symbol table: symbol_id → float value.
// Populated by bind_sas_symbol_table() at startup from the star network.
// symbol_id 0x00 = unresolved, 0x01-0x04 = X/Y/Z/W (geometric vars, not looked up here)
// 0x10 = SYM_PI (3.14159...), 0x11 = SYM_E (2.71828...)
// 0x20-0x7F = Reality Galaxy physical constants (G, c, h, k_B, ...)
// 0x80-0xFF = Math Galaxy symbols (reserved)
__constant__ float g_sas_symbol_values[256];    // float value for each interned symbol_id
__constant__ uint32_t g_sas_symbol_star_ids[256]; // star_id index in Galaxy (for traceability)
}
```

### Kernel 1: `k3d_canonicalize`

Normalizes a STAR DAG in-place by applying four rewrite passes in post-order, then
deduplicates identical subtrees via hashcons.

**Four normalization passes (in order, single-thread per expression):**

1. **Constant folding** — if both children are TAG_FLOAT leaf nodes, replace binary
   node with `cas_make_const(result)`. Handles OP_ADD, OP_SUB, OP_MUL, OP_DIV,
   OP_POWER. Same logic already in `cas_simple_simplify` — call it first.

2. **Identity removal** — x+0→x, x*1→x, x^0→1, x^1→x, x-0→x, 0*x→0, 1*x→x.

3. **Associativity flattening** — if `root = ADD(a, ADD(b,c))`, rebuild as
   n-ary form by chaining children through the existing binary node structure
   (no actual n-ary node type needed; flatten depth first so leftmost child is
   always the accumulated partial sum).

4. **Canonical sort for commutative ops (ADD, MUL)** — reorder children so that
   the canonical order is: CONST (TAG_FLOAT) < SYMBOL (TAG_SYMBOL) < EXPR (everything else).
   Within the same tag class, sort by pool index (deterministic but cheap).
   Use `__ballot_sync` to detect if a swap is needed, then atomicCAS to do it.

After all four passes, walk the tree post-order and for each node call `hashcons_insert`.
If `hashcons_insert` returns a pool index different from the current node's own index,
the parent's child pointer should be updated to the returned canonical index.

**Signature:**
```c
extern "C" __global__ void k3d_canonicalize(
    uint32_t  root_idx,        // root of STAR DAG in g_cas_pool
    uint32_t* out_canon_idx    // canonical root after normalization + hashcons
);
```

Single-block, single-thread (thread 0 of block 0 does the traversal).
Stack depth bounded by `kCasTraversalCap = 128` (same cap as k3d_diff).

### Kernel 2: `k3d_pattern_match`

One-way unification (NOT Robinson's full unification — no occurs check, no variable
substitution in the pattern itself). Innermost strategy (bottomup) — most GPU-friendly
because all threads converge before any backtracking.

**Input:** a pattern STAR DAG (from a Grammar Galaxy cas_rule `meaning_rpn`) and a
subject STAR DAG (from the expression being simplified). Both already in g_cas_pool.

**Output:** a binding table — up to 16 variable bindings, each binding is a pair
`(var_symbol_id, subject_pool_idx)`. Returns 1 if match succeeded, 0 if failed.

**Algorithm (single-thread, bottomup traversal):**
```
binding_table[16] = {EMPTY}
match(pattern_root, subject_root):
    if pattern is SYMBOL and TAG_SYMBOL:
        // Pattern variable — bind it.
        if already bound to different subject → FAIL
        bind(pattern.symbol_id → subject)
        return SUCCESS
    if pattern.opcode != subject.opcode → FAIL
    if ARITY(pattern) != ARITY(subject) → FAIL
    if ARITY == 0:
        if TAG_FLOAT: compare floats with epsilon → SUCCESS/FAIL
        if TAG_SYMBOL: compare symbol_ids → SUCCESS/FAIL
    recurse on children (innermost = visit children first before root)
```

**Signature:**
```c
extern "C" __global__ void k3d_pattern_match(
    uint32_t  pattern_root_idx,
    uint32_t  subject_root_idx,
    uint32_t* out_binding_var_ids,    // [16] symbol_id per binding slot
    uint32_t* out_binding_subj_idxs, // [16] subject pool_idx per binding slot
    uint32_t* out_binding_count,
    uint32_t* out_matched            // 1 = success, 0 = fail
);
```

### Kernel 3: `k3d_rule_apply`

Given a binding table from `k3d_pattern_match` and a replacement STAR DAG template
(from a Grammar Galaxy cas_rule `behavior_rpn`), materialize the rewritten expression
in g_cas_pool.

**Algorithm (single-thread, post-order copy of replacement template):**
- Walk replacement template in post-order.
- For each node:
  - If SYMBOL (pattern variable): look up in binding table, emit the bound subject subtree.
  - Otherwise: copy node to a fresh pool slot, update child pointers.
- Return root of the materialized rewrite result.

After applying, call `k3d_canonicalize` on the result to normalize the new expression.

**Signature:**
```c
extern "C" __global__ void k3d_rule_apply(
    uint32_t  replacement_template_idx,
    const uint32_t* binding_var_ids,
    const uint32_t* binding_subj_idxs,
    uint32_t  binding_count,
    uint32_t* out_result_idx
);
```

---

## Boot-time Symbol Table

### Python: `sas_symbol_bootstrap.py`

Create `knowledge3d/cranium/sas_symbol_bootstrap.py`.

**Purpose:** At startup, scan the Reality Galaxy + Math Galaxy stars for entries with
`meaning_class` in `{"physical_constant", "mathematical_constant", "variable"}` and
build a compact `symbol_id → float_value` array for upload to `g_sas_symbol_values`
in `__constant__` memory.

Symbol IDs use the `SYM_*` constants already defined in `cas_star_node.h`. Extended IDs
`0x20-0x7F` are for physical constants (G, c, h, k_B, N_A, etc.) indexed by a small
compile-time enum defined in a new companion header section of `sas_kernels.cu`.

```python
"""Boot-time symbol table: scan Galaxy → build symbol_id → float array."""

from __future__ import annotations
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_SEMANTIC_RESOLVE,
)

# Canonical symbol registry (symbol_id → (name, default_value))
# IDs 0x01-0x04 are X/Y/Z/W geometric vars — not numeric constants.
# IDs 0x10-0x11 are mathematical constants baked into cas_star_node.h.
# IDs 0x20+ are physical constants loaded from Reality Galaxy at boot.
SYMBOL_REGISTRY: dict[int, tuple[str, float]] = {
    0x10: ("PI",  3.141592653589793),
    0x11: ("E",   2.718281828459045),
    0x20: ("G",   6.67430e-11),   # gravitational constant [m³ kg⁻¹ s⁻²]
    0x21: ("c",   299792458.0),   # speed of light [m/s]
    0x22: ("h",   6.62607015e-34),# Planck constant [J·s]
    0x23: ("hbar", 1.054571817e-34),
    0x24: ("k_B", 1.380649e-23),  # Boltzmann constant [J/K]
    0x25: ("N_A", 6.02214076e23), # Avogadro number [mol⁻¹]
    0x26: ("e",   1.602176634e-19),# elementary charge [C]
    0x27: ("eps0", 8.8541878128e-12),# vacuum permittivity [F/m]
    0x28: ("mu0", 1.25663706212e-6), # vacuum permeability [H/m]
    # Add more from CODATA/NIST as Reality Galaxy is populated.
}


def build_symbol_table(galaxy_manager=None) -> tuple[list[float], list[int]]:
    """
    Returns (values[256], star_id_indices[256]) for upload to __constant__ memory.
    If galaxy_manager is provided, live star values override SYMBOL_REGISTRY defaults.
    """
    values = [0.0] * 256
    star_ids = [0] * 256
    for sym_id, (name, default_val) in SYMBOL_REGISTRY.items():
        values[sym_id] = default_val
        if galaxy_manager is not None:
            # Look up the canonical star for this symbol name in Reality/Math Galaxy.
            star = galaxy_manager.find_star_by_meaning_class("physical_constant", name)
            if star is None:
                star = galaxy_manager.find_star_by_meaning_class("mathematical_constant", name)
            if star is not None:
                # Extract the numeric value from the star's reality_refs or meta_refs.
                numeric_val = _extract_numeric(star)
                if numeric_val is not None:
                    values[sym_id] = numeric_val
    return values, star_ids


def _extract_numeric(star) -> float | None:
    """Pull a float value out of a MeaningCentricStar's meta_refs or reality_refs."""
    for ref in (star.meta_refs or []):
        if isinstance(ref, (int, float)):
            return float(ref)
        if isinstance(ref, str):
            try:
                return float(ref)
            except ValueError:
                continue
    return None


__all__ = ["SYMBOL_REGISTRY", "build_symbol_table"]
```

---

## Bridge Additions: `sovereign_bridges.py`

Add to `ModularRPNEngine` in `knowledge3d/cranium/bridges/sovereign_bridges.py`.

Follow exactly the pattern of the existing CAS bridge methods (`bind_cas_pool`,
`launch_k3d_simplify`, etc.):

```python
def bind_sas_symbol_table(self, values: list[float], star_ids: list[int]) -> None:
    """Upload the boot-time symbol table to g_sas_symbol_values in __constant__ memory."""
    # Load sas_kernels.cu (lazy-compile alongside cas_kernels.cu).
    # Use cuMemcpyHtoD to upload values[256] → g_sas_symbol_values.
    # Use cuMemcpyHtoD to upload star_ids[256] → g_sas_symbol_star_ids.
    ...

def launch_k3d_canonicalize(self, root_idx: int) -> int:
    """Normalize STAR DAG rooted at root_idx. Returns canonical root pool index."""
    ...

def launch_k3d_pattern_match(
    self, pattern_root_idx: int, subject_root_idx: int
) -> tuple[list[int], list[int], bool]:
    """
    Returns (var_ids, subj_idxs, matched).
    var_ids and subj_idxs are parallel arrays of the binding table.
    matched is True if pattern matched, False otherwise.
    """
    ...

def launch_k3d_rule_apply(
    self,
    replacement_template_idx: int,
    var_ids: list[int],
    subj_idxs: list[int],
) -> int:
    """Materialize the rewritten expression. Returns root pool index of the result."""
    ...
```

---

## Modular Kernel: New SAS Cases in `modular_rpn_kernel.cu`

Add a `SAS` globals block near the CAS globals (same `extern "C"` pattern):

```c
// SAS device globals (defined in sas_kernels.cu)
extern __device__ HashconsSlot g_sas_hashcons[];
extern __constant__ float g_sas_symbol_values[];
```

Add switch cases in the main dispatch loop for the six new opcodes:

```c
// ── SAS opcodes (0x238-0x23D) ────────────────────────────────────────────
case 0x238: /* OP_CANONICALIZE */
    // Pop STAR handle from stack, call cas_simple_simplify inline (re-use existing helper),
    // then push canonical handle.  Full k3d_canonicalize is a dedicated kernel —
    // this modular case is the lightweight in-executor path.
    if (sp > 0) {
        uint32_t handle = (uint32_t)stack[sp - 1];
        stack[sp - 1] = (float)cas_simple_simplify(handle);
    }
    break;
case 0x239: /* OP_CAS_HASH */
    // Pop STAR handle, compute a deterministic uint32 hash from (opcode, child0, child1, flags),
    // push hash as float bits.  This is the lightweight inline version.
    if (sp > 0) {
        uint32_t idx = (uint32_t)stack[sp - 1];
        if (idx < CAS_POOL_SIZE) {
            const StarNode& n = g_cas_pool[idx];
            uint32_t h = hashcons_slot(n.opcode, STAR_CHILD0(n), STAR_CHILD1(n), n.flags);
            stack[sp - 1] = __uint_as_float(h);
        }
    }
    break;
case 0x23A: /* OP_SEMANTIC_RESOLVE */
    // Pop symbol_id (as float int-cast), look up g_sas_symbol_values, push float value.
    if (sp > 0) {
        uint32_t sym_id = (uint32_t)stack[sp - 1];
        float val = (sym_id < 256u) ? g_sas_symbol_values[sym_id] : 0.0f;
        stack[sp - 1] = val;
    }
    break;
case 0x23B: /* OP_RULE_SELECT */
    // Lightweight stub — push 0 (no matching rule found).
    // Full Grammar Galaxy rule selection is a dedicated kernel (k3d_pattern_match).
    if (sp < STACK_DEPTH) stack[sp++] = 0.0f;
    break;
case 0x23C: /* OP_CONTEXTUAL_REWRITE */
    // Lightweight stub — pop rule_handle and subject_handle, push subject_handle unchanged.
    // Full rewrite is a dedicated kernel (k3d_rule_apply).
    if (sp >= 2) { sp--; }
    break;
case 0x23D: /* OP_SEMANTIC_EQUIV */
    // Pop two STAR handles, push 1.0 if canonically equal (same pool index after
    // canonicalization), else 0.0.
    if (sp >= 2) {
        uint32_t b = (uint32_t)stack[--sp];
        uint32_t a = (uint32_t)stack[sp - 1];
        // Lightweight check: same pool index = definitely equal.
        // Different index = unknown (full check needs k3d_canonicalize first).
        stack[sp - 1] = (a == b) ? 1.0f : 0.0f;
    }
    break;
```

**Important:** Keep the modular kernel cases truthful — they are the lightweight
inline path. The full semantics (Hashcons dedup, Grammar rule lookup, actual
pattern matching) live in the dedicated SAS kernels. Add a comment above each stub
pointing to the corresponding dedicated kernel.

---

## SAS Grammar Bootstrap: `sas_grammar_bootstrap.py`

Create `knowledge3d/cranium/sas_grammar_bootstrap.py`.

Same structure as `cas_grammar_bootstrap.py`. Add canonicalization meta-rules and
identity equivalence rules as MeaningCentricStars with `meaning_class="sas_rule"`,
`domain="grammar"`, `galaxy_ref="Grammar"`.

```python
"""Bootstrap Grammar Galaxy with SAS canonicalization and equivalence rules."""

from __future__ import annotations
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar

CANONICALIZATION_RULES = [
    MeaningCentricStar(
        star_id="sas_rule:commute_add",
        meaning_class="sas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_VAR_Y OP_ADD",
        behavior_rpn="OP_VAR_Y OP_VAR_X OP_ADD",
        taxonomy_refs=["algebra", "commutativity", "addition"],
        grammar_refs=["canonical_form", "commutative_sort"],
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="sas_rule:commute_mul",
        meaning_class="sas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_VAR_Y OP_MUL",
        behavior_rpn="OP_VAR_Y OP_VAR_X OP_MUL",
        taxonomy_refs=["algebra", "commutativity", "multiplication"],
        grammar_refs=["canonical_form", "commutative_sort"],
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="sas_rule:identity_add_zero",
        meaning_class="sas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_CONST 0.0 OP_ADD",
        behavior_rpn="OP_VAR_X",
        taxonomy_refs=["algebra", "identity", "addition"],
        grammar_refs=["canonical_form", "identity_removal"],
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="sas_rule:identity_mul_one",
        meaning_class="sas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_CONST 1.0 OP_MUL",
        behavior_rpn="OP_VAR_X",
        taxonomy_refs=["algebra", "identity", "multiplication"],
        grammar_refs=["canonical_form", "identity_removal"],
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="sas_rule:annihilate_mul_zero",
        meaning_class="sas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_CONST 0.0 OP_MUL",
        behavior_rpn="OP_CONST 0.0",
        taxonomy_refs=["algebra", "annihilator", "multiplication"],
        grammar_refs=["canonical_form", "identity_removal"],
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="sas_rule:power_zero",
        meaning_class="sas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_CONST 0.0 OP_POWER",
        behavior_rpn="OP_CONST 1.0",
        taxonomy_refs=["algebra", "exponentiation", "identity"],
        grammar_refs=["canonical_form", "identity_removal"],
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="sas_rule:power_one",
        meaning_class="sas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_CONST 1.0 OP_POWER",
        behavior_rpn="OP_VAR_X",
        taxonomy_refs=["algebra", "exponentiation", "identity"],
        grammar_refs=["canonical_form", "identity_removal"],
        confidence=1,
        polarity=1,
    ),
]

DEFEASIBLE_METADATA = {
    # All SAS rules are strict (+1) — they are mathematical identities, not
    # defeasible heuristics.  PMR conflict resolution uses rule_strength to
    # determine priority when multiple rules match the same redex.
    "rule_strength": 1,
    "trust_weight": 1.0,
    "superior_to": [],
}


def build_sas_rule_stars() -> list[MeaningCentricStar]:
    """Return the foundational SAS canonicalization and identity rules."""
    return [MeaningCentricStar.from_dict(r.to_dict()) for r in CANONICALIZATION_RULES]


__all__ = ["CANONICALIZATION_RULES", "DEFEASIBLE_METADATA", "build_sas_rule_stars"]
```

---

## Ingestion Wiring

### `knowledge3d/ingestion/__init__.py`

Add:
```python
from knowledge3d.cranium.sas_symbol_bootstrap import build_symbol_table
from knowledge3d.cranium.sas_grammar_bootstrap import build_sas_rule_stars

def ingest_sas_bootstrap(galaxy_manager=None):
    """Bootstrap the SAS layer: symbol table + canonicalization grammar rules."""
    values, star_ids = build_symbol_table(galaxy_manager)
    # Store symbol table for bridge upload at GPU bind time.
    # Store SAS grammar rules as MeaningCentricStars in Grammar Galaxy.
    stars = build_sas_rule_stars()
    ...
    return values, star_ids, stars
```

### `scripts/fundamental_ingest_payloads.py` and `knowledge3d/tools/ingest_from_manifest.py`

Wire `ingest_sas_bootstrap()` in the same place `ingest_cas_grammar()` is called.
Add after the CAS grammar ingest call, keep the pattern identical.

---

## Tests

Create `tests/test_sovereign_sas_surface.py`.

Minimum test surface (CPU-side, no GPU required for most):

```python
def test_sas_opcodes_in_correct_range():
    # All 0x238-0x23D must be within 0x238-0x25F (SAS block)
    from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op
    sas_ops = [op.OP_CANONICALIZE, op.OP_CAS_HASH, op.OP_SEMANTIC_RESOLVE,
               op.OP_RULE_SELECT, op.OP_CONTEXTUAL_REWRITE, op.OP_SEMANTIC_EQUIV]
    for v in sas_ops:
        assert 0x238 <= v <= 0x25F

def test_sas_no_collision_with_cas():
    from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op
    cas_range = set(range(0x220, 0x238))
    sas_range = set(range(0x238, 0x240))
    assert cas_range.isdisjoint(sas_range)

def test_symbol_table_has_physical_constants():
    from knowledge3d.cranium.sas_symbol_bootstrap import build_symbol_table, SYMBOL_REGISTRY
    values, _ = build_symbol_table(galaxy_manager=None)
    # G must be present
    assert values[0x20] == SYMBOL_REGISTRY[0x20][1]
    # c must be present
    assert values[0x21] == SYMBOL_REGISTRY[0x21][1]

def test_sas_grammar_stars_are_valid_meaning_stars():
    from knowledge3d.cranium.sas_grammar_bootstrap import build_sas_rule_stars
    stars = build_sas_rule_stars()
    assert len(stars) >= 7
    for star in stars:
        assert star.meaning_class == "sas_rule"
        assert star.domain == "grammar"
        assert star.galaxy_ref == "Grammar"
        assert star.meaning_rpn
        assert star.behavior_rpn

def test_sas_rule_stars_star_ids_unique():
    from knowledge3d.cranium.sas_grammar_bootstrap import build_sas_rule_stars
    stars = build_sas_rule_stars()
    ids = [s.star_id for s in stars]
    assert len(ids) == len(set(ids))

def test_sas_mnemonics_compile_in_rpn_engine():
    from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNProgram
    prog = ModularRPNProgram()
    prog.u16(0x238)  # OP_CANONICALIZE
    prog.u16(0x23A)  # OP_SEMANTIC_RESOLVE
    prog.u16(0x23D)  # OP_SEMANTIC_EQUIV
    assert len(prog.bytecode()) > 0

def test_sovereign_sas_no_hot_path_imports():
    # None of the SAS kernel/bridge/opcode files must import symengine
    import subprocess, sys
    files = [
        "knowledge3d/cranium/sas_symbol_bootstrap.py",
        "knowledge3d/cranium/sas_grammar_bootstrap.py",
        "knowledge3d/cranium/ptx_runtime/rpn_opcodes.py",
    ]
    for f in files:
        result = subprocess.run(
            [sys.executable, "-c", f"import ast, sys; ast.parse(open('{f}').read())"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
    # Grep check: none of the above import symengine
    import re
    for f in files:
        src = open(f).read()
        assert "symengine" not in src, f"{f} must not import symengine in hot path"
```

Also **update** `tests/test_opcode_namespace_integrity.py` to add:

```python
def test_sas_ops_in_dedicated_range():
    from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op
    sas_vals = [op.OP_CANONICALIZE, op.OP_CAS_HASH, op.OP_SEMANTIC_RESOLVE,
                op.OP_RULE_SELECT, op.OP_CONTEXTUAL_REWRITE, op.OP_SEMANTIC_EQUIV]
    for v in sas_vals:
        assert 0x238 <= v <= 0x25F, f"SAS opcode 0x{v:X} out of range"

def test_no_cross_domain_conflicts_sas():
    from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op
    sas_set = {op.OP_CANONICALIZE, op.OP_CAS_HASH, op.OP_SEMANTIC_RESOLVE,
               op.OP_RULE_SELECT, op.OP_CONTEXTUAL_REWRITE, op.OP_SEMANTIC_EQUIV}
    drawing_set = {v for k, v in vars(op).items() if k.startswith("OP_DRAW_")}
    ternary_set = {op.OP_TADD, op.OP_TMUL, op.OP_TNOT, op.OP_TCOMP,
                   op.OP_TQUANT, op.OP_TPACK, op.OP_TUNPACK}
    trm_set = {op.OP_TRM_MATVEC_512x1024, op.OP_TRM_MATVEC_1024x512,
               op.OP_TRM_VEC_ADD3_512, op.OP_TRM_SWIGLU_512, op.OP_TRM_SWIGLU_1024}
    assert sas_set.isdisjoint(drawing_set)
    assert sas_set.isdisjoint(ternary_set)
    assert sas_set.isdisjoint(trm_set)
```

---

## PTX Rebuild

After all files pass `py_compile`:

```bash
nvcc --ptx -arch=sm_86 -O3 --use_fast_math \
  -I knowledge3d/cranium/kernels \
  knowledge3d/cranium/kernels/modular_rpn_kernel.cu \
  -o knowledge3d/cranium/ptx/modular_rpn_kernel.ptx
```

The dedicated SAS kernels compile separately through the sovereign bridge lazy-path
(same pattern as `cas_kernels.cu`):

```bash
nvcc -arch=sm_86 -O3 --use_fast_math \
  -I knowledge3d/cranium/kernels \
  --ptx knowledge3d/cranium/kernels/sas_kernels.cu \
  -o knowledge3d/cranium/ptx/sas_kernels.ptx
```

---

## Implementation Order

| Step | Task |
|------|------|
| SAS-A | Add six SAS opcodes to `rpn_opcodes.py` + `__all__` |
| SAS-B | Create `sas_hashcons.h` |
| SAS-C | Create `sas_grammar_bootstrap.py` with 7 canonical-form rules |
| SAS-D | Create `sas_symbol_bootstrap.py` with 9 physical constants |
| SAS-E | Add SAS globals + six modular kernel switch cases to `modular_rpn_kernel.cu` |
| SAS-F | Create `sas_kernels.cu` with `k3d_canonicalize`, `k3d_pattern_match`, `k3d_rule_apply` |
| SAS-G | Add bridge methods to `sovereign_bridges.py` |
| SAS-H | Wire `ingest_sas_bootstrap()` into ingestion surfaces |
| SAS-I | Create `tests/test_sovereign_sas_surface.py`; update `test_opcode_namespace_integrity.py` |
| SAS-J | Rebuild modular PTX; compile `sas_kernels.cu` |

---

## Sovereign Compliance Checklist

- [ ] `symengine` appears ONLY in `cas_ingestion.py` (already enforced by CAS step)
- [ ] `sas_symbol_bootstrap.py` imports ZERO non-stdlib, non-K3D packages
- [ ] `sas_grammar_bootstrap.py` imports ZERO non-stdlib, non-K3D packages
- [ ] `sas_kernels.cu` uses ONLY K3D pool allocators — no `malloc`, no `new`
- [ ] `g_sas_symbol_values` is `__constant__` memory — zero runtime allocation
- [ ] `g_sas_hashcons` is `__device__` memory — allocated once at GPU init
- [ ] All SAS opcodes in `0x238-0x25F` — no overlap with CAS `0x220-0x237`,
      drawing `0x200-0x21F`, TRM `0x300-0x304`, ternary `0x70-0x76`
- [ ] `test_opcode_namespace_integrity.py` updated to cover SAS range

---

## What NOT To Do In This Pass

- Do NOT implement NTT-accelerated poly_mul (that is CAS Step 2)
- Do NOT implement `OP_SYMBOLIC_DIFF` bridge continuation (CAS Step 2)
- Do NOT touch `TRMLauncher` (Phase D)
- Do NOT implement full Grammar Galaxy ANN rule lookup in the modular kernel
  (the `OP_RULE_SELECT` modular case is a bounded stub — the full kernel is
  `k3d_pattern_match` launched from the bridge)
- Do NOT remove the protected encyclopedia ingest (PID 101379)

---

## Validation Gate

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/pytest -q \
  tests/test_sovereign_sas_surface.py \
  tests/test_opcode_namespace_integrity.py \
  tests/test_sovereign_cas_surface.py \
  tests/test_sovereign_entity_surface.py \
  tests/test_sovereign_physics_surface.py \
  -x
```

Expected: all passing. No regressions on CAS, entity, or physics surfaces.

Report back with `CODEX_TO_CLAUDE_SAS_REPORT_2026-04-08.md`.
