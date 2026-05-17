# Supersession Patches — 2026-04-18 v4 (Turn-6 Rulings Consolidation)

**Date**: 2026-04-18
**Author**: Claude (architecture, connective-tissue lane)
**Supersedes**: `supersession_patches_04.18.2026_v3.md` (additive; v3 + v2 + v1 patches stand)
**Rulings covered**: Turn-6 Rulings 1–4
**Sibling patches**: Lane A (kernel-implementation lane) handles Path A / Path B kernel variants and the in-kernel silent-rescale logic. v4 does NOT patch kernel code — it locks the spec semantics.

**Codex instructions**: Apply v4 patches atomically per section. Where v4 conflicts with v3, v4 wins. Where v4 is silent, v3 stands.

---

## Turn-6 Ruling Summary

| Ruling | Topic | Status | Owner |
|---|---|---|---|
| 1 | Path B (MARGIN_SCALED) smem-prefetch = MANDATORY on every Path B kernel | PATCHED HERE | connective tissue (spec) + Lane A (kernel) |
| 2 | d-mismatch handling = silent in-kernel rescale (no warning, no log, no exit) | PATCHED HERE | connective tissue (spec) + Lane A (kernel) |
| 3 | `CONTRASTIVE_RANK_TOPK` (0x1A9) default margin path = Path A (SHIFT); lane-switch to Path B mid-RPN-execution via opcode argument flag | PATCHED HERE | connective tissue |
| 4 | Sub-reservation for "Attention Future Expansion — halting gate variants / sparse-K attention" | PATCHED HERE (with block-owner collision note — see §4 below) | connective tissue |

---

## 1. Ruling 1 — Path B (MARGIN_SCALED) Mandatory smem Prefetch

Ruling 1 (verbatim intent): **Every Path B kernel MUST prefetch `confidence_margin` into shared memory at tile start. No non-prefetch variant is admissible.**

### 1.1 Rationale

Path B's cost model (per `attention_score_normalization_dual_path_spec_04.18.2026.md` §4) shows that global-memory loads of star metadata dominate the comparison cost (~100-150 cycles uncached vs. 2-3 cycles with smem prefetch). Allowing a non-prefetch variant would produce a silent performance-regression footgun: a lane could invoke Path B without prefetch and get 30-50× worse throughput than Path A, silently, with no warning. Ruling 1 closes that footgun by making prefetch structural, not optional.

### 1.2 Files to Patch

#### `TEMP/attention_score_normalization_dual_path_spec_04.18.2026.md §4 (Path B)`

**Supersede** — replace the subsection header and body of the "Optimization: Pre-fetch the star metadata into shared memory" block (starting around spec line 266) with:

```markdown
### Mandatory Shared-Memory Prefetch (Ruling 1 v4)

Path B has NO non-prefetch variant. Every Path B kernel MUST prefetch the
`confidence_margin` slab for the active tile into shared memory before the
top-K comparison loop opens. A kernel that reads `confidence_margin` via
global-memory load inside the per-candidate loop is NOT a valid Path B
implementation.

Structure:

```cuda
// Shared memory: 46 stars × 32 bytes metadata (margin + trit + tier + reserved)
__shared__ StarMetadata smem_stars[MAX_STARS_PER_BLOCK];

// Cooperative prefetch — happens ONCE per tile, before the loop
for (int idx = threadIdx.x; idx < num_stars; idx += blockDim.x) {
    smem_stars[idx] = global_stars[idx];
}
__syncthreads();

// Top-K loop — all reads from smem (1-cycle broadcast)
for (...) {
    int8_t margin = smem_stars[star_idx].confidence_margin;
    // ... compare ...
}
```

Acceptance gate (Gate R-prefetch): `grep -A40 "0x1AF\|ATTENTION_MARGIN_SCALED" <kernel.cu>`
must contain `__shared__` AND `__syncthreads()` AND a load-from-smem before
the scoring loop. Hard fail otherwise.
```

**Also replace** the "Cons" bullet in §4 ("Cache misses: Metadata load can be slow if not prefetched.") with:

```markdown
- **Not applicable (Ruling 1 v4)**: prefetch is mandatory. The cache-miss
  scenario cannot occur in a valid Path B implementation.
```

#### `TEMP/attention_opcode_expansion_v2.md §6 (ATTENTION_MARGIN_SCALED, 0x1AF)`

**Supersede** — replace the "Priority" line and add a new "Mandatory Prefetch Rule" subsection at the top of §6:

```markdown
**Priority**: P0 (Ruling 1 v4 — prefetch is structural, not optional)

### Mandatory Prefetch Rule (Ruling 1 v4)

0x1AF is defined to REQUIRE shared-memory prefetch of `confidence_margin`
at the start of every enclosing kernel tile. The opcode's contract is
"compare against a smem-resident margin". Any kernel that invokes 0x1AF
without a preceding cooperative prefetch + `__syncthreads` is out of spec.

Lane A (kernel-implementation lane) MUST emit the prefetch in the kernel
body surrounding every 0x1AF invocation. There is no opt-out flag.
```

**Also patch** the "Cycle Cost on sm_86" block in §6: remove the "without prefetch" sub-branch entirely (retain only the smem-prefetch cycle figures).

#### `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §7.x (0x1AF entry)`

**Append to the existing `0x1AF` entry** (or add as a normative note beneath it):

```markdown
**Ruling 1 v4 (2026-04-18) — Mandatory Prefetch**: Every kernel that invokes
0x1AF MUST cooperatively prefetch `confidence_margin` into shared memory
before the scoring loop and issue `__syncthreads()`. This is part of the
opcode's contract, not an optimization. See Gate R-prefetch in
`TEMP/attention_score_normalization_dual_path_spec_04.18.2026.md §4`.
```

---

## 2. Ruling 2 — d-Mismatch = Silent In-Kernel Rescale

Ruling 2 (verbatim intent): **When a Path B consumer reads a `confidence_margin` that was pre-computed for a different `d_tier` than the active query's `d`, apply a silent scale factor in-kernel. No warning. No log. No exit.**

### 2.1 Rationale

Matryoshka tier switching is a frequent, expected operation. A loud warning on every mismatch would flood logs during normal tier cycling. The scale factor is a single `int` multiply (1 cycle), cheaper than the log-emission path. Ruling 2 explicitly bans warning/logging on d-mismatch — this is a silent correction, not an error condition.

### 2.2 Scale Formula

```
d_active  = query's active dimension (32, 64, 128, 512)
d_stored  = star metadata field d_tier (what the margin was pre-computed for)

if (d_active == d_stored):
    margin_effective = confidence_margin   // no rescale
else:
    scale_num = d_active
    scale_den = d_stored
    margin_effective = (int32_t)confidence_margin * scale_num / scale_den
```

For all supported tier pairs (32 ↔ 64 ↔ 128 ↔ 512), the numerator and denominator are powers of 2, so the division compiles to a right-shift. Total cost: 1 IMUL + 1 SHR = 2 cycles. Well below the smem-load dominated comparison cost.

Range safety: `confidence_margin` is `int8` (|margin| ≤ 127). With scale ratios bounded by 512/32 = 16×, the intermediate INT32 product stays within `int32` range without saturation. After the shift, the result is truncated back to an effective `int32` margin; downstream `ATTENTION_MARGIN_SCALED` comparison consumes it as INT32.

### 2.3 Files to Patch

#### `TEMP/attention_score_normalization_dual_path_spec_04.18.2026.md §4 (Tier-Coupling Cons bullet)`

**Supersede** the "Tier-coupling" Cons bullet:

```markdown
- **Tier-coupling handled silently (Ruling 2 v4)**: If a star's
  `confidence_margin` was pre-computed for a different `d_tier` than the
  active query's `d`, the kernel applies a silent in-kernel scale factor
  `(d_active / d_stored)` as a 1-cycle IMUL + 1-cycle SHR (tier ratios are
  powers of 2). No warning, no log, no exit. d-mismatch is NOT an error
  condition — it is an expected Matryoshka tier-switch operation.
```

**Also add** a new subsection §4.5 immediately after the "Cons" bullet list:

```markdown
### 4.5 Silent d-Mismatch Rescale (Ruling 2 v4)

When `d_active != d_stored`, the kernel MUST apply:

```cuda
int32_t margin_effective = (int32_t)confidence_margin
                          * d_active / d_stored;
```

Implementation: since all supported tiers are powers of 2, the division is
a right-shift. The kernel emits the rescale inline without branching on
the mismatch (predicated multiply by `d_active/d_stored` ratio; the ratio
is 1 when tiers match). No log statement, no printf, no early exit.

Acceptance gate (Gate R-dmatch): `grep -n "printf\|fprintf\|stderr" <path_b_kernel.cu>`
must return zero matches inside the d-mismatch code region. Hard fail otherwise.
```

#### `TEMP/attention_opcode_expansion_v2.md §6 (ATTENTION_MARGIN_SCALED, 0x1AF)`

**Append** a new "d-Mismatch Handling" subsection at the end of §6 (before the "When to Use" block):

```markdown
### d-Mismatch Handling (Ruling 2 v4)

If `d_active != star.d_tier`, 0x1AF applies a silent in-kernel rescale:
`margin_effective = confidence_margin × d_active / d_tier`. The rescale is
a 2-cycle compile-time optimization (IMUL + SHR; ratios are powers of 2).
No warning, no log, no exit path. d-mismatch is an expected Matryoshka
tier-switch event, not an error.
```

#### `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §7.x (0x1A9 and 0x1AF entries)`

**Append to the `0x1A9 CONTRASTIVE_RANK_TOPK` entry**:

```markdown
**Ruling 2 v4 (2026-04-18) — Silent d-Mismatch Rescale**: When 0x1A9 is
invoked in Path B mode (via flag per Ruling 3) and `d_active != star.d_tier`,
the kernel applies `margin × d_active / d_tier` inline. No warning, no log.
```

**Append to the `0x1AF` entry**:

```markdown
**Ruling 2 v4 (2026-04-18) — Silent d-Mismatch Rescale**: 0x1AF handles
d-mismatch via silent in-kernel rescale (1 IMUL + 1 SHR). No log output.
```

---

## 3. Ruling 3 — 0x1A9 Default Margin Path = Path A (SHIFT), Lane-Switchable via Opcode Flag

Ruling 3 (verbatim intent): **`CONTRASTIVE_RANK_TOPK` (0x1A9) defaults to Path A (SHIFT). Support lane-switch to Path B mid-RPN-execution via an opcode argument flag.**

### 3.1 Rationale

Path A has zero preconditions and 1-cycle cost. Path B has a prefetch requirement, metadata-layout requirement, and 2-cycle cost per comparison. Defaulting to Path A makes 0x1A9 usable in simple RPN programs without any Galaxy preprocessing. The lane-switch flag lets a single RPN program start in Path A mode for coarse filtering and upgrade to Path B for fine-grained ranking within the same execution, without needing two different opcodes in the program text.

### 3.2 Operand Encoding

0x1A9 gains a new 1-bit operand: `margin_path`.

```
[opcode:       16 bits = 0x1A9]
[operand_0:     4 bits = bank_scores]
[operand_1:     4 bits = bank_stars]
[operand_2:     4 bits = bank_topk_out]
[operand_3:     8 bits = K_topk]
[operand_4:     1 bit  = margin_path]   ; 0 = Path A (SHIFT, default), 1 = Path B (SCALED)
[operand_5:     3 bits = reserved]      ; must be zero
```

**Default semantics**: if `margin_path` is omitted in an RPN program, the assembler emits `margin_path = 0` (Path A).

**Lane-switch semantics**: a single RPN program MAY invoke 0x1A9 twice with different `margin_path` values — the kernel's dispatch is purely argument-driven; there is no global state carrying a "current path" setting.

### 3.3 Files to Patch

#### `TEMP/attention_opcode_expansion_v2.md` (0x1A9 section in §0 "Existing Opcodes — Preserved Unchanged")

**Supersede** — replace the `CONTRASTIVE_RANK_TOPK — 0x1A9` subsection with:

```markdown
### `CONTRASTIVE_RANK_TOPK` — 0x1A9
**Status**: EXTENDED (v4 Ruling 3) with `margin_path` flag. v1 semantics preserved for `margin_path = 0`.

Warp-cooperative bitonic top-K + margin gate. v4 adds a 1-bit `margin_path`
operand that selects the margin comparison pathway:

- `margin_path = 0` (default, Ruling 3 v4): **Path A** — uses 0x1AE
  (ATTENTION_MARGIN_SHIFT) semantics inline. 1-cycle SHR. No Galaxy
  metadata load required. Coarse-grained filtering.

- `margin_path = 1` (opt-in): **Path B** — uses 0x1AF
  (ATTENTION_MARGIN_SCALED) semantics inline. Requires mandatory smem
  prefetch per Ruling 1 v4. Silent d-mismatch rescale per Ruling 2 v4.
  Fine-grained ranking.

Lane-switch: a single RPN program may call 0x1A9 twice with different
`margin_path` values. The kernel dispatches per-invocation; no global
"current path" state.

See v1 `attention_opcode_expansion.md` §3 for the bitonic top-K body.
See Rulings 1, 2, 3 above for the Path B + d-mismatch + default-path
additions.
```

#### `TEMP/attention_score_normalization_dual_path_spec_04.18.2026.md §5 (Selection Criteria)`

**Prepend** a new subsection §5.0 at the top of §5 (before the Decision Tree):

```markdown
### 5.0 Default Path (Ruling 3 v4)

`CONTRASTIVE_RANK_TOPK` (0x1A9) defaults to **Path A (SHIFT)** when the
`margin_path` operand is omitted or zero. Path B is opt-in via
`margin_path = 1` and requires smem-prefetched `confidence_margin` per
Ruling 1 v4.

The Decision Tree below guides the selection within a single RPN program:
Path A for the default filter stage, Path B for fine-grained ranking
stages where confidence-aware margins matter. Lane-switching within a
single program is supported via separate 0x1A9 invocations with
different `margin_path` values.
```

#### `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §7.x (0x1A9 entry)`

**Append to the `0x1A9` entry**:

```markdown
**Ruling 3 v4 (2026-04-18) — Default Path A**: 0x1A9 accepts a new 1-bit
`margin_path` operand. Default `0` = Path A (SHIFT, 1-cycle SHR, no metadata
load). Opt-in `1` = Path B (SCALED, smem-prefetch-mandatory per Ruling 1 v4,
silent d-mismatch rescale per Ruling 2 v4). Lane-switchable mid-program.
```

#### `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` (constants module)

**Note for Codex (no Claude edit)**: add two named flag constants next to the 0x1A9 opcode:

```
CONTRASTIVE_RANK_TOPK_PATH_A = 0  # default, Ruling 3 v4
CONTRASTIVE_RANK_TOPK_PATH_B = 1  # opt-in, requires smem prefetch
```

---

## 4. Ruling 4 — Sub-Reservation for "Attention Future Expansion"

Ruling 4 (verbatim intent): **Seal an explicit block owner for "Attention Future Expansion — halting gate variants / sparse-K attention".**

### 4.1 Block Collision with Daniel's Proposed Range

Daniel proposed sub-reserving `0x1B6-0x1B9` for this block owner. This range is **already minted** by `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` to:

- `0x1B6` — `TENSOR_INTERPOLATE`
- `0x1B7` — `KMEANS_PLUS_INIT`
- `0x1B8` — `CTYPES_VIEW_AS_PTX`
- `0x1B9` — `CUDA_MALLOC_ASYNC`

Per **expand-not-replace doctrine**, these minted opcodes cannot be renumbered or re-assigned. Claude therefore **cannot** accept 0x1B6-0x1B9 as the attention-future-expansion block.

**Alternative sub-reservation proposed (v4)**: `0x1B1-0x1B5`. This range is currently listed in the v3 Reservation Table as "future normalization/attention family (headroom)" with no minted opcodes inside it. Converting it from generic "normalization/attention headroom" to a named sub-reservation for "Attention Future Expansion — halting gate variants / sparse-K attention" is an append-not-replace operation (tightens the owner; does not renumber anything).

**Pending Daniel confirmation**: Claude proceeds with `0x1B1-0x1B5` as the sub-reservation. If Daniel intended a different range, v5 will rename the block with zero opcode renumbering cost (since no opcode in this range is yet minted).

### 4.2 Files to Patch

#### `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §11 (Reservation Table)`

**Append** a new row after the existing `0x1B0` row and before the `0x1B1-0x1BF` row. **Also supersede** the `0x1B1-0x1BF` row to tighten its scope (expand-not-replace is preserved because no opcode in that range is minted).

Replace the existing row:
```
| `0x1B1` | `0x1BF` | future normalization/attention family (headroom) | 2026-04-18 | active |
```

With two rows:
```
| `0x1B1` | `0x1B5` | **Attention Future Expansion** — halting gate variants, sparse-K attention, per `TEMP/supersession_patches_04.18.2026_v4.md §4` (Ruling 4) | 2026-04-18 | active |
| `0x1BA` | `0x1BF` | future normalization/attention family (headroom — narrowed from 0x1B1-0x1BF by v4 sub-reservation) | 2026-04-18 | active |
```

**Rationale**: the row narrowing is expand-not-replace because the pre-v4 row had no minted opcodes inside it; tightening the owner is additive, not a renumber.

#### `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md`

**Append** a new "Case Study" section at the end:

```markdown
## Case Study — Turn-6 Ruling 4 Block Collision (2026-04-18)

When a proposed sub-reservation range collides with already-minted opcodes
(as happened with the turn-6 proposal for 0x1B6-0x1B9 where bulk-lib-purge
had minted TENSOR_INTERPOLATE, KMEANS_PLUS_INIT, CTYPES_VIEW_AS_PTX,
CUDA_MALLOC_ASYNC), the correct resolution is:

1. Do NOT renumber the minted opcodes. Expand-not-replace forbids this.
2. Propose the nearest unminted range with matching intent.
3. Document the collision and the resolution in the supersession patch.
4. Leave the decision to Daniel for explicit confirmation.

For turn-6 Ruling 4, the resolution was to use 0x1B1-0x1B5 (unminted,
previously "future normalization/attention headroom") as the sub-reservation
block. See `TEMP/supersession_patches_04.18.2026_v4.md §4` for the full
collision analysis.
```

---

## 5. OPCODE SLOT SUMMARY — Post v1+v2+v3+v4 State

(Carries forward v3's summary; v4 changes marked with **v4** annotation.)

| Range | Status |
|-------|--------|
| 0x100–0x108 | 0x100–0x107 existing ternary; 0x108 = TERNARY_XNOR_POPCOUNT |
| 0x170–0x177 | Yard ops |
| 0x178–0x17A | Queue ops |
| 0x17B–0x17F | Reserved (yard family future) |
| 0x180–0x18F | WINE I/O contract block |
| 0x190 | PHYSICS_EMIT_VISUAL |
| 0x191–0x19F | Reserved (physics visual future) |
| 0x1A0–0x1A6 | Bulk-lib purge math/utility |
| 0x1A7 | ATTENTION_FWD (_BASE) |
| 0x1A8 | ATTENTION_FWD_TERNARY |
| 0x1A9 | CONTRASTIVE_RANK_TOPK (**v4**: gains `margin_path` 1-bit operand, default=0=Path A) |
| 0x1AA | TERNARY_MATMUL_ADDSUB |
| 0x1AB | TERNARY_PACK5 |
| 0x1AC | TERNARY_UNPACK5 |
| 0x1AD | VEC_NORM_L2_INT8 (v2 spec — note: also listed at 0x1B0 in v3 registry §11; see §7 "Known Divergence" below) |
| 0x1AE | ATTENTION_MARGIN_SHIFT (Path A) |
| 0x1AF | ATTENTION_MARGIN_SCALED (Path B — **v4**: smem prefetch MANDATORY) |
| 0x1B0 | VEC_NORM_L2_INT8 (per v3 Reservation Table; scale=64) |
| 0x1B1–0x1B5 | **v4**: Attention Future Expansion (halting gate variants, sparse-K attention) — Ruling 4 |
| 0x1B6 | TENSOR_INTERPOLATE (bulk-lib purge; held against renumber per expand-not-replace) |
| 0x1B7 | KMEANS_PLUS_INIT |
| 0x1B8 | CTYPES_VIEW_AS_PTX |
| 0x1B9 | CUDA_MALLOC_ASYNC |
| 0x1BA–0x1BF | Reserved (future normalization/attention headroom — narrowed by v4) |
| 0x1C0–0x1C5 | IMAGE/SPARSE (relocated per v2) |
| 0x1C6–0x1CF | Reserved (physics expansion headroom) |

---

## 6. File Change Manifest — v4

| File | Change | Section(s) |
|---|---|---|
| `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` | APPEND Ruling 1 note to 0x1AF entry; APPEND Ruling 2 notes to 0x1A9 and 0x1AF entries; APPEND Ruling 3 note to 0x1A9 entry; SUPERSEDE §11 row `0x1B1-0x1BF` with two rows (`0x1B1-0x1B5` + `0x1BA-0x1BF`) | §7.x (0x1A9, 0x1AF); §11 |
| `TEMP/attention_score_normalization_dual_path_spec_04.18.2026.md` | SUPERSEDE §4 prefetch subsection (optional → mandatory); SUPERSEDE §4 tier-coupling Cons bullet; APPEND §4.5 silent rescale; PREPEND §5.0 default path | §4, §4.5, §5.0 |
| `TEMP/attention_opcode_expansion_v2.md` | SUPERSEDE 0x1A9 subsection in §0 (add `margin_path` flag); SUPERSEDE §6 priority + mandatory prefetch rule; SUPERSEDE §6 cycle-cost block (remove non-prefetch branch); APPEND §6 d-mismatch subsection | §0 (0x1A9); §6 |
| `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md` | APPEND case-study for block collision | tail |
| `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_attention_margin_dual_path_rulings.md` | NEW MEMORY (all three turn-6 rulings on attention margin) | all |
| `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/MEMORY.md` | UPDATE — add pointer to new memory under attention section | attention block |
| `TEMP/consistency_sweep_v4_04.18.2026.md` | NEW (this v4 sweep report) | all |

**Not touched by v4**:
- `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` — 0x1B6-0x1B9 opcodes stand (block-collision note in §4 above).
- `knowledge3d/cranium/**` — kernel code is Lane A's responsibility.
- Tier 2/3 yard kernel specs — unchanged.

---

## 7. Known Divergence — VEC_NORM Opcode Number (0x1AD vs 0x1B0)

The `attention_opcode_expansion_v2.md` (§4) assigns `VEC_NORM_L2_INT8` to `0x1AD`. The `supersession_patches_04.18.2026_v3.md` §5 summary and the registry §11 Reservation Table assign it to `0x1B0`. Both references exist in the current tree.

**v4 position**: v4 does NOT resolve this divergence — the collision predates v4 and belongs to the consistency sweep. Both opcode numbers refer to the same semantic operation (L2 normalize INT8 to scale 64). v4 flags this in the consistency sweep §Known-Divergences block. Daniel's turn-7 input is required to pick one; the other becomes `RESERVED (alias)` per expand-not-replace.

---

## 8. Codex Acceptance Checklist — v4

1. [ ] `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §11` has the new `0x1B1-0x1B5` row AND the narrowed `0x1BA-0x1BF` row; the old `0x1B1-0x1BF` row is replaced, not removed from history (keep a changelog entry if registry §11 grows a changelog subsection).
2. [ ] `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §7.x` has the Ruling 1 note on 0x1AF.
3. [ ] `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §7.x` has the Ruling 2 note on 0x1A9 AND 0x1AF.
4. [ ] `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §7.x` has the Ruling 3 note on 0x1A9 with `margin_path` operand spec.
5. [ ] `TEMP/attention_score_normalization_dual_path_spec_04.18.2026.md §4` prefetch block says MANDATORY, not optional.
6. [ ] `TEMP/attention_score_normalization_dual_path_spec_04.18.2026.md §4.5` (new) describes silent d-mismatch rescale with no printf.
7. [ ] `TEMP/attention_score_normalization_dual_path_spec_04.18.2026.md §5.0` (new) documents default Path A.
8. [ ] `TEMP/attention_opcode_expansion_v2.md §0 (0x1A9)` describes the `margin_path` flag.
9. [ ] `TEMP/attention_opcode_expansion_v2.md §6` lists P0 priority and mandatory prefetch rule.
10. [ ] `TEMP/attention_opcode_expansion_v2.md §6` has a d-mismatch subsection.
11. [ ] `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md` has the turn-6 case-study.
12. [ ] New memory file `feedback_attention_margin_dual_path_rulings.md` exists.
13. [ ] `MEMORY.md` has a pointer to the new memory file.
14. [ ] Gate R-prefetch and Gate R-dmatch grep checks pass against Lane A's kernel deliverables when they land.
15. [ ] VEC_NORM opcode-number divergence (0x1AD vs 0x1B0) escalated to Daniel for turn-7 ruling. v4 does NOT resolve.

---

## 9. References

- `TEMP/supersession_patches_04.18.2026.md` (v1)
- `TEMP/supersession_patches_04.18.2026_v2.md` (v2)
- `TEMP/supersession_patches_04.18.2026_v3.md` (v3)
- `TEMP/attention_score_normalization_dual_path_spec_04.18.2026.md`
- `TEMP/attention_opcode_expansion_v2.md`
- `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md`
- `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` (block-collision source)
- `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_expand_not_replace_opcodes.md`
- `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_hyper_modular_symlink_architecture.md`
