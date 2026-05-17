# Supersession Patches — 2026-04-18 v3 (Turn-5 Rulings Consolidation)

**Date**: 2026-04-18
**Author**: Claude (architecture, connective-tissue lane)
**Supersedes**: `supersession_patches_04.18.2026_v2.md` (additive; v2 remains authoritative for patches not re-issued here)
**Rulings covered**: Turn-5 Rulings 1–5
**Sibling patches**: Lane 1 handles Ruling 1 (Q·K^T normalization dual path), Lane 3 handles Ruling 4 Gate 7 (Matryoshka pack-order verification). This file does NOT patch those items — it references them.

**Codex instructions**: Apply v3 patches atomically per section. Where v3 conflicts with v2, v3 wins. Where v3 is silent, v2 stands.

---

## Turn-5 Ruling Summary

| Ruling | Topic | Status | Owner |
|---|---|---|---|
| 1 | Q·K^T normalization: keep BOTH shift-down and scale-up paths | REFERENCED (Lane 1) | parallel lane |
| 2 | `confidence_trit` + `confidence_margin` as paired coexisting fields | PATCHED HERE | connective tissue |
| 3 | `VEC_NORM_L2_INT8` default scale = 64 (headroom, not unit-sphere 127) | PATCHED HERE | connective tissue |
| 4 | Bulk-lib purge Gate 7: Matryoshka pack-order verification | REFERENCED (Lane 3) | parallel lane |
| 5 | Opcode range-reservation protocol formalized as standing doctrine | PATCHED HERE (new doctrine + registry §11) | connective tissue |

---

## 1. Ruling 2 — Paired Fields: `confidence_trit` + `confidence_margin`

Ruling 2 (verbatim): **Keep both as coexisting paired fields per expand-not-replace.**

- `confidence_trit`: raw balanced-ternary Galaxy field `{-1, 0, +1}`, 2-bit packed slot in star metadata (per `galaxy_confidence_trit_field_spec.md` and v2 §CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md §3.1).
- `confidence_margin`: derived `int8` threshold for dp4a-scale comparisons. Computed at load time from `confidence_trit` and the active opcode's scale-bits configuration.

### 1.1 Derivation Rule

```
margin = trit × scale_bits_at_load_time
```

Where `scale_bits_at_load_time` is the INT8 scale (typically the contrastive-margin base `m_base` for attention, or the post-quantization scale for a given opcode's lane). This is a one-line compile-time multiplication performed during program load, not at hot-path execution time.

**Semantics table:**

| `confidence_trit` | `scale_bits` | `confidence_margin` (int8) | Meaning |
|---|---|---|---|
| +1 | k | +k | high confidence → tight margin |
| 0 | k | 0 | unknown → default (no-adjust) margin |
| −1 | k | −k | low confidence → loose margin (widen acceptance) |

**Range discipline**: `scale_bits` is chosen per consumer such that `|margin| ≤ 127` always holds. For the default attention pipeline with `m_base ≤ 64`, `scale_bits = m_base` keeps margins inside int8 range without saturation.

### 1.2 Which Opcodes Consume Which

| Opcode | Consumes | Pathway |
|---|---|---|
| `0x1A8` `ATTENTION_FWD_TERNARY` | `confidence_trit` (raw) | Loads trit via `YARD_PEEK_ADDR` per candidate; computes `m_effective = m_base + (1 − trit) × m_delta` inline. Unchanged from v2. |
| `0x1A9` `CONTRASTIVE_RANK_TOPK` | `confidence_margin` (derived int8) | Uses pre-computed `confidence_margin` loaded once at program-start for dp4a-scale partial-sort comparisons. No per-candidate trit expansion inside the TopK loop. |
| `0x1A7` `ATTENTION_FWD` (_BASE variant) | neither (float32 path) | Training-lane reference; reads the full float contrastive score without trit gating. |
| `0x1B0` `VEC_NORM_L2_INT8` | neither | Post-attention magnitude bounding; independent of confidence fields. |

**Rule**: a lane may consume either the raw `confidence_trit` (expressive, per-candidate) or the derived `confidence_margin` (fast, pre-computed) — but it must choose **one**, and must declare which in the opcode's spec entry. Consuming both in the same opcode is forbidden (double-counts confidence).

### 1.3 Files to Patch

#### `TEMP/CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md §3.1`

**Superseded (from v2)**: the §3.1 body describes only `confidence_trit`.

**Corrected (append after the current §3.1 body)**:
```markdown
### 3.1.1 Paired Field — `confidence_margin` (Ruling 2, v3)

The star schema also carries `confidence_margin: int8`, derived at program-load time from `confidence_trit` and the consumer opcode's `scale_bits` configuration.

- Storage: contiguous 8-bit slot adjacent to the `confidence_trit` 2-bit slot in the star metadata word (Region 2).
- Derivation: `margin = trit × scale_bits_at_load_time`. Computed once at load; immutable for the program's active lifetime.
- Consumer: fast-path opcodes that need dp4a-scale comparisons without per-candidate trit expansion (e.g., `CONTRASTIVE_RANK_TOPK`).
- Expand-not-replace: `confidence_margin` is a NEW derived field appended to the schema. It does NOT replace `confidence_trit`. Opcodes that need the raw trit continue to read it via `YARD_PEEK_ADDR`.

Opcode MUST declare in its spec entry which field it consumes (trit or margin); consuming both is forbidden.
```

#### `TEMP/CLAUDE_CODEX_PHASE_B_NATIVE_EMBEDDING_04.18.2026.md §3.5.1`

**Superseded (from v2)**: the v2 patch describes only storing `confidence_trit`.

**Corrected (append as §3.5.2)**:
```markdown
### 3.5.2 confidence_margin Derivation (Ruling 2 paired field)

Phase B MUST also populate `confidence_margin: int8` as a derived field at the point of star write.

- Formula: `confidence_margin = confidence_trit × scale_bits`, where `scale_bits = m_base` for attention consumers (default 64 per Ruling 3 v3, see §VEC_NORM_L2_INT8 scale). If the star is not destined for an attention consumer, `scale_bits = DEFAULT_SCALE` (64).
- Storage: int8 slot adjacent to the 2-bit trit slot in star metadata.
- Acceptance: a Phase B star is not complete until both `confidence_trit` and `confidence_margin` are present.
```

---

## 2. Ruling 3 — `VEC_NORM_L2_INT8` Default Scale = 64

Ruling 3 (verbatim): **Default scale = 64 (headroom), not 127 (unit-sphere). Prevents INT8 overflow in downstream accumulations.**

The v2 patch landed `0x1B0 VEC_NORM_L2_INT8` as the mandatory post-attention pass but did not specify the normalization target magnitude. v3 fixes this.

### 2.1 Files to Patch

#### `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §7.x (VEC_NORM_L2_INT8 entry)`

**Add the opcode entry** (if not already landed from v2) under the next available section (the registry structure will pick the right position):

```markdown
| `0x1B0` | `VEC_NORM_L2_INT8` | Normalization / post-attention | `[int8_vec_tile] -> [int8_vec_tile]` | Compute L2 norm of int8 vector tile in active yard bank; rescale elements so that output L2 norm equals `NORM_SCALE` (default 64, NOT 127). Output remains int8. | Confidence preserved from input; polarity preserved element-wise. Default scale 64 leaves 1 bit of headroom for downstream add/sub/skip accumulations without saturation. |
```

**Add a normative note** after the opcode table entry:

```markdown
#### 7.x.1 NORM_SCALE Constant

`NORM_SCALE = 64` (default). Rationale (Ruling 3): normalizing to `|v|_2 = 127` (int8 unit sphere) leaves no headroom; a subsequent add/sub/skip accumulation of two normalized vectors can saturate int8 range and silently clip. Normalizing to `|v|_2 = 64` reserves 1 bit (factor-of-2) of accumulation headroom. This is the correct tradeoff for K3D's multi-layer attention stack where each layer emits into the next without intermediate quantization rescaling.

`NORM_SCALE` is a compile-time constant on the kernel; it is NOT an opcode operand. Lanes that require a non-default scale must emit a distinct opcode variant (`VEC_NORM_L2_INT8_UNIT` for 127, `VEC_NORM_L2_INT8_HALF` for 32, etc.) per expand-not-replace.
```

#### `TEMP/reference_attention_rpn_program.md §6 (post-attention VEC_NORM pass)`

**Superseded (from v2)**: the v2 patch inserts the `0x1B0 VEC_NORM_L2_INT8` line but does not document the scale.

**Corrected (replace the inline comment immediately after the `0x1B0` line)**:
```
0x1B0  VEC_NORM_L2_INT8                  ; [0x1B0] L2-normalize to NORM_SCALE=64 (Ruling 3 v3).
                                          ; NOT unit sphere (127). Reserves headroom for downstream
                                          ; add/sub/skip accumulations (BitNet b1.58 kernels).
                                          ; NOT optional. Required after EVERY attention layer.
                                          ; Acceptance gate: grep every RPN program that calls
                                          ; 0x1A8 for an immediately following 0x1B0.
```

#### `TEMP/attention_opcode_expansion.md` and `TEMP/attention_opcode_expansion_v2.md`

**Superseded (from v2)**: the v2 partial-supersession header references 0x1B0 as "mandatory" but does not pin the scale.

**Corrected (extend the v2 partial-supersession header with)**:
```
> - **VEC_NORM_L2_INT8 scale = 64** (Ruling 3 v3, 2026-04-18). Default `NORM_SCALE` is 64, NOT the int8 unit sphere (127). See `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §7.x.1`.
```

---

## 3. Ruling 5 — Opcode Range Reservation Protocol

Ruling 5 (verbatim): **Formalize opcode range-reservation protocol as standing rule.**

### 3.1 New Doctrine File (companion to this patch)

Doctrine written to: `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md`

Full text of the doctrine: see that file. Summary:
1. Pre-reserve range blocks in `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` §11 before dispatching parallel-lane opcode work.
2. Lanes allocate within their reserved block; cross-block writes require re-reservation.
3. Registry is the single source of truth. Design-doc reservations are not reservations.
4. A reservation is a symlink to future kernel capability (ties to hyper-modular symlink doctrine).
5. Gate R is the automated enforcement check.

### 3.2 Registry Patch — New §11 "Reserved Future Blocks"

#### `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`

**Superseded**: no §11 exists; the registry ends at §10 References.

**Corrected**: append a new §11 after §10. **Append-only; does not modify any existing section.**

```markdown

---

## 11. Reserved Future Blocks (Range Reservation Table)

**Authority**: `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md`
**Status**: Normative — the table is the single source of truth for opcode range reservations.

Per the Opcode Range Reservation Doctrine, any parallel-lane task that will mint new opcodes MUST pre-reserve its block in this table before dispatching spec or implementation work. Opcodes assigned outside a reserved block fail the Gate R acceptance check.

**Schema:**

| Field | Meaning |
|---|---|
| `block_start` | Lowest opcode number in the block (inclusive, hex) |
| `block_end` | Highest opcode number in the block (inclusive, hex) |
| `owner_spec` | Spec file governing opcode assignments inside this block |
| `date_reserved` | Date the reservation was appended (YYYY-MM-DD) |
| `status` | `active` (lane working), `released` (lane done, opcodes permanent), `superseded` (spec withdrawn, range free) |

**Initial Reservations (reconstructed from 2026-04-18 state):**

| block_start | block_end | owner_spec | date_reserved | status |
|---|---|---|---|---|
| `0x100` | `0x10F` | `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §7.1` | 2026-04-13 | released |
| `0x170` | `0x17F` | `TEMP/CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md` | 2026-04-18 | active |
| `0x178` | `0x17A` | `TEMP/CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md` (queue ops — sub-reservation within 0x170–0x17F) | 2026-04-18 | active |
| `0x180` | `0x18F` | `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §4.2–4.3` (WINE I/O contract block) | 2026-04-18 | active |
| `0x190` | `0x19F` | `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §12` (physics-to-visual bridge) | 2026-04-18 | active |
| `0x1A0` | `0x1A6` | `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` (bulk-lib purge math/utility ops) | 2026-04-18 | active |
| `0x1A7` | `0x1AF` | `TEMP/attention_opcode_expansion.md` + `TEMP/ternary_contrastive_attention_design.md` (attention family — ATTENTION_FWD, ATTENTION_FWD_TERNARY, CONTRASTIVE_RANK_TOPK, and 0x1AA–0x1AF future attention headroom) | 2026-04-18 | active |
| `0x1B0` | `0x1B0` | `TEMP/supersession_patches_04.18.2026_v2.md` (VEC_NORM_L2_INT8) | 2026-04-18 | active |
| `0x1B1` | `0x1BF` | future normalization/attention family (headroom reservation) | 2026-04-18 | active |
| `0x1C0` | `0x1C5` | `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` (IMAGE/SPARSE — relocated from 0x1AA–0x1AF per v2) | 2026-04-18 | active |
| `0x1C6` | `0x1CF` | future physics expansion (headroom reservation, tied to §7.4) | 2026-04-18 | active |

**Maintenance rules:**
1. Additions are append-only. Never delete a row.
2. Status transitions `active → released → superseded` are one-way; once `superseded`, the range may be re-reserved by a new owner in a new row (existing opcodes inside it remain permanent per expand-not-replace).
3. Overlapping `active` rows are invalid. Gate R rejects them.
4. A reservation of size 1 (emergency single-opcode insertion) must still appear as a row.
```

### 3.3 Cross-Reference Patch in §6

**Add the following note at the end of §6.3 (Stage 3: PTX Kernel Admission)**:

```markdown
**Pre-reservation prerequisite (Ruling 5, 2026-04-18)**: before a Stage 3 admission writes an opcode number into the registry, the lane must verify its reservation exists in §11. If no reservation covers the target number, the admission is blocked until the Reservation Table is amended. See `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md` for the full workflow.
```

---

## 4. Rulings 1 and 4 — Cross-Lane References (No Patch in This File)

### 4.1 Ruling 1 — Q·K^T Dual Normalization Path

**Owner**: Lane 1 (parallel attention-normalization lane).
**Scope**: both shift-down (range compression) and scale-up (precision preservation) paths remain in the attention kernel. Lane 1 delivers the implementation; v3 does not patch the kernel.
**Acknowledgement**: `ATTENTION_FWD_TERNARY` (0x1A8) in `attention_opcode_expansion.md` remains the entry point; the dual path is internal to the kernel and does not surface new opcodes.

### 4.2 Ruling 4 — Bulk-Lib Purge Gate 7

**Owner**: Lane 3 (Matryoshka pack-order verification lane).
**Scope**: Gate 7 is added to the bulk-lib purge hard-acceptance gate sequence in `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md §N (gates section)`. Verifies Matryoshka weight-matrix pack-order: rows must be in ascending row-index order so tier-prefix truncation stays valid.
**Acknowledgement**: this file references Gate 7 but does NOT rewrite the gate body. Lane 3 owns the patch.

---

## 5. OPCODE SLOT SUMMARY — Post v1+v2+v3 State

(Carries forward v2's summary; no opcode reassignments in v3.)

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
| 0x1A9 | CONTRASTIVE_RANK_TOPK |
| 0x1AA–0x1AF | ATTENTION FAMILY RESERVED (clean — IMAGE/SPARSE relocated to 0x1C0–0x1C5 per v2) |
| 0x1B0 | VEC_NORM_L2_INT8 (scale=64, Ruling 3 v3) |
| 0x1B1–0x1B5 | Reserved (future normalization/attention) |
| 0x1B6–0x1B9 | TENSOR_INTERPOLATE, KMEANS_PLUS_INIT, CTYPES_VIEW_AS_PTX, CUDA_MALLOC_ASYNC |
| 0x1BA–0x1BF | Reserved (future) |
| 0x1C0–0x1C5 | IMAGE_DECODE_JPEG, RESIZE_BILINEAR_F32, NORMALIZE_IMAGE, STRIDED_GATHER, SPARSE_MATMUL, SPARSE_EIGSH (relocated per v2) |
| 0x1C6–0x1CF | Reserved (physics expansion headroom) |

---

## 6. File Change Manifest — v3

| File | Change | Section |
|---|---|---|
| `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` | ADD §11 (Reservation Table); ADD note at end of §6.3 (Stage 3 pre-reservation); ADD 0x1B0 opcode entry + §7.x.1 NORM_SCALE note if not yet present | §6.3, §7.x, §11 |
| `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md` | NEW FILE (doctrine, companion to this patch) | all |
| `TEMP/CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md` | ADD §3.1.1 (confidence_margin paired field) | §3.1.1 |
| `TEMP/CLAUDE_CODEX_PHASE_B_NATIVE_EMBEDDING_04.18.2026.md` | ADD §3.5.2 (confidence_margin derivation) | §3.5.2 |
| `TEMP/reference_attention_rpn_program.md` | PATCH post-attention VEC_NORM comment (scale=64 notes) | §6 |
| `TEMP/attention_opcode_expansion.md` | EXTEND partial-supersession header (scale=64 line) | header |
| `TEMP/attention_opcode_expansion_v2.md` | EXTEND partial-supersession header (scale=64 line) | header |
| `TEMP/supersession_patches_04.18.2026_v2.md` | No patch; v2 remains authoritative for its patches. v3 is additive. | — |
| `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_opcode_range_reservation_protocol.md` | NEW MEMORY (this ruling) | all |
| `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_bitnet_b158_ternary_pattern.md` | UPDATE (add NORM_SCALE=64 note) | §add |
| `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/MEMORY.md` | UPDATE (add pointer to new memory file) | § "Expand-Not-Replace …" block |

**Not touched by v3 (Lane 1, Lane 3 owners):**
- `TEMP/ternary_contrastive_attention_design.md` — Q·K^T dual-path semantics (Lane 1)
- `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md §N (gates section)` — Gate 7 Matryoshka pack-order (Lane 3)

---

## 7. Codex Acceptance Checklist — v3

1. [ ] `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §11` exists and contains the initial reservation table.
2. [ ] `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §6.3` has the pre-reservation cross-reference note.
3. [ ] `0x1B0 VEC_NORM_L2_INT8` entry and `NORM_SCALE = 64` note are in the registry.
4. [ ] `CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION §3.1.1` describes `confidence_margin` as a paired derived field.
5. [ ] `CLAUDE_CODEX_PHASE_B_NATIVE_EMBEDDING §3.5.2` describes `confidence_margin` derivation in Phase B.
6. [ ] `reference_attention_rpn_program §6` post-VEC_NORM comment includes scale=64.
7. [ ] `attention_opcode_expansion.md` and `_v2.md` supersession headers mention scale=64.
8. [ ] New doctrine file `CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md` exists.
9. [ ] Gate R is runnable against the current tree with no open violations.
10. [ ] Lane 1 (Q·K^T dual path) and Lane 3 (Gate 7) deliverables remain their owners' responsibility — not checked by v3.

---

## 8. References

- `TEMP/supersession_patches_04.18.2026.md` (v1)
- `TEMP/supersession_patches_04.18.2026_v2.md` (v2)
- `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md` (new doctrine)
- `TEMP/CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md`
- `TEMP/CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md`
- `TEMP/galaxy_confidence_trit_field_spec.md` (referenced by Ruling 2 pathway)
- `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_bitnet_b158_ternary_pattern.md`
