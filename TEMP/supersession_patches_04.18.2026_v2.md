# Supersession Patches — 2026-04-18 v2 (Rulings 2, 3, 4 Follow-Up)

**Date**: 2026-04-18
**Author**: Claude (architecture, connective-tissue lane)
**Extends**: `supersession_patches_04.18.2026.md` (v1 — do not remove; this file is additive)
**Rulings covered**:
- Ruling 2: contrastive margin `m` loaded from `star.confidence_trit` via `YARD_PEEK_ADDR`
- Ruling 3: mandatory `VEC_NORM_L2_INT8` pass after every attention output
- Ruling 4: vocabulary lockdown — "core" = CUDA block (46), "instance" = warp within a core (414 total)

**Codex instructions**: Read this file before acting on any spec listed below. Corrections here take precedence over the spec body. Apply atomically: do not partially apply patches from a single file section.

---

## VOCABULARY REFERENCE (Ruling 4 — applies across all files)

**Locked definitions (from `feedback_core_vs_instance_vocabulary.md`):**

| Term | Definition | RTX 3070 count |
|------|-----------|----------------|
| **core** | One CUDA block = one SM-exclusive math engine. Isolation boundary. Cross-core communication only via QUEUE_PUSH/POP/PEEK (0x178-0x17A). | **46** |
| **instance** | One warp within a core. 9 instances per core, share the yard substrate `float4 yards[9][9][69]`. | **414 total** (46 × 9) |
| **tier** | Capability layer inside an instance (Tier 1/2/3). Tiers share the yard freely. Isolation is at core boundary, not tier boundary. | N/A (per-program) |

**Wrong phrase → corrected phrase:**
- "414 cores" → "414 instances (across 46 isolated cores)"
- "cores per SM = 9" → "instances per core = 9" (or: "warps per block = 9")
- "core concurrency" (when meaning parallelism within a block) → "instance concurrency"

---

## CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md

### Section 2 — Constants table, row `MAX_INSTANCES` and `cores_per_sm`

**Superseded:**
```
| `MAX_INSTANCES` (per-engine hard-cap) | **delete** | Replaced by dynamic `sm_count × cores_per_sm` from `MicroSpecialistPool` |
| `cores_per_sm` | **9** | Bump from existing 10 → 9 for Tesla compliance; 46 SM × 9 = 414 concurrent cores on RTX 3070 (digit sum 9) |
```

**Corrected:**
```
| `MAX_INSTANCES` (per-engine hard-cap) | **delete** | Replaced by dynamic `sm_count × instances_per_core` from `MicroSpecialistPool` |
| `instances_per_core` | **9** | Tesla-9 compliance; one warp per instance; 46 SM × 1 core/SM × 9 instances/core = 414 concurrent instances. Note: each SM hosts exactly 1 isolated core (block) at the float4 ABI budget. |
```

**Reason:** Ruling 4 vocabulary lockdown. "core" = CUDA block (46 total). "instance" = warp within a core (414 total). `feedback_core_vs_instance_vocabulary.md`.

---

### Section 4.3 — Shared-memory layout comment

**Superseded:**
```
Per Ampere sm_86: 22.3 KB shared per block fits within the 100 KB/SM budget...
```

**Corrected:**
```
Per Ampere sm_86: 87.4 KB shared per block (per `yard_kernel_design_memo.md §1`). This exceeds the 82 KB half-budget, so occupancy is 1 block per SM = 1 isolated core per SM = 46 cores / 414 instances concurrently. This is the accepted trade-off for float4 ABI precision.
```

**Reason:** The 22.3 KB figure was corrected in `yard_kernel_design_memo.md §1` (float vs float4). Cross-referencing the correct value here.

---

## CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md

### Section 3 — Core-Count Math table, label `Concurrent cores`

**Superseded:**
```
| Concurrent cores | **414** | 46 × 9 |
```

**Corrected:**
```
| Isolated cores (CUDA blocks) | **46** | 1 block per SM × 46 SMs (at float4 ABI budget) |
| Instances per core (warps per block) | **9** | Tesla-9 compliance |
| Concurrent instances | **414** | 46 × 9 |
```

**Reason:** Ruling 4. "cores" = 46. "instances" = 414. The table previously listed 414 under "Concurrent cores" which conflated the two levels.

---

### Section 3 — CoreRegistry ceiling description

**Superseded:**
```
`CoreRegistry` tracks allocated cores against the 414-core hard ceiling (from `query_sm_count() * 9`).
```

**Corrected:**
```
`CoreRegistry` tracks allocated instances against the 414-instance hard ceiling (from `query_sm_count() * 9`). The 46 isolated cores are the physical SM allocation units; each core hosts 9 instances sharing the yard substrate. `spawn()` allocates one CUDA block (= one isolated core) only when all 9 of its instance slots are consumed, not per-instance. Attempting to spawn beyond 414 total active instances raises `CorePoolExhausted`.
```

**Reason:** Ruling 4. CoreRegistry ceiling is per-instance (414), but the CUDA block allocation is per-core (46).

---

### Section 4 — Instantiation API contract paragraph

**Superseded:**
```
`block_group` in CUDA = one core instance. Each `spawn()` allocates one CUDA block from the pool.
```

**Corrected:**
```
`block_group` in CUDA = one isolated core. Each `spawn()` allocates one CUDA block (= one core) from the pool; within that block, 9 instances (warps) run concurrently sharing the yard substrate. The `CoreHandle.instance_id` field identifies which of the 9 instances within the core is the primary program executor for this handle.
```

**Reason:** Ruling 4. A CUDA block is a core, not an instance.

---

### New subsection — Star Schema: `confidence_trit` field (add as §3.1 after the core-count table)

**Superseded:** (section not present)

**Corrected (add as §3.1):**
```markdown
### 3.1 Galaxy Star Schema — `confidence_trit` Field

Per Ruling 2 (2026-04-18): the contrastive margin parameter `m` in `ATTENTION_FWD_TERNARY` (0x1A8) and `ATTENTION_CONTRASTIVE_MARGIN` is loaded from each star's `confidence_trit` Galaxy field via `YARD_PEEK_ADDR` (0x173) inside the attention scoring loop. The margin is not a hardcoded integer operand.

The `confidence_trit` field is defined in full in `TEMP/galaxy_confidence_trit_field_spec.md`. Summary for Core Isolation implementers:

- **Storage**: 2-bit slot in the star's metadata word in Region 2 (Galaxy Universe VRAM). Located at a fixed byte offset within the star record, addressable by `YARD_PEEK_ADDR` with `bank_id = STAR_META_BANK` (bank 8 by convention, the register-store bank).
- **Values**: balanced ternary {-1, 0, +1}. +1 = high confidence (tight margin). 0 = unknown (default margin). -1 = low confidence (wide margin).
- **Derivation**: set by Phase B (`rpn_meaning_project.cu`) as a byproduct of chunk-variance during the meaning_rpn projection. Low variance across chunks → +1; high variance → -1.
- **Expand-not-replace**: this is a NEW field appended to the star schema. No existing star schema fields are modified or renamed.

**Attention scoring integration** (Codex note):
```
// Inside ATTENTION_FWD_TERNARY scoring loop, per-candidate:
YARD_PEEK_ADDR bank=STAR_META_BANK slot=CONFIDENCE_TRIT_OFFSET  ; push trit ∈ {-1,0,+1}
// m_effective = m_base + (1 - trit) * m_delta
//   trit=+1 → m_effective = m_base          (tight margin, high confidence)
//   trit= 0 → m_effective = m_base + m_delta (default margin)
//   trit=-1 → m_effective = m_base + 2*m_delta (wide margin, low confidence)
```

See `galaxy_confidence_trit_field_spec.md` for the full integration and sleep-time update protocol.
```

**Reason:** Ruling 2 requires `confidence_trit` to be a first-class Galaxy star schema field. The Core Isolation spec is the canonical location for the star record layout in VRAM Region 2, so the field definition anchor belongs here as a subsection.

---

## CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md

### Section 3 — Proposed New Opcodes table, attention family entries 0x1A7-0x1AF

**Superseded:**
```
| `0x1AA` | `IMAGE_DECODE_JPEG` | ...
| `0x1AB` | `RESIZE_BILINEAR_F32` | ...
| `0x1AC` | `NORMALIZE_IMAGE` | ...
| `0x1AD` | `STRIDED_GATHER` | ...
| `0x1AE` | `SPARSE_MATMUL` | ...
| `0x1AF` | `SPARSE_EIGSH` | ...
```

**Corrected:**
```
| `0x1AA` | IMAGE_DECODE_JPEG | RESERVED — this slot is in the attention family range (0x1AA-0x1AF). IMAGE_DECODE_JPEG must be relocated. See patch note below. | — | — |
| `0x1AB` | RESIZE_BILINEAR_F32 | RESERVED — same. Must be relocated. | — | — |
| `0x1AC` | NORMALIZE_IMAGE | RESERVED — same. Must be relocated. | — | — |
| `0x1AD` | STRIDED_GATHER | RESERVED — same; additionally 0x1AD is proposed for VEC_NORM_L2_INT8 (Ruling 3). See collision resolution below. | — | — |
| `0x1AE` | SPARSE_MATMUL | RESERVED — attention family range. Must be relocated. | — | — |
| `0x1AF` | SPARSE_EIGSH | RESERVED — attention family range. Must be relocated. | — | — |
```

**Collision resolution — IMAGE/SPARSE opcodes relocated to 0x1C0-0x1C5:**
```
| `0x1C0` | `IMAGE_DECODE_JPEG` | (relocated from 0x1AA) Pop encoded JPEG byte buffer, push decoded float32 RGB tensor (via nvjpeg) | P0 | cv2.imdecode, PIL.Image.open |
| `0x1C1` | `RESIZE_BILINEAR_F32` | (relocated from 0x1AB) Pop image tensor + (H_out, W_out); push resized float32 tensor | P0 | cv2.resize |
| `0x1C2` | `NORMALIZE_IMAGE` | (relocated from 0x1AC) Pop image tensor, pop mean[3], pop std[3]; push per-channel normalized tensor | P0 | cv2/PIL normalize |
| `0x1C3` | `STRIDED_GATHER` | (relocated from 0x1AD) Pop indices buffer, pop source tensor; push gathered elements | P0 | arr[mask], arr[:, indices] |
| `0x1C4` | `SPARSE_MATMUL` | (relocated from 0x1AE) Pop CSR format; push dense result | P1 | scipy.sparse.csr_matrix.dot |
| `0x1C5` | `SPARSE_EIGSH` | (relocated from 0x1AF) Pop CSR matrix, pop n_eigenpairs; push eigenvalues + vectors | P2 | scipy.sparse.linalg.eigsh |
```

**VEC_NORM_L2_INT8 — Ruling 3 assignment:**
```
| `0x1B0` | `VEC_NORM_L2_INT8` | Pop int8 vector tile from active bank; compute L2 norm; push L2-normalized int8 tile. Mandatory post-attention pass per Ruling 3. N=1 depth: called after every attention layer output, not only the final one. Replaces the "Optional L2 normalize" note in reference_attention_rpn_program.md §6. | P0 | Post-attention magnitude bounding |
```

Note: The parallel attention lane proposed 0x1AD for `VEC_NORM_L2_INT8`. That slot was already assigned to `STRIDED_GATHER` in this spec (written earlier 2026-04-18). Per the expand-not-replace doctrine (append-only, no slot reuse), `VEC_NORM_L2_INT8` takes 0x1B0, which is the first slot in the previously-reserved block 0x1B0-0x1B5. The 0x1B0-0x1B5 reservation is not broken — 0x1B0 is now assigned, 0x1B1-0x1B5 remain reserved.

**Reason:** Ruling 4 (attention family range 0x1AA-0x1AF is protected for attention ops per expand-not-replace doctrine); Ruling 3 (`VEC_NORM_L2_INT8` as mandatory post-attention pass); expand-not-replace (no slot collisions, relocate rather than conflict).

---

### Sections 2.4 context — STRIDED_GATHER reference update

**Superseded:**
```
| `arr[mask]` boolean indexing | PTX scatter/gather — **`STRIDED_GATHER`** | **0x1AD** (new) | Not yet |
| `arr[:, indices]` fancy indexing | Same: `STRIDED_GATHER` | **0x1AD** | Not yet |
| `np.ix_(a, b)` outer-product index | Decompose to two `STRIDED_GATHER` calls | Same | Not yet |
```

**Corrected:**
```
| `arr[mask]` boolean indexing | PTX scatter/gather — **`STRIDED_GATHER`** | **0x1C3** (relocated from 0x1AD) | Not yet |
| `arr[:, indices]` fancy indexing | Same: `STRIDED_GATHER` | **0x1C3** | Not yet |
| `np.ix_(a, b)` outer-product index | Decompose to two `STRIDED_GATHER` calls | **0x1C3** | Not yet |
```

**Reason:** Opcode relocation per attention-range reservation.

---

## CLAUDE_CODEX_PHASE_B_NATIVE_EMBEDDING_04.18.2026.md

### Section 3.5 — Chunk folding, add `confidence_trit` derivation paragraph

**Superseded:** §3.5 ends after the accumulation formula. No `confidence_trit` output declared.

**Corrected (add as the final paragraph of §3.5):**
```
### 3.5.1 confidence_trit Derivation (Ruling 2 dependency)

As a byproduct of chunk folding, `rpn_meaning_project.cu` MUST also compute and store a `confidence_trit` value for every star it processes.

**Derivation rule:**

1. Compute `chunk_variance = var({chunk_vec_0, chunk_vec_1, ..., chunk_vec_k})` across the k chunk vectors (before the final L2_NORM). This is a scalar: the mean squared deviation of per-chunk contribution magnitudes from the mean.
2. Map variance to trit:
   - `chunk_variance < LOW_VAR_THRESHOLD` (default: 0.05) → `confidence_trit = +1` (low variance = stable meaning across chunks = high confidence)
   - `chunk_variance > HIGH_VAR_THRESHOLD` (default: 0.30) → `confidence_trit = -1` (high variance = unstable = low confidence)
   - Otherwise → `confidence_trit = 0`
3. Store the 2-bit trit encoding ({-1→0b00, 0→0b01, +1→0b10}) in the star's metadata slot at `STAR_META_BANK` (see `galaxy_confidence_trit_field_spec.md`).

For programs with only one chunk (length ≤ 69 tokens), `chunk_variance = 0` → `confidence_trit = +1` always. This is correct: a single-chunk program is maximally self-consistent.

**This field is required by `ATTENTION_FWD_TERNARY` (0x1A8).** The attention scoring loop loads `confidence_trit` via `YARD_PEEK_ADDR` to set the effective contrastive margin per candidate. Phase B is not complete until `confidence_trit` is populated for all stars in the embedding table.
```

**Reason:** Ruling 2 requires `confidence_trit` to be produced by Phase B meaning_rpn projection. Without this field, attention margin loading fails.

---

## CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md

### Section 4.2 — Attention family reserved range, update to reflect 0x1AA-0x1AF protection

**Superseded:**
```
| `0x1AA` — `0x1AF` | Reserved | Attention family reserved range. Do NOT assign without cross-referencing `ternary_contrastive_attention_design.md` which is being finalized in a parallel architecture lane. |
```

**Corrected:**
```
| `0x1AA` — `0x1AF` | ATTENTION FAMILY RESERVED | Protected range. The Bulk-Lib Purge spec (written before this reservation was declared) incorrectly assigned IMAGE_DECODE_JPEG (0x1AA), RESIZE_BILINEAR_F32 (0x1AB), NORMALIZE_IMAGE (0x1AC), STRIDED_GATHER (0x1AD), SPARSE_MATMUL (0x1AE), and SPARSE_EIGSH (0x1AF) into this range. Those assignments are superseded by `supersession_patches_04.18.2026_v2.md`: those opcodes are relocated to 0x1C0-0x1C5. The 0x1AA-0x1AF range is now clean for attention family use. |
| `0x1A8` | `ATTENTION_FWD_TERNARY` | Assigned (parallel lane, 2026-04-18). Sovereign ternary Q·K + contrastive margin attention. Margin `m` loaded from `star.confidence_trit` per Ruling 2. |
| `0x1A9` | `CONTRASTIVE_RANK_TOPK` | Assigned (parallel lane, 2026-04-18). Top-K with contrastive margin gate. |
| `0x1AA-0x1AF` | RESERVED — attention family only | Available for future attention-family ops (multi-head coordinator, cross-attention, causal mask, etc.). |
| `0x1B0` | `VEC_NORM_L2_INT8` | Assigned (connective-tissue lane, 2026-04-18). Mandatory post-attention L2 normalization. Ruling 3. |
```

**Reason:** Ruling 3 (VEC_NORM_L2_INT8), Ruling 4 (vocabulary), and the attention-range collision resolution from the Bulk-Lib Purge spec.

---

## reference_modular_rpn_kernel_transfer_yard.cu

### File header comment — shared memory size and terminology

**Superseded:**
```
//   float4  yards[9][9][69]   //  [lane][bank][slot]  = 22,176 bytes
...
//   Total                                              ≈ 22,275 bytes  (< 100 KB Ampere budget)
```

**Corrected:**
```
//   float4  yards[9][9][69]   //  [lane][bank][slot]  = 87,264 bytes  (9×9×69×16 B)
//   uint8_t sp[9][9]          //  [lane][bank]        =     81 bytes
//   uint8_t active_bank[9]    //  [lane]              =      9 bytes
//   uint8_t error_code[9]     //  [lane]              =      9 bytes
//   ──────────────────────────────────────────────────────────────────
//   Total                                              ≈ 87,363 bytes  (< 164 KB sm_86 max)
//   NOTE: Exceeds 82 KB half-budget → 1 block per SM (1 isolated CORE per SM = 46 cores total).
//   TERMINOLOGY: each warp (threadIdx.y 0-8) = one INSTANCE. Block = one CORE.
//   46 cores × 9 instances/core = 414 concurrent instances.
```

**Reason:** Ruling 4 vocabulary; the 22 KB figure was documented as wrong in `yard_kernel_design_memo.md §1`.

---

## reference_advanced_rpn_kernel_transfer_yard.cu

### File header comment — terminology only

**Superseded:** No explicit "core vs instance" terminology in header.

**Corrected (add after the SHARED MEMORY block):**
```
// TERMINOLOGY (Ruling 4, 2026-04-18):
//   CORE     = one CUDA block = one SM-exclusive isolated compute unit. 46 on RTX 3070.
//   INSTANCE = one warp (threadIdx.y 0-8) within a block. 9 per core, 414 total.
//   TIER     = capability layer within an instance (Tier 1/2/3 share the yard).
//   Never say "414 cores" — the correct phrase is "414 instances across 46 cores".
```

**Reason:** Ruling 4. Reference kernels are read by Codex and must carry the locked vocabulary.

---

## yard_kernel_design_memo.md

### Section 3 — "9 Yards" rationale, concurrency claim

**Superseded:** The memo uses "cores" and "instances" inconsistently in the rationale paragraphs.

**Corrected (add as a terminology note at the top of §3):**
```
**Terminology note (Ruling 4):** In this memo, "lane" = "instance" (one warp within a block). Each block (= one isolated core) has 9 lanes/instances. The 46 SMs each host 1 block, giving 46 cores and 414 instances total. The memo's existing use of "lane" is correct for the warp-level analysis; "core" in this memo refers to the block/SM unit (not the instance/warp).
```

**Reason:** Ruling 4. The memo's internal "lane" terminology is functionally correct (it refers to `threadIdx.y` = instance). The note disambiguates for readers who arrive with the locked vocabulary.

---

## ternary_contrastive_attention_design.md

### File header — supersession note only (leave body intact)

**Superseded:** No supersession header.

**Corrected (prepend to file, before §1):**
```
> **PARTIAL SUPERSESSION — 2026-04-18 v2 (connective-tissue lane)**
> The body of this document remains authoritative for:
> - §2 (Q·K via XNOR+popcount derivation)
> - §3 (zero-mass correction, four-mask method)
> - §4+ (V-mix, contrastive margin semantics)
>
> The following sections are superseded or augmented by `supersession_patches_04.18.2026_v2.md`:
> - **Margin parameter `m`**: is no longer a static integer operand. It is loaded from `star.confidence_trit` via `YARD_PEEK_ADDR` per Ruling 2. See §galaxy_confidence_trit_field_spec.md for the derivation and `m_effective` formula.
> - **VEC_NORM post-pass**: Ruling 3 mandates `VEC_NORM_L2_INT8` (0x1B0) after every attention output. N=1 depth (normalize after every layer, not only final). Add `0x1B0 VEC_NORM_L2_INT8` as the last opcode in the RPN trace for every attention head.
> - **Opcode 0x1AD**: was proposed here for `VEC_NORM_L2_INT8`. That slot is occupied by `STRIDED_GATHER` (relocated to 0x1C3). `VEC_NORM_L2_INT8` is assigned 0x1B0. See collision resolution in `supersession_patches_04.18.2026_v2.md §BULK_LIB_PURGE`.
```

**Reason:** Rulings 2 and 3. The document body is correct; only the margin source and the post-pass requirement are new.

---

## attention_opcode_expansion.md

### File header — supersession note only (leave body intact)

**Superseded:** No supersession header.

**Corrected (prepend to file, before §0):**
```
> **PARTIAL SUPERSESSION — 2026-04-18 v2 (connective-tissue lane)**
> This document remains authoritative for the implementation specifications of 0x1A7, 0x1A8 (ATTENTION_FWD_TERNARY), 0x108 (TERNARY_XNOR_POPCOUNT), and 0x1A9 (CONTRASTIVE_RANK_TOPK).
>
> Augmented by `supersession_patches_04.18.2026_v2.md`:
> - `ATTENTION_FWD_TERNARY` operand `margin_m`: this is now a DERIVED value, not a static operand. The opcode loads `confidence_trit` from the candidate star's metadata via `YARD_PEEK_ADDR` and computes `m_effective` dynamically. The static `margin_m` operand field remains for use as `m_base`. See `galaxy_confidence_trit_field_spec.md`.
> - **VEC_NORM_L2_INT8 (0x1B0)** is the mandatory next step after any `ATTENTION_FWD_TERNARY` invocation. Every RPN program calling 0x1A8 must follow it with `0x1B0 VEC_NORM_L2_INT8`. This is a grep-checkable acceptance gate.
> - The proposed opcode number for `VEC_NORM_L2_INT8` was 0x1AD in this lane's earlier work. That number is occupied by `STRIDED_GATHER` (now at 0x1C3). The correct number is **0x1B0**.
```

**Reason:** Rulings 2, 3, and the 0x1AD collision resolution.

---

## reference_attention_rpn_program.md

### Section 6 — Phase 5 Output Readout, insert VEC_NORM_L2_INT8

**Superseded:**
```
0x170  YARD_SELECT    bank_id=3          ; [0x170] activate output bank
; (Optional) L2 normalize output using matryoshka_prefix_dot pattern
; The composed head pipeline continues: output is passed to Halting Gate or next layer
```

**Corrected:**
```
0x170  YARD_SELECT    bank_id=3          ; [0x170] activate output bank
; ── Ruling 3: MANDATORY VEC_NORM pass, N=1 depth ──────────────────────────────────────────
0x1B0  VEC_NORM_L2_INT8                  ; [0x1B0] L2-normalize attention output in-place.
                                          ; NOT optional. Required after EVERY attention layer.
                                          ; Bounds output magnitude; prevents activation blowup
                                          ; in multi-layer or recursive TRM tick scenarios.
                                          ; Acceptance gate: grep every RPN program that calls
                                          ; 0x1A8 for an immediately following 0x1B0. If absent,
                                          ; the program fails the VEC_NORM gate.
; ────────────────────────────────────────────────────────────────────────────────────────────
; The composed head pipeline continues: output is passed to Halting Gate or next layer
```

**Reason:** Ruling 3 (mandatory VEC_NORM_L2_INT8, N=1 depth, after every attention output). The prior "Optional" label is wrong.

---

## CLAUDE_CODEX_GPU_NATIVE_ASYNC_DOCTRINE_04.18.2026.md

**Vocabulary sweep result**: Document uses only generic GPU concurrency terminology ("block", "warp", "lane") without asserting specific counts. No "414 cores" pattern found.

**Status**: OK — no patches required.

---

## CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md

**Vocabulary sweep result**: References "414 cores" in the dependency chain description at §5 (approximate location — Codex should grep for "414").

**Superseded (grep: "414 cores" in the doc):**
```
...414 cores on RTX 3070...
```

**Corrected:**
```
...414 instances (across 46 isolated cores) on RTX 3070...
```

**Reason:** Ruling 4. If the exact phrase is not present, this patch is a no-op; Codex must confirm with grep.

---

## CLAUDE_CODEX_OLD_ATTEMPTS_MIGRATION_04.18.2026.md

**Vocabulary sweep result**: No "cores" count claims found. `enhanced_fallback.py` direct-archive ruling was already patched in v1 of this file.

**Status**: OK — no new patches required. V1 patches for `enhanced_fallback.py` stand.

---

## OPCODE SLOT SUMMARY — Post v1+v2 State

| Range | Status |
|-------|--------|
| 0x100-0x108 | 0x100-0x107 existing ternary; 0x108 = TERNARY_XNOR_POPCOUNT (parallel lane) |
| 0x170-0x177 | Yard ops (Transfer Yard spec) |
| 0x178-0x17A | Queue ops (Core Isolation spec) |
| 0x17B-0x17F | Reserved (yard family future) |
| 0x1A0-0x1A9 | New math/utility (0x1A0-0x1A6) + attention family (0x1A7-0x1A9) |
| 0x1AA-0x1AF | ATTENTION FAMILY RESERVED (clean — IMAGE/SPARSE opcodes relocated to 0x1C0-0x1C5) |
| 0x1B0 | VEC_NORM_L2_INT8 (Ruling 3, this patch) |
| 0x1B1-0x1B5 | Reserved (future) |
| 0x1B6-0x1B9 | TENSOR_INTERPOLATE, KMEANS_PLUS_INIT, CTYPES_VIEW_AS_PTX, CUDA_MALLOC_ASYNC |
| 0x1BA-0x1BF | Reserved (future) |
| 0x1C0-0x1C5 | IMAGE_DECODE_JPEG, RESIZE_BILINEAR_F32, NORMALIZE_IMAGE, STRIDED_GATHER, SPARSE_MATMUL, SPARSE_EIGSH (relocated) |
