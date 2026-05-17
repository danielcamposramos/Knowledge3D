# Doctrine: Opcode Range Reservation Protocol

**Date**: 2026-04-18
**Author**: Claude (architecture partner)
**Status**: DOCTRINE — standing rule, enforcement at the registry level
**Authority**: Daniel's turn-5 ruling, 2026-04-18
**Sibling doctrines**: `CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md`, `CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md`
**Canonical registry**: `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`

---

## 1. Problem Statement — The 0x1AD Collision

On 2026-04-18 two parallel architecture lanes dispatched in the same day minted opcodes into the same byte-range without coordination:

- The **attention-family lane** (`ternary_contrastive_attention_design.md`, `attention_opcode_expansion.md`) allocated `0x1A7` (`ATTENTION_FWD`), `0x1A8` (`ATTENTION_FWD_TERNARY`), `0x1A9` (`CONTRASTIVE_RANK_TOPK`) and carved out `0x1AA–0x1AF` as an **attention-family reserved block** — but this reservation was written only in the design doc, not in the canonical registry.
- The **bulk-lib purge lane** (`CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md`) concurrently landed `IMAGE_DECODE_JPEG` (0x1AA), `RESIZE_BILINEAR_F32` (0x1AB), `NORMALIZE_IMAGE` (0x1AC), `STRIDED_GATHER` (0x1AD), `SPARSE_MATMUL` (0x1AE), `SPARSE_EIGSH` (0x1AF) — fully occupying the attention family's intended reserved range.

The collision was discovered only after both specs were written. Resolution (documented in `supersession_patches_04.18.2026_v2.md`) required relocating IMAGE/SPARSE opcodes to `0x1C0–0x1C5` and issuing patch notes to four downstream files. A further collision occurred at `0x1AD`: the attention lane had also proposed it for `VEC_NORM_L2_INT8`, which was reassigned to `0x1B0`.

**Root cause**: neither lane pre-reserved its range block in the canonical registry before dispatching implementation work. Both treated their design doc as authoritative; neither was.

**Meta-insight**: when parallel lanes mint opcodes without prior registry reservation, collisions are not a risk — they are the default outcome. The expand-not-replace doctrine prevents renumbering after the fact; this doctrine prevents the collision in the first place.

---

## 2. Rule — Pre-Reserve Before Dispatch

**Mandatory rule for every parallel-lane task that will mint one or more new opcodes:**

> **Before** any parallel lane begins spec-writing or implementation work that will allocate opcode numbers, the orchestrator (Claude, acting as architecture partner) **must** append an entry to the Reservation Table in `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` that:
>
> 1. Declares the `[block_start, block_end]` range (inclusive, hex)
> 2. Names the owner spec file (the design doc that will define the opcodes inside the block)
> 3. Records the reservation date
> 4. Marks the block `active`

Lanes then allocate **within** their reserved block. A lane may **not** write an opcode number outside its reserved range.

**Cross-block writes require re-reservation.** If a lane discovers it needs more opcodes than its block holds, it must either (a) extend the existing block by appending a contiguous reservation for the extension, or (b) reserve a new non-contiguous block. In both cases the registry is edited first, before any further spec body text is written.

**Registry is the single source of truth.** A reservation written only in a design doc is not a reservation. A reservation written only in a memory note is not a reservation. The registry's Reservation Table is the sole authoritative list.

---

## 3. Reservation Table Schema

Append this table to `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` as a new section titled **"11. Reserved Future Blocks"** (after §10 References). The table has five columns:

| Column | Type | Meaning |
|---|---|---|
| `block_start` | hex u16 | Lowest opcode number in the block (inclusive) |
| `block_end` | hex u16 | Highest opcode number in the block (inclusive) |
| `owner_spec` | path | Relative path to the spec file that governs opcode assignments inside this block |
| `date_reserved` | YYYY-MM-DD | Date the reservation was appended |
| `status` | enum | `active` \| `released` \| `superseded` |

**Status semantics:**
- `active` — lane is currently minting opcodes within the block; other lanes must not write inside it
- `released` — lane has completed work; opcodes assigned inside the block remain permanent (expand-not-replace) but the reservation no longer blocks neighboring lanes from adjacent blocks
- `superseded` — the owner spec was withdrawn; the reservation is dropped and the range is once again free to reserve (opcodes that were already assigned before supersession remain permanent)

**A `released` block is NOT a free-for-all.** Opcodes inside the block are permanent by expand-not-replace. "Released" only means neighboring lanes no longer need to cross-reference the owner spec before working adjacent to the block.

---

## 4. Workflow for a Parallel-Lane Dispatch

The orchestrator dispatching a parallel lane **must** perform these steps in order, before any spec body is authored:

1. **Scan** the Reservation Table for active blocks that overlap the lane's anticipated range.
2. **Identify** the lowest free range that fits the lane's opcode count plus a 25%–50% headroom margin (lanes routinely undercount; headroom avoids repeat re-reservations).
3. **Append** a new row to the Reservation Table with status `active`. The edit is a single-line registry patch, not a rewrite of the section.
4. **Reference** the reservation in the lane's spec file header: `Reserved block: [block_start, block_end] (see RPN_DOMAIN_OPCODE_REGISTRY.md §11)`.
5. **Dispatch** the lane. The lane author is now free to assign any opcode number inside the reserved block.

At completion, the orchestrator updates the status to `released`. If the lane was withdrawn, the status becomes `superseded`.

---

## 5. Symlink Interpretation — Tie to Hyper-Modular Symlink Doctrine

Per `CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md`: phases stand on each other like symlinks; breaking a link reduces everything above it to a fallback in disguise.

**A range reservation IS a symlink to future kernel capability.** The reservation names a block of opcode numbers that downstream RPN programs, test traces, and kernel-wiring plans may reference *before the kernels inside the block exist*. The reservation is the symlink target; the eventual kernel is the linked resource.

Treating the reservation as anything less than a load-bearing symlink — i.e., writing opcodes outside the reservation, or treating design-doc text as a substitute for a registry edit — breaks the link. Every downstream program that assumed the reservation held is retroactively a fallback path. The 0x1AD collision is the canonical example: the attention design doc claimed `0x1AA–0x1AF`, but because the link was not registered, the bulk-lib lane wrote through it, and every downstream attention-family program briefly became fallback code until the supersession patch landed.

**Applied rule:** a lane cannot symlink to an opcode block that is not in the Reservation Table. The registry edit is the act of creating the symlink.

---

## 6. Enforcement — Grep-Checkable Acceptance Gate

The doctrine is enforceable by a single automated check, runnable by Codex as an acceptance gate:

**Gate R (Range Reservation) — to be added to the bulk-lib purge hard-acceptance gates:**

For every opcode assignment `0x{NNNN}` introduced in any file modified during a lane's work:
1. Parse the lane's spec header for `Reserved block: [block_start, block_end]`.
2. Assert `block_start <= 0x{NNNN} <= block_end`.
3. Assert the registry's Reservation Table contains a matching active row with `owner_spec` equal to the lane's spec path.

Violations fail the lane's acceptance gate. The patch must be reverted or the Reservation Table amended (the latter is permitted mid-lane via step 4 of the workflow in §4).

Claude (architecture partner) runs Gate R during consistency sweeps. Codex runs Gate R as part of the hard-acceptance sweep before any parallel-lane merge.

---

## 7. Initial Reservation Table Contents (2026-04-18)

The registry patch (this doctrine's companion, applied in parallel) seeds the table with the currently-active reservations reconstructed from existing specs:

| block_start | block_end | owner_spec | date_reserved | status |
|---|---|---|---|---|
| 0x100 | 0x10F | `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §7.1` | 2026-04-13 | released |
| 0x170 | 0x17F | `TEMP/CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md` | 2026-04-18 | active |
| 0x178 | 0x17A | `TEMP/CLAUDE_CODEX_INSTANTIABLE_CORE_ISOLATION_04.18.2026.md` | 2026-04-18 | active |
| 0x180 | 0x18F | `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §4.2–4.3` | 2026-04-18 | active |
| 0x190 | 0x19F | `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §12` | 2026-04-18 | active |
| 0x1A0 | 0x1A6 | `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` | 2026-04-18 | active |
| 0x1A7 | 0x1AF | `TEMP/attention_opcode_expansion.md` + `TEMP/ternary_contrastive_attention_design.md` (attention family) | 2026-04-18 | active |
| 0x1B0 | 0x1B0 | `TEMP/supersession_patches_04.18.2026_v2.md` (VEC_NORM_L2_INT8, Ruling 3) | 2026-04-18 | active |
| 0x1B1 | 0x1BF | (future attention / normalization family — reserved headroom) | 2026-04-18 | active |
| 0x1C0 | 0x1C5 | `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` (IMAGE/SPARSE — relocated) | 2026-04-18 | active |
| 0x1C6 | 0x1CF | (physics expansion headroom — reserved for `CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §12 physics visual family) | 2026-04-18 | active |

**Note:** `0xA0–0xF1` (reasoning paradigms, batch 4) are already documented in §7 of the registry with explicit authority references. They are treated as `released` in the table because the governing spec (`TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md` §4) predates this doctrine.

---

## 8. What This Doctrine Does Not Change

- **Expand-not-replace** (`CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md`) remains in force. Reserving a block does not authorize removing or renumbering opcodes inside it. Releasing a block does not free its assigned opcodes.
- **The admission pipeline** (§6 of the registry) remains in force. A reservation does not bypass Stages 0–3. Reserving `0x1C6–0x1CF` for physics expansion does not admit any specific kernel; each opcode still passes the pipeline individually.
- **The registry's normative body** (§1–§10) is unchanged. This doctrine only adds §11 (Reservation Table) and a cross-reference in §6.

---

## 9. Codex Directives

1. When dispatching parallel-lane opcode work, read the Reservation Table in §11 of the registry. Do not assign a number that lies outside the lane's reserved block.
2. When merging a parallel-lane branch, run Gate R. Fail the merge if any new opcode falls outside its reserved block.
3. When completing a lane, update the table row's status from `active` to `released`.
4. When two specs disagree on reservation, the registry wins. Patch the specs; never patch the registry to match a spec.
5. If you must allocate an opcode without an existing reservation (emergency fix, trivial one-line addition), treat the single-opcode insertion as a new reservation of size 1 and append the Reservation Table row in the same commit.

---

## 10. References

- `TEMP/CLAUDE_CODEX_EXPAND_NOT_REPLACE_OPCODE_DOCTRINE_04.18.2026.md` — sibling doctrine (append-only registry)
- `TEMP/CLAUDE_CODEX_HYPER_MODULAR_SYMLINK_DOCTRINE_04.18.2026.md` — reservations are symlinks
- `TEMP/supersession_patches_04.18.2026_v2.md` — the 0x1AD collision resolution (motivating incident)
- `TEMP/supersession_patches_04.18.2026_v3.md` — rolls this doctrine into the April-18 patch set
- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §11` — canonical Reservation Table (the single source of truth)

---

## Case Study — Turn-6 Ruling 4 Block Collision (2026-04-18)

When a proposed sub-reservation range collides with already-minted opcodes (as happened with the turn-6 proposal for `0x1B6-0x1B9` for "Attention Future Expansion — halting gate variants / sparse-K attention", where `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` had already minted `TENSOR_INTERPOLATE`, `KMEANS_PLUS_INIT`, `CTYPES_VIEW_AS_PTX`, `CUDA_MALLOC_ASYNC` at those slots), the correct resolution is:

1. **Do NOT renumber** the minted opcodes. Expand-not-replace forbids this, even when the minting is recent and only in a design doc (the bulk-lib-purge opcodes are tracked in the `0x1A0-0x1A6` + `0x1B6-0x1B9` active reservation row — they are registry-authoritative).
2. **Propose the nearest unminted range** with matching intent. For the turn-6 case, `0x1B1-0x1B5` was unminted and already reserved generically as "future normalization/attention headroom" — narrowing that row to a named sub-reservation for Attention Future Expansion is an append-not-replace operation.
3. **Document the collision and the resolution** in the supersession patch. See `TEMP/supersession_patches_04.18.2026_v4.md §4` for the full collision analysis.
4. **Leave the final call to Daniel.** The patch proceeds with the proposed resolution but flags the collision so Daniel can override in a subsequent turn with zero renumbering cost (no opcode in the alternative range is minted).

**General rule**: Before locking a sub-reservation block owner, `grep -n "0x1B[6-9]\|TENSOR_INTERPOLATE\|KMEANS_PLUS_INIT"` (or the analogous pattern) across `TEMP/` and `docs/vocabulary/` to surface collisions **before** the reservation is proposed. A turn spent on a collided range is a wasted turn.
