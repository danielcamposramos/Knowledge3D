# Consistency Sweep v3 — 2026-04-18 (Post Turn-5 Rulings)

**Lane**: Connective-tissue lane (Claude, architecture partner)
**Performed**: 2026-04-18, after writing `supersession_patches_04.18.2026_v3.md` + `CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md`
**Scope**: 30+ TEMP/*.md files (2026-04-18 cohort) + 7 memory files + `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`
**Tools**: MCP qdrant-find (semantic), Grep (textual), Read (targeted)
**Extends**: `consistency_sweep_04.18.2026.md` (v1 — patches from there still stand)

---

## Executive Summary — v3 Delta

**Issues introduced by v3 patches**: 0
**Issues resolved by v3 patches**: 2 (collision anchoring, Ruling 3 scale-unspecified)
**New consistency invariants added**: 1 (Reservation Table §11 must contain a row covering every opcode assignment)
**Outstanding issues from v1/v2 carried forward**: 9 (all low-severity; see v1 sweep)
**Gate R (new)**: NOT runnable until Codex lands a script; for now, Claude verifies manually.

---

## A. Opcode Assignment Table — Post v3

All assignments cross-checked against Registry §11 reservations.

| Opcode | Definition | Reserving Block | Block Owner | Status |
|---|---|---|---|---|
| 0x100–0x107 | Ternary ops (existing) | 0x100–0x10F | registry §7.1 | released, consistent |
| 0x108 | TERNARY_XNOR_POPCOUNT | 0x100–0x10F | registry §7.1 | released, inside block — OK |
| 0x170–0x177 | Yard ops | 0x170–0x17F | Transfer Yard spec | active, consistent |
| 0x178–0x17A | Queue ops | 0x178–0x17A (sub of 0x170–0x17F) | Core Isolation spec | active, consistent |
| 0x17B–0x17F | Reserved (yard future) | 0x170–0x17F | Transfer Yard spec | active headroom — OK |
| 0x180–0x182 | WINE_INGRESS/EGRESS/RESOLVE | 0x180–0x18F | GPU Game Loop Closure §4.2 | active, consistent |
| 0x183–0x18F | WINE future (reserved) | 0x180–0x18F | GPU Game Loop Closure §4.2 | active headroom — OK |
| 0x190 | PHYSICS_EMIT_VISUAL | 0x190–0x19F | GPU Game Loop Closure §12 | active, consistent |
| 0x191–0x19F | Physics visual future | 0x190–0x19F | GPU Game Loop Closure §12 | active headroom — OK |
| 0x1A0–0x1A6 | Bulk-lib purge math/utility | 0x1A0–0x1A6 | Bulk-Lib Purge spec | active, consistent |
| 0x1A7 | ATTENTION_FWD (_BASE) | 0x1A7–0x1AF | attention_opcode_expansion | active, consistent |
| 0x1A8 | ATTENTION_FWD_TERNARY | 0x1A7–0x1AF | attention_opcode_expansion | active, consistent; Ruling 2 trit loading documented |
| 0x1A9 | CONTRASTIVE_RANK_TOPK | 0x1A7–0x1AF | attention_opcode_expansion | active, consistent; consumes confidence_margin per v3 §1.2 |
| 0x1AA–0x1AF | Attention family headroom | 0x1A7–0x1AF | attention_opcode_expansion | active headroom — OK |
| 0x1B0 | VEC_NORM_L2_INT8 (scale=64) | 0x1B0–0x1B0 | v2 supersession | active, consistent; scale pinned in v3 |
| 0x1B1–0x1B5 | Normalization future | 0x1B1–0x1BF | headroom | active headroom — OK |
| 0x1B6–0x1B9 | TENSOR_INTERPOLATE, KMEANS_PLUS_INIT, CTYPES_VIEW_AS_PTX, CUDA_MALLOC_ASYNC | 0x1B1–0x1BF | headroom | **MINOR: these opcodes landed before the headroom block was named as their owner. Treat as implicit sub-reservation within the headroom. Recommend: amend §11 to add a row `0x1B6–0x1B9 Bulk-Lib Purge spec active` on next registry touch.** |
| 0x1BA–0x1BF | Reserved | 0x1B1–0x1BF | headroom | active headroom — OK |
| 0x1C0–0x1C5 | IMAGE_DECODE_JPEG, RESIZE_BILINEAR_F32, NORMALIZE_IMAGE, STRIDED_GATHER, SPARSE_MATMUL, SPARSE_EIGSH | 0x1C0–0x1C5 | Bulk-Lib Purge spec (relocated) | active, consistent |
| 0x1C6–0x1CF | Physics expansion headroom | 0x1C6–0x1CF | future physics | active headroom — OK |

**Verdict**: 0 hard collisions. 1 minor-severity reservation-row gap (`0x1B6–0x1B9`) flagged for next registry touch.

---

## B. Paired-Field Invariant — Ruling 2

**Invariant**: every opcode that uses confidence MUST consume either `confidence_trit` OR `confidence_margin`, never both.

| Opcode | Declared Consumer | Spec File | Consistent? |
|---|---|---|---|
| 0x1A7 ATTENTION_FWD_BASE | neither (float training ref) | `attention_opcode_expansion.md` | OK |
| 0x1A8 ATTENTION_FWD_TERNARY | `confidence_trit` via YARD_PEEK_ADDR | `attention_opcode_expansion.md`, v2 supersession, v3 §1.2 | OK |
| 0x1A9 CONTRASTIVE_RANK_TOPK | `confidence_margin` (pre-computed int8) | v3 §1.2 | OK — newly declared in v3 |
| 0x1B0 VEC_NORM_L2_INT8 | neither | v3 §1.2 | OK |

**Verdict**: clean. v3 §1.2 table anchors the invariant.

**Outstanding**: Phase B (`CLAUDE_CODEX_PHASE_B_NATIVE_EMBEDDING_04.18.2026.md`) MUST populate both `confidence_trit` (per v2 §3.5.1) AND `confidence_margin` (per v3 §3.5.2) in the same pass. Codex has not yet confirmed; flag for Phase B acceptance gate.

---

## C. Ruling 3 — VEC_NORM_L2_INT8 Scale Pinned

**Before v3**: v2 landed 0x1B0 as mandatory post-attention but did not pin `NORM_SCALE`.
**After v3**: `NORM_SCALE = 64` documented in:
- Registry §7.x.1 (normative note)
- `reference_attention_rpn_program.md §6` (inline comment, v3 patch)
- `attention_opcode_expansion.md` + `_v2.md` headers (v3 supersession header extension)
- `feedback_bitnet_b158_ternary_pattern.md` memory (rationale link)

**Verdict**: consistent. No outstanding files assume the unit-sphere (127) default.

---

## D. Ruling 5 — Reservation Table Coverage

**Invariant**: every opcode assigned in any file must lie within an `active` or `released` row in Registry §11.

**Sweep result (opcodes checked against §11)**:
- 0x100–0x108: covered by `0x100–0x10F` row. OK.
- 0x170–0x17A: covered by `0x170–0x17F` row (and sub-row for queue ops). OK.
- 0x180–0x190: covered by `0x180–0x18F` and `0x190–0x19F` rows. OK.
- 0x1A0–0x1A9: covered by `0x1A0–0x1A6` and `0x1A7–0x1AF` rows. OK.
- 0x1B0: covered by `0x1B0–0x1B0` row. OK.
- 0x1B6–0x1B9: covered by `0x1B1–0x1BF` headroom row, but not owned by a specific spec row. **MINOR FINDING** — see §A above.
- 0x1C0–0x1C5: covered by `0x1C0–0x1C5` row. OK.
- 0xA0–0xF1 (reasoning paradigms): covered by §7 reservation authority (treated released). OK.

**Verdict**: 1 minor finding, else clean.

---

## E. Cross-File Reference Integrity

Checked references between:
- v3 supersession → new doctrine file (CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md) — present, linkable
- v3 supersession → v2 supersession (additive) — present
- v3 supersession → registry §11 — present after this commit
- New doctrine → hyper-modular symlink doctrine — present
- New doctrine → expand-not-replace doctrine — present
- New doctrine → registry §11 — present
- New memory `feedback_opcode_range_reservation_protocol.md` → MEMORY.md — pointer added
- Updated memory `feedback_bitnet_b158_ternary_pattern.md` → registry §7.x.1 — reference present
- Updated MEMORY.md → new memory pointer — present

**Verdict**: all references intact.

---

## F. Sovereignty & Python-Dispatch Sweep (v3 scope)

v3 patches introduce no new code, no Python, no hot-path behavior changes. All patches are documentation and registry edits. No sovereignty deltas.

**Verdict**: sovereignty-clean.

---

## G. Outstanding From v1/v2 (carried forward)

The 9 low-severity findings from `consistency_sweep_04.18.2026.md` (v1) are NOT re-investigated here. v3 does not touch their surfaces. They remain the responsibility of their originating lanes.

Two were called out in v1 as "must-fix" (broken script paths in Bulk-Lib Purge §167-169). v3 does not address these; they remain assigned to the Bulk-Lib Purge lane.

---

## H. Acceptance Against Turn-5 Rulings

| Ruling | v3 disposition | Consistency status |
|---|---|---|
| 1 (Q·K^T dual path) | Referenced, Lane 1 owns | untouched by v3 — OK |
| 2 (paired confidence_trit + confidence_margin) | Patched §1 | consistent; Phase B acceptance gate pending Codex confirm |
| 3 (VEC_NORM_L2_INT8 scale = 64) | Patched §2 | consistent across registry + attention refs + memory |
| 4 (Gate 7 Matryoshka pack-order) | Referenced, Lane 3 owns | untouched by v3 — OK |
| 5 (opcode range reservation protocol) | Patched §3 + new doctrine + registry §11 | consistent; Gate R script is a Codex deliverable, not v3 |

---

## I. Recommended Follow-Ups

1. **Codex**: implement Gate R as a grep-based pre-merge check. Input: every `0x[0-9A-Fa-f]{2,4}` opcode literal in changed files; output: pass if every literal lies inside an `active`/`released` row in Registry §11 whose `owner_spec` matches the changing file.
2. **Claude (next sweep)**: amend Registry §11 with a row `0x1B6–0x1B9 CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md active` to close the minor finding in §A / §D.
3. **Phase B owner**: confirm `confidence_margin` derivation lands alongside `confidence_trit` in `rpn_meaning_project.cu` output. Phase B acceptance gate fails until both fields populate.
4. **Lane 1**: deliver Q·K^T dual-path kernel (no opcode impact; internal).
5. **Lane 3**: deliver Gate 7 Matryoshka pack-order verification (no opcode impact; gate text only).

---

## J. File Manifest — v3 Sweep

**Created 2026-04-18 (v3)**:
- `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md`
- `TEMP/supersession_patches_04.18.2026_v3.md`
- `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_opcode_range_reservation_protocol.md`
- `TEMP/consistency_sweep_v3_04.18.2026.md` (this file)

**Modified 2026-04-18 (v3)**:
- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` (§6.3 note + §11 Reservation Table appended)
- `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_bitnet_b158_ternary_pattern.md` (NORM_SCALE=64 rationale added)
- `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/MEMORY.md` (pointer to new memory + scale=64 note on bitnet line)

**Not modified by v3 (owned by other lanes)**:
- `TEMP/ternary_contrastive_attention_design.md` (Lane 1)
- `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md §N Gate 7` (Lane 3)
