# Consistency Sweep v4 — 2026-04-18

**Date**: 2026-04-18
**Author**: Claude (architecture, connective-tissue lane)
**Scope**: Re-scan `TEMP/*.md` + `docs/vocabulary/*.md` for drift against turn-6 rulings (Path B prefetch mandatory, silent d-mismatch rescale, 0x1A9 default Path A, Attention Future Expansion sub-reservation).
**Companion patch**: `TEMP/supersession_patches_04.18.2026_v4.md`

---

## 1. Sweep Targets

| Target pattern | Purpose | Action |
|---|---|---|
| "optional prefetch" / "recommended prefetch" / "prefetch.*optional" | Ruling 1 v4 made prefetch MANDATORY for Path B. Any remaining "optional" wording is drift. | Patch or flag. |
| "TBD d-mismatch" / "d-mismatch Daniel ruling" / "d-mismatch TBD" | Ruling 2 v4 resolved d-mismatch to silent rescale. Any pending TBD is drift. | Patch or flag. |
| "default margin" / "margin default path" | Ruling 3 v4 locked default to Path A. Ambiguous "default margin" wording outside opcode-flag context is drift. | Verify each instance names Path A. |
| "confidence_trit/margin ambiguity" (both fields in one opcode, or unclear which is consumed) | v3 Ruling 2 requires opcodes to declare which field they consume. | Verify declarations. |
| Opcode range collisions (0x1A9, 0x1AF, 0x1B0, 0x1B1-0x1B5, 0x1B6-0x1B9) | v4 introduces a sub-reservation; must not renumber or overlap. | Verify registry §11. |
| Symlink integrity — every attention-family opcode has a documented source | No orphans; every opcode referenced in a spec has an owner_spec row in §11. | Verify ownership. |

---

## 2. Findings

### 2.1 "optional prefetch" / "recommended prefetch"

Grep: `optional prefetch|recommended prefetch|prefetch.*optional|prefetch.*recommended` (case-insensitive) across `/K3D/GitHub/Knowledge3D/**/*.md`.

| File | Line | Match | Status |
|---|---|---|---|
| `TEMP/supersession_patches_04.18.2026_v4.md` | 30, 83, 408, 436 | "not optional" / "optional → mandatory" | **OK** — v4 patch text itself; describes the supersession. |
| `TEMP/attention_opcode_expansion_v2.md` | 523 | "Priority: P0 … prefetch is structural, not optional" | **OK** — v4 patch landed. |

**Residual drift**: NONE. All remaining mentions of "optional" are in v4's own corrective prose.

**Verdict**: CLEAN.

### 2.2 "TBD d-mismatch" / pending Daniel ruling on d-mismatch

Grep: `TBD.*d.mismatch|d.mismatch.*TBD|d.mismatch.*Daniel|Daniel.*d.mismatch|d.mismatch.*ruling` across `/K3D/GitHub/Knowledge3D/**/*.md`.

All matches are now resolving references to Ruling 2 v4 itself (descriptive, not pending):
- `attention_score_normalization_dual_path_spec_04.18.2026.md §4.5` — Silent d-Mismatch Rescale (resolved).
- `attention_opcode_expansion_v2.md §6` — d-Mismatch Handling (resolved).
- `supersession_patches_04.18.2026_v4.md` — patch text describing the ruling.

**Residual drift**: NONE. No "TBD", "pending", or "Daniel's ruling needed" strings remain on d-mismatch.

**Verdict**: CLEAN.

### 2.3 "default margin" wording

Grep: `default margin|margin.*default path`.

| File | Line | Context | Status |
|---|---|---|---|
| `TEMP/supersession_patches_04.18.2026_v4.md` | 19, 207 | v4 patch header; Ruling 3 body | **OK** — names Path A explicitly. |
| `TEMP/supersession_patches_04.18.2026_v2.md` | 133, 143 | Describes trit values; "trit=0 → m_effective = m_base + m_delta (default margin)" | **OK** — describes trit-based margin computation inside 0x1A8 ATTENTION_FWD_TERNARY (which consumes the raw trit, not the int8 margin field). Different "default margin" — not the path-selection default. No drift. |
| `TEMP/galaxy_confidence_trit_field_spec.md` | 19, 65 | "Default margin `m_base`" / "default margin" | **OK** — describes trit=0 fallback value, not path selection. Not drift. |

**Cross-check**: no file describes 0x1A9's default as "Path B" or leaves the default ambiguous. All references to 0x1A9 default path that exist (v4 patch text) name Path A.

**Verdict**: CLEAN. The "default margin" phrases in v2 and `galaxy_confidence_trit_field_spec.md` describe the trit-zero fallback inside `m_effective = m_base + (1 − trit) × m_delta`, not the path-selection default. No renaming needed; they are orthogonal semantic spaces.

### 2.4 confidence_trit vs confidence_margin disambiguation

Grep: `confidence_trit.*confidence_margin|confidence_margin.*confidence_trit`.

Files: `TEMP/consistency_sweep_v3_04.18.2026.md`, `TEMP/supersession_patches_04.18.2026_v3.md`.

Both describe the paired-field doctrine (Ruling 2 v3): trit is the raw 2-bit field; margin is the derived int8 field. Opcodes must declare which they consume.

- 0x1A8 `ATTENTION_FWD_TERNARY` → consumes trit (per v3 §1.2).
- 0x1A9 `CONTRASTIVE_RANK_TOPK` → consumes margin (per v3 §1.2); v4 extends with `margin_path` flag (Path A = internal shift, no metadata; Path B = smem-prefetched margin).
- 0x1A7 `ATTENTION_FWD` (BASE / float32) → neither.
- 0x1B0 `VEC_NORM_L2_INT8` → neither.
- 0x1AF `ATTENTION_MARGIN_SCALED` → consumes margin (Path B semantics).

**Verdict**: CLEAN. No opcode declares consumption of both. No ambiguous consumer declarations.

### 2.5 Opcode range collisions

Registry §11.2 post-v4 state:

| Range | Owner | Collision Check |
|---|---|---|
| `0x1A7-0x1AF` | attention family | **OK** — includes 0x1A9 (extended), 0x1AF (Ruling 1), 0x1AE. |
| `0x1B0-0x1B0` | VEC_NORM_L2_INT8 (v3) | **KNOWN DIVERGENCE** — see §3 below. |
| `0x1B1-0x1B5` | **v4 sub-reservation**: Attention Future Expansion | Newly minted; no overlap with existing rows. OK. |
| `0x1B6-0x1B9` | bulk-lib purge (minted) | **COLLISION with Daniel's proposed block** — see §3 below. |
| `0x1BA-0x1BF` | future normalization/attention headroom | Narrowed from pre-v4 `0x1B1-0x1BF`. OK (expand-not-replace preserved since no opcode was minted in the withdrawn portion). |

**Verdict**: No overlapping `active` rows. 0x1B1-0x1B5 is newly exclusive to Attention Future Expansion. Gate R would pass.

### 2.6 Symlink integrity — every attention-family opcode has a source

| Opcode | Documented source | Status |
|---|---|---|
| 0x108 | `RPN_DOMAIN_OPCODE_REGISTRY.md §7.1` | OK |
| 0x1A7 | `attention_opcode_expansion.md` + `attention_opcode_expansion_v2.md §0` | OK |
| 0x1A8 | `attention_opcode_expansion.md` + `attention_opcode_expansion_v2.md §0` | OK |
| 0x1A9 | `attention_opcode_expansion.md` + `attention_opcode_expansion_v2.md §0` + `supersession_patches_04.18.2026_v4.md §3` | OK — v4 extension documented |
| 0x1AA | `attention_opcode_expansion_v2.md §1` | OK |
| 0x1AB | `attention_opcode_expansion_v2.md §2` | OK |
| 0x1AC | `attention_opcode_expansion_v2.md §3` | OK |
| 0x1AD | `attention_opcode_expansion_v2.md §4` | **DIVERGENT** — see §3 below |
| 0x1AE | `attention_opcode_expansion_v2.md §5` + `attention_score_normalization_dual_path_spec §6` | OK |
| 0x1AF | `attention_opcode_expansion_v2.md §6` + `supersession_patches_04.18.2026_v4.md §1, §2` | OK |
| 0x1B0 | `supersession_patches_04.18.2026_v2.md` + registry §11 row | **DIVERGENT** with 0x1AD — see §3 |
| 0x1B1-0x1B5 | `supersession_patches_04.18.2026_v4.md §4` | OK — v4 sub-reservation |

**Verdict**: No orphans. One known divergence (0x1AD vs 0x1B0 for VEC_NORM_L2_INT8), pre-existing before v4. Flagged for Daniel.

---

## 3. Known Divergences (Flagged, Not Resolved by v4)

### 3.1 VEC_NORM_L2_INT8 Opcode Number: 0x1AD vs 0x1B0

Two incompatible assignments exist in the tree:

- `TEMP/attention_opcode_expansion_v2.md §4` assigns `VEC_NORM_L2_INT8` to `0x1AD` (at least 15 in-file references).
- `TEMP/supersession_patches_04.18.2026_v2.md` and `v3.md §5 OPCODE SLOT SUMMARY` and `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §11.2` (row for `0x1B0`) assign it to `0x1B0`.

Both refer to the same semantic operation (L2 normalize INT8 to scale 64, per v3 Ruling 3). The `attention_opcode_expansion_v2.md` assignment also conflicts with `0x1AE ATTENTION_MARGIN_SHIFT` if both coexist (0x1AD would be VEC_NORM AND 0x1AE would be MARGIN_SHIFT — but registry §11.2 treats 0x1A7-0x1AF as the attention family, leaving 0x1AD unassigned and 0x1B0 as VEC_NORM's home).

**v4 position**: v4 does NOT resolve this. Both references coexist; kernel implementation (Lane A) must pick one before kernel-emit. Escalated to Daniel for turn-7 ruling. The other opcode number becomes `RESERVED (alias)` per expand-not-replace (no renumbering, just deprecation of one mention path).

**Recommended resolution (non-binding, for Daniel)**: keep `0x1B0` as the canonical assignment (matches registry §11.2 row; matches v3 summary). Patch `attention_opcode_expansion_v2.md §4` to say "0x1B0" and leave a `0x1AD (withdrawn alias — see v3/v4 supersession)` note in the opcode-summary table.

### 3.2 Daniel's Proposed 0x1B6-0x1B9 for Attention Future Expansion

**Collision**: `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md` has already minted:
- `0x1B6` — `TENSOR_INTERPOLATE`
- `0x1B7` — `KMEANS_PLUS_INIT`
- `0x1B8` — `CTYPES_VIEW_AS_PTX`
- `0x1B9` — `CUDA_MALLOC_ASYNC`

Per expand-not-replace, these cannot be renumbered. v4 sub-reserves `0x1B1-0x1B5` as the nearest unminted range with matching intent ("future normalization/attention headroom" → tightened to "Attention Future Expansion").

**v4 position**: proceeded with `0x1B1-0x1B5`. The collision and alternative-range choice are documented in `supersession_patches_04.18.2026_v4.md §4.1` and in the new case-study section of `CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md`. If Daniel intended a different range, v5 can re-sub-reserve at zero renumber cost (no opcode in `0x1B1-0x1B5` is minted).

---

## 4. Files Modified by v4 — Verified Landed

| File | Patch site | Verified |
|---|---|---|
| `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` | §11.2 row split + §11.4 normative notes appended | YES |
| `TEMP/attention_score_normalization_dual_path_spec_04.18.2026.md` | §4 prefetch mandatory, §4.5 silent rescale, §5.0 default path, §4 Cons bullet | YES |
| `TEMP/attention_opcode_expansion_v2.md` | §0 (0x1A9 extended), §6 (0x1AF priority + prefetch rule + cycle-cost branch removed + d-mismatch appendix) | YES |
| `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md` | case-study appended | YES |
| `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_attention_margin_dual_path_rulings.md` | new file | YES |
| `/home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/MEMORY.md` | pointer added under Expand-Not-Replace/Attention block | YES |
| `TEMP/supersession_patches_04.18.2026_v4.md` | new file | YES |

---

## 5. Acceptance Summary

| Target | Status |
|---|---|
| No "optional prefetch" wording remains in attention-family specs | CLEAN |
| No "TBD d-mismatch" remains | CLEAN |
| 0x1A9 default path documented as Path A in all three normative files (dual-path spec, v2 opcode spec, registry §11.4) | CLEAN |
| confidence_trit / confidence_margin disambiguation preserved, no opcode consumes both | CLEAN |
| No overlapping `active` reservation rows in registry §11.2 | CLEAN |
| Every attention-family opcode has a documented source file | CLEAN (with 1 known divergence on 0x1AD/0x1B0, pre-v4) |
| 0x1B1-0x1B5 is v4's Attention Future Expansion sub-reservation | LANDED |
| 0x1B6-0x1B9 collision with Daniel's proposed range is documented and alternative proposed | FLAGGED |

**Collisions introduced by v4**: ZERO.
**Pre-existing divergences surfaced**: 1 (VEC_NORM_L2_INT8 at 0x1AD vs 0x1B0).
**Pending Daniel input**: 2 (confirm 0x1B1-0x1B5 for Attention Future Expansion; resolve 0x1AD/0x1B0 for VEC_NORM_L2_INT8).

---

## 6. References

- `TEMP/supersession_patches_04.18.2026_v4.md`
- `TEMP/supersession_patches_04.18.2026_v3.md`
- `TEMP/consistency_sweep_v3_04.18.2026.md`
- `TEMP/attention_score_normalization_dual_path_spec_04.18.2026.md`
- `TEMP/attention_opcode_expansion_v2.md`
- `TEMP/CLAUDE_CODEX_OPCODE_RANGE_RESERVATION_DOCTRINE_04.18.2026.md`
- `TEMP/CLAUDE_CODEX_BULK_LIB_PURGE_HARD_ACCEPTANCE_04.18.2026.md`
- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §11`
