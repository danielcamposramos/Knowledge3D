# Codex Directions: Opcode Audit + Drawing Migration
**Date:** 2026-04-08
**Priority:** P0 — architectural correctness; run before Step 4

---

## Problem Statement

The Step 1 "fix" renamed drawing constants with `*_LEGACY` suffix but kept the **same byte values**. This is a fake fix. The raw byte conflicts still exist in `procedural_drawing_bridge.py` (which hardcodes raw values) and in `trm_rpn_program.py` (which uses wrong byte values for TRM ops).

### Confirmed conflicts (from audit)

| Byte | `rpn_opcodes.py` says | Kernel does | `procedural_drawing_bridge.py` uses |
|------|----------------------|-------------|--------------------------------------|
| 0x60 | OP_TRM_MATVEC_512x1024 | CHECKPOINT | — |
| 0x61 | OP_TRM_MATVEC_1024x512 | ROLLBACK | — |
| 0x62 | OP_TRM_VEC_ADD3_512 | VERIFY | — |
| 0x63 | OP_TRM_SWIGLU_512 | (no case) | — |
| 0x64 | OP_TRM_SWIGLU_1024 + OP_DRAW_MOVE_LEGACY | (no case) | MOVE (active!) |
| 0x70 | OP_TADD + OP_DRAW_PUSH_STATE_LEGACY | TADD (live) | PUSH_STATE (active!) |
| 0x71 | OP_TMUL + OP_DRAW_POP_STATE_LEGACY | TMUL (live) | POP_STATE (active!) |
| 0x72 | OP_TNOT + OP_DRAW_TRANSLATE_LEGACY | TNOT (live) | TRANSLATE (active!) |
| 0x73 | OP_TCOMP + OP_DRAW_ROTATE_LEGACY | TCOMP (live) | ROTATE (active!) |
| 0x74 | OP_TQUANT + OP_DRAW_SCALE_LEGACY | TQUANT (live) | SCALE (active!) |
| 0x75 | OP_TPACK + OP_DRAW_SET_STROKE_COLOR_LEGACY | TPACK (live) | SET_STROKE_COLOR (active!) |
| 0x76 | OP_TUNPACK + OP_DRAW_SET_FILL_COLOR_LEGACY | TUNPACK (live) | SET_FILL_COLOR (active!) |

**Drawing ops at 0x64-0x78 are ACTIVELY USED** by:
- `knowledge3d/cranium/bridges/procedural_drawing_bridge.py` — glyph/path rendering
- `knowledge3d/cranium/action_primitives_bootstrap.py` — ARC action visual_rpn atoms
- `scripts/populate_math_symbols.py` — math symbol glyphs (∑, ∏, ∫, etc.)

**TRM ops (0x60-0x64) are WRONG** — they collide with CHECKPOINT/ROLLBACK/VERIFY (0x60-0x62) that the kernel actually implements, and the TRM operations run via `trm_step_fused.ptx`, not via the modular dispatch.

**OP_CHECKPOINT/ROLLBACK/VERIFY are MISSING from rpn_opcodes.py** — the kernel has them at 0x60-0x62 but there are no Python constants for them.

---

## Resolution

### Rule: what moves, what stays

| Group | Action |
|-------|--------|
| Ternary ops (0x70-0x76): TADD, TMUL, TNOT, TCOMP, TQUANT, TPACK, TUNPACK | **STAY** — sovereign GPU, live kernel cases |
| Checkpoint/Rollback/Verify (0x60-0x62) in kernel | **STAY** — add missing constants to rpn_opcodes.py |
| TRM forward-pass ops (OP_TRM_MATVEC, OP_TRM_SWIGLU) | **MOVE to 0x300-0x304** — these are TRM-internal ops, not modular dispatch |
| Drawing path + state ops (0x64-0x78) | **MOVE to 0x200-0x210** — dedicated drawing range, no conflicts |
| Phase-2 drawing ops that conflict with ternary (REL_LINE=0x70, FIELD_COEF=0x71, DOT_EMIT=0x72, VECTORDOTMAP 0x73-0x74) | **MOVE to 0x211-0x215** |
| LAYER_NEW (0x78, conflicts with DRAW_SET_TERNARY_HINT) | **MOVE to 0x216** |
| BEZIER_EVAL (0x6C), SHAPE_UNION/INTERSECT/SUBTRACT (0x6D-0x6F) | **STAY** — clean range, no conflict |

---

## New Opcode Map

### Drawing path primitives — 0x200-0x207 (was 0x64-0x6B)
```
0x200 = OP_DRAW_MOVE              (was OP_DRAW_MOVE_LEGACY = 0x64)
0x201 = OP_DRAW_LINE              (was OP_DRAW_LINE_LEGACY = 0x65)
0x202 = OP_DRAW_QUAD              (was OP_DRAW_QUAD_LEGACY = 0x66)
0x203 = OP_DRAW_CUBIC             (was OP_DRAW_CUBIC_LEGACY = 0x67)
0x204 = OP_DRAW_ARC               (was OP_DRAW_ARC_LEGACY = 0x68)
0x205 = OP_DRAW_CLOSE             (was OP_DRAW_CLOSE_LEGACY = 0x69)
0x206 = OP_DRAW_STROKE            (was OP_DRAW_STROKE_LEGACY = 0x6A)
0x207 = OP_DRAW_FILL              (was OP_DRAW_FILL_LEGACY = 0x6B)
```

### Drawing state — 0x208-0x210 (was 0x70-0x78)
```
0x208 = OP_DRAW_PUSH_STATE        (was OP_DRAW_PUSH_STATE_LEGACY = 0x70)
0x209 = OP_DRAW_POP_STATE         (was OP_DRAW_POP_STATE_LEGACY = 0x71)
0x20A = OP_DRAW_TRANSLATE         (was OP_DRAW_TRANSLATE_LEGACY = 0x72)
0x20B = OP_DRAW_ROTATE            (was OP_DRAW_ROTATE_LEGACY = 0x73)
0x20C = OP_DRAW_SCALE             (was OP_DRAW_SCALE_LEGACY = 0x74)
0x20D = OP_DRAW_SET_STROKE_COLOR  (was OP_DRAW_SET_STROKE_COLOR_LEGACY = 0x75)
0x20E = OP_DRAW_SET_FILL_COLOR    (was OP_DRAW_SET_FILL_COLOR_LEGACY = 0x76)
0x20F = OP_DRAW_SET_LINE_WIDTH    (was OP_DRAW_SET_LINE_WIDTH_LEGACY = 0x77)
0x210 = OP_DRAW_SET_TERNARY_HINT  (was OP_DRAW_SET_TERNARY_HINT_LEGACY = 0x78)
```

### Drawing Phase 2 — 0x211-0x215 (was conflicting 0x70-0x74)
```
0x211 = OP_DRAW_REL_LINE              (was OP_REL_LINE_LEGACY = 0x70, conflicted with OP_TADD)
0x212 = OP_DRAW_FIELD_COEF            (was OP_FIELD_COEF_LEGACY = 0x71, conflicted with OP_TMUL)
0x213 = OP_DRAW_DOT_EMIT              (was OP_DOT_EMIT = 0x72, conflicted with OP_TNOT)
0x214 = OP_DRAW_VECTORDOTMAP_ENCODE   (was OP_VECTORDOTMAP_ENCODE = 0x73, conflicted with OP_TCOMP)
0x215 = OP_DRAW_VECTORDOTMAP_DECODE   (was OP_VECTORDOTMAP_DECODE = 0x74, conflicted with OP_TQUANT)
0x216 = OP_DRAW_LAYER_NEW             (was OP_LAYER_NEW = 0x78, conflicted with OP_DRAW_SET_TERNARY_HINT)
```

### New constants for existing kernel ops — 0x60-0x62
```
OP_CHECKPOINT = 0x60   (kernel case already live, missing from rpn_opcodes.py)
OP_ROLLBACK   = 0x61   (kernel case already live, missing from rpn_opcodes.py)
OP_VERIFY     = 0x62   (kernel case already live, missing from rpn_opcodes.py)
```

### TRM forward-pass ops moved — 0x300-0x304 (was 0x60-0x64)
```
0x300 = OP_TRM_MATVEC_512x1024  (was 0x60, collided with OP_CHECKPOINT)
0x301 = OP_TRM_MATVEC_1024x512  (was 0x61, collided with OP_ROLLBACK)
0x302 = OP_TRM_VEC_ADD3_512     (was 0x62, collided with OP_VERIFY)
0x303 = OP_TRM_SWIGLU_512       (was 0x63)
0x304 = OP_TRM_SWIGLU_1024      (was 0x64, collided with OP_DRAW_MOVE)
```
**Note:** TRM ops are dispatched by `trm_step_fused.ptx`, NOT via the modular RPN dispatch switch.
Do NOT add case 0x300-0x304 to `modular_rpn_kernel.cu` in this step. The value change ensures
future wiring lands in a clean range.

---

## Files to Modify

### 1. `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`

Changes:
- Replace the `# Phase 1A – TRM integration opcodes` block: change 0x60-0x64 to 0x300-0x304, keep names
- Add OP_CHECKPOINT/ROLLBACK/VERIFY = 0x60/0x61/0x62 after the TRM block (new section: `# Sovereign stack checkpoint ops`)
- Replace the `# Legacy procedural drawing primitives` block (0x64-0x6B, 0x70-0x78): change values to 0x200-0x210, remove `_LEGACY` suffix from constant names (they are the real drawing ops, not legacy)
- Replace Phase 2 conflicting drawing ops: REL_LINE_LEGACY→OP_DRAW_REL_LINE=0x211, FIELD_COEF_LEGACY→OP_DRAW_FIELD_COEF=0x212, DOT_EMIT→OP_DRAW_DOT_EMIT=0x213, VECTORDOTMAP_ENCODE→OP_DRAW_VECTORDOTMAP_ENCODE=0x214, VECTORDOTMAP_DECODE→OP_DRAW_VECTORDOTMAP_DECODE=0x215
- Replace OP_LAYER_NEW=0x78 with OP_DRAW_LAYER_NEW=0x216
- Remove the `# CONFLICT` comment lines (conflicts are now resolved)
- Update `__all__` to match: remove old _LEGACY names, add new names, add OP_CHECKPOINT/ROLLBACK/VERIFY

### 2. `knowledge3d/cranium/bridges/procedural_drawing_bridge.py`

The `OPCODES` dict (line ~865) hardcodes raw byte values. Update ALL of them:

```python
OPCODES = {
    "MOVE":             0x200,   # was 0x64
    "LINE":             0x201,   # was 0x65
    "QUAD":             0x202,   # was 0x66
    "CUBIC":            0x203,   # was 0x67
    "ARC":              0x204,   # was 0x68
    "CLOSE":            0x205,   # was 0x69
    "STROKE":           0x206,   # was 0x6A
    "FILL":             0x207,   # was 0x6B
    "PUSH_STATE":       0x208,   # was 0x70
    "POP_STATE":        0x209,   # was 0x71
    "TRANSLATE":        0x20A,   # was 0x72
    "ROTATE":           0x20B,   # was 0x73
    "SCALE":            0x20C,   # was 0x74
    "SET_STROKE_COLOR": 0x20D,   # was 0x75
    "SET_COLOR":        0x20D,   # alias for SET_STROKE_COLOR
    "SET_FILL_COLOR":   0x20E,   # was 0x76
    "SET_LINE_WIDTH":   0x20F,   # was 0x77
    "STROKE_WIDTH":     0x20F,   # alias for SET_LINE_WIDTH
}
```

Also add `DRAW_MOVE`, `DRAW_LINE`, `DRAW_STROKE`, `DRAW_ARC`, `DRAW_QUAD`, `DRAW_CUBIC`, `DRAW_CLOSE`, `DRAW_FILL` as aliases (with `DRAW_` prefix) pointing to the same values. These aliases make the visual_rpn strings from `action_primitives_bootstrap.py` and `populate_math_symbols.py` (which use `DRAW_MOVE`, `DRAW_LINE` etc.) compile correctly through this executor:

```python
    # DRAW_* aliases — match mnemonics in visual_rpn strings
    "DRAW_MOVE":             0x200,
    "DRAW_LINE":             0x201,
    "DRAW_QUAD":             0x202,
    "DRAW_CUBIC":            0x203,
    "DRAW_ARC":              0x204,
    "DRAW_CLOSE":            0x205,
    "DRAW_STROKE":           0x206,
    "DRAW_FILL":             0x207,
    "CURVE":                 0x202,   # CURVE = QUAD in populate_math_symbols.py usage
```

Update `OPERAND_COUNTS` dict to use new byte values (replace all 0x64→0x200, 0x65→0x201, ..., 0x70→0x208, etc.)

### 3. `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`

The drawing mnemonic entries (`MOVE_LEGACY`, `LINE_LEGACY`, etc.) at lines 257-293 do NOT belong in the modular engine. Remove them entirely:
- Remove lines for: MOVE_LEGACY through SET_TERNARY_HINT_LEGACY (and their lowercase aliases)
- These programs go through `procedural_drawing_bridge.py`, not the modular kernel

Update the Phase 2 conflicting entries in the mnemonic table:
```python
"rel_line":              0x211,   # was OP_REL_LINE_LEGACY = 0x70
"field_coef":            0x212,   # was OP_FIELD_COEF_LEGACY = 0x71
"dot_emit":              0x213,   # was OP_DOT_EMIT = 0x72
"vectordotmap_encode":   0x214,   # was OP_VECTORDOTMAP_ENCODE = 0x73
"vectordotmap_decode":   0x215,   # was OP_VECTORDOTMAP_DECODE = 0x74
"layer_new":             0x216,   # was OP_LAYER_NEW = 0x78
```

Add ternary op mnemonics using the ACTUAL symbolic name (not numeric):
```python
"checkpoint": OP_CHECKPOINT,   # 0x60
"rollback":   OP_ROLLBACK,     # 0x61
"verify":     OP_VERIFY,       # 0x62
```

### 4. `knowledge3d/cranium/ptx_runtime/trm_rpn_program.py`

Update imports at top of file:
```python
from .rpn_opcodes import (
    OP_TRM_MATVEC_512x1024,   # now 0x300
    OP_TRM_MATVEC_1024x512,   # now 0x301
    OP_TRM_VEC_ADD3_512,      # now 0x302
    OP_TRM_SWIGLU_1024,       # now 0x304
)
```
No logic changes needed — the constant names stay the same, only their values change.
Update the module docstring: change "0x60–0x64" to "0x300–0x304".

### 5. `knowledge3d/cranium/sovereign/trm_launcher.py` and `knowledge3d/cranium/sovereign_trm.py`

Search for any usage of `OP_TRM_MATVEC`, `OP_TRM_SWIGLU`, `OP_TRM_VEC_ADD3` and update imports.
The constant names stay the same — Python-side code using these constants needs no logic changes.

### 6. `knowledge3d/cranium/kernels/modular_rpn_kernel.cu`

**NO LOGIC CHANGES** in this step. The kernel is already correct:
- 0x60/0x61/0x62 = CHECKPOINT/ROLLBACK/VERIFY — stay as-is
- 0x70-0x76 = TADD through TUNPACK — stay as-is
- Drawing ops never went through this kernel anyway

Only add comment alignment above the 0x60-0x62 cases:
```c
case 0x60: {  // OP_CHECKPOINT — see rpn_opcodes.py
```

**Do NOT rebuild PTX in this step.** The kernel logic is unchanged.

---

## What NOT to do

- **Do NOT** touch ternary op values (0x70-0x76) — they are sovereign GPU hot path, live in kernel
- **Do NOT** add TRM dispatch cases (0x300-0x304) to `modular_rpn_kernel.cu` — TRM runs via trm_step_fused.ptx, a separate execution path
- **Do NOT** modify `action_primitives_bootstrap.py` or `populate_math_symbols.py` — they use text mnemonics which will now resolve correctly via the DRAW_* alias additions to the drawing bridge
- **Do NOT** change `BEZIER_EVAL` (0x6C), `SHAPE_UNION` (0x6D), `SHAPE_INTERSECT` (0x6E), `SHAPE_SUBTRACT` (0x6F) — they are clean (no conflicts)
- **Do NOT** attempt to address the broader 0x80-0x88 conflicts (Phase 5 3D vs Phase 2 bitwise logic) — that is a deferred audit task

---

## Deferred: 0x80-0x88 Audit Note

The following also have conflicting assignments but are NOT in scope for this step:

| Value | Conflict |
|-------|---------|
| 0x80 | OP_AND vs OP_NURBS_EVAL |
| 0x81 | OP_OR vs OP_MARCHING_CUBES |
| 0x82 | OP_XOR vs OP_LSYSTEM_GENERATE |
| 0x83 | OP_NOT vs OP_PARAMETRIC_SURFACE |
| 0x84 | OP_GEN_TORUS vs OP_CSG_UNION_3D |
| 0x85 | OP_GEN_ICOSPHERE vs OP_CSG_INTERSECT_3D |
| 0x86 | OP_CSG_UNION vs OP_CSG_SUBTRACT_3D |
| 0x87 | OP_CSG_SUBTRACT vs OP_CROSS_MODAL_LINK |
| 0x88 | OP_CSG_INTERSECT vs OP_PROCEDURAL_TEXTURE |

These will need a follow-up audit to determine which are live in each executor.

---

## Tests

Add `tests/test_opcode_namespace_integrity.py`:

```python
"""Verify the opcode namespace has no cross-domain byte conflicts."""
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op


def test_drawing_ops_in_dedicated_range():
    draw_ops = [
        op.OP_DRAW_MOVE, op.OP_DRAW_LINE, op.OP_DRAW_QUAD, op.OP_DRAW_CUBIC,
        op.OP_DRAW_ARC, op.OP_DRAW_CLOSE, op.OP_DRAW_STROKE, op.OP_DRAW_FILL,
        op.OP_DRAW_PUSH_STATE, op.OP_DRAW_POP_STATE, op.OP_DRAW_TRANSLATE,
        op.OP_DRAW_ROTATE, op.OP_DRAW_SCALE, op.OP_DRAW_SET_STROKE_COLOR,
        op.OP_DRAW_SET_FILL_COLOR, op.OP_DRAW_SET_LINE_WIDTH,
        op.OP_DRAW_SET_TERNARY_HINT,
    ]
    for v in draw_ops:
        assert 0x200 <= v <= 0x21F, f"Drawing op {hex(v)} outside dedicated range 0x200-0x21F"


def test_ternary_ops_in_dedicated_range():
    ternary = [op.OP_TADD, op.OP_TMUL, op.OP_TNOT, op.OP_TCOMP, op.OP_TQUANT, op.OP_TPACK, op.OP_TUNPACK]
    for v in ternary:
        assert 0x70 <= v <= 0x76, f"Ternary op {hex(v)} out of range 0x70-0x76"


def test_trm_ops_in_dedicated_range():
    trm = [op.OP_TRM_MATVEC_512x1024, op.OP_TRM_MATVEC_1024x512, op.OP_TRM_VEC_ADD3_512,
           op.OP_TRM_SWIGLU_512, op.OP_TRM_SWIGLU_1024]
    for v in trm:
        assert 0x300 <= v <= 0x30F, f"TRM op {hex(v)} outside dedicated range 0x300-0x30F"


def test_checkpoint_constants_exist():
    assert op.OP_CHECKPOINT == 0x60
    assert op.OP_ROLLBACK == 0x61
    assert op.OP_VERIFY == 0x62


def test_no_cross_domain_byte_conflicts():
    drawing = set(range(0x200, 0x220))
    ternary = {0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76}
    trm_internal = set(range(0x300, 0x310))
    checkpoint = {0x60, 0x61, 0x62}

    assert not drawing & ternary
    assert not drawing & trm_internal
    assert not drawing & checkpoint
    assert not ternary & trm_internal
    assert not ternary & checkpoint
```

Run:
```bash
pytest tests/test_opcode_namespace_integrity.py tests/test_procedural_texture_surface.py tests/test_sovereign_entity_surface.py tests/test_sovereign_physics_surface.py -x -q
```

All tests must pass. No PTX rebuild unless `modular_rpn_kernel.cu` changes (it should not).

---

## Handoff

After this step passes, report back for **Step 4 (2D Physics)**. That step is
architecturally self-contained and does not require further opcode work.
