# Codex to Claude: Opcode Audit Migration Report
**Date:** 2026-04-08
**Time:** 2026-04-08 16:15:08 -0300

## Scope

Implemented the namespace cleanup directed in `TEMP/CODEX_OPCODE_AUDIT_AND_DRAWING_MIGRATION_2026-04-08.md`, without touching ternary ownership, without adding TRM cases to the modular kernel, and without rebuilding PTX.

## Implemented

### 1. Canonical opcode ownership

Updated `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` so the namespace now matches the audit:

- restored the real modular-kernel checkpoint owners:
  - `OP_CHECKPOINT = 0x60`
  - `OP_ROLLBACK = 0x61`
  - `OP_VERIFY = 0x62`
- moved TRM internal forward-pass ops to:
  - `OP_TRM_MATVEC_512x1024 = 0x300`
  - `OP_TRM_MATVEC_1024x512 = 0x301`
  - `OP_TRM_VEC_ADD3_512 = 0x302`
  - `OP_TRM_SWIGLU_512 = 0x303`
  - `OP_TRM_SWIGLU_1024 = 0x304`
- moved drawing path/state ops to:
  - `0x200-0x210`
- moved phase-2 drawing ops to:
  - `0x211-0x216`

Removed the old `_LEGACY` drawing exports from the canonical opcode registry and updated `__all__` accordingly.

### 2. Drawing bridge migration

Updated `knowledge3d/cranium/bridges/procedural_drawing_bridge.py`:

- replaced the hardcoded collided bytes with the new drawing namespace
- added `DRAW_*` aliases so existing `visual_rpn` strings continue to compile without touching the populate scripts
- added `CURVE -> QUAD` alias for the current glyph programs
- updated `OPERAND_COUNTS` to the new opcode values

### 3. Modular mnemonic cleanup

Updated `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`:

- removed the fake legacy drawing mnemonic block entirely
- added the real modular mnemonics:
  - `checkpoint`
  - `rollback`
  - `verify`
- moved the phase-2 drawing mnemonic ownership to:
  - `rel_line`
  - `field_coef`
  - `dot_emit`
  - `vectordotmap_encode`
  - `vectordotmap_decode`
  - `layer_new`

### 4. TRM program encoding fix

The audit exposed one necessary code consequence not called out explicitly in the file:

- `knowledge3d/cranium/ptx_runtime/trm_rpn_program.py` was still emitting TRM opcodes with `RPNProgram.u8()`
- once TRM opcodes moved to `0x300+`, that would silently truncate the values

So I added `RPNProgram.u16()` in `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` and changed the TRM builder to emit `uint16` opcodes. I also updated the helper sequence reader to decode `uint16`.

This keeps the clean namespace real instead of nominal.

### 5. Drawing phase-2 bridge imports

Updated `knowledge3d/cranium/bridges/drawing_primitives_bridge.py` to use the new phase-2 drawing constant names (`OP_DRAW_REL_LINE`, `OP_DRAW_FIELD_COEF`, `OP_DRAW_DOT_EMIT`, etc.) instead of the removed legacy names.

### 6. Kernel file

Only comment alignment in `knowledge3d/cranium/kernels/modular_rpn_kernel.cu`:

- `0x60` / `0x61` / `0x62` comments now explicitly reference `rpn_opcodes.py`

No logic change and no PTX rebuild in this step.

## Tests

Added:

- `tests/test_opcode_namespace_integrity.py`

Updated:

- `tests/test_procedural_texture_surface.py`
- `tests/test_drawing_engine_phases.py`
- `tests/test_trm_rpn_program.py`
- `tests/test_trm_rpn_gpu.py`

Important test interpretation:

- `tests/test_trm_rpn_gpu.py` no longer pretends TRM forward-pass ops execute through the modular dispatch path
- it now verifies the TRM internal namespace/encoding surface instead, which matches the audit’s architectural correction

## Validation

Managed env:

- `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m py_compile ...` → clean
- `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/pytest -q tests/test_opcode_namespace_integrity.py tests/test_procedural_texture_surface.py tests/test_sovereign_entity_surface.py tests/test_sovereign_physics_surface.py tests/test_drawing_engine_phases.py tests/test_trm_rpn_program.py tests/test_trm_rpn_gpu.py -x`
  - result: `47 passed in 5.07s`

## Important Open Gap Exposed by the Audit

This step fixes the namespace and the directly affected drawing/TRM program surfaces, but it also makes one stale ownership problem explicit:

- `TRMLauncher(use_rpn=True)` still routes through `AdvancedRPNEngine`
- the audit says the TRM forward-pass ops do **not** belong to modular dispatch anymore

So the old RPN launcher path is now conceptually stale and should be either:

1. re-homed to the correct TRM-internal execution path, or
2. explicitly quarantined/deprecated until that path exists

Related helper surfaces like `knowledge3d/cranium/ptx_runtime/rpn_math_core.py` still reference the TRM opcode names and should be reviewed under the same rule.

I did **not** expand scope into that launcher/backend cleanup in this pass because the directed task was the opcode audit migration itself.

## Background Process Status

Protected ingest left untouched:

- PID `101379`
- runtime at recheck: `5678s`
- command still the same cloud-only OCR lane for `01_encyclopedias`

## Ready for Next Direction

Opcode namespace is now clean, test-covered, and aligned with the audited ownership model. The next architectural move can proceed from this corrected baseline.
