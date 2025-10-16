# RPN Phase 1A Progress – TRM Opcode Integration (2025-10-15)

## Objective
Lay the groundwork for executing the TRM refinement loop through Tier‑3 RPN opcodes (`0x60`–`0x64`) before the CUDA/PTX implementation lands. This ensures downstream modules can compile programs and the swarm can coordinate pointer layouts and tests ahead of the kernel work.

## Deliverables
- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` – Centralised definition of the new TRM opcodes alongside existing sparse operations.
- `knowledge3d/cranium/ptx_runtime/trm_rpn_program.py` – Generates bytecode for the recursive TRM loop using the new opcodes; returns helper templates for unit inspection.
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` – Pointer literals inside `RPNProgram` are now resolved once, preventing stale placeholders when the byte array is re-used.
- `tests/test_trm_rpn_program.py` – Validates the opcode schedule, error handling for invalid step counts, and pointer relocation semantics.
- `knowledge3d/cranium/ptx_runtime/__init__.py` – Exposes the TRM builders via package imports and relaxes optional import guards to cope with environments lacking `cuda-python`.

## Current Status
- ✅ Opcode constants mirrored in Python runtime (PTX additions pending).
- ✅ Deterministic TRM bytecode generator in place; per-step pattern covered by tests.
- ✅ Pointer relocation tested, matching future GPU expectations.
- ⏳ Modular PTX kernel (`modular_rpn_kernel_extended.ptx`) still needs handlers for opcodes `0x60`–`0x64`.
- ⏳ Tier‑3 bridge (`AdvancedRPNEngine`) will require bindings for the new literals once PTX support lands.
- ⏳ TRM launcher to be extended with an RPN execution path after kernel support arrives.

## Next Steps
1. Implement Phase 1A opcodes inside `modular_rpn_kernel_extended.cu` and regenerate PTX.
2. Extend `AdvancedRPNEngine` to decode the new vector/matrix literals and route the TRM matvec ops.
3. Add GPU-backed tests that execute the generated TRM bytecode against small dummy weights, comparing against the existing PTX TRM launcher.
4. Wire the new RPN program into `TRMLauncher` behind a feature flag (`K3D_USE_RPN_TRM`).

Document owner: Codex (2025-10-15).
