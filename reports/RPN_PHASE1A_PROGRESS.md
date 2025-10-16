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
- ✅ Modular PTX kernel (`modular_rpn_kernel_extended.ptx`) now implements the new opcodes plus a generic pointer literal (`0x03`) so TRM tensors can stay resident in GPU memory.
- ✅ Tier‑3 bridge (`AdvancedRPNEngine`) decodes tensor pointers and can execute the Phase 1A ops directly.
- ✅ TRM launcher now exposes an RPN execution path via `K3D_USE_RPN_TRM` (or the `use_rpn` flag) with GPU parity tests confirming equivalence to the PTX path.
- ✅ GPU regression: `tests/test_trm_rpn_gpu.py` covers `VEC_ADD3`, `MATVEC_512x1024`, `MATVEC_1024x512`, and `SWIGLU_1024` against NumPy references.
- ✅ End-to-end parity: `tests/test_trm_launcher_rpn.py` compares RPN vs PTX refinement outputs for multiple recursion depths.
- ❗ **Benchmark result (2025-10-15):** `tests/benchmarks/test_trm_launcher_performance.py` reports ~10.1 ms per refinement for the PTX backend versus ~503.8 ms for the RPN path (∼50× slower). The discrepancy likely stems from per-step PTX launches being far cheaper than the current Tier‑3 orchestration; optimisation is required before promoting RPN to default.

## Next Steps
1. Benchmark the RPN backend versus PTX kernels to quantify latency improvements and tune tolerances.
2. Profile and optimise the RPN execution path (kernel fusion, reduced pointer literal overhead) until latency is competitive with the PTX baseline.
3. Decide when to make the RPN path the default once performance targets are validated.

Document owner: Codex (2025-10-15).
