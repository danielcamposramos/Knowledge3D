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
- ✅ **Fused kernel success (2025-10-15):** `trm_step_fused.ptx` drives the full TRM step in a single launch; `tests/test_trm_fused_parity.py` matches the PTX baseline and the benchmark reports **9.29 ms** per refinement (vs. 10.39 ms PTX).
- ❗ **RPN still slow:** `tests/benchmarks/test_trm_launcher_performance.py` shows the current Tier‑3 interpreter at **≈504 ms** per refinement (~50× slower). Timing breakdown after pre-built programs: build/update ≈0.6 ms (0.1%), memcpy ≈0.15 ms, **execution ≈500 ms (99.8%)**. The bottleneck is squarely in the Tier‑3 interpreter loop.

## Next Steps
1. Implement the pre-built RPN program path (no per-step list building) and re-benchmark.
2. Profile the Tier‑3 interpreter to identify remaining hotspots (pointer literal packing, switch dispatch) and optimise until RPN < 50 ms.
3. Decide when to make the RPN or fused path the default once targets are met.

Document owner: Codex (2025-10-15).
