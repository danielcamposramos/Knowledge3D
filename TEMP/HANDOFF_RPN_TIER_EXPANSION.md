# RPN Tiered Architecture – Handoff Notes (Phase 2)

## Current State (2025-10-19)

### Tier 1 (Lightweight)
- Bridge implemented in `knowledge3d/cranium/bridges/lightweight_rpn.py`
  - Loads `modular_rpn_kernel_lite.ptx` when CUDA context available
  - Falls back to CPU interpreter for the Tier‑1 opcode subset
- Unit tests `tests/test_rpn_tier1.py` cover arithmetic, math, comparisons, stack ops, and GPU latency (skips when CUDA unavailable)
- **Missing:** actual `modular_rpn_kernel_lite.ptx` trimmed to 20 ops (current file not created yet)

### Tier 2 (Standard)
- Existing `ModularRPNEngine` remains untouched (Tier 2)
- All legacy tests still rely on this engine (252 passing baseline)

### Tier 3 (Advanced)
- Not started yet. Need to activate extended PTX and add bridge/tests.

### Orchestrator
- Not started yet (no `TieredRPNEngine`).

## Next Actions

1. **Finalize Tier 1 PTX**
   - Copy `cranium/ptx/modular_rpn_kernel.ptx` → `cranium/ptx/modular_rpn_kernel_lite.ptx`
   - Remove unused opcode branches, registers, shared-memory usage to target ~10 KB
   - Ensure entry symbol remains `modular_rpn_geometric_kernel`
   - Re-run `tests/test_rpn_tier1.py` with GPU to confirm <1 µs latency

2. **Implement Tier 3 (Advanced)**
   - Activate `cranium/ptx/modular_rpn_kernel_extended.ptx`
     - Add matrix ops (MATMUL/TRACE/DET/INV) and programmability opcodes (BRANCH/JUMP/LOOP/NEXT/STORE/RECALL/CALL/RET)
     - Extend instance state (variables + call stack + loop counters)
   - Create `cranium/bridges/advanced_rpn.py` mirroring the Tier‑2 bridge pattern
   - Add tests `tests/test_rpn_tier3.py` covering matrix math, reductions, programmability, stack extensions

3. **Build Tiered Orchestrator**
   - Implement `cranium/bridges/tiered_rpn.py` that dispatches to Tier 1/2/3 based on opcode analysis
   - Provide helper `execute_matrix` that delegates to Tier 3
   - Add coverage in `tests/test_tiered_rpn.py`

4. **Integration + Benchmarks**
   - Update existing modules (ActionBuffer, ThinkingTag, LED pathfinder) to use `TieredRPNEngine`
   - Re-run full suite to reach ≥280 tests
   - Capture latency benchmarks (<1 µs Tier‑1, ~3 µs Tier‑2, ~10 µs Tier‑3)
   - Verify GPU usage remains <300 MB during tests (`nvidia-smi`)

## Notes & References
- Strategy documents in TEMP:
  - `RPN_KERNEL_STRATEGY_ANALYSIS.md`
  - `RPN_HP50G_EXPANSION_STRATEGY.md`
- Keep Tier‑2 kernel unchanged for backward compatibility
- Ensure each new PTX file is added to repository and referenced by bridges
- CI environments without CUDA should gracefully skip latency assertions (already in Tier‑1 tests)

## Suggested Order for Next Session
1. Finish Tier‑1 PTX trimming + verify
2. Bring up Tier‑3 bridge with minimal matrix ops (MATMUL, TRACE) before programmability
3. Add orchestrator + targeted integration tests
4. Expand Tier‑3 opcodes/programming features iteratively, updating tests each step
