# Step 14 — Nine-Chain Swarm Prototype

**Date:** 2025-10-16  
**Author:** Codex (GPU PTX lane)  
**Status:** Prototype complete ✅

---

## Summary

- Implemented `nine_chain_swarm_kernel.cu`, a CUDA proof-of-concept that runs nine reasoning chains in parallel, exchanges resonance information, and aggregates a synthesis output.
- Added the Python bridge `nine_chain_swarm_bridge.py` providing `execute_swarm()` and diagnostic helpers for integration experiments.
- Created GPU-marked tests/benchmarks covering execution, resonance sanity, adaptation behaviour, latency targets, and ThinkingTag hand-off scenarios.
- Documentation updates record environment quirks (tmux + `CUDA_VISIBLE_DEVICES=0`) from Step 13-E and outline remaining Step 14 tasks.

---

## Prototype Architecture

| Chain ID | Role (prototype)             | Notes                                                        |
|----------|------------------------------|--------------------------------------------------------------|
| 0        | Ingest                       | Copies input embedding                                        |
| 1-2      | Fuse A/B                     | Currently use same tanh transform (unique logic pending)      |
| 3-5      | Spatial A/B/C                | Parallel copies; specialisation deferred                      |
| 6        | Reductionist reasoning       | Deterministic adaptation                                      |
| 7        | Creative reasoning           | Deterministic adaptation                                      |
| 8        | Synthesis                    | Weighted average using resonance-derived weights              |

**Resonance computation:** per-chain dot products (`NUM_CHAINS × 64`) with mean aggregation.  
**Adaptation:** low resonance (<0.8) blends towards swarm consensus (10% rate).  
**Synthesis:** normalised resonance weights to mix all chain states (prevents outliers dominating).

---

## Performance Snapshot

Benchmark suite is ready under GPU markers; run inside `tmux` + `k3d-cranium` with `CUDA_VISIBLE_DEVICES=0` to collect hardware numbers.

| Metric (num_iterations=3) | Placeholder | Target | Notes |
|---------------------------|-------------|--------|-------|
| Swarm latency (µs)        | _TBD on GPU_ | <95    | Warm-up + 500-run mean (`test_step14_swarm_performance.py::test_swarm_latency_budget`) |
| Iteration scaling (1→5)   | _TBD on GPU_ | ≈linear | Observed via `test_swarm_iteration_scaling`                                    |
| Parallel efficiency est.  | _TBD on GPU_ | Informational | `test_swarm_parallel_efficiency_estimate`                                   |

Local (CPU-fallback) smoke runs show much higher timings; ignore those in official reports.

---

## Test Coverage

- `tests/test_step14_swarm_prototype.py` — execution/resonance/adaptation diagnostics.
- `tests/benchmarks/test_step14_swarm_performance.py` — latency + scaling benchmarks.
- `tests/test_step14_thinkingtag_integration.py` — demonstrates swarm as a ThinkingTag reasoning surface.

All tests are marked `@pytest.mark.gpu`; they skip automatically on non-GPU CI until run in the sovereign environment.

---

## Next Steps Toward Full Step 14

1. **Chain-specialised logic:** replace tanh placeholders with modality-aware handlers (INGEST ↔ FUSE ↔ SPATIAL ↔ REASON lanes).
2. **Richer communication:** add pheromone/message buffers using Step 13-E matrix ops (`OP_MATMUL_SMALL`, `OP_DOT_BATCH`) and programmable flow.
3. **Program counter extensions:** upgrade `OP_BRANCH/LOOP` to support true jumps for adaptive control policies.
4. **ThinkingTag FSM integration:** embed the swarm in the production reasoning stage, wiring latency guards and telemetry.
5. **Monitoring & introspection:** expose real-time chain health, resonance trends, and diversity metrics via Tablet UX.

---

## How to Reproduce

```bash
# Compile kernel
cd knowledge3d/cranium/kernels
nvcc -ptx -arch=sm_86 -O3 nine_chain_swarm_kernel.cu -o ../ptx/nine_chain_swarm_kernel.ptx

# GPU session (tmux + conda env)
export CUDA_VISIBLE_DEVICES=0
tmux new -As k3d && conda activate k3d-cranium

# Run prototype tests
pytest tests/test_step14_swarm_prototype.py -v
pytest tests/test_step14_thinkingtag_integration.py -v
pytest tests/benchmarks/test_step14_swarm_performance.py -vs
```

---

## Notes

- Prototype intentionally deterministic; future iterations can introduce stochasticity for creative chains.
- No global barrier exists between chains; identical workloads keep them in lockstep for now. Production version may use multi-kernel phases if divergence is introduced.
- Keep `knowledge3d.cranium.ptx/nine_chain_swarm_kernel.ptx` in sync when modifying the kernel.
