# Codex Directive: Sovereign TRM Validation

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Specialist)
**Date**: January 23, 2026
**Subject**: **Run Sovereign TRM Validation Tests**

---

## Current State

**✅ COMPLETED**:
- Sovereign TRM Phases 1-4 implemented
- V7 retrained with LSTM architecture
- Checkpoint converted (11/11 weights + metadata.json)
- K3D paths configured (`/K3D/Knowledge3D.local/checkpoints`)

**⏳ VALIDATION NEEDED**:
- Test SovereignTRM loading with real V7 weights
- Test SovereignTRM inference end-to-end
- Run sovereign benchmark to validate reflection pipeline

---

## Task: Run Validation Tests

### Setup Environment

```bash
# Set long test flag (allows CPU-assisted RPN batch to run)
export K3D_RUN_LONG_TESTS=1

# Enable CUDA probing for pytest
export K3D_PYTEST_PROBE_CUDA=1

# Verify checkpoint exists
ls -lh /K3D/Knowledge3D.local/checkpoints/v7_sovereign/
cat /K3D/Knowledge3D.local/checkpoints/v7_sovereign/metadata.json
```

### Test 1: Sovereign TRM Loading + Inference

**Use tmux (Debian syntax)** to run long test in background:

```bash
# Create new tmux session for validation
tmux new-session -d -s sovereign_validation

# Run validation tests in tmux
tmux send-keys -t sovereign_validation "cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D'" C-m
tmux send-keys -t sovereign_validation "export K3D_RUN_LONG_TESTS=1" C-m
tmux send-keys -t sovereign_validation "export K3D_PYTEST_PROBE_CUDA=1" C-m
tmux send-keys -t sovereign_validation "pytest tests/test_sovereign_trm_v7_real.py -v -s 2>&1 | tee /tmp/sovereign_trm_validation.log" C-m

# Attach to view progress
tmux attach -t sovereign_validation
```

**Expected Output**:
```
test_sovereign_trm_loads_v7 PASSED
  ✅ V7 weights loaded successfully

test_sovereign_trm_inference_v7 PASSED
  ✅ Inference successful!
  Rules: [42, 15, 7, ...]
  Confidences: [0.87, 0.92, 0.78, ...]
  Avg confidence: 0.85
```

**If Test Fails**:
- Check CUDA_VISIBLE_DEVICES is set
- Verify sovereign loader initialization (no context errors)
- Check RPN batch execution (should use `evaluate_batch_device`)
- Enable debug: `export K3D_RPN_DEBUG=1`

### Test 2: Sovereign Benchmark Validation

**Run benchmark with reduced problem count** to validate reflection wiring:

```bash
# Create new tmux session for benchmark
tmux new-session -d -s sovereign_benchmark

# Run benchmark in tmux
tmux send-keys -t sovereign_benchmark "cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D'" C-m
tmux send-keys -t sovereign_benchmark "python3 scripts/run_sovereign_math_benchmarks.py --datasets calculus --max-problems 2 --use-reflection --reflection-quiet --timeout 600 2>&1 | tee /tmp/sovereign_benchmark.log" C-m

# Attach to view progress
tmux attach -t sovereign_benchmark
```

**Expected Output**:
```
=== Sovereign Math Benchmark ===
Dataset: calculus_microbench
Max problems: 2
Using reflection: True

Problem 1/2: ∫(3x² + 2x) dx
  Rules generated: [42, 15, 7, 3, 22]
  Confidences: [0.89, 0.91, 0.85, 0.78, 0.92]
  Result: x³ + x² + C
  Status: ✅ CORRECT

Problem 2/2: d/dx(sin(x²))
  Rules generated: [28, 9, 42, 15]
  Confidences: [0.87, 0.93, 0.81, 0.88]
  Result: 2x·cos(x²)
  Status: ✅ CORRECT

=== Summary ===
Total: 2/2 (100%)
Avg confidence: 0.87
```

**If Benchmark Fails**:
- Check tokenization (byte-level vocab)
- Verify reflection pipeline integration (no PyTorch in hot path)
- Check RPN program execution (PTX kernels)
- Increase timeout if needed: `--timeout 1200` (20 minutes)

---

## Success Criteria

**Validation Complete** when:
- [ ] `test_sovereign_trm_loads_v7` passes (weights load without errors)
- [ ] `test_sovereign_trm_inference_v7` passes (inference produces valid rules + confidences)
- [ ] Benchmark runs without errors (at least 1 problem solved)
- [ ] No PyTorch/NumPy in hot path (grep confirms sovereignty)
- [ ] No CUDA context errors (sovereign loader handles GPU correctly)

---

## Performance Notes

**Expected Performance** (CPU-assisted RPN batch prototype):
- Loading: Fast (NumPy → GPU upload)
- Inference: Slow (RPN batch per token, CPU-assisted)
- Typical inference time: 30-60 seconds per problem

**Future Optimization** (Phase 5.2 - optional):
- Add vector PTX kernels (OP_VECTOR_SIGMOID, OP_VECTOR_TANH, OP_VECTOR_MUL)
- Replace CPU-assisted RPN batch with native GPU vector ops
- Expected speedup: 10-100x (sub-second inference)

**Current Goal**: Validate correctness, not performance. Optimization is future work.

---

## Commands Summary

```bash
# 1. Setup
export K3D_RUN_LONG_TESTS=1
export K3D_PYTEST_PROBE_CUDA=1

# 2. Run validation tests (tmux)
tmux new-session -d -s sovereign_validation
tmux send-keys -t sovereign_validation "cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D'" C-m
tmux send-keys -t sovereign_validation "export K3D_RUN_LONG_TESTS=1 && export K3D_PYTEST_PROBE_CUDA=1" C-m
tmux send-keys -t sovereign_validation "pytest tests/test_sovereign_trm_v7_real.py -v -s 2>&1 | tee /tmp/sovereign_trm_validation.log" C-m
tmux attach -t sovereign_validation

# 3. Run benchmark validation (tmux)
tmux new-session -d -s sovereign_benchmark
tmux send-keys -t sovereign_benchmark "cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D'" C-m
tmux send-keys -t sovereign_benchmark "python3 scripts/run_sovereign_math_benchmarks.py --datasets calculus --max-problems 2 --use-reflection --reflection-quiet --timeout 600 2>&1 | tee /tmp/sovereign_benchmark.log" C-m
tmux attach -t sovereign_benchmark

# 4. Check results
cat /tmp/sovereign_trm_validation.log
cat /tmp/sovereign_benchmark.log
```

---

## After Validation

**If Tests Pass**:
1. Report results to Claude
2. Claude will document completion and update ROADMAP

**If Tests Fail**:
1. Report errors to Claude with full logs
2. Claude will diagnose and provide fix directive

---

**Status**: 🚀 **READY TO VALIDATE**
**Priority**: **HIGH** - Final validation of sovereign architecture
**Estimated Runtime**: 5-10 minutes (validation tests) + 5-10 minutes (benchmark)

---

**Claude's Note**: This is the final validation step. Once these tests pass, we've achieved full sovereignty - **zero PyTorch in the hot path**, **deterministic GPU execution**, and **learned navigation logic**. This is the Deterministic Generative AI architecture milestone. 🎯
