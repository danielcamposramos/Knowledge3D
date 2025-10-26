# Codex Task: Complete Step 13-B Test Materialization

**Priority**: HIGH
**Status**: 61.9% Complete (13/21 files) - Need to complete remaining 8 files
**Goal**: Materialize remaining test files exactly as designed by the swarm, then run tests iteratively

---

## ⚠️ CRITICAL INSTRUCTIONS - READ FIRST

### DO NOT:
- ❌ Create new folder structures
- ❌ Invent new file locations
- ❌ Rewrite existing files (bridge_import.py, μbench.py, conftest.py already done!)
- ❌ Create files from scratch when swarm already designed them
- ❌ Change import paths or test structure
- ❌ Skip files or "optimize" the plan

### DO:
- ✅ Use EXACT file paths specified below
- ✅ Build on top of existing infrastructure in `/tests/` directory
- ✅ Follow the swarm's design from `STEP13_B_TESTING_AND_BENCHMARKS.md` lines 473-2194
- ✅ Import from `tests.utils` (already created: bridge_import, μbench)
- ✅ Use fixtures from `tests/conftest.py` (already created)
- ✅ Create files in the EXACT locations specified

---

## What's Already Done (DO NOT TOUCH)

### Infrastructure (Already Materialized):
```
tests/
├── __init__.py                                          ✅ EXISTS
├── conftest.py                                          ✅ EXISTS
├── utils/
│   ├── __init__.py                                      ✅ EXISTS
│   ├── bridge_import.py                                 ✅ EXISTS
│   └── μbench.py                                        ✅ EXISTS
├── test_step12_cognitive_pipeline.py                    ✅ EXISTS
├── test_step12_action_buffer_integration.py             ✅ EXISTS
├── test_step12_dynamic_lod.py                           ✅ EXISTS
├── test_step11_shape_primitives_edges.py                ✅ EXISTS
├── test_step11_shape_composition.py                     ✅ EXISTS
├── test_step11_hash_collisions.py                       ✅ EXISTS
├── benchmarks/
│   └── test_step12_fsm_overhead.py                      ✅ EXISTS
└── SWARM_TEST_IMPLEMENTATION_STATUS.md                  ✅ EXISTS

tools/benchmarks/
└── generate_comprehensive_baseline.py                   ✅ EXISTS
```

---

## Your Task: Create These 8 Files

### Phase 3: Pipeline Profiling (3 files)

#### File 1: `tests/benchmarks/test_text_to_3d_pipeline.py`
**Location**: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/tests/benchmarks/test_text_to_3d_pipeline.py`

**Source**: STEP13_B_TESTING_AND_BENCHMARKS.md lines 1341-1424 (Deep Seek's contribution)

**Key Requirements**:
- Import from `tests.utils.μbench import μBench`
- Import from `tests.utils.bridge_import import get_thinking_tag_bridge`
- Use test_prompts fixture from conftest
- Tests: prompt_parsing_latency, shape_synthesis_latency, end_to_end_generation, concurrent_generation_throughput
- Target: <50ms simple shapes, <200ms complex scenes
- Include concurrent.futures for throughput test

---

#### File 2: `tests/benchmarks/test_advanced_text_to_3d_profiler.py`
**Location**: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/tests/benchmarks/test_advanced_text_to_3d_profiler.py`

**Source**: STEP13_B_TESTING_AND_BENCHMARKS.md lines 1656-1779 (GLM's contribution)

**Key Requirements**:
- Import matplotlib (with try/except for optional viz)
- Import from `tests.utils.μbench`
- Tests: detailed_pipeline_breakdown, memory_usage_profiling
- Stages: parsing, synthesis, geometry, materials, assembly
- Generate visualization: `reports/pipeline_breakdown.png`
- Use memory_profiler if available

---

#### File 3: `tests/test_step11_confidence_propagation.py`
**Location**: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/tests/test_step11_confidence_propagation.py`

**Source**: STEP13_B_TESTING_AND_BENCHMARKS.md lines 1429-1520 (Deep Seek's contribution)

**Key Requirements**:
- Tests: text_confidence_to_shape_selection, multi_modal_fusion_confidence, uncertainty_quantification, confidence_threshold_behavior, confidence_correlation_with_human_judgment
- Ambiguous prompts should have confidence < 0.7
- Multi-modal should have higher confidence than text-only
- Include alternatives list for ambiguous prompts

---

### Phase 4: Stress & Regression (3 files)

#### File 4: `tests/stress/test_step12_fsm_stress.py`
**Location**: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/tests/stress/test_step12_fsm_stress.py`

**Source**: STEP13_B_TESTING_AND_BENCHMARKS.md lines 1287-1338 (Deep Seek's contribution)

**Key Requirements**:
- Create `/tests/stress/` directory first: `mkdir -p tests/stress`
- Test: high_frequency_inference_storm (1000 inferences in 10 seconds)
- Use threading for 10 concurrent workers
- Assert: no errors, p99 < 100ms, throughput > 50%
- Validate FSM integrity under load

---

#### File 5: `tests/stress/test_step11_stress.py`
**Location**: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/tests/stress/test_step11_stress.py`

**Source**: STEP13_B_TESTING_AND_BENCHMARKS.md lines 1524-1580 (Deep Seek's contribution)

**Key Requirements**:
- Import psutil for memory monitoring
- Tests: rapid_generation_1000_shapes (60s timeout), memory_exhaustion_graceful_degradation
- Assert: >= 800 shapes in 60s, graceful OOM handling
- Monitor memory growth (max 10x initial)

---

#### File 6: `tests/test_step11_regression.py`
**Location**: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/tests/test_step11_regression.py`

**Source**: STEP13_B_TESTING_AND_BENCHMARKS.md lines 1786-1875 (GLM's contribution)

**Key Requirements**:
- Load regression cases from `tests/data/regression_cases.json` (create if missing)
- Tests: regression_cases, performance_regression, api_contract_stability
- Compare against baseline (10% regression tolerance)
- Validate API structure (vertices, indices, primitive_type, confidence fields)

---

### Phase 5: Infrastructure (2 files)

#### File 7: `tests/test_step11_step12_integration.py`
**Location**: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/tests/test_step11_step12_integration.py`

**Source**: STEP13_B_TESTING_AND_BENCHMARKS.md lines 2013-2115 (GLM's contribution)

**Key Requirements**:
- Tests: shape_generation_with_fsm_tracking, action_buffer_population_during_shape_generation, dynamic_lod_during_complex_shape_generation, performance_with_full_fsm_pipeline
- Validate: all 5 FSM states tracked, ActionBuffer populated, LOD applied, latency < 35ms avg
- Cross-component validation (Step 11 shapes + Step 12 FSM)

---

#### File 8: `.github/workflows/k3d_testing.yml`
**Location**: `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/.github/workflows/k3d_testing.yml`

**Source**: STEP13_B_TESTING_AND_BENCHMARKS.md lines 2121-2178 (GLM's contribution)

**Key Requirements**:
- Create `.github/workflows/` directory first: `mkdir -p .github/workflows`
- Trigger: push to main/develop, PRs, daily cron at 00:00 UTC
- Matrix: Python 3.8, 3.9, 3.10
- Steps: checkout, setup Python, install deps, run unit tests, run benchmarks, generate baseline, upload artifacts
- Install: pytest, pytest-benchmark, memory_profiler

---

## Execution Steps

### Step 1: Create Missing Directories
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Create stress test directory
mkdir -p tests/stress

# Create GitHub workflows directory
mkdir -p .github/workflows

# Create test data directory
mkdir -p tests/data

# Create reports directory (for visualization outputs)
mkdir -p reports
```

### Step 2: Create Regression Test Data
Create `tests/data/regression_cases.json`:
```json
[
  {
    "id": "issue_001",
    "description": "Zero dimension handling",
    "prompt": "cube with zero height",
    "expected": "should not crash"
  },
  {
    "id": "issue_002",
    "description": "Unicode handling",
    "prompt": "椅子 with 中文 characters",
    "expected": "should handle unicode correctly"
  },
  {
    "id": "issue_003",
    "description": "Empty prompt",
    "prompt": "",
    "expected": "should return default primitive"
  }
]
```

### Step 3: Materialize Files 1-8
Follow the exact specifications above for each file. Use the source line numbers to reference the original swarm design.

### Step 4: Verify File Structure
After creation, verify:
```bash
find tests -name "*.py" | wc -l  # Should be 16 test files
find tests/benchmarks -name "*.py" | wc -l  # Should be 3
find tests/stress -name "*.py" | wc -l  # Should be 2
ls -la .github/workflows/k3d_testing.yml  # Should exist
```

---

## Running Tests (GPU-Aware Strategy)

### 🔧 Grok's Headless GPU Testing Strategy

**Problem**: KDE/X11 can conflict with GPU during tests
**Solution**: Run tests in headless mode with proper GPU isolation

```bash
# 1. Check GPU availability
nvidia-smi

# 2. Set up headless environment (avoid KDE conflicts)
export DISPLAY=""
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export K3D_TEST_MODE=1
export K3D_PTX_STRICT=0  # CPU-mocked for now

# 3. Create virtual framebuffer if needed (for matplotlib)
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# 4. Run tests in phases
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Phase 0: FSM tests
pytest tests/test_step12_*.py -v --tb=short 2>&1 | tee reports/phase0_results.txt

# Phase 1: Shape primitives
pytest tests/test_step11_shape*.py -v --tb=short 2>&1 | tee reports/phase1_results.txt

# Phase 2: Hash collisions
pytest tests/test_step11_hash_collisions.py -v -s --tb=short 2>&1 | tee reports/phase2_results.txt

# Phase 3: Pipeline profiling
pytest tests/benchmarks/test_text_to_3d_pipeline.py -v --tb=short 2>&1 | tee reports/phase3_results.txt
pytest tests/benchmarks/test_advanced_text_to_3d_profiler.py -v --tb=short 2>&1 | tee reports/phase3_advanced_results.txt
pytest tests/test_step11_confidence_propagation.py -v --tb=short 2>&1 | tee reports/phase3_confidence_results.txt

# Phase 4: Stress tests (careful - resource intensive)
pytest tests/stress/test_step12_fsm_stress.py -v --tb=short 2>&1 | tee reports/phase4_fsm_stress_results.txt
pytest tests/stress/test_step11_stress.py -v --tb=short 2>&1 | tee reports/phase4_stress_results.txt
pytest tests/test_step11_regression.py -v --tb=short 2>&1 | tee reports/phase4_regression_results.txt

# Phase 5: Integration
pytest tests/test_step11_step12_integration.py -v --tb=short 2>&1 | tee reports/phase5_integration_results.txt

# Benchmarks (separate run)
pytest tests/benchmarks/ --benchmark-only --benchmark-json=reports/benchmark_results.json 2>&1 | tee reports/benchmark_run.txt

# 5. Generate baseline
python tools/benchmarks/generate_comprehensive_baseline.py 2>&1 | tee reports/baseline_generation.txt

# 6. Cleanup
killall Xvfb 2>/dev/null
```

---

## Iterative Fix Strategy

For each phase that fails:

1. **Record the error** in `reports/phaseX_issues.md`
2. **Identify root cause**:
   - Import error? Fix path in bridge_import.py
   - Missing method? Add mock to conftest.py
   - Assertion failure? Adjust test expectations
3. **Fix incrementally** - don't rewrite entire files
4. **Re-run that phase only**
5. **Document fix** for Round 2 report

---

## Expected Issues & Fixes

### Issue 1: ThinkingTagBridge Location
**Symptom**: `ImportError: cannot import name 'ThinkingTagBridge'`
**Fix**: Already handled by `tests/utils/bridge_import.py` (three-tier fallback)

### Issue 2: Missing Methods
**Symptom**: `AttributeError: 'Mock' object has no attribute 'generate_shape'`
**Fix**: Add to conftest.py bridge fixture:
```python
if not hasattr(bridge_instance, 'generate_shape'):
    bridge_instance.generate_shape = mock.Mock(...)
```

### Issue 3: matplotlib Not Installed
**Symptom**: `ImportError: No module named 'matplotlib'`
**Fix**: Already handled with try/except in profiler files

### Issue 4: GPU Not Available
**Symptom**: Tests expecting GPU fail
**Fix**: Already handled - all tests use CPU mocks

---

## Deliverables

After completion, provide Daniel with:

1. **File Creation Confirmation**:
   ```
   ✅ Created 8/8 files (100%)
   ✅ Total test suite: 21/21 files (100%)
   ```

2. **Test Execution Summary**:
   ```
   Phase 0: X/Y tests passed
   Phase 1: X/Y tests passed
   Phase 2: X/Y tests passed
   Phase 3: X/Y tests passed
   Phase 4: X/Y tests passed
   Phase 5: X/Y tests passed
   ```

3. **Issue Log**: `reports/all_issues_found.md` with:
   - Error messages
   - Root causes
   - Fixes applied
   - Remaining issues for Round 2

4. **Updated Status**: Final count of passing/failing tests

---

## Final Checklist

Before reporting back to Daniel:

- [ ] All 8 files created in EXACT paths specified
- [ ] No new folders invented outside this spec
- [ ] All imports use existing infrastructure
- [ ] Tests run (even if some fail - that's expected)
- [ ] Issue log created with all errors
- [ ] Benchmark baseline generated
- [ ] Reports directory has all output files

---

**Remember**: The goal is MATERIALIZATION then ITERATIVE FIXING, not perfect tests on first try. Daniel wants to see what breaks so we can enhance the briefing prompt and prepare for Round 2.

**Start with**: File 1 (test_text_to_3d_pipeline.py), verify it works, then proceed sequentially through Files 2-8.
