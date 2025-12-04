# Hybrid TRM+K3D Architecture — Adaptive Parallel Breadth + Sequential Depth

**Date:** December 4, 2025
**Author:** Claude (Architecture) + Daniel (User Vision)
**Status:** Design Specification (Implementation Pending)

---

## Executive Summary

**Problem:** Some ARC tasks plateau at high-but-not-perfect scores (85-94%), suggesting they need **sequential refinement** rather than more parallel exploration.

**Solution:** Hybrid architecture combining:
- **K3D's parallel breadth** (9 workers × 6 candidates = fast exploration)
- **TRM's sequential depth** (3 workers × 21 refinements = deep reasoning)
- **Adaptive routing** (only use deep workers if quick workers <95% after N attempts)

**Expected Benefit:**
- Easy/medium tasks: Solved by fast parallel (no extra cost)
- Hard tasks: Benefit from deep sequential refinement (targeted compute spend)
- Overall: Better accuracy with minimal overhead (adaptive gating)

---

## Daniel Directives (Sovereign Memory + Ternary + Symlinked Base)

- **Galaxy-first memory**: Latents and pattern lookups must come from Galaxy buffers (House-backed, tablet-loaded) instead of Python accumulators. Shadow/semantic contexts stay in VRAM, addressed by IDs; no JSON duplication.
- **Ternary ops where it counts**: Use SIGN/TQUANT/TCMP for gating (confidence, plateau detection, routing decisions) and for discrete pattern selection; keep magnitudes in binary where needed. Target: ternary router + ternary gating inside deep workers.
- **Math cores are cheap**: Spawn Tier-1/2/3 math cores per task or per worker as needed; reclaim after use. Favor lightweight per-task instances instead of global locks.
- **Sovereign loader**: Load kernels via `SovereignLoader` (one context per kernel) and reuse loaded modules; avoid ad-hoc CUDA contexts. Respect existing loader pattern to prevent context thrash.
- **Symlinked construction**: All new artifacts (shadow entries, refinements, vocab hits) reference existing House/Galaxy items (shape IDs, rule IDs). No duplication; use references/links to teach the system “compose, don’t copy.”
---

## Current K3D Architecture (Parallel Breadth)

### Worker Configuration

```python
# ParallelCandidateGenerator (current)
num_workers = 9                    # Parallel workers
candidates_per_worker = 6          # Candidates per worker
total_candidates = 54              # Per task attempt

# Strategy: Broad parallel exploration
# - Generate 54 candidates in parallel
# - Rank by PTX cosine similarity
# - Select top-k (k=3)
# - Fast but shallow (single generation step)
```

### Strengths
✅ **Fast:** 54 candidates generated simultaneously
✅ **Broad coverage:** Explores diverse solution space
✅ **PTX-native:** Zero Python overhead
✅ **Good for easy/medium tasks:** Most tasks solvable with breadth

### Weaknesses
⚠️ **Shallow:** Single generation step per candidate
⚠️ **No refinement:** Can't improve near-miss solutions
⚠️ **Plateau effect:** Tasks stuck at 85-94% need deeper reasoning

**Example plateau tasks (hypothetical):**
- Task gets 8/9 cells correct → 88.9% accuracy
- Quick parallel can't find the last cell
- Sequential refinement could iteratively improve the 88.9% solution

---

## Proposed Hybrid Architecture (Parallel + Sequential)

### Two-Tier Worker Pool

**Tier 1: Quick Parallel Workers (K3D Standard)**
- **Count:** 9 workers
- **Strategy:** Parallel breadth-first
- **Candidates:** 6 per worker = 54 total
- **Depth:** 1 generation step (shallow)
- **Use case:** Fast solve for easy/medium tasks

**Tier 2: Deep Sequential Workers (TRM-Style)**
- **Count:** 3 workers
- **Strategy:** Sequential depth-first refinement
- **Candidates:** 1 initial + 20 refinements = 21 per worker
- **Depth:** 7 recursions × 3 cycles = 21 steps (deep)
- **Use case:** Hard tasks needing iterative improvement

**Total candidates per task:**
- Quick path: 54 (if Tier 1 solves it)
- Deep path: 54 + 63 = 117 (if Tier 2 activated)

### Adaptive Routing Logic

```python
def generate_candidates_hybrid(task, train_examples, expected_output):
    """
    Hybrid TRM+K3D candidate generation with adaptive depth.

    Strategy:
    1. Try quick parallel workers (Tier 1)
    2. Check if best candidate ≥ 95% accuracy
    3. If yes: Return (task solved, no deep search needed)
    4. If no: Activate deep sequential workers (Tier 2)
    5. Combine and rank all candidates
    """

    # Phase 1: Quick parallel exploration (K3D standard)
    quick_candidates = parallel_generate(
        num_workers=9,
        candidates_per_worker=6,
        depth=1  # Single generation step
    )

    # Early stopping: Check if quick solve achieved
    best_quick = rank_candidates(quick_candidates)[0]
    quick_score = evaluate_candidate(best_quick, expected_output)

    if quick_score >= 0.95:
        # Task solved quickly, no need for deep search
        log(f"[QUICK SOLVE] Task solved with {quick_score:.1%} accuracy")
        return quick_candidates

    # Phase 2: Deep sequential refinement (TRM-style)
    log(f"[DEEP SEARCH] Quick solve {quick_score:.1%} < 95%, activating deep workers")

    deep_candidates = sequential_refine(
        num_workers=3,
        refinements_per_worker=21,  # TRM's n=6, T=3 → 7×3=21 steps
        initial_candidates=top_k(quick_candidates, k=3)  # Seed from quick
    )

    # Combine and re-rank
    all_candidates = quick_candidates + deep_candidates
    return rank_candidates(all_candidates)
```

### Tesla Alignment (6-3-9 Pattern)

**Quick Workers (Tier 1):**
- **6** candidates per worker
- **3** top candidates selected
- **9** workers total
- Pattern: 6 → 3 → 9 (Tesla alignment maintained)

**Deep Workers (Tier 2):**
- **6** latent recursions per cycle (n=6, TRM standard)
- **3** cycles per refinement (T=3, TRM standard)
- **9** total refinements per worker (3 workers × 3 refinements)
- Pattern: 6 → 3 → 9 (Tesla alignment extended to depth)

**Total System:**
- Quick: 9 × 6 = 54 candidates (breadth)
- Deep: 3 × 21 = 63 refinements (depth)
- Combined: 117 candidates (hybrid)
- Tesla pattern maintained at both breadth and depth levels

---

## TRM Sequential Refinement Implementation

### Recursion Structure (From Paper)

**TRM Standard (n=6, T=3):**

```python
def trm_sequential_refine(x, y, z, n=6, T=3):
    """
    TRM-style sequential refinement.

    Args:
        x: Input (embedded question)
        y: Current answer (embedded solution)
        z: Latent reasoning feature
        n: Number of latent recursions per cycle
        T: Number of cycles

    Returns:
        Refined answer y after T cycles of n recursions
    """

    # T-1 cycles WITHOUT gradients (improve latent)
    for cycle in range(T - 1):
        for i in range(n):
            z = net(x, y, z)  # Update latent reasoning
        y = net(y, z)  # Update answer
        z = z.detach()  # Detach for next cycle

    # Final cycle WITH gradients (backprop through)
    for i in range(n):
        z = net(x, y, z)  # Update latent reasoning
    y = net(y, z)  # Update answer

    return y, z
```

**K3D Adaptation:**

```python
def k3d_sequential_refine(
    input_grid,
    initial_candidate,
    shadow_copy,
    drawing_galaxy,
    galaxy_resonator,  # Galaxy-backed latent fetch (House→Galaxy, tablet path)
    sovereign_loader,  # Reuse loaded PTX modules; no extra contexts
    n=6,  # Latent recursions per cycle
    T=3   # Cycles per refinement
):
    """
    K3D-adapted TRM sequential refinement.

    Instead of pure latent z, we use:
    - z → Shadow Copy discoveries (latent reasoning memory)
    - y → Current candidate grid (solution)
    - x → Input grid + train examples (question)

    Refinement strategy:
    1. Start with best quick candidate as initial y
    2. Recursively improve using shadow patterns (z) fetched from Galaxy
    3. Each cycle: n latent updates + 1 answer update
    4. Return refined answer after T cycles
    """

    x = embed_input(input_grid)
    y = embed_candidate(initial_candidate)
    z = galaxy_resonator.fetch_latents(shadow_copy.ids())  # Galaxy-backed latent memory (no Python copies)

    # T-1 cycles without gradients (efficient exploration)
    for cycle in range(T - 1):
        # n latent recursions (explore shadow patterns)
        for i in range(n):
            z = apply_shadow_patterns(x, y, z, drawing_galaxy, ternary_gate=True)  # SIGN/TCMP gating

        # Update answer using refined latent
        y = refine_candidate(y, z)

    # Final cycle with gradients (learning signal)
    for i in range(n):
        z = apply_shadow_patterns(x, y, z, drawing_galaxy, ternary_gate=True)

    y = refine_candidate(y, z)

    return decode_candidate(y), z
```

**Key K3D Adaptations:**
1. **Latent z → Shadow Copy:** Use discovered patterns as reasoning memory
2. **Answer y → Candidate Grid:** Iteratively refine grid solution
3. **Input x → Task Context:** Input grid + train examples
4. **Refinement → Pattern Application:** Apply shadow patterns to improve candidate

### Effective Depth Calculation

**Per Deep Worker:**
- n latent updates = 6
- Answer updates per cycle = 1
- Total ops per cycle = 6 + 1 = 7
- Cycles = 3
- **Total refinement steps = 7 × 3 = 21 per worker**

**All Deep Workers:**
- Workers = 3
- Refinements per worker = 21
- **Total deep ops = 3 × 21 = 63**

**Combined System:**
- Quick candidates: 54 (Tier 1)
- Deep refinements: 63 (Tier 2, only if needed)
- **Total: 54-117 candidates** (adaptive)

---

## Adaptive Gating Strategy

### When to Activate Deep Workers

**Criterion 1: Quick Score Threshold**
```python
if best_quick_score < 0.95:
    activate_deep_workers()
```

**Criterion 2: Plateau Detection**
```python
if task_accuracy_history[-3:] == [0.88, 0.89, 0.88]:
    # Stuck at ~88% for 3 epochs
    activate_deep_workers()
```

**Criterion 3: Task Difficulty Heuristic**
```python
if task.grid_size > 20 and task.pattern_complexity > 0.7:
    activate_deep_workers()
```

**Criterion 4: Shadow Copy Confidence**
```python
if shadow_copy.get_confidence(task_pattern) < 0.80:
    # Shadow Copy not confident about this pattern
    activate_deep_workers()
```

### When to Skip Deep Workers

**Fast Solve (95%+ accuracy):**
- Quick parallel already solved it
- No need for expensive deep search
- **Time saved:** ~70% (skip 63 deep refinements)

**Example:**
```
Task 001: Quick = 98% → SKIP deep workers (time saved)
Task 002: Quick = 67% → ACTIVATE deep workers (needs depth)
Task 003: Quick = 92% → ACTIVATE deep workers (close but not quite)
Task 004: Quick = 96% → SKIP deep workers (good enough)
```

### Expected Activation Rate

**Hypothesis:**
- Easy tasks (30%): Quick ≥ 95% → Skip deep (30% time saved)
- Medium tasks (50%): Quick 70-94% → Use deep (50% use deep)
- Hard tasks (20%): Quick < 70% → Use deep (20% use deep)

**Total deep activation rate: ~70% of tasks**
**Time overhead: 70% × 117/54 = 1.52× compute** (vs always using deep)

**Compare to always-deep:**
- Always-deep: 2.17× compute (117/54 for all tasks)
- Adaptive: 1.52× compute (117/54 for 70% of tasks)
- **Savings: 30% compute** vs always-deep

---

## Task Difficulty Analysis (Post-Run 037)

### Analysis Script

```python
def analyze_task_difficulty(run_log):
    """
    Categorize tasks by score pattern to identify which need deep refinement.

    Categories:
    1. Easy: Score ≥ 95% consistently (quick parallel sufficient)
    2. Plateau High: Score 85-94% (close but needs refinement)
    3. Plateau Medium: Score 70-84% (needs depth)
    4. Hard: Score < 70% (needs more exploration or impossible)
    """

    tasks = {}

    for line in run_log:
        if "Task" in line and "Accuracy" in line:
            task_id = extract_task_id(line)
            accuracy = extract_accuracy(line)
            epoch = extract_epoch(line)

            if task_id not in tasks:
                tasks[task_id] = []

            tasks[task_id].append((epoch, accuracy))

    # Categorize tasks
    easy = []
    plateau_high = []
    plateau_medium = []
    hard = []

    for task_id, history in tasks.items():
        final_accuracy = history[-1][1]
        max_accuracy = max(h[1] for h in history)

        if max_accuracy >= 0.95:
            easy.append(task_id)
        elif max_accuracy >= 0.85:
            plateau_high.append(task_id)
        elif max_accuracy >= 0.70:
            plateau_medium.append(task_id)
        else:
            hard.append(task_id)

    return {
        "easy": easy,
        "plateau_high": plateau_high,
        "plateau_medium": plateau_medium,
        "hard": hard
    }
```

### Expected Patterns

**Easy Tasks (Quick Parallel Sufficient):**
- Accuracy ≥ 95% within first 20 epochs
- Stable trajectory (no plateau)
- **Example:** Simple color fill, basic flip/rotate
- **Action:** Skip deep workers (time saved)

**Plateau High Tasks (Need Sequential Refinement):**
- Accuracy stuck at 85-94%
- Multiple near-miss attempts
- **Example:** Complex pattern with subtle rule
- **Action:** Activate deep workers (likely to improve)

**Plateau Medium Tasks (Need More Depth):**
- Accuracy stuck at 70-84%
- Some pattern recognition but incomplete
- **Example:** Multi-step transformation
- **Action:** Activate deep workers (may improve)

**Hard Tasks (Fundamental Difficulty):**
- Accuracy < 70% throughout training
- No clear pattern found
- **Example:** Novel concept not in training
- **Action:** Activate deep workers (low success chance but worth trying)

### Post-Run 037 Analysis Commands

```bash
# Extract per-task accuracy trajectories
grep -E "Task.*Accuracy" /K3D/Knowledge3D.local/logs/run_037.log | \
  awk '{print $2, $4, $6}' > task_trajectories.txt

# Categorize tasks
python scripts/analyze_task_difficulty.py task_trajectories.txt

# Output:
# Easy (≥95%): 32 tasks (30%)
# Plateau High (85-94%): 28 tasks (26%)
# Plateau Medium (70-84%): 26 tasks (24%)
# Hard (<70%): 22 tasks (20%)
#
# Recommendation: Activate deep workers for 76 tasks (70%)
# Expected accuracy gain: +3-5% on plateau tasks
```

### Identifying Deep-Worker Candidates

**High-Priority Tasks (Plateau High):**
```bash
# Tasks stuck at 85-94% (most likely to benefit)
grep "Plateau High" task_analysis.txt | head -n 10

# Example output:
# Task abc123: 88.9% (8/9 cells correct, needs 1 more)
# Task def456: 91.7% (11/12 cells correct, close!)
# Task ghi789: 85.0% (17/20 cells correct, pattern almost there)
```

**Medium-Priority Tasks (Plateau Medium):**
```bash
# Tasks stuck at 70-84% (may benefit)
grep "Plateau Medium" task_analysis.txt | head -n 10

# Example output:
# Task jkl012: 75.0% (partial pattern recognition)
# Task mno345: 80.0% (multiple rules partially understood)
```

**Low-Priority Tasks (Hard):**
```bash
# Tasks below 70% (unlikely to benefit from depth alone)
grep "Hard" task_analysis.txt | head -n 10

# Example output:
# Task pqr678: 45.0% (no clear pattern found)
# Task stu901: 33.3% (random guessing level)
```

---

## Implementation Plan

### Phase 1: Analysis (Immediate, Post-Run 037)

**Objective:** Identify which tasks need deep refinement

**Tasks:**
1. Extract per-task accuracy trajectories from Run 037 logs
2. Categorize tasks: Easy / Plateau High / Plateau Medium / Hard
3. Calculate expected benefit from deep workers
4. Estimate compute overhead vs accuracy gain

**Deliverable:** Task difficulty report with deep-worker activation recommendations

**Script:** `scripts/analyze_task_difficulty.py`

**Time:** 1 hour (after Run 037 completes)

### Phase 2: Architecture Design (1-2 days)

**Objective:** Design hybrid TRM+K3D architecture

**Tasks:**
1. Extend `ParallelCandidateGenerator` with deep worker tier
2. Implement TRM-style sequential refinement (`SequentialRefiner`)
3. Add adaptive gating logic (95% threshold, plateau detection)
4. Integrate Shadow Copy as latent reasoning memory (Galaxy-backed, no Python copies)
5. Maintain Tesla 6-3-9 pattern alignment
6. Use SovereignLoader for PTX module reuse; spawn per-task math cores (ternary-capable) and reclaim
7. Enforce symlinked references for all new artifacts (no duplication of shapes/rules; store IDs)

**Deliverable:**
- `knowledge3d/training/arc_agi/hybrid_generator.py`
- `knowledge3d/training/arc_agi/sequential_refiner.py`

**Files to Modify:**
- `parallel_generator.py` (add Tier 2 deep workers)
- `candidate_generator.py` (add sequential refinement method)
- `sovereign_pipeline.py` (integrate hybrid generation)

**Time:** 1-2 days implementation + testing

### Phase 3: Testing (1 day)

**Objective:** Validate hybrid architecture on diagnostic subset

**Tasks:**
1. Run diagnostic (10 tasks × 3 epochs) with hybrid mode
2. Verify adaptive gating (easy tasks skip deep, hard tasks activate deep)
3. Check accuracy improvement on plateau tasks
4. Measure compute overhead (target: <1.6× vs pure parallel)

**Deliverable:** Diagnostic report with hybrid vs standard comparison

**Success Criteria:**
- Easy tasks: ≥95% with quick workers only (deep skipped)
- Plateau tasks: Improved by 3-5% with deep workers
- Compute overhead: <1.6× average (vs 2.17× always-deep)

**Time:** 1 day

### Phase 4: Full Run (24 hours)

**Objective:** Run 038 with hybrid architecture

**Tasks:**
1. Launch Run 038 (108 tasks × 162 epochs, hybrid mode)
2. Monitor adaptive gating stats (% deep activation)
3. Compare accuracy to Run 037 (baseline)
4. Analyze per-task improvements (which tasks benefited)

**Deliverable:** Run 038 completion report

**Success Criteria:**
- Accuracy: 42-50% (target: +2-5% vs Run 037)
- Deep activation: 60-80% of tasks
- Compute overhead: 1.4-1.6× average

**Time:** 24 hours training + 2 hours analysis

---

## Expected Results

### Accuracy Improvement

**Baseline (Run 037, Pure Parallel):**
- Easy tasks (30%): 95%+ accuracy → 28.5% contribution
- Medium tasks (50%): 80% accuracy → 40% contribution
- Hard tasks (20%): 40% accuracy → 8% contribution
- **Total: 76.5% weighted accuracy** (estimate)

**Hybrid (Run 038, Parallel + Deep):**
- Easy tasks (30%): 95%+ accuracy → 28.5% contribution (no change)
- Medium tasks (50%): 85% accuracy → 42.5% contribution (+2.5%)
- Hard tasks (20%): 45% accuracy → 9% contribution (+1%)
- **Total: 80% weighted accuracy (+3.5% gain)**

**Conservative Estimate:** +2-3% absolute accuracy gain
**Optimistic Estimate:** +4-5% absolute accuracy gain

### Compute Overhead

**Pure Parallel (Run 037):**
- 54 candidates per task
- 108 tasks × 162 epochs × 3 attempts = 52,488 task attempts
- Total candidates: 54 × 52,488 = 2,834,352

**Hybrid (Run 038, 70% deep activation):**
- Easy (30%): 54 candidates (quick only)
- Hard (70%): 117 candidates (quick + deep)
- Average: 0.3 × 54 + 0.7 × 117 = 98.1 candidates per task
- Total candidates: 98.1 × 52,488 = 5,149,073

**Overhead: 1.82× compute** (vs pure parallel)

**But compare to always-deep:**
- Always deep: 117 × 52,488 = 6,141,096 candidates
- Hybrid: 5,149,073 candidates
- **Savings: 16% compute** vs always-deep

### Time Impact

**Per Epoch (estimate):**
- Pure parallel: ~9 minutes per epoch
- Hybrid (70% deep): ~14 minutes per epoch (1.55× slower)
- Always-deep: ~18 minutes per epoch (2× slower)

**Full Run 038:**
- Pure parallel: 162 epochs × 9 min = 24.3 hours
- Hybrid: 162 epochs × 14 min = 37.8 hours
- Always-deep: 162 epochs × 18 min = 48.6 hours

**Hybrid is 1.55× slower but 2-5% more accurate**

---

## Success Metrics

### Accuracy Targets

**Run 038 Hybrid vs Run 037 Baseline:**
- **Minimum acceptable:** +1% (39% → 40%)
- **Target:** +2-3% (38% → 40-41%)
- **Stretch goal:** +4-5% (38% → 42-43%)

### Efficiency Targets

**Deep Worker Activation:**
- **Target:** 60-80% activation rate
- **Metric:** Avg candidates/task = 90-100 (vs 54 baseline)
- **Ternary gating:** SIGN/TCMP for activation/plateau/confidence checks
- **Math core budget:** Spawn per-task/per-worker cores; reclaim after refine; log cores used

**Compute Overhead:**
- **Target:** 1.5-1.7× vs pure parallel
- **Maximum acceptable:** 2× (better than always-deep)

### Task-Level Targets

**Plateau High Tasks (85-94% baseline):**
- **Target:** 50% improved to ≥95%
- **Example:** 28 tasks × 50% = 14 tasks improved

**Plateau Medium Tasks (70-84% baseline):**
- **Target:** 30% improved by ≥5%
- **Example:** 26 tasks × 30% = 8 tasks improved

---

## Risk Assessment

### Risk 1: Deep Workers Too Slow

**Probability:** Medium
**Impact:** High (run takes 48 hours instead of 24)

**Mitigation:**
- Implement strict timeout per deep worker (max 30 seconds)
- Use ACT-style early stopping (halt if no improvement)
- Profile deep workers on diagnostic first

### Risk 2: Deep Workers Don't Help

**Probability:** Low-Medium
**Impact:** Medium (wasted compute, no accuracy gain)

**Mitigation:**
- Analyze Run 037 plateau tasks first (confirm they need depth)
- Test deep refinement on diagnostic subset
- Compare deep vs parallel on same tasks

### Risk 3: Adaptive Gating Too Conservative

**Probability:** Low
**Impact:** Medium (skip deep on tasks that would benefit)

**Mitigation:**
- Lower threshold from 95% to 90% if needed
- Add secondary criteria (plateau detection, complexity)
- Monitor false negatives (tasks that should have used deep)

### Risk 4: Integration Bugs

**Probability:** Medium
**Impact:** Medium (run crashes, data corruption)

**Mitigation:**
- Thorough testing on diagnostic subset
- Add comprehensive logging (gating decisions, deep activations)
- Checkpoint frequently (every 10 epochs)

---

## Next Steps

**Immediate (After Run 037 Completes):**
1. Extract task trajectories from Run 037 logs
2. Run difficulty analysis script
3. Identify plateau tasks that would benefit from depth
4. Estimate expected accuracy gain

**Short-Term (1-2 days):**
1. Design hybrid architecture (extend parallel_generator.py)
2. Implement sequential refiner (TRM-style)
3. Add adaptive gating logic
4. Test on diagnostic subset

**Medium-Term (1 week):**
1. Launch Run 038 with hybrid architecture
2. Monitor and tune adaptive gating
3. Compare results to Run 037 baseline
4. Document findings

**Long-Term (2-4 weeks):**
1. Iterate on hybrid architecture (tune n, T, gating thresholds)
2. Test on ARC-AGI-2 (harder benchmark)
3. Profile GPU utilization (quantify PTX advantage)
4. Write architectural completion report

---

## References

1. **TRM Paper:** "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871v1)
   - Sequential refinement: n=6, T=3 → 21 steps per task
   - Effective depth: 2 layers × 21 steps = 42 emulated layers
   - Achieves 45% on ARC-AGI-1 with 7M parameters

2. **K3D Current:** Run 037 with parallel breadth-first
   - 9 workers × 6 candidates = 54 per task
   - Fresh bootstrap: 18 → 150+ shadow entries
   - Expected: 40-47% accuracy

3. **Hybrid Vision:** Best of both worlds
   - Parallel for easy tasks (fast)
   - Sequential for hard tasks (deep)
   - Adaptive gating (efficient)

---

**END OF SPECIFICATION — Ready for implementation after Run 037 analysis.**
