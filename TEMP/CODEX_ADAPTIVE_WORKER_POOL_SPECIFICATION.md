# Adaptive Worker Pool Architecture Specification

**Date:** December 3, 2025
**Version:** 1.0
**Status:** Pending Implementation (Post Run 036)
**Authors:** Claude (Architecture), Daniel (Lead Architect)
**Priority:** High - GPU Under-Utilization Issue

---

## Executive Summary

**Problem Identified:** Live monitoring of Run 036 reveals critical inefficiencies in parallel candidate generation:

1. **Uneven Load Balancing:** Last worker gets 2-3× more tasks than others (11 hints vs 4)
2. **Massive GPU Under-Utilization:** RTX 3060 using only 182 MiB of 12GB VRAM (1.5% capacity)
3. **Sequential Execution:** Workers run one-at-a-time in a loop (not truly parallel)
4. **Static Configuration:** Hardcoded 9 workers, no adaptation to GPU capacity

**Impact:**
- Wasted GPU resources (98.5% idle VRAM)
- Reduced throughput (54 candidates/task instead of 500+)
- Uneven worker utilization (last worker overloaded)

**Solution:** Adaptive Worker Pool with dynamic scaling, even load balancing, true parallel execution, and configurable execution strategies.

**Expected Benefits:**
- **10-20× throughput increase** (54 → 500+ candidates per task)
- **GPU utilization** from 1.5% → 30-40%
- **Even load distribution** (all workers get equal tasks ±1)
- **Adaptive scaling** (auto-detects optimal worker count for GPU)

---

## Current State Analysis

### Hardware Capacity

**GPU:** NVIDIA GeForce RTX 3060
**VRAM:** 12,288 MiB total
**Compute Capability:** 8.6
**CUDA Cores:** 3,584

**Current Usage:**
- Training VRAM: **182 MiB** (~1.5% of capacity)
- Idle VRAM: **11,806 MiB** (~98.5% wasted)
- Conclusion: **Massive under-utilization**

### Current Architecture

**File:** `knowledge3d/training/arc_agi/parallel_generator.py`

**Configuration:**
```python
num_workers = 9                # Hardcoded (line 23, 32)
candidates_per_worker = 6      # Static (line 24)
# Total: 54 candidates maximum per task
```

**Worker Allocation Logic (Lines 66-78):**
```python
hints_per_worker = max(1, total_hints // num_worker_slots)  # Integer division
for worker_idx in range(num_worker_slots):
    start_idx = worker_idx * hints_per_worker
    end_idx = start_idx + hints_per_worker if worker_idx < num_worker_slots - 1 else total_hints
    # ⚠️ BUG: Last worker gets ALL remainder instead of fair distribution
```

**Live Example from Run 036:**
```
[WORKER 0] Assigned hints 0:4 (4 hints)
[WORKER 1] Assigned hints 4:8 (4 hints)
[WORKER 2] Assigned hints 8:12 (4 hints)
[WORKER 3] Assigned hints 12:16 (4 hints)
[WORKER 4] Assigned hints 16:20 (4 hints)
[WORKER 5] Assigned hints 20:24 (4 hints)
[WORKER 6] Assigned hints 24:28 (4 hints)
[WORKER 7] Assigned hints 28:32 (4 hints)
[WORKER 8] Assigned hints 32:43 (11 hints)  ⚠️ 2.75× more than others!
```

**Execution Pattern (Lines 86-108):**
```python
for worker_idx, (core_id, executor, worker_hints) in enumerate(pairs):
    # Sequential iteration - NOT parallel!
    cand_list = gen.generate_candidates(...)  # Blocking call
    all_candidates.extend(cand_list)
```

**Problem:** Despite name "ParallelCandidateGenerator", workers execute **sequentially** in a loop. GPU waits for each worker to finish before starting the next.

---

## Architectural Design

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│  1. GPU Capacity Profiler                                   │
│     - Auto-detect VRAM per worker                           │
│     - Calculate optimal worker count                        │
│     - Runtime monitoring and adjustment                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Adaptive Worker Pool Manager                            │
│     - Dynamic worker allocation (9 → 50-100 workers)        │
│     - Even load balancing (fix remainder distribution)      │
│     - Configurable via CLI args                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Parallel Execution Scheduler                            │
│     - ThreadPoolExecutor for true concurrency               │
│     - Execution strategies: parallel, pipeline, hybrid      │
│     - Depth control (chain vs breadth)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Runtime Monitoring & Logging                            │
│     - VRAM usage per worker batch                           │
│     - Worker utilization metrics                            │
│     - Throughput statistics                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Tasks

### **Task 1: Fix Load Balancing Bug (Priority: Critical)**

**File:** `knowledge3d/training/arc_agi/parallel_generator.py`

**Current Bug (Lines 66-78):**
```python
# Remainder goes entirely to last worker
hints_per_worker = max(1, total_hints // num_worker_slots)
for worker_idx in range(num_worker_slots):
    start_idx = worker_idx * hints_per_worker
    end_idx = start_idx + hints_per_worker if worker_idx < num_worker_slots - 1 else total_hints
    # Last worker: end_idx = total_hints (gets all remainder!)
```

**Fix: Even Distribution**

```python
def _distribute_hints_evenly(
    self,
    hints: List[str],
    num_workers: int
) -> List[List[str]]:
    """
    Distribute hints evenly across workers.

    Algorithm:
    - base_count = total // num_workers
    - remainder = total % num_workers
    - First 'remainder' workers get (base_count + 1) hints
    - Remaining workers get base_count hints

    Example:
        43 hints, 9 workers
        base_count = 43 // 9 = 4
        remainder = 43 % 9 = 7

        Workers 0-6: get 5 hints each (7 × 5 = 35)
        Workers 7-8: get 4 hints each (2 × 4 = 8)
        Total: 43 ✓ (evenly distributed, max difference = 1)

    Args:
        hints: List of semantic hints to distribute
        num_workers: Number of workers to distribute across

    Returns:
        List of hint partitions, one per worker
    """
    if not hints:
        return [[] for _ in range(num_workers)]

    total_hints = len(hints)
    base_count = total_hints // num_workers
    remainder = total_hints % num_workers

    partitions = []
    start_idx = 0

    for worker_idx in range(num_workers):
        # First 'remainder' workers get +1 extra hint
        count_for_worker = base_count + (1 if worker_idx < remainder else 0)
        end_idx = start_idx + count_for_worker

        worker_partition = hints[start_idx:end_idx]
        partitions.append(worker_partition)

        print(f"  [WORKER {worker_idx}] Assigned hints {start_idx}:{end_idx} ({len(worker_partition)} hints)")

        start_idx = end_idx

    # Verification
    total_assigned = sum(len(p) for p in partitions)
    assert total_assigned == total_hints, f"Load balancing error: {total_assigned} != {total_hints}"

    return partitions
```

**Integration Point (Line 68):**

Replace:
```python
semantic_partitions: List[Optional[List[str]]] = []
if semantic_hints:
    total_hints = len(semantic_hints)
    hints_per_worker = max(1, total_hints // num_worker_slots)
    for worker_idx in range(num_worker_slots):
        start_idx = worker_idx * hints_per_worker
        end_idx = start_idx + hints_per_worker if worker_idx < num_worker_slots - 1 else total_hints
        worker_hints = semantic_hints[start_idx:end_idx] if start_idx < total_hints else []
        semantic_partitions.append(worker_hints)
        print(f"  [WORKER {worker_idx}] Assigned hints {start_idx}:{end_idx} ({len(worker_hints)} hints)")
else:
    semantic_partitions = [None for _ in range(num_worker_slots)]
```

With:
```python
if semantic_hints:
    semantic_partitions = self._distribute_hints_evenly(semantic_hints, num_worker_slots)
else:
    semantic_partitions = [[] for _ in range(num_worker_slots)]
```

**Test Cases:**

```python
# Test 1: Even division
hints = list(range(45))  # 45 hints
partitions = _distribute_hints_evenly(hints, 9)  # 9 workers
# Expected: all workers get 5 hints each

# Test 2: With remainder
hints = list(range(43))  # 43 hints
partitions = _distribute_hints_evenly(hints, 9)  # 9 workers
# Expected: 7 workers get 5, 2 workers get 4

# Test 3: More workers than hints
hints = list(range(5))  # 5 hints
partitions = _distribute_hints_evenly(hints, 9)  # 9 workers
# Expected: 5 workers get 1, 4 workers get 0

# Verify all
for hints, num_workers in [(45, 9), (43, 9), (5, 9), (100, 9)]:
    partitions = _distribute_hints_evenly(list(range(hints)), num_workers)
    total = sum(len(p) for p in partitions)
    max_diff = max(len(p) for p in partitions) - min(len(p) for p in partitions)
    print(f"{hints} hints, {num_workers} workers: total={total}, max_diff={max_diff}")
    assert total == hints
    assert max_diff <= 1  # Fair distribution
```

---

### **Task 2: GPU Capacity Profiler**

**Create:** `knowledge3d/training/arc_agi/gpu_capacity_profiler.py`

```python
"""
GPU Capacity Profiler for Adaptive Worker Pools.

Profiles GPU VRAM usage to determine optimal worker count for parallel
candidate generation without exceeding memory constraints.
"""

from __future__ import annotations

import subprocess
from typing import Optional


class GPUCapacityProfiler:
    """
    Profile GPU capacity and recommend optimal worker count.

    Strategy:
    1. Measure baseline VRAM usage (idle state)
    2. Run test batches with increasing worker counts
    3. Measure VRAM per worker
    4. Calculate optimal workers within target VRAM budget
    """

    def __init__(
        self,
        gpu_index: int = 0,
        target_vram_gb: float = 4.0,
        safety_margin_gb: float = 2.0,
        max_workers_cap: int = 100,
    ):
        """
        Initialize profiler.

        Args:
            gpu_index: GPU device index (default 0)
            target_vram_gb: Target VRAM usage in GB (default 4.0)
            safety_margin_gb: Safety margin to reserve (default 2.0)
            max_workers_cap: Hard cap on workers (default 100)
        """
        self.gpu_index = gpu_index
        self.target_vram_gb = target_vram_gb
        self.safety_margin_gb = safety_margin_gb
        self.max_workers_cap = max_workers_cap

        self.total_vram_mb = self._get_total_vram()
        self.baseline_vram_mb = self._get_current_vram_usage()

    def _get_total_vram(self) -> float:
        """Get total GPU VRAM in MB."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except Exception as e:
            print(f"[GPU PROFILER] Warning: Could not query GPU VRAM: {e}")
            return 12288.0  # Default to RTX 3060 spec

    def _get_current_vram_usage(self) -> float:
        """Get current GPU VRAM usage in MB."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except Exception as e:
            print(f"[GPU PROFILER] Warning: Could not query GPU usage: {e}")
            return 0.0

    def estimate_optimal_workers(
        self,
        vram_per_worker_mb: Optional[float] = None,
    ) -> int:
        """
        Estimate optimal worker count based on VRAM constraints.

        Args:
            vram_per_worker_mb: Measured VRAM per worker (if None, use heuristic)

        Returns:
            Recommended worker count
        """
        if vram_per_worker_mb is None:
            # Heuristic based on current 9 workers using 182 MiB
            # 182 MiB / 9 workers ≈ 20 MiB per worker
            vram_per_worker_mb = 20.0

        available_vram_mb = (self.target_vram_gb * 1024) - self.baseline_vram_mb

        # Reserve safety margin
        available_vram_mb -= (self.safety_margin_gb * 1024)

        if available_vram_mb <= 0:
            print(f"[GPU PROFILER] Warning: Insufficient VRAM available")
            return 9  # Fall back to current default

        optimal_workers = int(available_vram_mb / vram_per_worker_mb)

        # Apply cap
        optimal_workers = min(optimal_workers, self.max_workers_cap)
        optimal_workers = max(optimal_workers, 1)  # At least 1 worker

        print(f"\n[GPU CAPACITY PROFILER]")
        print(f"  Total VRAM: {self.total_vram_mb:.0f} MiB")
        print(f"  Baseline usage: {self.baseline_vram_mb:.0f} MiB")
        print(f"  Target VRAM: {self.target_vram_gb * 1024:.0f} MiB")
        print(f"  Safety margin: {self.safety_margin_gb * 1024:.0f} MiB")
        print(f"  Available for workers: {available_vram_mb:.0f} MiB")
        print(f"  VRAM per worker (estimated): {vram_per_worker_mb:.1f} MiB")
        print(f"  Optimal workers: {optimal_workers}")
        print(f"  Expected total VRAM: {self.baseline_vram_mb + (optimal_workers * vram_per_worker_mb):.0f} MiB")

        return optimal_workers

    def profile_worker_batch(
        self,
        test_worker_counts: list[int] = [1, 10, 25, 50],
        test_fn = None,
    ) -> dict[int, float]:
        """
        Profile VRAM usage for different worker counts.

        Args:
            test_worker_counts: Worker counts to test
            test_fn: Function to run for profiling (callable taking num_workers)

        Returns:
            Dict mapping worker_count → vram_usage_mb
        """
        if test_fn is None:
            print(f"[GPU PROFILER] No test function provided, using estimation")
            return {}

        results = {}
        baseline = self._get_current_vram_usage()

        print(f"\n[GPU PROFILER] Running worker batch profiling...")
        print(f"  Baseline VRAM: {baseline:.0f} MiB")

        for num_workers in test_worker_counts:
            # Run test batch
            test_fn(num_workers)

            # Measure VRAM
            current_vram = self._get_current_vram_usage()
            delta_vram = current_vram - baseline
            vram_per_worker = delta_vram / num_workers if num_workers > 0 else 0

            results[num_workers] = current_vram

            print(f"  {num_workers:3d} workers: {current_vram:6.0f} MiB total, {vram_per_worker:5.1f} MiB/worker, delta={delta_vram:6.0f} MiB")

            # Stop if we exceed target
            if current_vram > self.target_vram_gb * 1024:
                print(f"  ⚠️ Exceeded target VRAM ({self.target_vram_gb * 1024:.0f} MiB), stopping profiling")
                break

        return results


def get_optimal_worker_count(
    target_vram_gb: float = 4.0,
    safety_margin_gb: float = 2.0,
    max_cap: int = 100,
) -> int:
    """
    Convenience function to get optimal worker count.

    Args:
        target_vram_gb: Target VRAM budget
        safety_margin_gb: Safety margin to reserve
        max_cap: Maximum workers to allow

    Returns:
        Recommended worker count
    """
    profiler = GPUCapacityProfiler(
        target_vram_gb=target_vram_gb,
        safety_margin_gb=safety_margin_gb,
        max_workers_cap=max_cap,
    )

    return profiler.estimate_optimal_workers()


__all__ = ["GPUCapacityProfiler", "get_optimal_worker_count"]
```

---

### **Task 3: True Parallel Execution**

**File:** `knowledge3d/training/arc_agi/parallel_generator.py`

**Current Sequential Execution (Lines 86-108):**

```python
for worker_idx, (core_id, executor, worker_hints) in enumerate(pairs):
    # Sequential - workers wait for each other
    try:
        gen = CandidateGenerator(...)
        cand_list = gen.generate_candidates(...)  # Blocking
        all_candidates.extend(cand_list)
    finally:
        if core_id is not None:
            self.core_pool.release_core(core_id, pool=True)
```

**New: Concurrent Execution**

Add import at top:
```python
import concurrent.futures
import threading
```

**Replace sequential loop with ThreadPoolExecutor:**

```python
def _worker_task(
    self,
    worker_id: int,
    hints_subset: List[str],
    input_grid,
    train_examples,
    expected_output,
) -> tuple[int, List[Candidate]]:
    """
    Single worker task (runs in thread).

    Args:
        worker_id: Worker index
        hints_subset: Semantic hints for this worker
        input_grid: Input grid for task
        train_examples: Training examples
        expected_output: Expected output (for validation)

    Returns:
        (worker_id, candidates) tuple
    """
    core_id = None
    try:
        # Allocate math core
        core_id = self.core_pool.spawn_core(tier=1, reuse=True)
        executor = ARCRPNExecutor(pool=self.core_pool, instance_id=core_id)

        # Generate candidates
        gen = CandidateGenerator(
            matryoshka_dim=self.matryoshka_dim,
            max_candidates=self.candidates_per_worker,
            shadow_copy=self.shadow_copy,
            executor=executor,
            codec_embedder=self.codec_embedder,
            embedding_galaxy=self.embedding_galaxy,
            cosine_bridge=self.cosine_bridge,
        )

        candidates = gen.generate_candidates(
            input_grid=input_grid,
            train_examples=train_examples,
            semantic_hints=hints_subset,
            expected_output=expected_output,
        )

        hint_count = len(hints_subset) if hints_subset else 0
        print(f"  [WORKER {worker_id}] Generated {len(candidates)} candidates from {hint_count} hints")

        return (worker_id, candidates)

    finally:
        if core_id is not None:
            self.core_pool.release_core(core_id, pool=True)


def generate_parallel(
    self,
    input_grid: Sequence[Sequence[int]],
    train_examples: List[Dict[str, Any]],
    semantic_hints: Optional[List[str]],
    expected_output: Optional[Sequence[Sequence[int]]],
) -> List[Candidate]:
    """
    Generate candidates in TRUE PARALLEL using ThreadPoolExecutor.

    Workers execute concurrently, maximizing GPU utilization.
    """
    # Distribute hints evenly
    num_worker_slots = self.num_workers
    if semantic_hints:
        semantic_partitions = self._distribute_hints_evenly(semantic_hints, num_worker_slots)
    else:
        semantic_partitions = [[] for _ in range(num_worker_slots)]

    # Parallel execution
    all_candidates: List[Candidate] = []
    ptx_success = 0
    ptx_fallback = 0

    print(f"  [PARALLEL GEN] Launching {num_worker_slots} workers concurrently...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker_slots) as thread_pool:
        # Submit all worker tasks
        futures = [
            thread_pool.submit(
                self._worker_task,
                worker_id,
                hints_subset,
                input_grid,
                train_examples,
                expected_output,
            )
            for worker_id, hints_subset in enumerate(semantic_partitions)
        ]

        # Gather results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                worker_id, candidates = future.result()
                all_candidates.extend(candidates)
            except Exception as e:
                print(f"  [PARALLEL GEN] Worker failed: {e}")

    # Instrumentation (collect from all executors if accessible)
    # Note: PTX stats collection needs refactoring for thread-safe access
    print(f"  [PARALLEL GEN] Total candidates before dedup: {len(all_candidates)}")
    print(f"  [PARALLEL GEN] Returning all {len(all_candidates)} candidates for semantic ranking")

    return all_candidates
```

**Benefits:**
- All workers run **simultaneously** (true parallelism)
- GPU can batch-process multiple workers
- ThreadPoolExecutor handles thread safety
- Exceptions in one worker don't crash others

---

### **Task 4: Adaptive Worker Configuration**

**File:** `knowledge3d/training/arc_agi/parallel_generator.py`

**Add auto-detection mode:**

```python
class ParallelCandidateGenerator:
    def __init__(
        self,
        *,
        num_workers: int | str = "auto",  # Allow "auto" mode
        candidates_per_worker: int = 6,
        top_k: int = 3,
        matryoshka_dim: int = 512,
        target_vram_gb: float = 4.0,
        shadow_copy: Optional[DualShadowCopy] = None,
        codec_embedder: Any | None = None,
        embedding_galaxy: Optional[Dict[int, List[float]]] = None,
        cosine_bridge: Any | None = None,
    ) -> None:
        # Auto-detect optimal workers if "auto"
        if num_workers == "auto":
            from knowledge3d.training.arc_agi.gpu_capacity_profiler import get_optimal_worker_count
            self.num_workers = get_optimal_worker_count(
                target_vram_gb=target_vram_gb,
                safety_margin_gb=2.0,
                max_cap=100,
            )
            print(f"  [PARALLEL GEN] Auto-detected {self.num_workers} workers (target {target_vram_gb}GB VRAM)")
        else:
            self.num_workers = num_workers

        self.candidates_per_worker = candidates_per_worker
        self.top_k = top_k
        self.matryoshka_dim = matryoshka_dim
        self.target_vram_gb = target_vram_gb
        # ... rest of init
```

**File:** `knowledge3d/training/arc_agi/sovereign_pipeline.py`

**Update initialization (Line 291):**

```python
par_gen = ParallelCandidateGenerator(
    num_workers="auto",  # Changed from hardcoded 9
    candidates_per_worker=6,
    top_k=3,
    matryoshka_dim=self.matryoshka_dim,
    target_vram_gb=4.0,  # Configurable
    shadow_copy=self.shadow_copy,
    codec_embedder=self.codec_embedder,
    embedding_galaxy=self.embedding_galaxy,
    cosine_bridge=self.cosine_bridge,
)
```

**File:** `scripts/train_arc_sovereign_loop.py`

**Add CLI arguments:**

```python
parser.add_argument(
    "--num-workers",
    type=str,
    default="auto",
    help="Number of parallel workers ('auto' or integer, default: auto)"
)
parser.add_argument(
    "--target-vram-gb",
    type=float,
    default=4.0,
    help="Target VRAM usage in GB (default: 4.0)"
)
parser.add_argument(
    "--candidates-per-worker",
    type=int,
    default=6,
    help="Candidates per worker (default: 6)"
)
```

**Pass to pipeline:**

```python
pipeline = SovereignAIPipeline(
    matryoshka_dim=args.matryoshka_dim,
    num_workers=args.num_workers,
    target_vram_gb=args.target_vram_gb,
    candidates_per_worker=args.candidates_per_worker,
    # ... existing args
)
```

---

### **Task 5: Execution Strategies (Advanced)**

**Create:** `knowledge3d/training/arc_agi/worker_scheduler.py`

```python
"""
Worker execution strategies for adaptive candidate generation.

Provides:
- Parallel: All workers run simultaneously (maximize breadth)
- Pipeline: Workers in sequential stages (maximize depth/composition)
- Hybrid: Parallel generation + pipeline refinement
"""

from __future__ import annotations

from typing import List, Callable, Any
from enum import Enum


class ExecutionStrategy(Enum):
    """Worker execution strategies."""
    PARALLEL = "parallel"      # All workers run simultaneously
    PIPELINE = "pipeline"      # Workers in sequential stages
    HYBRID = "hybrid"          # Parallel breadth + pipeline depth


class WorkerScheduler:
    """
    Schedule worker execution with different strategies.

    Strategies:
    1. PARALLEL: Maximize breadth (many diverse candidates)
    2. PIPELINE: Maximize depth (compositional refinement)
    3. HYBRID: Breadth-first, then depth refinement
    """

    def __init__(self, strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL):
        self.strategy = strategy

    def execute(
        self,
        workers: List[Any],
        task_fn: Callable,
        *args,
        **kwargs,
    ) -> List[Any]:
        """
        Execute workers with configured strategy.

        Args:
            workers: List of worker instances
            task_fn: Function to execute (takes worker, *args, **kwargs)
            *args, **kwargs: Passed to task_fn

        Returns:
            List of results from all workers
        """
        if self.strategy == ExecutionStrategy.PARALLEL:
            return self._execute_parallel(workers, task_fn, *args, **kwargs)
        elif self.strategy == ExecutionStrategy.PIPELINE:
            return self._execute_pipeline(workers, task_fn, *args, **kwargs)
        elif self.strategy == ExecutionStrategy.HYBRID:
            return self._execute_hybrid(workers, task_fn, *args, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _execute_parallel(self, workers, task_fn, *args, **kwargs) -> List:
        """
        Parallel execution: all workers run simultaneously.

        Use case: Maximize diversity of candidates
        """
        import concurrent.futures

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = [
                executor.submit(task_fn, worker, *args, **kwargs)
                for worker in workers
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.extend(result if isinstance(result, list) else [result])
                except Exception as e:
                    print(f"  [SCHEDULER PARALLEL] Worker failed: {e}")

        return results

    def _execute_pipeline(self, workers, task_fn, *args, **kwargs) -> List:
        """
        Pipeline execution: workers in sequential stages.

        Use case: Compositional depth (refine previous stage output)

        Flow:
        Worker 0 → generates candidates
        Worker 1 → refines Worker 0 output
        Worker 2 → refines Worker 1 output
        ...
        """
        intermediate_results = None

        for stage_idx, worker in enumerate(workers):
            print(f"  [SCHEDULER PIPELINE] Stage {stage_idx}: running worker")

            # First stage: use original args
            # Later stages: use previous stage output as input
            if stage_idx == 0:
                stage_results = task_fn(worker, *args, **kwargs)
            else:
                # Refinement: use previous output as new input
                stage_results = task_fn(worker, intermediate_results, **kwargs)

            intermediate_results = stage_results
            print(f"  [SCHEDULER PIPELINE] Stage {stage_idx}: {len(stage_results)} candidates")

        return intermediate_results

    def _execute_hybrid(
        self,
        workers,
        task_fn,
        *args,
        parallel_batch_size: int = 10,
        **kwargs,
    ) -> List:
        """
        Hybrid execution: parallel breadth + pipeline depth.

        Use case: Best of both worlds

        Flow:
        Stage 1: First N workers run in parallel (breadth)
        Stage 2: Remaining workers refine in pipeline (depth)
        """
        if len(workers) < 2:
            return self._execute_parallel(workers, task_fn, *args, **kwargs)

        # Stage 1: Parallel breadth
        parallel_workers = workers[:parallel_batch_size]
        breadth_results = self._execute_parallel(parallel_workers, task_fn, *args, **kwargs)

        print(f"  [SCHEDULER HYBRID] Stage 1 (parallel): {len(breadth_results)} candidates from {len(parallel_workers)} workers")

        # Stage 2: Pipeline depth
        pipeline_workers = workers[parallel_batch_size:]
        if pipeline_workers:
            depth_results = self._execute_pipeline(pipeline_workers, task_fn, breadth_results, **kwargs)
            print(f"  [SCHEDULER HYBRID] Stage 2 (pipeline): {len(depth_results)} refined candidates")
            return depth_results
        else:
            return breadth_results


__all__ = ["WorkerScheduler", "ExecutionStrategy"]
```

**Integration (optional - for future experimentation):**

```python
# In parallel_generator.py

from knowledge3d.training.arc_agi.worker_scheduler import WorkerScheduler, ExecutionStrategy

class ParallelCandidateGenerator:
    def __init__(
        self,
        *,
        execution_strategy: str = "parallel",  # "parallel", "pipeline", "hybrid"
        # ... other args
    ):
        self.scheduler = WorkerScheduler(
            strategy=ExecutionStrategy(execution_strategy)
        )
```

---

### **Task 6: Runtime Monitoring & Logging**

**Add VRAM monitoring during training:**

**File:** `knowledge3d/training/arc_agi/parallel_generator.py`

```python
def _log_gpu_usage(self, phase: str = ""):
    """Log current GPU VRAM usage."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        used, free, util = result.stdout.strip().split(", ")
        print(f"  [GPU {phase}] VRAM: {used} MiB used, {free} MiB free, {util}% GPU util")
    except Exception:
        pass  # Silent fail if nvidia-smi unavailable

# Call before/after worker batch
def generate_parallel(self, ...):
    self._log_gpu_usage("BEFORE WORKERS")

    # ... worker execution ...

    self._log_gpu_usage("AFTER WORKERS")
```

**Add worker utilization metrics:**

```python
def _log_worker_statistics(self, worker_timings: List[float]):
    """
    Log worker performance statistics.

    Args:
        worker_timings: List of execution times per worker
    """
    if not worker_timings:
        return

    avg_time = sum(worker_timings) / len(worker_timings)
    min_time = min(worker_timings)
    max_time = max(worker_timings)
    std_dev = (sum((t - avg_time)**2 for t in worker_timings) / len(worker_timings))**0.5

    print(f"  [WORKER STATS]")
    print(f"    Workers: {len(worker_timings)}")
    print(f"    Avg time: {avg_time:.2f}s")
    print(f"    Min time: {min_time:.2f}s")
    print(f"    Max time: {max_time:.2f}s")
    print(f"    Std dev: {std_dev:.2f}s")
    print(f"    Utilization: {(min_time / max_time * 100):.1f}% (ideal: 100%)")
```

---

## Testing Plan

### Test 1: Load Balancing Fix

**Verify even distribution:**

```bash
# Run short training with logging
PYTHONPATH=. python scripts/train_arc_sovereign_loop.py \
  --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
  --max-tasks 5 \
  --epochs 1 \
  > /tmp/test_load_balancing.log 2>&1

# Check worker assignments
grep "WORKER.*Assigned hints" /tmp/test_load_balancing.log

# Expected: max difference of 1 hint between workers
# Example:
#   [WORKER 0] Assigned hints 0:5 (5 hints)
#   [WORKER 1] Assigned hints 5:10 (5 hints)
#   ...
#   [WORKER 8] Assigned hints 40:44 (4 hints)  ✓ Only 1 less, not 3× more
```

### Test 2: GPU Capacity Profiling

**Test auto-detection:**

```bash
# Run with auto worker detection
PYTHONPATH=. python -c "
from knowledge3d.training.arc_agi.gpu_capacity_profiler import get_optimal_worker_count

optimal = get_optimal_worker_count(target_vram_gb=4.0, safety_margin_gb=2.0)
print(f'Optimal workers: {optimal}')
"

# Expected output:
# [GPU CAPACITY PROFILER]
#   Total VRAM: 12288 MiB
#   Baseline usage: ~180 MiB
#   Target VRAM: 4096 MiB
#   Available for workers: 1916 MiB
#   VRAM per worker (estimated): 20.0 MiB
#   Optimal workers: 95
```

### Test 3: Parallel Execution Throughput

**Compare sequential vs parallel:**

```bash
# Test with old sequential code (baseline)
# Measure time for 1 task with 9 workers
time PYTHONPATH=. python scripts/test_worker_throughput.py --mode sequential --workers 9

# Test with new parallel code
time PYTHONPATH=. python scripts/test_worker_throughput.py --mode parallel --workers 9

# Test with auto-scaled workers
time PYTHONPATH=. python scripts/test_worker_throughput.py --mode parallel --workers auto

# Expected improvements:
# Sequential 9 workers: ~10s per task, 54 candidates
# Parallel 9 workers: ~2-3s per task, 54 candidates (3-5× faster)
# Parallel 50 workers: ~3-4s per task, 300 candidates (10× throughput)
```

### Test 4: Full Training Run

**Run short training with new architecture:**

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/train_arc_sovereign_loop.py \
  --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
  --max-tasks 10 \
  --epochs 3 \
  --num-workers auto \
  --target-vram-gb 4.0 \
  > /tmp/test_adaptive_workers.log 2>&1

# Check logs for:
grep "Auto-detected.*workers" /tmp/test_adaptive_workers.log
grep "GPU.*VRAM" /tmp/test_adaptive_workers.log
grep "Total candidates before dedup" /tmp/test_adaptive_workers.log

# Expected:
# Auto-detected 95 workers (target 4.0GB VRAM)
# GPU AFTER WORKERS: 3800 MiB used, 8400 MiB free, 45% GPU util
# Total candidates before dedup: 570
```

---

## Rollout Strategy

### Phase 1: Fix Critical Bug (Immediate)

**Priority: Critical**
**Risk: Low**
**Impact: High**

1. Implement `_distribute_hints_evenly()` method
2. Replace load balancing logic in `parallel_generator.py`
3. Test with 5-task run
4. Deploy to Run 037 (after Run 036 completes)

**Success Criteria:**
- No worker gets >1 more hint than others
- Log shows even distribution
- No performance regression

---

### Phase 2: GPU Capacity Profiling (Low Risk)

**Priority: High**
**Risk: Low**
**Impact: Medium**

1. Create `gpu_capacity_profiler.py`
2. Test auto-detection standalone
3. Integrate into `ParallelCandidateGenerator.__init__()`
4. Run 10-task test with auto mode
5. Deploy if stable

**Success Criteria:**
- Auto-detection recommends 50-100 workers
- VRAM usage stays within target
- No CUDA OOM errors

---

### Phase 3: Parallel Execution (Medium Risk)

**Priority: High**
**Risk: Medium** (threading complexity)
**Impact: Very High**

1. Implement `_worker_task()` method
2. Replace sequential loop with ThreadPoolExecutor
3. Add thread-safe PTX stats collection
4. Test with 9 workers (same as current)
5. Test with 25 workers
6. Test with 50 workers
7. Deploy if no deadlocks/crashes

**Success Criteria:**
- No deadlocks or race conditions
- GPU utilization increases (1.5% → 30%+)
- Candidate count increases proportionally
- No accuracy regression

---

### Phase 4: Runtime Monitoring (Optional)

**Priority: Medium**
**Risk: Low**
**Impact: Low** (visibility only)

1. Add `_log_gpu_usage()` calls
2. Add `_log_worker_statistics()`
3. Integrate into training loop
4. Deploy

**Success Criteria:**
- VRAM logs appear before/after workers
- Worker utilization metrics show balance

---

### Phase 5: Execution Strategies (Future Research)

**Priority: Low**
**Risk: Low**
**Impact: Unknown** (experimental)

1. Create `worker_scheduler.py`
2. Test PARALLEL strategy (baseline)
3. Test PIPELINE strategy (compositional depth)
4. Test HYBRID strategy
5. Compare accuracy across strategies
6. Deploy best-performing strategy

**Success Criteria:**
- All strategies work without crashes
- PIPELINE/HYBRID show accuracy improvements
- Throughput trade-offs understood

---

## Expected Outcomes

### Immediate (Phase 1-2)

**After Load Balancing Fix:**
- ✅ Even worker distribution (all workers ±1 hint)
- ✅ Last worker no longer overloaded
- ✅ Fairer GPU utilization across workers

**After GPU Profiling:**
- ✅ Automatic worker scaling (9 → 50-100 workers)
- ✅ VRAM usage increases to target (4GB)
- ✅ 10-20× candidate throughput (54 → 500+)

### Medium-Term (Phase 3-4)

**After Parallel Execution:**
- ✅ True concurrent worker execution
- ✅ GPU utilization: 1.5% → 30-40%
- ✅ Reduced wall-clock time per task
- ✅ Visible VRAM/utilization monitoring

### Long-Term (Phase 5)

**After Execution Strategies:**
- ✅ Experimental pipeline/hybrid modes
- ✅ Compositional depth exploration
- ✅ Accuracy improvements from depth composition

---

## Risks and Mitigations

### Risk 1: ThreadPoolExecutor Deadlocks

**Mitigation:**
- Start with same worker count as current (9)
- Verify no deadlocks before scaling up
- Add timeouts to worker tasks
- Monitor for hung threads

### Risk 2: CUDA OOM Errors

**Mitigation:**
- Conservative safety margin (2GB)
- Gradual scaling (9 → 25 → 50 → 100)
- Monitor VRAM in real-time
- Fall back to lower worker count on OOM

### Risk 3: Accuracy Regression

**Mitigation:**
- Compare Run 037 (with fixes) to Run 036 (baseline)
- If accuracy drops, roll back and investigate
- Track vocabulary quality metrics
- Ensure dedup/ranking still works with more candidates

### Risk 4: MathCorePool Capacity

**Mitigation:**
- Check `MathCorePool.spawn_core()` limits
- Handle exceptions when pool exhausted
- Fall back to available cores gracefully
- Document pool capacity constraints

---

## CLI Usage Examples

### Auto Mode (Recommended)

```bash
# Let system auto-detect optimal workers
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/train_arc_sovereign_loop.py \
  --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
  --max-tasks 108 \
  --epochs 162 \
  --num-workers auto \
  --target-vram-gb 4.0
```

### Manual Override

```bash
# Manually set 50 workers
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python scripts/train_arc_sovereign_loop.py \
  --num-workers 50 \
  --candidates-per-worker 10 \
  --target-vram-gb 6.0
```

### Conservative Mode

```bash
# Conservative: 25 workers, 2GB target
python scripts/train_arc_sovereign_loop.py \
  --num-workers 25 \
  --target-vram-gb 2.0
```

---

## Performance Expectations

### Current Baseline (Run 036)

- Workers: 9 (hardcoded)
- Candidates per task: 54
- VRAM usage: 182 MiB (~1.5% of 12GB)
- GPU utilization: ~1% (idle most of time)
- Wall-clock time: ~18-24 hours for 108 tasks × 162 epochs

### After Phase 1-2 (Load Balancing + Auto-Scaling)

- Workers: ~50-100 (auto-detected)
- Candidates per task: 300-600
- VRAM usage: 3-4 GB (~30-35% of 12GB)
- GPU utilization: ~25-35% (better but still sequential)
- Wall-clock time: Similar (more candidates, same sequential bottleneck)

### After Phase 3 (True Parallel)

- Workers: ~50-100
- Candidates per task: 300-600
- VRAM usage: 3-4 GB
- GPU utilization: ~40-60% (concurrent execution)
- Wall-clock time: **12-16 hours** (25-33% faster)

---

## Code Review Checklist

Before deploying to production runs:

**Load Balancing:**
- [ ] `_distribute_hints_evenly()` passes all test cases
- [ ] Verification assertion checks total == input
- [ ] Max difference between workers ≤ 1

**GPU Profiling:**
- [ ] Auto-detection doesn't crash on nvidia-smi errors
- [ ] Falls back gracefully if profiling unavailable
- [ ] Safety margin prevents OOM

**Parallel Execution:**
- [ ] ThreadPoolExecutor properly releases threads
- [ ] Math cores released in `finally` blocks
- [ ] No race conditions in candidate collection
- [ ] Exception handling per-worker (doesn't crash all)

**Integration:**
- [ ] CLI args parsed correctly ("auto" vs integer)
- [ ] Backward compatible (existing runs still work)
- [ ] Logging doesn't spam (reasonable verbosity)

**Testing:**
- [ ] Load balancing test passes
- [ ] Auto-detection test shows reasonable worker count
- [ ] Parallel execution completes without deadlocks
- [ ] 10-task smoke test runs successfully

---

## References

**Existing Files:**
- `knowledge3d/training/arc_agi/parallel_generator.py` (current implementation)
- `knowledge3d/training/arc_agi/candidate_generator.py` (single worker)
- `knowledge3d/cranium/ptx_runtime/math_core_pool.py` (core allocation)
- `scripts/train_arc_sovereign_loop.py` (training loop)

**W3C Standards:**
- [docs/W3C/K3D_W3C_STANDARDS_ALIGNMENT.md](../docs/W3C/K3D_W3C_STANDARDS_ALIGNMENT.md) (WebGPU parallel execution)

**Architecture Specs:**
- [docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md](../docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md) (Shadow Copy learning)
- [docs/vocabulary/SOVEREIGN_TRAINING_SPECIFICATION.md](../docs/vocabulary/SOVEREIGN_TRAINING_SPECIFICATION.md) (GPU sovereignty)

---

## Appendix: Detailed Calculations

### VRAM Capacity Analysis

**RTX 3060 Specs:**
- Total VRAM: 12,288 MiB
- Current baseline: ~180 MiB (desktop + minimal training)
- Available: 12,108 MiB

**Current Usage (9 workers):**
- Training VRAM: 182 MiB
- VRAM per worker: 182 / 9 ≈ 20 MiB
- Candidates per worker: 6
- VRAM per candidate: 20 / 6 ≈ 3.3 MiB

**Projected Usage (50 workers, 4GB target):**
- Target VRAM: 4,096 MiB
- Safety margin: 2,048 MiB
- Available for workers: 4,096 - 180 = 3,916 MiB
- Workers: 3,916 / 20 ≈ 195 workers (capped at 100 for safety)
- Conservative estimate: 50 workers
- Expected VRAM: 180 + (50 × 20) = 1,180 MiB
- Headroom: 4,096 - 1,180 = 2,916 MiB (plenty of safety margin)

**Projected Usage (100 workers, aggressive):**
- Expected VRAM: 180 + (100 × 20) = 2,180 MiB
- Still under 4GB target ✓

### Throughput Analysis

**Current (9 workers × 6 candidates):**
- Total candidates: 54 per task
- Post-dedup: ~40-50 (depends on semantic overlap)

**Projected (50 workers × 6 candidates):**
- Total candidates: 300 per task
- Post-dedup: ~200-250 (5× increase)

**Projected (100 workers × 6 candidates):**
- Total candidates: 600 per task
- Post-dedup: ~400-500 (10× increase)

**Accuracy Impact:**
- More diverse candidates → better coverage of solution space
- Attractor tracking becomes more meaningful (higher sample size)
- Semantic ranking has more options to choose from
- Expected accuracy improvement: +5-10% (hypothesis, needs validation)

---

**End of Specification**

**Next Steps:**
1. Wait for Run 036 completion
2. Review Run 036 results with Daniel + Claude
3. If architecture is sound, implement Phase 1-2
4. Test with short run (10 tasks × 3 epochs)
5. Deploy to Run 037 if stable
