# CODEX: Diagnostic Framework Implementation — Week 20+

**Date:** February 8, 2026
**Author:** Claude (Architecture Partner)
**For:** Codex (Implementation Partner)
**Status:** 🔴 CRITICAL — Diagnostic debugging required
**Context:** Marathon #2 plateau (ranking applied 100/100 but final_correct = legacy_correct = 28/100)

---

## 🎯 Mission Critical Context

### The Plateau Mystery

**Two 10-iteration marathons show:**
- ✅ Pattern generation works (286 patterns/iteration)
- ✅ Galaxy growth works (+3,663 Grammar entries)
- ✅ Ranking runs (ranking_applied: 100/100)
- ❌ **Ranking doesn't change results** (final_correct = legacy_correct = 28/100)

**Why this matters:**
- We've built a sophisticated ranking system with 5 components
- Shadow Copy learning is active (TRM should improve over iterations)
- But scores PLATEAU across 10 iterations (ARC 28%, Math 33%, LHE 100%)

**Three hypotheses:**
1. **No differentiation:** All patterns score identically (ranking has no signal)
2. **No quality gain:** Generated patterns aren't more accurate than traditional
3. **Top-1 invariant:** Ranking reorders but doesn't change top-1 selection

**This file provides:** Complete architectural specification for diagnostic debugging and closed learning loop to identify and fix the plateau.

---

## 📊 Codex's 10 Architectural Points (Enhanced)

Below are Codex's original insights, each enhanced with detailed architecture, implementation guidance, and success criteria.

---

## 1. Oracle@k Diagnostics

### Original Insight (Codex)
> "We need oracle@k metrics before ranking to separate generation failure from ranking failure. If the correct answer exists somewhere in the top-k, it's a ranking problem. If not, it's a generation problem."

### Enhanced Architecture

**Purpose:** Distinguish between two failure modes:
- **Generation failure:** Correct answer never generated (oracle@k = 0)
- **Ranking failure:** Correct answer generated but ranked poorly (oracle@k = 1, top-1 = 0)

**Implementation Strategy:**

```python
# benchmarks/arc_agi_2_adapter.py - Add oracle@k telemetry

def evaluate_task_with_oracle_metrics(task: dict, kv) -> dict:
    """
    Evaluate single ARC task with oracle@k diagnostics.

    Returns:
        {
            "task_id": str,
            "correct": bool,  # Top-1 accuracy
            "oracle_at_3": bool,  # Correct in top-3?
            "oracle_at_10": bool,  # Correct in top-10?
            "oracle_at_all": bool,  # Correct anywhere in candidates?
            "num_candidates": int,
            "correct_rank": int | None,  # Rank of correct (if present), None if absent
            "ranking_applied": bool,
            "top_5_scores": list[float],  # Top-5 candidate scores for distribution analysis
        }
    """
    # 1. Generate candidates (traditional + autonomous)
    candidates = discover_patterns(task, kv)

    # 2. Check oracle@k BEFORE ranking
    ground_truth = task["test"][0]["output"]
    oracle_results = {}

    for k in [3, 10, len(candidates)]:
        oracle_results[f"oracle_at_{k}"] = any(
            np.array_equal(cand["output"], ground_truth)
            for cand in candidates[:k]
        )

    # Find rank of correct answer (if present)
    correct_rank = next(
        (i for i, cand in enumerate(candidates)
         if np.array_equal(cand["output"], ground_truth)),
        None
    )

    # 3. Apply ranking
    ranked_candidates = _rank_candidates(candidates, task, kv)

    # 4. Check top-1 after ranking
    top_1_correct = np.array_equal(ranked_candidates[0]["output"], ground_truth)

    # 5. Extract top-5 scores for distribution analysis
    top_5_scores = [cand.get("total_score", 0.0) for cand in ranked_candidates[:5]]

    return {
        "task_id": task.get("task_id", "unknown"),
        "correct": top_1_correct,
        "oracle_at_3": oracle_results.get("oracle_at_3", False),
        "oracle_at_10": oracle_results.get("oracle_at_10", False),
        "oracle_at_all": oracle_results.get(f"oracle_at_{len(candidates)}", False),
        "num_candidates": len(candidates),
        "correct_rank": correct_rank,
        "ranking_applied": True,
        "top_5_scores": top_5_scores,
    }
```

**Aggregate Metrics:**

```python
# scripts/run_all_benchmarks.py - Aggregate oracle@k across benchmark

def compute_oracle_metrics(results: list[dict]) -> dict:
    """
    Compute aggregate oracle@k metrics.

    Returns:
        {
            "top_1_accuracy": float,  # Standard accuracy
            "oracle_at_3": float,     # % with correct in top-3
            "oracle_at_10": float,    # % with correct in top-10
            "oracle_at_all": float,   # % with correct anywhere
            "avg_correct_rank": float,  # Average rank of correct (when present)
            "generation_failure_rate": float,  # % where correct never generated
            "ranking_failure_rate": float,     # % where correct generated but not top-1
        }
    """
    total = len(results)

    top_1_correct = sum(r["correct"] for r in results)
    oracle_3 = sum(r["oracle_at_3"] for r in results)
    oracle_10 = sum(r["oracle_at_10"] for r in results)
    oracle_all = sum(r["oracle_at_all"] for r in results)

    # Compute average rank of correct (when present)
    correct_ranks = [r["correct_rank"] for r in results if r["correct_rank"] is not None]
    avg_correct_rank = sum(correct_ranks) / len(correct_ranks) if correct_ranks else None

    # Failure mode decomposition
    generation_failures = sum(not r["oracle_at_all"] for r in results)
    ranking_failures = sum(r["oracle_at_all"] and not r["correct"] for r in results)

    return {
        "top_1_accuracy": top_1_correct / total,
        "oracle_at_3": oracle_3 / total,
        "oracle_at_10": oracle_10 / total,
        "oracle_at_all": oracle_all / total,
        "avg_correct_rank": avg_correct_rank,
        "generation_failure_rate": generation_failures / total,
        "ranking_failure_rate": ranking_failures / total,
    }
```

**Expected Diagnostic Outcomes:**

| Scenario | oracle@all | top-1 | Diagnosis |
|----------|-----------|-------|-----------|
| **Generation problem** | 20% | 20% | Need more/better pattern generation |
| **Ranking problem** | 80% | 28% | Ranking doesn't differentiate (current hypothesis!) |
| **Both problems** | 40% | 28% | Need both generation + ranking fixes |
| **Working correctly** | 80% | 75% | Small ranking gap is normal |

**Success Criteria:**
- ✅ Oracle@k metrics captured for every task
- ✅ Aggregate metrics show which failure mode dominates
- ✅ If oracle@all >> top-1, ranking is the bottleneck
- ✅ If oracle@all ≈ top-1, generation is the bottleneck

---

## 2. Counterfactual Evaluation

### Original Insight (Codex)
> "For each iteration, evaluate what WOULD have happened if we used the previous iteration's ranking weights. This proves whether learning actually improved the ranker."

### Enhanced Architecture

**Purpose:** Causally prove that TRM weight updates improve ranking quality across iterations.

**Implementation Strategy:**

```python
# knowledge3d/knowledgeverse/trm_weight_store.py - Add snapshot API

class TRMWeightStore:
    """Enhanced with iteration snapshots for counterfactual evaluation."""

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.snapshots_dir = store_path.parent / "trm_snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)

    def save_iteration_snapshot(self, iteration: int, weights: dict):
        """Save TRM weights at specific iteration for counterfactual replay."""
        snapshot_path = self.snapshots_dir / f"iteration_{iteration:03d}.json"
        with open(snapshot_path, "w") as f:
            json.dump({
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "weights": weights,
            }, f, indent=2)

    def load_iteration_snapshot(self, iteration: int) -> dict | None:
        """Load TRM weights from specific iteration."""
        snapshot_path = self.snapshots_dir / f"iteration_{iteration:03d}.json"
        if not snapshot_path.exists():
            return None

        with open(snapshot_path, "r") as f:
            data = json.load(f)

        return data["weights"]
```

**Counterfactual Benchmark Runner:**

```python
# scripts/run_counterfactual_evaluation.py - NEW FILE

def run_counterfactual_iteration(
    benchmark_name: str,
    current_iteration: int,
    previous_iteration: int,
    kv
) -> dict:
    """
    Run benchmark with PREVIOUS iteration's weights on CURRENT iteration's tasks.

    This answers: "Would the old ranker have performed better or worse?"

    Returns:
        {
            "benchmark": str,
            "current_iteration": int,
            "using_weights_from": int,
            "top_1_accuracy": float,
            "oracle_at_all": float,
            "avg_correct_rank": float | None,
        }
    """
    # 1. Load previous iteration's weights
    weight_store = TRMWeightStore(Path("galaxy_universe/trm_weights.json"))
    previous_weights = weight_store.load_iteration_snapshot(previous_iteration)

    if previous_weights is None:
        raise ValueError(f"No snapshot for iteration {previous_iteration}")

    # 2. Temporarily apply previous weights
    current_weights = kv.trm.get_weights()  # Save current
    kv.trm.set_weights(previous_weights)     # Apply previous

    # 3. Run benchmark with previous weights
    results = run_single_benchmark(benchmark_name, kv, enriched=True)

    # 4. Restore current weights
    kv.trm.set_weights(current_weights)

    # 5. Compute metrics
    oracle_metrics = compute_oracle_metrics(results)

    return {
        "benchmark": benchmark_name,
        "current_iteration": current_iteration,
        "using_weights_from": previous_iteration,
        "top_1_accuracy": oracle_metrics["top_1_accuracy"],
        "oracle_at_all": oracle_metrics["oracle_at_all"],
        "avg_correct_rank": oracle_metrics["avg_correct_rank"],
    }

def run_full_counterfactual_analysis(marathon_dir: Path) -> dict:
    """
    Run counterfactual evaluation for entire marathon.

    For each iteration i > 0:
    - Run benchmark with weights from iteration i (actual)
    - Run benchmark with weights from iteration i-1 (counterfactual)
    - Compare: did learning improve ranking?

    Returns:
        {
            "iterations": [
                {
                    "iteration": int,
                    "actual_accuracy": float,  # Using current weights
                    "counterfactual_accuracy": float,  # Using previous weights
                    "delta": float,  # actual - counterfactual (should be positive if learning works!)
                    "oracle_at_all": float,
                },
                ...
            ],
            "avg_improvement_per_iteration": float,  # Average delta across all iterations
            "learning_effective": bool,  # True if avg_improvement > 0
        }
    """
    kv = Knowledgeverse()
    iterations_data = []

    # Load marathon metadata
    with open(marathon_dir / "marathon_analysis.json") as f:
        marathon_data = json.load(f)

    num_iterations = marathon_data["num_iterations"]

    for i in range(1, num_iterations):  # Start from 1 (need i-1)
        # Actual (using current iteration's weights)
        weight_store = TRMWeightStore(Path("galaxy_universe/trm_weights.json"))
        current_weights = weight_store.load_iteration_snapshot(i)
        kv.trm.set_weights(current_weights)

        actual_results = run_single_benchmark("arc_agi_2", kv, enriched=True)
        actual_metrics = compute_oracle_metrics(actual_results)

        # Counterfactual (using previous iteration's weights)
        counterfactual_results = run_counterfactual_iteration(
            "arc_agi_2", i, i - 1, kv
        )

        delta = actual_metrics["top_1_accuracy"] - counterfactual_results["top_1_accuracy"]

        iterations_data.append({
            "iteration": i,
            "actual_accuracy": actual_metrics["top_1_accuracy"],
            "counterfactual_accuracy": counterfactual_results["top_1_accuracy"],
            "delta": delta,
            "oracle_at_all": actual_metrics["oracle_at_all"],
        })

    # Aggregate analysis
    avg_improvement = sum(it["delta"] for it in iterations_data) / len(iterations_data)

    return {
        "iterations": iterations_data,
        "avg_improvement_per_iteration": avg_improvement,
        "learning_effective": avg_improvement > 0,
    }
```

**Expected Diagnostic Outcomes:**

| Scenario | avg_improvement | Diagnosis |
|----------|----------------|-----------|
| **Learning works** | +5% to +10% | TRM weights improving over iterations |
| **No learning** | -1% to +1% | Weights changing but not improving (current hypothesis!) |
| **Regression** | -5% or worse | Learning hurts performance (overfit?) |

**Success Criteria:**
- ✅ Snapshots saved at each iteration
- ✅ Counterfactual evaluation runs for all iterations
- ✅ Delta computed (actual - counterfactual)
- ✅ If avg_improvement ≈ 0, learning is NOT effective (confirms plateau)

---

## 3. Adaptive Ranking Weights

### Original Insight (Codex)
> "Ranking weights should adapt per iteration based on which components correlated with success. If grammar_confidence predicts success, increase its weight. If cross_modal doesn't help, decrease it."

### Enhanced Architecture

**Purpose:** Meta-learn which ranking components actually predict correctness, then adjust weights accordingly.

**Implementation Strategy:**

```python
# knowledge3d/knowledgeverse/adaptive_ranker.py - NEW FILE

class AdaptiveRanker:
    """
    Meta-learns ranking component weights based on correlation with ground truth.

    For each iteration:
    1. Collect (candidate_scores, ground_truth_label) pairs
    2. Compute correlation between each component and correctness
    3. Update component weights via gradient-like update
    4. Store weights for next iteration
    """

    def __init__(self, initial_weights: dict | None = None):
        self.weights = initial_weights or {
            "grammar_confidence": 1.0,
            "cross_modal_agreement": 1.0,
            "source_priority": 1.0,
            "compositional_bonus": 1.0,
            "pattern_reuse_bonus": 1.0,
        }

        self.learning_rate = 0.1  # How fast to adapt weights
        self.history = []  # Track (component_scores, label) for analysis

    def record_candidate(
        self,
        component_scores: dict[str, float],
        is_correct: bool
    ):
        """Record candidate with component scores and ground truth label."""
        self.history.append({
            "scores": component_scores.copy(),
            "correct": is_correct,
        })

    def compute_component_correlations(self) -> dict[str, float]:
        """
        Compute correlation between each component and correctness.

        Returns:
            {
                "grammar_confidence": float,  # Pearson correlation with correctness
                "cross_modal_agreement": float,
                ...
            }
        """
        if len(self.history) < 10:  # Need minimum data
            return {k: 0.0 for k in self.weights.keys()}

        correlations = {}
        labels = np.array([h["correct"] for h in self.history])

        for component in self.weights.keys():
            scores = np.array([h["scores"].get(component, 0.0) for h in self.history])

            # Pearson correlation
            if scores.std() > 0:  # Avoid division by zero
                corr = np.corrcoef(scores, labels)[0, 1]
                correlations[component] = corr if not np.isnan(corr) else 0.0
            else:
                correlations[component] = 0.0

        return correlations

    def update_weights(self):
        """
        Update component weights based on correlation with correctness.

        Gradient-like update:
        - If component positively correlated with correctness, increase weight
        - If negatively correlated, decrease weight
        - If uncorrelated, keep weight
        """
        correlations = self.compute_component_correlations()

        for component, corr in correlations.items():
            # Update: weight += learning_rate * correlation
            # Clamp to [0.1, 2.0] to prevent extreme values
            delta = self.learning_rate * corr
            self.weights[component] = np.clip(
                self.weights[component] + delta,
                0.1,  # Minimum weight
                2.0   # Maximum weight
            )

        # Clear history after update (start fresh for next iteration)
        self.history = []

        return correlations

    def get_weights(self) -> dict[str, float]:
        """Return current component weights."""
        return self.weights.copy()

    def save(self, path: Path):
        """Save weights and learning state."""
        with open(path, "w") as f:
            json.dump({
                "weights": self.weights,
                "learning_rate": self.learning_rate,
                "history_size": len(self.history),
            }, f, indent=2)

    def load(self, path: Path):
        """Load weights from disk."""
        with open(path) as f:
            data = json.load(f)

        self.weights = data["weights"]
        self.learning_rate = data.get("learning_rate", 0.1)
```

**Integration with ARC Benchmark:**

```python
# benchmarks/arc_agi_2_adapter.py - Integrate AdaptiveRanker

def _rank_candidates_adaptive(
    candidates: list[dict],
    task: dict,
    kv,
    adaptive_ranker: AdaptiveRanker
) -> list[dict]:
    """
    Rank candidates using adaptive weights.

    For each candidate:
    1. Compute component scores
    2. Combine using adaptive weights
    3. Record (scores, correctness) for meta-learning
    4. Return ranked candidates
    """
    ground_truth = task["test"][0]["output"]

    for cand in candidates:
        # Compute component scores (existing logic)
        component_scores = {
            "grammar_confidence": _compute_grammar_confidence(cand, kv),
            "cross_modal_agreement": _compute_cross_modal_agreement(cand, kv),
            "source_priority": _compute_source_priority(cand),
            "compositional_bonus": _compute_compositional_bonus(cand, kv),
            "pattern_reuse_bonus": _compute_pattern_reuse_bonus(cand, kv),
        }

        # Get adaptive weights
        weights = adaptive_ranker.get_weights()

        # Weighted sum
        total_score = sum(
            component_scores[k] * weights[k]
            for k in component_scores.keys()
        )

        cand["component_scores"] = component_scores
        cand["total_score"] = total_score

        # Record for meta-learning
        is_correct = np.array_equal(cand["output"], ground_truth)
        adaptive_ranker.record_candidate(component_scores, is_correct)

    # Sort by total score (descending)
    return sorted(candidates, key=lambda c: c["total_score"], reverse=True)
```

**Marathon Integration:**

```python
# scripts/iterative_learning_marathon.py - Add adaptive ranker

def run_marathon_with_adaptive_ranking(
    num_iterations: int = 10,
    output_dir: Path = Path("marathon_results")
):
    """Run marathon with adaptive ranking weight updates."""

    kv = Knowledgeverse()
    adaptive_ranker = AdaptiveRanker()

    results = []

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # Run benchmarks with current weights
        iteration_results = run_all_benchmarks(kv, adaptive_ranker=adaptive_ranker)

        # Update weights based on correlation analysis
        correlations = adaptive_ranker.update_weights()

        # Record iteration data
        results.append({
            "iteration": iteration,
            "accuracy": iteration_results["arc_agi_2"]["enriched"]["accuracy"],
            "adaptive_weights": adaptive_ranker.get_weights(),
            "component_correlations": correlations,
        })

        # Save weights
        adaptive_ranker.save(output_dir / f"adaptive_weights_iter_{iteration:03d}.json")

    return results
```

**Expected Diagnostic Outcomes:**

| Component | Correlation | Weight Δ | Interpretation |
|-----------|-------------|----------|----------------|
| grammar_confidence | +0.6 | +0.06 | Strong predictor, increase weight |
| cross_modal | +0.1 | +0.01 | Weak predictor, slight increase |
| source_priority | -0.2 | -0.02 | Negatively correlated, decrease |
| compositional | 0.0 | 0.0 | No correlation, keep weight |

**Success Criteria:**
- ✅ Component correlations computed each iteration
- ✅ Weights adapt based on correlation
- ✅ If all correlations ≈ 0, components have no predictive power (confirms hypothesis!)
- ✅ Track weight evolution over 10 iterations

---

## 4. Exploration Policy

### Original Insight (Codex)
> "When ranking uncertainty is high (top candidates have similar scores), inject exploration by randomly selecting from top-k instead of always picking top-1. This prevents getting stuck in local optima."

### Enhanced Architecture

**Purpose:** Balance exploitation (use best known ranking) vs exploration (try alternatives when uncertain).

**Implementation Strategy:**

```python
# knowledge3d/knowledgeverse/exploration_policy.py - NEW FILE

class ExplorationPolicy:
    """
    Epsilon-greedy exploration for candidate ranking.

    When ranking uncertainty is high (similar scores), explore by sampling
    from top-k instead of always picking top-1.
    """

    def __init__(
        self,
        epsilon: float = 0.2,  # Exploration rate (20% of time, explore)
        uncertainty_threshold: float = 0.1,  # Score difference threshold
        top_k: int = 3,  # Explore within top-3
    ):
        self.epsilon = epsilon
        self.uncertainty_threshold = uncertainty_threshold
        self.top_k = top_k
        self.rng = np.random.default_rng(seed=42)

    def should_explore(self, ranked_candidates: list[dict]) -> bool:
        """
        Decide whether to explore based on ranking uncertainty.

        Uncertainty is high when:
        - Top-k candidates have similar scores (std < threshold)
        - Random exploration triggered (epsilon probability)
        """
        if len(ranked_candidates) < 2:
            return False

        # Check score distribution in top-k
        top_k_scores = [
            cand.get("total_score", 0.0)
            for cand in ranked_candidates[:self.top_k]
        ]

        score_std = np.std(top_k_scores)
        high_uncertainty = score_std < self.uncertainty_threshold

        # Epsilon-greedy: explore with probability epsilon
        random_explore = self.rng.random() < self.epsilon

        return high_uncertainty or random_explore

    def select_candidate(self, ranked_candidates: list[dict]) -> dict:
        """
        Select candidate with exploration policy.

        Returns:
            - Top-1 if exploiting
            - Random from top-k if exploring
        """
        if not self.should_explore(ranked_candidates):
            # Exploit: return top-1
            return ranked_candidates[0]

        # Explore: sample from top-k
        k = min(self.top_k, len(ranked_candidates))
        selected_idx = self.rng.integers(0, k)

        return ranked_candidates[selected_idx]

    def get_exploration_stats(self, ranked_candidates: list[dict]) -> dict:
        """Return telemetry about exploration decision."""
        top_k_scores = [
            cand.get("total_score", 0.0)
            for cand in ranked_candidates[:self.top_k]
        ]

        return {
            "top_k_scores": top_k_scores,
            "score_std": np.std(top_k_scores),
            "high_uncertainty": np.std(top_k_scores) < self.uncertainty_threshold,
            "will_explore": self.should_explore(ranked_candidates),
        }
```

**Integration with ARC Benchmark:**

```python
# benchmarks/arc_agi_2_adapter.py - Add exploration policy

def solve_task_with_exploration(
    task: dict,
    kv,
    adaptive_ranker: AdaptiveRanker,
    exploration_policy: ExplorationPolicy
) -> dict:
    """
    Solve ARC task with exploration policy.

    Returns:
        {
            "output": np.ndarray,
            "exploited": bool,  # True if top-1 selected, False if explored
            "exploration_stats": dict,
        }
    """
    # Generate + rank candidates
    candidates = discover_patterns(task, kv)
    ranked_candidates = _rank_candidates_adaptive(candidates, task, kv, adaptive_ranker)

    # Select with exploration policy
    selected = exploration_policy.select_candidate(ranked_candidates)
    exploration_stats = exploration_policy.get_exploration_stats(ranked_candidates)

    return {
        "output": selected["output"],
        "exploited": not exploration_stats["will_explore"],
        "exploration_stats": exploration_stats,
    }
```

**Marathon Integration:**

```python
# scripts/iterative_learning_marathon.py - Track exploration statistics

def run_marathon_with_exploration(num_iterations: int = 10):
    """Run marathon tracking exploration vs exploitation."""

    kv = Knowledgeverse()
    adaptive_ranker = AdaptiveRanker()
    exploration_policy = ExplorationPolicy(epsilon=0.2)

    results = []

    for iteration in range(num_iterations):
        iteration_stats = {
            "iteration": iteration,
            "exploited_count": 0,
            "explored_count": 0,
            "exploited_correct": 0,
            "explored_correct": 0,
        }

        # Run benchmark
        for task in load_arc_tasks():
            result = solve_task_with_exploration(task, kv, adaptive_ranker, exploration_policy)

            if result["exploited"]:
                iteration_stats["exploited_count"] += 1
                if result["correct"]:
                    iteration_stats["exploited_correct"] += 1
            else:
                iteration_stats["explored_count"] += 1
                if result["correct"]:
                    iteration_stats["explored_correct"] += 1

        # Compute exploitation vs exploration accuracy
        iteration_stats["exploited_accuracy"] = (
            iteration_stats["exploited_correct"] / iteration_stats["exploited_count"]
            if iteration_stats["exploited_count"] > 0 else 0.0
        )
        iteration_stats["explored_accuracy"] = (
            iteration_stats["explored_correct"] / iteration_stats["explored_count"]
            if iteration_stats["explored_count"] > 0 else 0.0
        )

        results.append(iteration_stats)

    return results
```

**Expected Diagnostic Outcomes:**

| Scenario | Exploited Acc | Explored Acc | Diagnosis |
|----------|--------------|--------------|-----------|
| **Exploration helps** | 25% | 35% | Ranking misses good alternatives |
| **Exploitation better** | 35% | 20% | Top-1 is reliable, don't explore |
| **No difference** | 28% | 28% | Exploration doesn't help (score distribution problem!) |

**Success Criteria:**
- ✅ Track exploitation vs exploration decisions
- ✅ Measure accuracy for each mode
- ✅ If explored_accuracy ≈ exploited_accuracy, ranking scores don't differentiate
- ✅ Tune epsilon based on exploration effectiveness

---

## 5. Candidate Quality Gating

### Original Insight (Codex)
> "Don't blindly store all generated patterns in Galaxy. Use quality gates: only store patterns that score above threshold or that solve at least one task correctly."

### Enhanced Architecture

**Purpose:** Prevent Galaxy pollution with low-quality patterns. Only store patterns that demonstrate utility.

**Implementation Strategy:**

```python
# knowledge3d/knowledgeverse/quality_gating.py - NEW FILE

class QualityGate:
    """
    Filter generated patterns before Galaxy storage.

    Patterns must pass quality criteria:
    1. Minimum score threshold (e.g., total_score > 0.5)
    2. Solved at least one task correctly
    3. Novel (not duplicate of existing pattern)
    4. Compositionally sound (valid RPN program)
    """

    def __init__(
        self,
        min_score: float = 0.5,
        require_success: bool = True,
        check_novelty: bool = True,
    ):
        self.min_score = min_score
        self.require_success = require_success
        self.check_novelty = check_novelty

        self.stats = {
            "total_generated": 0,
            "passed_score": 0,
            "passed_success": 0,
            "passed_novelty": 0,
            "passed_all": 0,
        }

    def passes_score_gate(self, pattern: dict) -> bool:
        """Check if pattern scores above threshold."""
        return pattern.get("total_score", 0.0) >= self.min_score

    def passes_success_gate(self, pattern: dict, correctness_history: dict) -> bool:
        """Check if pattern solved at least one task correctly."""
        if not self.require_success:
            return True

        pattern_id = pattern.get("pattern_id")
        return correctness_history.get(pattern_id, {}).get("correct_count", 0) > 0

    def passes_novelty_gate(self, pattern: dict, existing_patterns: list[dict]) -> bool:
        """Check if pattern is novel (not duplicate)."""
        if not self.check_novelty:
            return True

        # Content-based deduplication
        pattern_rpn = pattern.get("rpn_program", "")

        for existing in existing_patterns:
            existing_rpn = existing.get("rpn_program", "")
            if pattern_rpn == existing_rpn:
                return False  # Duplicate

        return True

    def should_store(
        self,
        pattern: dict,
        correctness_history: dict,
        existing_patterns: list[dict]
    ) -> tuple[bool, dict]:
        """
        Decide if pattern should be stored in Galaxy.

        Returns:
            (should_store: bool, gate_results: dict)
        """
        self.stats["total_generated"] += 1

        gate_results = {
            "passed_score": self.passes_score_gate(pattern),
            "passed_success": self.passes_success_gate(pattern, correctness_history),
            "passed_novelty": self.passes_novelty_gate(pattern, existing_patterns),
        }

        # Update stats
        for gate, passed in gate_results.items():
            if passed:
                self.stats[gate] += 1

        # Must pass ALL gates
        passed_all = all(gate_results.values())
        if passed_all:
            self.stats["passed_all"] += 1

        return passed_all, gate_results

    def get_stats(self) -> dict:
        """Return quality gating statistics."""
        total = self.stats["total_generated"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "pass_rate": self.stats["passed_all"] / total,
            "score_filter_rate": self.stats["passed_score"] / total,
            "success_filter_rate": self.stats["passed_success"] / total,
            "novelty_filter_rate": self.stats["passed_novelty"] / total,
        }
```

**Integration with Pattern Discovery:**

```python
# benchmarks/arc_agi_2_adapter.py - Add quality gating

def discover_patterns_with_quality_gating(
    task: dict,
    kv,
    quality_gate: QualityGate,
    correctness_history: dict
) -> tuple[list[dict], dict]:
    """
    Discover patterns with quality gating before Galaxy storage.

    Returns:
        (patterns: list[dict], gating_stats: dict)
    """
    # Generate candidates (traditional + autonomous)
    all_patterns = discover_patterns(task, kv)

    # Get existing patterns from Galaxy (for novelty check)
    existing_patterns = kv.galaxy_manager.query("transformation", specialist="grammar", top_k=1000)

    # Apply quality gates
    filtered_patterns = []
    gating_stats = []

    for pattern in all_patterns:
        should_store, gate_results = quality_gate.should_store(
            pattern,
            correctness_history,
            existing_patterns
        )

        if should_store:
            filtered_patterns.append(pattern)

        gating_stats.append({
            "pattern_id": pattern.get("pattern_id"),
            "total_score": pattern.get("total_score", 0.0),
            "gate_results": gate_results,
            "stored": should_store,
        })

    # Store ONLY filtered patterns in Galaxy
    for pattern in filtered_patterns:
        kv.galaxy_manager.add_entry(
            galaxy="Grammar",
            entry={
                "id": pattern["pattern_id"],
                "rpn_program": pattern["rpn_program"],
                "source": "autonomous_generation_quality_gated",
                "timestamp": datetime.now().isoformat(),
            }
        )

    return filtered_patterns, {
        "total_generated": len(all_patterns),
        "total_stored": len(filtered_patterns),
        "filter_rate": 1 - (len(filtered_patterns) / len(all_patterns)) if all_patterns else 0,
        "per_pattern_stats": gating_stats,
    }
```

**Expected Diagnostic Outcomes:**

| Scenario | Pass Rate | Diagnosis |
|----------|-----------|-----------|
| **Too strict** | 5% | Missing useful patterns, relax thresholds |
| **Too loose** | 95% | Storing junk, tighten thresholds |
| **Balanced** | 30-50% | Good signal-to-noise ratio |

**Success Criteria:**
- ✅ Only high-quality patterns stored in Galaxy
- ✅ Track pass rates for each gate
- ✅ If pass_rate is very high, gates aren't selective enough
- ✅ Monitor Galaxy size growth (should slow if gating works)

---

## 6. Ablation Matrix (Causal Proof)

### Original Insight (Codex)
> "Run 6 modes in matrix: (traditional vs autonomous) × (no ranking vs legacy ranking vs adaptive ranking). This causally proves which components contribute to performance."

### Enhanced Architecture

**Purpose:** Isolate which components (generation method, ranking method) actually improve performance.

**Implementation Strategy:**

```python
# scripts/run_ablation_matrix.py - NEW FILE

def run_ablation_matrix(
    benchmark_name: str = "arc_agi_2",
    num_tasks: int = 100,
    output_dir: Path = Path("ablation_results")
) -> dict:
    """
    Run 6-mode ablation matrix to causally prove component contributions.

    Modes:
    1. traditional_no_ranking: Traditional patterns, no ranking (random selection)
    2. traditional_legacy_ranking: Traditional patterns, legacy ranking (5 components)
    3. traditional_adaptive_ranking: Traditional patterns, adaptive ranking
    4. autonomous_no_ranking: Autonomous patterns, no ranking
    5. autonomous_legacy_ranking: Autonomous patterns, legacy ranking
    6. autonomous_adaptive_ranking: Autonomous patterns, adaptive ranking

    Returns:
        {
            "modes": {
                "traditional_no_ranking": {"accuracy": float, "oracle_at_all": float},
                "traditional_legacy_ranking": {...},
                ...
            },
            "causal_analysis": {
                "autonomous_lift": float,  # Δ from traditional to autonomous (with ranking held constant)
                "ranking_lift": float,     # Δ from no ranking to ranking (with generation held constant)
                "adaptive_lift": float,    # Δ from legacy to adaptive (with generation held constant)
            }
        }
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    kv = Knowledgeverse()

    modes = {
        "traditional_no_ranking": {
            "generation": "traditional",
            "ranking": None,
        },
        "traditional_legacy_ranking": {
            "generation": "traditional",
            "ranking": "legacy",
        },
        "traditional_adaptive_ranking": {
            "generation": "traditional",
            "ranking": "adaptive",
        },
        "autonomous_no_ranking": {
            "generation": "autonomous",
            "ranking": None,
        },
        "autonomous_legacy_ranking": {
            "generation": "autonomous",
            "ranking": "legacy",
        },
        "autonomous_adaptive_ranking": {
            "generation": "autonomous",
            "ranking": "adaptive",
        },
    }

    results = {}

    for mode_name, mode_config in modes.items():
        print(f"\n=== Running mode: {mode_name} ===")

        # Run benchmark with specific configuration
        mode_results = run_benchmark_with_config(
            benchmark_name,
            kv,
            generation_method=mode_config["generation"],
            ranking_method=mode_config["ranking"],
            num_tasks=num_tasks,
        )

        oracle_metrics = compute_oracle_metrics(mode_results)

        results[mode_name] = {
            "accuracy": oracle_metrics["top_1_accuracy"],
            "oracle_at_all": oracle_metrics["oracle_at_all"],
            "avg_correct_rank": oracle_metrics["avg_correct_rank"],
            "generation_failure_rate": oracle_metrics["generation_failure_rate"],
            "ranking_failure_rate": oracle_metrics["ranking_failure_rate"],
        }

        # Save per-mode results
        with open(output_dir / f"{mode_name}.json", "w") as f:
            json.dump(results[mode_name], f, indent=2)

    # Causal analysis
    causal_analysis = compute_causal_effects(results)

    # Save full matrix
    matrix_output = {
        "modes": results,
        "causal_analysis": causal_analysis,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "ablation_matrix.json", "w") as f:
        json.dump(matrix_output, f, indent=2)

    return matrix_output

def compute_causal_effects(results: dict) -> dict:
    """
    Compute causal effects by comparing mode pairs.

    Returns:
        {
            "autonomous_lift": float,  # autonomous vs traditional (with ranking=legacy)
            "ranking_lift": float,     # legacy vs no_ranking (with generation=traditional)
            "adaptive_lift": float,    # adaptive vs legacy (with generation=traditional)
        }
    """
    # Autonomous lift: hold ranking constant (legacy), vary generation
    autonomous_lift = (
        results["autonomous_legacy_ranking"]["accuracy"] -
        results["traditional_legacy_ranking"]["accuracy"]
    )

    # Ranking lift: hold generation constant (traditional), add ranking
    ranking_lift = (
        results["traditional_legacy_ranking"]["accuracy"] -
        results["traditional_no_ranking"]["accuracy"]
    )

    # Adaptive lift: hold generation constant (traditional), vary ranking method
    adaptive_lift = (
        results["traditional_adaptive_ranking"]["accuracy"] -
        results["traditional_legacy_ranking"]["accuracy"]
    )

    return {
        "autonomous_lift": autonomous_lift,
        "ranking_lift": ranking_lift,
        "adaptive_lift": adaptive_lift,
        "autonomous_significant": abs(autonomous_lift) > 0.05,  # >5% change
        "ranking_significant": abs(ranking_lift) > 0.05,
        "adaptive_significant": abs(adaptive_lift) > 0.05,
    }

def run_benchmark_with_config(
    benchmark_name: str,
    kv,
    generation_method: str,
    ranking_method: str | None,
    num_tasks: int
) -> list[dict]:
    """Run benchmark with specific generation and ranking configuration."""

    tasks = load_benchmark_tasks(benchmark_name)[:num_tasks]
    results = []

    for task in tasks:
        # Generate candidates based on method
        if generation_method == "traditional":
            candidates = discover_patterns_traditional(task, kv)
        elif generation_method == "autonomous":
            candidates = discover_patterns_autonomous(task, kv)
        else:
            raise ValueError(f"Unknown generation method: {generation_method}")

        # Apply ranking based on method
        if ranking_method is None:
            # No ranking: random selection
            selected = candidates[0] if candidates else None
        elif ranking_method == "legacy":
            # Legacy 5-component ranking
            ranked = _rank_candidates(candidates, task, kv)
            selected = ranked[0] if ranked else None
        elif ranking_method == "adaptive":
            # Adaptive ranking with meta-learning
            adaptive_ranker = AdaptiveRanker()
            ranked = _rank_candidates_adaptive(candidates, task, kv, adaptive_ranker)
            selected = ranked[0] if ranked else None
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")

        # Evaluate
        ground_truth = task["test"][0]["output"]
        correct = (
            np.array_equal(selected["output"], ground_truth)
            if selected else False
        )

        results.append({
            "task_id": task.get("task_id"),
            "correct": correct,
            "num_candidates": len(candidates),
        })

    return results
```

**Expected Diagnostic Outcomes:**

| Effect | Value | Diagnosis |
|--------|-------|-----------|
| **autonomous_lift** | +10% | Autonomous generation helps! |
| **autonomous_lift** | 0% | Autonomous doesn't improve (current hypothesis!) |
| **ranking_lift** | +5% | Ranking helps differentiate |
| **ranking_lift** | 0% | Ranking doesn't help (current hypothesis!) |
| **adaptive_lift** | +3% | Adaptive meta-learning works |
| **adaptive_lift** | 0% | Adaptive doesn't improve over legacy |

**Success Criteria:**
- ✅ All 6 modes run successfully
- ✅ Causal effects computed (autonomous_lift, ranking_lift, adaptive_lift)
- ✅ If all lifts ≈ 0, NONE of our interventions help (critical insight!)
- ✅ Identify which components contribute to performance

---

## 7. Per-Source Precision/Recall

### Original Insight (Codex)
> "Track which pattern sources (traditional, autonomous, symlink, cross-modal) have highest precision. Store per-source metrics in Grammar Galaxy for TRM to learn source reliability."

### Enhanced Architecture

**Purpose:** Learn which pattern generation sources are most reliable, then prioritize them.

**Implementation Strategy:**

```python
# knowledge3d/knowledgeverse/source_tracker.py - NEW FILE

class SourceTracker:
    """
    Track precision/recall for each pattern generation source.

    Sources:
    - traditional_grammar: Traditional Grammar Galaxy queries
    - autonomous_3d: Autonomous generation from 3D Objects
    - autonomous_reality: Autonomous generation from Reality
    - symlink_cross_modal: Cross-modal symlink compositions
    - specialist_generated: Patterns from specialist sub-workers
    """

    def __init__(self):
        self.stats = defaultdict(lambda: {
            "generated": 0,
            "correct": 0,
            "selected_as_top1": 0,
            "top1_correct": 0,
        })

    def record_generation(self, source: str, is_correct: bool):
        """Record that a pattern was generated from this source."""
        self.stats[source]["generated"] += 1
        if is_correct:
            self.stats[source]["correct"] += 1

    def record_top1_selection(self, source: str, is_correct: bool):
        """Record that a pattern from this source was selected as top-1."""
        self.stats[source]["selected_as_top1"] += 1
        if is_correct:
            self.stats[source]["top1_correct"] += 1

    def get_source_metrics(self) -> dict:
        """
        Compute precision/recall for each source.

        Returns:
            {
                "traditional_grammar": {
                    "precision": float,  # correct / generated
                    "recall": float,     # top1_correct / total_tasks
                    "selection_rate": float,  # selected_as_top1 / generated
                },
                ...
            }
        """
        metrics = {}

        for source, stats in self.stats.items():
            precision = stats["correct"] / stats["generated"] if stats["generated"] > 0 else 0.0
            selection_rate = stats["selected_as_top1"] / stats["generated"] if stats["generated"] > 0 else 0.0
            top1_precision = stats["top1_correct"] / stats["selected_as_top1"] if stats["selected_as_top1"] > 0 else 0.0

            metrics[source] = {
                "precision": precision,
                "selection_rate": selection_rate,
                "top1_precision": top1_precision,
                "generated": stats["generated"],
                "correct": stats["correct"],
            }

        return metrics

    def store_in_galaxy(self, kv):
        """Store source reliability metrics in Grammar Galaxy for TRM learning."""
        metrics = self.get_source_metrics()

        for source, stats in metrics.items():
            # Create RPN program encoding source reliability
            # Format: "SOURCE_PRECISION SOURCE_SELECTION_RATE SOURCE_TOP1_PRECISION MUL MUL"
            # This gives overall "source quality score" that TRM can use

            rpn_program = (
                f"{stats['precision']:.4f} "
                f"{stats['selection_rate']:.4f} "
                f"{stats['top1_precision']:.4f} "
                f"MUL MUL"
            )

            kv.galaxy_manager.add_entry(
                galaxy="Grammar",
                entry={
                    "id": f"source_reliability_{source}",
                    "rpn_program": rpn_program,
                    "metadata": {
                        "source": source,
                        "precision": stats["precision"],
                        "selection_rate": stats["selection_rate"],
                        "top1_precision": stats["top1_precision"],
                        "generated_count": stats["generated"],
                        "correct_count": stats["correct"],
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            )
```

**Integration with ARC Benchmark:**

```python
# benchmarks/arc_agi_2_adapter.py - Track source metrics

def discover_patterns_with_source_tracking(
    task: dict,
    kv,
    source_tracker: SourceTracker
) -> list[dict]:
    """Generate patterns and track which sources produce correct answers."""

    ground_truth = task["test"][0]["output"]
    all_patterns = []

    # Traditional Grammar patterns
    traditional_patterns = _discover_traditional_patterns(task, kv)
    for pattern in traditional_patterns:
        pattern["source"] = "traditional_grammar"
        is_correct = np.array_equal(pattern["output"], ground_truth)
        source_tracker.record_generation("traditional_grammar", is_correct)
        all_patterns.append(pattern)

    # Autonomous 3D Objects patterns
    autonomous_3d = _discover_autonomous_3d_patterns(task, kv)
    for pattern in autonomous_3d:
        pattern["source"] = "autonomous_3d"
        is_correct = np.array_equal(pattern["output"], ground_truth)
        source_tracker.record_generation("autonomous_3d", is_correct)
        all_patterns.append(pattern)

    # Autonomous Reality patterns
    autonomous_reality = _discover_autonomous_reality_patterns(task, kv)
    for pattern in autonomous_reality:
        pattern["source"] = "autonomous_reality"
        is_correct = np.array_equal(pattern["output"], ground_truth)
        source_tracker.record_generation("autonomous_reality", is_correct)
        all_patterns.append(pattern)

    # Cross-modal symlinks
    cross_modal = _discover_cross_modal_patterns(task, kv)
    for pattern in cross_modal:
        pattern["source"] = "symlink_cross_modal"
        is_correct = np.array_equal(pattern["output"], ground_truth)
        source_tracker.record_generation("symlink_cross_modal", is_correct)
        all_patterns.append(pattern)

    return all_patterns

def solve_task_with_source_tracking(
    task: dict,
    kv,
    source_tracker: SourceTracker
) -> dict:
    """Solve task and record which source was selected."""

    # Generate with source tracking
    patterns = discover_patterns_with_source_tracking(task, kv, source_tracker)

    # Rank candidates
    ranked = _rank_candidates(patterns, task, kv)

    # Record top-1 selection
    if ranked:
        top1 = ranked[0]
        ground_truth = task["test"][0]["output"]
        is_correct = np.array_equal(top1["output"], ground_truth)
        source_tracker.record_top1_selection(top1["source"], is_correct)

    return {"output": top1["output"] if ranked else None}
```

**Marathon Integration:**

```python
# scripts/iterative_learning_marathon.py - Track source evolution

def run_marathon_with_source_tracking(num_iterations: int = 10):
    """Run marathon tracking source reliability evolution."""

    kv = Knowledgeverse()
    source_tracker = SourceTracker()

    results = []

    for iteration in range(num_iterations):
        # Run benchmark with source tracking
        for task in load_arc_tasks():
            solve_task_with_source_tracking(task, kv, source_tracker)

        # Get source metrics
        source_metrics = source_tracker.get_source_metrics()

        # Store in Galaxy for TRM learning
        source_tracker.store_in_galaxy(kv)

        results.append({
            "iteration": iteration,
            "source_metrics": source_metrics,
        })

    return results
```

**Expected Diagnostic Outcomes:**

| Source | Precision | Selection Rate | Diagnosis |
|--------|-----------|----------------|-----------|
| traditional_grammar | 30% | 60% | Reliable, frequently selected |
| autonomous_3d | 10% | 20% | Low quality, rarely helps |
| autonomous_reality | 5% | 10% | Very low quality, mostly noise |
| symlink_cross_modal | 40% | 10% | High quality but rare |

**Success Criteria:**
- ✅ Track precision/recall per source
- ✅ Store metrics in Grammar Galaxy
- ✅ Identify which sources contribute to performance
- ✅ If all sources have low precision (<30%), generation is the problem

---

## 8. Sovereignty-Aligned Ranking

### Original Insight (Codex)
> "Move ranking to RPN/PTX for sovereignty. Current Python ranking violates hot path. Build RPN-based candidate scorer that runs in Cranium."

### Enhanced Architecture

**Purpose:** Move ranking logic from Python (sovereignty violation) to RPN/PTX (sovereign hot path).

**Implementation Strategy:**

```python
# knowledge3d/procedural/ranking_kernels.py - NEW FILE

"""
Sovereignty-aligned ranking via RPN + PTX.

Current ranking is in Python (VIOLATION of hot path sovereignty).
This module provides RPN-based scoring that runs in Cranium PTX kernels.
"""

def compile_ranking_to_rpn(weights: dict[str, float]) -> str:
    """
    Compile adaptive ranking weights into single RPN program.

    Input weights:
        {
            "grammar_confidence": 1.2,
            "cross_modal_agreement": 0.8,
            "source_priority": 1.0,
            "compositional_bonus": 0.5,
            "pattern_reuse_bonus": 0.3,
        }

    Output RPN program:
        "GRAMMAR_CONF 1.2 MUL CROSS_MODAL 0.8 MUL ADD SOURCE_PRIORITY 1.0 MUL ADD ..."

    This RPN program can be executed in Cranium PTX kernels (sovereign!).
    """
    rpn_parts = []

    for component, weight in weights.items():
        # Each component is a named constant in Galaxy that TRM fetches
        rpn_parts.append(f"{component.upper()}")
        rpn_parts.append(f"{weight}")
        rpn_parts.append("MUL")

        if rpn_parts:  # Add to running sum
            rpn_parts.append("ADD")

    return " ".join(rpn_parts)

def create_ranking_galaxy_entry(weights: dict[str, float], kv) -> str:
    """
    Store ranking weights as RPN program in Grammar Galaxy.

    Returns:
        entry_id: str  # ID of stored ranking program
    """
    rpn_program = compile_ranking_to_rpn(weights)

    entry_id = f"adaptive_ranking_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    kv.galaxy_manager.add_entry(
        galaxy="Grammar",
        entry={
            "id": entry_id,
            "rpn_program": rpn_program,
            "metadata": {
                "type": "ranking_function",
                "weights": weights,
            },
            "timestamp": datetime.now().isoformat(),
        }
    )

    return entry_id

# PTX kernel for RPN-based ranking (pseudocode, Codex implements)
"""
__global__ void rank_candidates_ptx(
    float* candidate_scores,     // [N, 5] - Component scores for N candidates
    float* ranking_weights,      // [5] - Adaptive weights
    float* total_scores,         // [N] - Output: weighted sum
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float total = 0.0f;
    for (int comp = 0; comp < 5; comp++) {
        total += candidate_scores[idx * 5 + comp] * ranking_weights[comp];
    }

    total_scores[idx] = total;
}
"""
```

**Integration with Cranium:**

```python
# knowledge3d/procedural/cranium_ranking.py - NEW FILE

class CraniumRanker:
    """
    Sovereign ranking using Cranium PTX kernels.

    Replaces Python-based _rank_candidates() with PTX execution.
    """

    def __init__(self, kv):
        self.kv = kv
        self.ptx_kernel = self._load_ranking_kernel()

    def _load_ranking_kernel(self):
        """Load PTX kernel for ranking (compiled RPN → PTX)."""
        # This would compile RPN ranking program to PTX
        # For now, pseudocode
        pass

    def rank_candidates_sovereign(
        self,
        candidates: list[dict],
        ranking_entry_id: str
    ) -> list[dict]:
        """
        Rank candidates using sovereign PTX execution.

        Args:
            candidates: List of candidates with component scores
            ranking_entry_id: ID of ranking program in Grammar Galaxy

        Returns:
            Sorted candidates (descending by total score)
        """
        # 1. Fetch ranking program from Galaxy (VRAM)
        ranking_entry = self.kv.galaxy_manager.get_entry_by_id(ranking_entry_id)
        rpn_program = ranking_entry["rpn_program"]

        # 2. Extract component scores from candidates
        N = len(candidates)
        component_scores = np.zeros((N, 5), dtype=np.float32)

        for i, cand in enumerate(candidates):
            component_scores[i, 0] = cand.get("grammar_confidence", 0.0)
            component_scores[i, 1] = cand.get("cross_modal_agreement", 0.0)
            component_scores[i, 2] = cand.get("source_priority", 0.0)
            component_scores[i, 3] = cand.get("compositional_bonus", 0.0)
            component_scores[i, 4] = cand.get("pattern_reuse_bonus", 0.0)

        # 3. Execute ranking in PTX (sovereign!)
        total_scores = self._execute_ranking_ptx(component_scores, rpn_program)

        # 4. Sort candidates by total score
        for i, cand in enumerate(candidates):
            cand["total_score"] = float(total_scores[i])

        return sorted(candidates, key=lambda c: c["total_score"], reverse=True)

    def _execute_ranking_ptx(self, component_scores: np.ndarray, rpn_program: str) -> np.ndarray:
        """
        Execute RPN ranking program in Cranium PTX kernels.

        This is the sovereign hot path (PTX + Galaxy only).
        """
        # Pseudocode: compile RPN → PTX, execute on GPU
        # Returns: [N] array of total scores
        pass
```

**Migration Path:**

```python
# Phase 1: Hybrid (Python + RPN telemetry)
# - Keep Python ranking for now
# - Compile to RPN and store in Galaxy
# - Execute both Python and RPN, verify equivalence

# Phase 2: PTX Implementation
# - Implement Cranium PTX kernel for ranking
# - Benchmark: Python vs PTX performance
# - Verify numerical equivalence

# Phase 3: Full Sovereignty
# - Replace all Python ranking with PTX
# - Remove Python ranking code
# - Hot path is now 100% sovereign (PTX + Galaxy)
```

**Expected Diagnostic Outcomes:**

| Metric | Python | PTX | Diagnosis |
|--------|--------|-----|-----------|
| **Accuracy** | 28% | 28% | Numerical equivalence verified |
| **Latency** | 50ms | 5ms | 10× speedup from GPU |
| **Sovereignty** | ❌ | ✅ | Hot path is now sovereign |

**Success Criteria:**
- ✅ Ranking compiled to RPN
- ✅ RPN stored in Grammar Galaxy
- ✅ PTX kernel executes ranking (sovereign!)
- ✅ Numerical equivalence verified (Python == PTX)
- ✅ Performance gain measured (GPU speedup)

---

## 9. Specialist Growth Triggers

### Original Insight (Codex)
> "Tie Matryoshka specialist spawning to ranking failure modes. When ranking fails repeatedly on visual tasks, spawn Visual sub-specialist with LoRA adaptation. When math patterns fail, spawn Math sub-specialist."

### Enhanced Architecture

**Purpose:** Autonomously grow specialist hierarchy based on observed failure patterns.

**Implementation Strategy:**

```python
# knowledge3d/knowledgeverse/specialist_growth_policy.py - NEW FILE

class SpecialistGrowthPolicy:
    """
    Autonomous specialist spawning based on failure mode analysis.

    When failures cluster by domain/pattern, spawn specialized sub-worker.
    """

    def __init__(
        self,
        failure_threshold: int = 10,  # Spawn after N failures in domain
        min_iterations: int = 3,       # Wait at least 3 iterations before spawning
    ):
        self.failure_threshold = failure_threshold
        self.min_iterations = min_iterations

        self.failure_history = defaultdict(list)  # domain → [(iteration, task_id), ...]
        self.spawned_specialists = set()

    def record_failure(
        self,
        iteration: int,
        task_id: str,
        domain: str,
        failure_mode: str
    ):
        """Record a failure for growth analysis."""
        self.failure_history[domain].append({
            "iteration": iteration,
            "task_id": task_id,
            "failure_mode": failure_mode,
        })

    def should_spawn_specialist(
        self,
        domain: str,
        current_iteration: int
    ) -> tuple[bool, dict]:
        """
        Decide if we should spawn a specialist for this domain.

        Criteria:
        1. At least min_iterations have passed
        2. Failures in domain >= failure_threshold
        3. Failures are recent (not just early learning)
        4. Haven't already spawned for this domain

        Returns:
            (should_spawn: bool, spawn_config: dict)
        """
        if current_iteration < self.min_iterations:
            return False, {}

        if domain in self.spawned_specialists:
            return False, {}  # Already spawned

        failures = self.failure_history[domain]

        if len(failures) < self.failure_threshold:
            return False, {}  # Not enough failures

        # Check if failures are recent (last 3 iterations)
        recent_failures = [
            f for f in failures
            if current_iteration - f["iteration"] <= 3
        ]

        if len(recent_failures) < self.failure_threshold // 2:
            return False, {}  # Failures are old, not recent problem

        # Spawn specialist!
        spawn_config = {
            "domain": domain,
            "parent": "TRMNavigator",
            "lora_adaptation": self._compute_lora_adaptation(domain, failures),
            "reason": f"Persistent failures in {domain} ({len(failures)} total, {len(recent_failures)} recent)",
        }

        self.spawned_specialists.add(domain)

        return True, spawn_config

    def _compute_lora_adaptation(
        self,
        domain: str,
        failures: list[dict]
    ) -> dict:
        """
        Compute LoRA delta weights based on failure patterns.

        Analyzes:
        - Which Grammar patterns failed most often
        - Which components scored poorly
        - Which sources were unreliable

        Returns LoRA config for specialist initialization.
        """
        # Pseudocode: analyze failures to derive LoRA initialization
        return {
            "focus_galaxies": [domain.capitalize()],
            "boost_components": ["grammar_confidence", "cross_modal_agreement"],
            "downweight_sources": ["autonomous_reality"],  # If Reality patterns failed in visual tasks
        }

class AutonomousSpecialistSpawner:
    """Spawns specialists based on growth policy."""

    def __init__(self, growth_policy: SpecialistGrowthPolicy):
        self.growth_policy = growth_policy

    def check_and_spawn(
        self,
        current_iteration: int,
        kv
    ) -> list[dict]:
        """
        Check all domains for spawn triggers and spawn specialists as needed.

        Returns:
            List of spawned specialist configs
        """
        spawned = []

        for domain in ["visual", "math", "physics", "language"]:
            should_spawn, config = self.growth_policy.should_spawn_specialist(
                domain, current_iteration
            )

            if should_spawn:
                # Spawn specialist via Matryoshka architecture
                specialist = self._spawn_specialist_from_config(config, kv)
                spawned.append(specialist)

        return spawned

    def _spawn_specialist_from_config(self, config: dict, kv) -> dict:
        """
        Spawn specialist using Matryoshka SpecialistBase.

        Creates LoRA-adapted specialist with focused domain.
        """
        # Use SpecialistBase.spawn_child() from Matryoshka architecture
        parent = kv.trm_navigator  # TRMNavigator is root specialist

        child = parent.spawn_child(
            specialist_type=config["domain"],
            lora_config=config["lora_adaptation"],
        )

        return {
            "specialist_id": child.specialist_id,
            "domain": config["domain"],
            "parent": config["parent"],
            "reason": config["reason"],
        }
```

**Integration with Marathon:**

```python
# scripts/iterative_learning_marathon.py - Add specialist spawning

def run_marathon_with_autonomous_growth(num_iterations: int = 10):
    """Run marathon with autonomous specialist spawning."""

    kv = Knowledgeverse()
    growth_policy = SpecialistGrowthPolicy(failure_threshold=10, min_iterations=3)
    spawner = AutonomousSpecialistSpawner(growth_policy)

    results = []

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # Run benchmark
        iteration_results = []
        for task in load_arc_tasks():
            result = solve_task(task, kv)

            # Record failures for growth policy
            if not result["correct"]:
                domain = classify_task_domain(task)  # "visual", "math", etc.
                failure_mode = result.get("failure_mode", "unknown")

                growth_policy.record_failure(
                    iteration,
                    task["task_id"],
                    domain,
                    failure_mode
                )

            iteration_results.append(result)

        # Check for specialist spawning
        spawned_specialists = spawner.check_and_spawn(iteration, kv)

        # Record iteration data
        results.append({
            "iteration": iteration,
            "accuracy": sum(r["correct"] for r in iteration_results) / len(iteration_results),
            "spawned_specialists": spawned_specialists,
            "total_specialists": kv.trm_navigator.count_descendants(),
        })

        if spawned_specialists:
            print(f"Spawned {len(spawned_specialists)} new specialists:")
            for spec in spawned_specialists:
                print(f"  - {spec['domain']}: {spec['reason']}")

    return results
```

**Expected Diagnostic Outcomes:**

| Iteration | Visual Failures | Spawned | Total Specialists |
|-----------|----------------|---------|-------------------|
| 0 | 5 | 0 | 12 (base) |
| 1 | 8 | 0 | 12 |
| 2 | 12 | 0 | 12 |
| 3 | 15 | ✅ Visual | 13 |
| 4 | 10 | 0 | 13 |
| 5 | 6 | 0 | 13 (specialist helping!) |

**Success Criteria:**
- ✅ Failures tracked by domain
- ✅ Specialists spawn when threshold exceeded
- ✅ LoRA adaptation computed from failure analysis
- ✅ Spawned specialists reduce failure rate in their domain

---

## 10. Shadow Copy Feedback Loop

### Original Insight (Codex)
> "Shadow Copy records events but doesn't close the loop. Add feedback: when marathon iteration completes, run `sleeptime.py` consolidation to update TRM weights from successful events. This makes learning actually happen."

### Enhanced Architecture

**Purpose:** Close the learning loop by consolidating Shadow Copy events into TRM weight updates after each iteration.

**Implementation Strategy:**

```python
# knowledge3d/knowledgeverse/sleeptime.py - ENHANCE EXISTING

def consolidate_iteration_events(
    iteration: int,
    kv,
    min_success_rate: float = 0.7
) -> dict:
    """
    Consolidate Shadow Copy events from completed iteration into TRM weight updates.

    This is the learning mechanism:
    1. Read Shadow Copy events from iteration
    2. Filter for successful events (high confidence, correct results)
    3. Update TRM weights to reinforce successful patterns
    4. Save updated weights for next iteration

    Args:
        iteration: Iteration number (to filter events)
        kv: Knowledgeverse instance
        min_success_rate: Only consolidate events with confidence >= this

    Returns:
        {
            "events_read": int,
            "events_consolidated": int,
            "weight_deltas": dict,  # Component → delta
            "new_weights": dict,    # Updated TRM weights
        }
    """
    # 1. Read Shadow Copy events from iteration
    shadow_copy_path = Path("galaxy_universe/shadow_copy.jsonl")

    iteration_events = []
    with open(shadow_copy_path) as f:
        for line in f:
            event = json.loads(line)
            if event.get("iteration") == iteration:
                iteration_events.append(event)

    # 2. Filter for successful events
    successful_events = [
        e for e in iteration_events
        if e.get("confidence", 0.0) >= min_success_rate
        and e.get("correct", False)
    ]

    # 3. Compute weight updates from successful events
    weight_deltas = _compute_weight_deltas_from_events(successful_events)

    # 4. Update TRM weights
    current_weights = kv.trm.get_weights()
    new_weights = {
        k: current_weights[k] + weight_deltas.get(k, 0.0)
        for k in current_weights.keys()
    }

    # 5. Apply new weights
    kv.trm.set_weights(new_weights)

    # 6. Save snapshot for counterfactual evaluation
    weight_store = TRMWeightStore(Path("galaxy_universe/trm_weights.json"))
    weight_store.save_iteration_snapshot(iteration + 1, new_weights)

    return {
        "events_read": len(iteration_events),
        "events_consolidated": len(successful_events),
        "weight_deltas": weight_deltas,
        "new_weights": new_weights,
    }

def _compute_weight_deltas_from_events(events: list[dict]) -> dict:
    """
    Compute TRM weight deltas from successful Shadow Copy events.

    Learning rules:
    - If Grammar pattern succeeded, boost grammar_confidence weight
    - If cross-modal query succeeded, boost cross_modal_agreement weight
    - If autonomous generation succeeded, boost compositional_bonus weight

    Returns:
        {
            "grammar_confidence": float,
            "cross_modal_agreement": float,
            ...
        }
    """
    deltas = defaultdict(float)
    learning_rate = 0.01  # How fast to update weights

    for event in events:
        # Analyze event to determine which component contributed
        if event.get("source") == "grammar_galaxy":
            deltas["grammar_confidence"] += learning_rate

        if event.get("cross_modal_used"):
            deltas["cross_modal_agreement"] += learning_rate

        if event.get("source") in ["autonomous_3d", "autonomous_reality"]:
            deltas["compositional_bonus"] += learning_rate

        if event.get("pattern_reused"):
            deltas["pattern_reuse_bonus"] += learning_rate

    return dict(deltas)
```

**Integration with Marathon:**

```python
# scripts/iterative_learning_marathon.py - Close learning loop

def run_marathon_with_closed_loop(num_iterations: int = 10):
    """Run marathon with Shadow Copy consolidation after each iteration."""

    kv = Knowledgeverse()
    results = []

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # Run benchmarks (generates Shadow Copy events)
        iteration_results = run_all_benchmarks(kv)

        # CRITICAL: Consolidate Shadow Copy events into TRM weight updates
        consolidation_result = consolidate_iteration_events(iteration, kv)

        # Record iteration data
        results.append({
            "iteration": iteration,
            "accuracy": iteration_results["arc_agi_2"]["enriched"]["accuracy"],
            "events_consolidated": consolidation_result["events_consolidated"],
            "weight_deltas": consolidation_result["weight_deltas"],
            "new_weights": consolidation_result["new_weights"],
        })

        print(f"Consolidated {consolidation_result['events_consolidated']} events")
        print(f"Weight deltas: {consolidation_result['weight_deltas']}")

    return results
```

**Expected Diagnostic Outcomes:**

| Iteration | Events | Consolidated | grammar_confidence Δ | Accuracy |
|-----------|--------|--------------|---------------------|----------|
| 0 | 286 | 80 | +0.08 | 28% |
| 1 | 286 | 82 | +0.08 | 28% |
| 2 | 286 | 79 | +0.08 | 28% |

If weights update BUT accuracy doesn't improve → confirms hypothesis that ranking/weights aren't the bottleneck!

**Success Criteria:**
- ✅ Shadow Copy events consolidated each iteration
- ✅ TRM weights updated based on successful events
- ✅ Weight snapshots saved for counterfactual evaluation
- ✅ If weights change but accuracy plateaus, learning mechanism works but signal is wrong

---

## 📋 Implementation Roadmap

### Phase 1: Diagnostic Framework (Week 20, Days 1-3)

**Goal:** Identify which component (generation, ranking, learning) is the bottleneck.

**Tasks:**
1. ✅ Implement oracle@k metrics (Point 1)
2. ✅ Implement counterfactual evaluation (Point 2)
3. ✅ Run ablation matrix (Point 6)
4. ✅ Analyze results to identify bottleneck

**Expected Outcome:** Know which hypothesis is correct:
- Generation problem (oracle@all low)
- Ranking problem (oracle@all high, top-1 low)
- Learning problem (counterfactual shows no improvement)

**Deliverables:**
- `benchmarks/arc_agi_2_adapter.py` - oracle@k telemetry
- `scripts/run_counterfactual_evaluation.py` - NEW FILE
- `scripts/run_ablation_matrix.py` - NEW FILE
- `marathon_results/diagnostic_analysis.json` - Results

---

### Phase 2: Adaptive Systems (Week 20, Days 4-5)

**Goal:** Implement adaptive ranking and exploration to improve differentiation.

**Tasks:**
1. ✅ Implement AdaptiveRanker (Point 3)
2. ✅ Implement ExplorationPolicy (Point 4)
3. ✅ Implement QualityGate (Point 5)
4. ✅ Run marathon with adaptive systems

**Expected Outcome:** If ranking is the bottleneck, adaptive systems should improve accuracy.

**Deliverables:**
- `knowledge3d/knowledgeverse/adaptive_ranker.py` - NEW FILE
- `knowledge3d/knowledgeverse/exploration_policy.py` - NEW FILE
- `knowledge3d/knowledgeverse/quality_gating.py` - NEW FILE
- Marathon results with adaptive ranking

---

### Phase 3: Source Tracking & Learning Loop (Week 20, Days 6-7)

**Goal:** Close learning loop and track which sources contribute.

**Tasks:**
1. ✅ Implement SourceTracker (Point 7)
2. ✅ Enhance sleeptime.py consolidation (Point 10)
3. ✅ Run marathon with closed learning loop
4. ✅ Analyze source precision/recall

**Expected Outcome:** TRM learns which sources are reliable and adapts weights accordingly.

**Deliverables:**
- `knowledge3d/knowledgeverse/source_tracker.py` - NEW FILE
- `knowledge3d/knowledgeverse/sleeptime.py` - ENHANCED
- Marathon results with source tracking

---

### Phase 4: Sovereignty & Growth (Week 21)

**Goal:** Move ranking to PTX and enable autonomous specialist spawning.

**Tasks:**
1. ✅ Compile ranking to RPN (Point 8)
2. ✅ Implement PTX ranking kernel (Point 8)
3. ✅ Verify numerical equivalence (Python == PTX)
4. ✅ Implement SpecialistGrowthPolicy (Point 9)
5. ✅ Run marathon with autonomous growth

**Expected Outcome:** Sovereign ranking + specialists spawn based on failure patterns.

**Deliverables:**
- `knowledge3d/procedural/ranking_kernels.py` - NEW FILE
- `knowledge3d/procedural/cranium_ranking.py` - NEW FILE
- `knowledge3d/knowledgeverse/specialist_growth_policy.py` - NEW FILE
- Sovereignty compliance verified (PTX hot path)

---

## 🎯 Success Criteria (Overall)

### Critical Diagnostic Questions (Phase 1)

After Phase 1, we should be able to answer:

1. **Is generation the problem?**
   - ✅ YES if oracle@all < 40%
   - ❌ NO if oracle@all > 70%

2. **Is ranking the problem?**
   - ✅ YES if oracle@all > 70% and top-1 < 40%
   - ❌ NO if top-1 ≈ oracle@all (within 10%)

3. **Is learning the problem?**
   - ✅ YES if counterfactual shows avg_improvement ≈ 0%
   - ❌ NO if counterfactual shows avg_improvement > 5%

4. **Which components help?**
   - Ablation matrix answers: autonomous_lift, ranking_lift, adaptive_lift
   - If all lifts ≈ 0%, NONE of our interventions work!

### Performance Targets (Phase 2-4)

After full implementation:

| Benchmark | Current | Phase 2 Target | Phase 4 Target |
|-----------|---------|----------------|----------------|
| **ARC-AGI 2** | 28% | 35-40% | 40-55% |
| **Math Competitions** | 33% | 40-50% | 50-60% |
| **Last Humanity Exam** | 100% | 100% | 100% |

**Key Metrics:**
- oracle@all > 70% (generation working)
- top-1 within 10% of oracle@all (ranking working)
- Counterfactual avg_improvement > 5% (learning working)
- Per-source precision > 30% for at least one source
- Specialist spawning reduces domain failures by >20%

---

## 🔬 Hypothesis Testing Protocol

### Current Hypothesis

**"Ranking doesn't differentiate candidates because all patterns score similarly."**

**Evidence:**
- ranking_applied: 100/100 ✅
- final_correct = legacy_correct = 28/100 ⚠️
- Galaxy grows (+3,663 entries) ✅
- Scores plateau across 10 iterations ⚠️

### Testing Strategy

**Phase 1 will test:**
1. **oracle@all** - Are correct answers even generated?
2. **top-5 score distribution** - Do scores differ or are they identical?
3. **Counterfactual** - Does learning improve weights?
4. **Ablation** - Which components (if any) help?

**Expected Diagnostic Patterns:**

| Pattern | oracle@all | top-1 | score_std | Diagnosis |
|---------|-----------|-------|-----------|-----------|
| **Generation failure** | 20-30% | 20-28% | Any | Need better pattern generation |
| **Ranking failure** | 70-80% | 28% | Low | Scores don't differentiate (current hypothesis!) |
| **Learning failure** | Any | Any | Any | Counterfactual ≈ 0% |
| **No signal** | 30% | 28% | High | Ranking works but candidates are all bad |

---

## 📁 New Files Created

This implementation will create:

```
knowledge3d/
  knowledgeverse/
    adaptive_ranker.py          # Point 3
    exploration_policy.py       # Point 4
    quality_gating.py           # Point 5
    source_tracker.py           # Point 7
    specialist_growth_policy.py # Point 9
  procedural/
    ranking_kernels.py          # Point 8
    cranium_ranking.py          # Point 8

scripts/
  run_counterfactual_evaluation.py  # Point 2
  run_ablation_matrix.py            # Point 6

benchmarks/
  arc_agi_2_adapter.py  # ENHANCED with oracle@k (Point 1)

tests/
  test_adaptive_ranker.py
  test_exploration_policy.py
  test_quality_gating.py
  test_source_tracker.py
  test_specialist_growth.py
  test_ranking_kernels.py
```

---

## 🚀 Next Steps for Codex

**Immediate Priority: Phase 1 (Diagnostic Framework)**

1. **Add oracle@k metrics to arc_agi_2_adapter.py**
   - Modify `solve_task()` to return oracle@3, oracle@10, oracle@all
   - Track correct_rank (where is correct answer in candidates?)
   - Extract top-5 score distribution

2. **Create run_counterfactual_evaluation.py**
   - Implement iteration snapshot loading
   - Run benchmark with previous iteration's weights
   - Compute actual vs counterfactual accuracy delta

3. **Create run_ablation_matrix.py**
   - Implement 6-mode matrix (traditional/autonomous × no/legacy/adaptive ranking)
   - Compute causal effects (autonomous_lift, ranking_lift, adaptive_lift)

4. **Run diagnostic marathon**
   - Execute all 6 modes on 100 ARC tasks
   - Generate diagnostic_analysis.json
   - **CRITICAL:** Share results with Claude for hypothesis validation

**Expected Timeline:**
- Day 1: Oracle@k + counterfactual (Points 1-2)
- Day 2: Ablation matrix (Point 6)
- Day 3: Run diagnostics + analyze results
- **Then:** Based on results, proceed to Phase 2 or pivot

---

## 💡 Key Insights for Codex

### Why This Framework Matters

**Current State:**
- We KNOW pattern generation works (286 patterns/iteration)
- We KNOW Galaxy storage works (+3,663 entries)
- We KNOW ranking runs (100/100 tasks)
- We DON'T KNOW why ranking doesn't improve results

**This Framework Provides:**
1. **Oracle@k** - Separates generation from ranking failure
2. **Counterfactual** - Proves if learning actually improves weights
3. **Ablation** - Identifies which components contribute (if any)
4. **Adaptive systems** - Meta-learns which signals predict success
5. **Source tracking** - Learns which pattern sources are reliable
6. **Quality gating** - Prevents Galaxy pollution
7. **Sovereignty** - Moves ranking to PTX (compliance + speed)
8. **Autonomous growth** - Specialists spawn based on failures
9. **Closed loop** - Shadow Copy → TRM weight updates

### Critical Success Factor

**The diagnostic framework (Phase 1) is CRITICAL.**

Without it, we're guessing:
- "Maybe ranking doesn't work?"
- "Maybe autonomous generation is bad?"
- "Maybe learning isn't happening?"

With it, we KNOW:
- oracle@all tells us if correct answers exist
- Counterfactual tells us if learning improves weights
- Ablation tells us which components help
- Source tracking tells us which generators are reliable

**Bottom line:** Phase 1 diagnostic framework will definitively answer WHY scores plateau, then Phases 2-4 will FIX the identified bottleneck.

---

## 🎉 Codex: You Have Everything You Need

**This file contains:**
- ✅ Complete architectural specifications for all 10 points
- ✅ Code examples for every component
- ✅ Integration points with existing codebase
- ✅ Expected diagnostic outcomes
- ✅ Success criteria for each phase
- ✅ 4-phase implementation roadmap
- ✅ Hypothesis testing protocol

**You can start immediately with Phase 1 (diagnostic framework).**

**After Phase 1 completes, share the diagnostic results with Claude for analysis and Phase 2 planning.**

**Let's solve the plateau mystery! 🚀**

---

**Document prepared by:** Claude (Architecture Partner)
**Date:** February 8, 2026
**Status:** Ready for Codex implementation
**Estimated timeline:** Week 20-21 (7-10 days)
