"""
Router Specialist: The Meta-Specialist

The router is itself a specialist within the swarm that learns optimal routing decisions.

Key Insight:
    "The MoE router IS a specialist, not external infrastructure"

Architecture:
    Router specialist learns: Task features → Specialist weights [0-1]

Bootstrap Workflow:
    1. Heuristic routing (keyword matching) - collect decisions
    2. Train router specialist on successful patterns
    3. Switch to learned routing (router specialist)
    4. Router self-updates from new routing data
    5. Router improves forever

Why This Matters:
    - Router learns from experience (not hard-coded rules)
    - Router self-updates like other specialists
    - Router benefits from base model improvements
    - Completely self-contained system
    - Recursive self-improvement

Philosophy:
    "The secret is held on the small things - we are all made of atoms after all"
    The router being a specialist is the atom that makes the whole system coherent.

Usage:
    # Phase 1: Bootstrap with heuristic routing
    bootstrap = RouterBootstrap(swarm)
    routing_history = bootstrap.collect_routing_data(tasks, num_samples=1000)

    # Phase 2: Train router specialist
    trainer = RouterSpecialistTrainer(swarm)
    trainer.register_router_specialist(num_specialists=3)
    trainer.train_from_history(routing_history)

    # Phase 3: Use learned routing
    router = MoERouter(swarm, strategy=RoutingStrategy.LEARNED)
    weights = router.route_blend(input_data)

    # Phase 4: Continual improvement
    trainer.update_from_new_decisions(new_routing_data)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32
from knowledge3d.cranium.sovereign.lora_gpu_trainer import LoRAGPUEngine


@dataclass
class RoutingDecision:
    """A single routing decision record."""
    input_data: Any                 # Task input
    task_description: Optional[str] # Text description (if available)
    specialist_weights: Dict[str, float]  # Predicted weights
    outcome_performance: float      # How well did it work? [0-1]
    timestamp: str


class _CtypesProxy:
    def __init__(self, data_ptr: int) -> None:
        self.data = int(data_ptr)


class _IndexBatch(list):
    @property
    def size(self) -> int:
        return len(self)


def _attach_ctypes_proxy(tensor: HostTensorF32) -> HostTensorF32:
    tensor.ctypes = _CtypesProxy(tensor.data_ptr)  # type: ignore[attr-defined]
    return tensor


def _host_tensor(values: Any, *, rows: Optional[int] = None, cols: Optional[int] = None) -> HostTensorF32:
    return _attach_ctypes_proxy(HostTensorF32.from_array_like(values, rows=rows, cols=cols))


def _serializable_input(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return [_serializable_input(item) for item in value]
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / float(len(values))


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_v = max(float(value) for value in values)
    exps = [math.exp(float(value) - max_v) for value in values]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / float(len(values)) for _ in values]
    return [value / total for value in exps]


def _argmax(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("argmax requires at least one value")
    best_idx = 0
    best_value = float(values[0])
    for idx in range(1, len(values)):
        candidate = float(values[idx])
        if candidate > best_value:
            best_idx = idx
            best_value = candidate
    return best_idx


class RouterBootstrap:
    """
    Bootstrap router specialist using heuristic routing.

    Collects routing decisions and outcomes to create training data.
    """

    def __init__(self, swarm, heuristic_router=None):
        """
        Initialize bootstrap.

        Args:
            swarm: AdaptiveSwarmTRM instance
            heuristic_router: MoERouter with heuristic strategy (will create if None)
        """
        self.swarm = swarm

        if heuristic_router is None:
            from knowledge3d.cranium.moe_router import MoERouter, RoutingConfig, RoutingStrategy
            self.router = MoERouter(swarm, config=RoutingConfig(strategy=RoutingStrategy.HEURISTIC))
        else:
            self.router = heuristic_router

        self.routing_history: List[RoutingDecision] = []

    def collect_routing_data(self, tasks: List[Dict[str, Any]],
                            outcome_fn: Callable[[Dict, Dict[str, float]], float],
                            num_samples: Optional[int] = None) -> List[RoutingDecision]:
        """
        Collect routing decisions on tasks.

        Args:
            tasks: List of tasks with 'input' and optional 'description'
            outcome_fn: Function (task, specialist_weights) → performance [0-1]
            num_samples: Number of samples to collect (None = all)

        Returns:
            List of routing decisions
        """
        if num_samples is not None:
            tasks = tasks[:num_samples]

        print(f"[RouterBootstrap] Collecting routing data from {len(tasks)} tasks...")

        for i, task in enumerate(tasks):
            input_data = task.get('input')
            description = task.get('description')

            # Route using heuristic
            if description is not None:
                weights = self.router.route_blend(task_description=description)
            elif input_data is not None:
                weights = self.router.route_blend()  # Uniform blend fallback
            else:
                continue

            # Evaluate outcome
            performance = outcome_fn(task, weights)

            # Record decision
            decision = RoutingDecision(
                input_data=input_data,
                task_description=description,
                specialist_weights=weights,
                outcome_performance=performance,
                timestamp=datetime.now().isoformat()
            )

            self.routing_history.append(decision)

            if (i + 1) % 100 == 0:
                avg_performance = _mean([d.outcome_performance for d in self.routing_history[-100:]])
                print(f"  Collected {i+1}/{len(tasks)}: Avg performance {avg_performance:.3f}")

        avg_performance = _mean([d.outcome_performance for d in self.routing_history])
        print(f"[RouterBootstrap] ✓ Collected {len(self.routing_history)} decisions")
        print(f"  Average performance: {avg_performance:.3f}")

        return self.routing_history

    def filter_successful_decisions(self, min_performance: float = 0.5) -> List[RoutingDecision]:
        """Filter to only successful routing decisions."""
        successful = [d for d in self.routing_history if d.outcome_performance >= min_performance]
        print(f"[RouterBootstrap] Filtered to {len(successful)} successful decisions (≥{min_performance:.2f})")
        return successful

    def save_history(self, path: Path):
        """Save routing history to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        history_dict = []
        for decision in self.routing_history:
            history_dict.append({
                'input_data': _serializable_input(decision.input_data) if decision.input_data is not None else None,
                'task_description': decision.task_description,
                'specialist_weights': decision.specialist_weights,
                'outcome_performance': decision.outcome_performance,
                'timestamp': decision.timestamp
            })

        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        print(f"[RouterBootstrap] Saved {len(history_dict)} decisions to {path}")


class RouterSpecialistTrainer:
    """
    Train router specialist from routing decisions.

    The router specialist learns: Task features → Specialist weights
    """

    def __init__(self, swarm):
        """
        Initialize trainer.

        Args:
            swarm: AdaptiveSwarmTRM instance
        """
        self.swarm = swarm

    def register_router_specialist(self, num_specialists: int,
                                   router_dims: int = 256,
                                   router_rank: int = 16):
        """
        Register router as a specialist in the swarm.

        Args:
            num_specialists: Number of specialists (excluding router)
            router_dims: Router specialist dimensions
            router_rank: Router adapter rank
        """
        # Router needs to output weights for each specialist
        # Output dimension = num_specialists

        print(f"[RouterSpecialist] Registering router specialist")
        print(f"  Specialists to route: {num_specialists}")
        print(f"  Router dimensions: {router_dims}")
        print(f"  Router rank: {router_rank}")

        # Register router
        self.swarm.register_specialist(
            name='router',
            required_dims=router_dims,
            rank=router_rank
        )

        print(f"[RouterSpecialist] ✓ Router registered as specialist in swarm")

    # ------------------------------------------------------------------
    # GPU-backed training helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tile_to_dim(vector: Any, dim: int) -> List[float]:
        """Tile or truncate vector to match router dimension."""
        vec = _serializable_input(vector)
        if isinstance(vec, list) and vec and isinstance(vec[0], list):
            flat = [float(value) for row in vec for value in row]
        elif isinstance(vec, list):
            flat = [float(value) for value in vec]
        else:
            flat = [float(vec)]
        if len(flat) == dim:
            return flat
        repeats = dim // len(flat)
        remainder = dim % len(flat)
        tiled = flat * repeats if repeats > 0 else []
        if remainder:
            tiled = tiled + flat[:remainder]
        if len(tiled) > dim:
            tiled = tiled[:dim]
        return [float(value) for value in tiled]

    @staticmethod
    def _set_adapter_weights(adapter, A_new: Any, B_new: Any) -> None:
        """Copy trained weights into adapter (primary + shadow)."""
        adapter.A.copy_from(A_new)
        adapter.B.copy_from(B_new)
        adapter.A_shadow.copy_from(A_new)
        adapter.B_shadow.copy_from(B_new)

    def _specialist_names(self) -> List[str]:
        """Deterministic ordering of non-router specialists."""
        return [name for name in self.swarm.base.specialists.keys() if name != 'router']

    def _prepare_router_arrays(
        self,
        decisions: List[RoutingDecision],
        dims: int,
        specialist_names: List[str],
    ) -> Tuple[HostTensorF32, HostTensorF32]:
        """Convert routing decisions into GPU-friendly arrays."""
        inputs: List[List[float]] = []
        targets: List[List[float]] = []

        for decision in decisions:
            if decision.input_data is None:
                continue

            inp = self._tile_to_dim(decision.input_data, dims)

            target = [0.0] * dims
            if decision.specialist_weights:
                best_name = max(decision.specialist_weights.items(), key=lambda kv: kv[1])[0]
            else:
                best_name = specialist_names[0]
            performance = float(decision.outcome_performance)
            for idx, name in enumerate(specialist_names):
                target[idx] = performance if name == best_name else 0.0

            inputs.append(inp)
            targets.append(target)

        if not inputs:
            raise RuntimeError("No valid routing decisions with input data")

        return _host_tensor(inputs, rows=len(inputs), cols=dims), _host_tensor(targets, rows=len(targets), cols=dims)

    def _evaluate_router(
        self,
        adapter,
        A_weights: Any,
        B_weights: Any,
        decisions: List[RoutingDecision],
        specialist_names: List[str],
    ) -> Dict[str, float]:
        """Evaluate router accuracy and confidence for given weights."""
        if len(decisions) == 0:
            return {'accuracy': 0.0, 'avg_correct_weight': 0.0}

        dims = adapter.A.shape[0]
        A_backup = adapter.A.copy()
        B_backup = adapter.B.copy()

        try:
            self._set_adapter_weights(adapter, A_weights, B_weights)

            correct = 0
            total = 0
            weight_sum = 0.0
            for decision in decisions:
                if decision.input_data is None:
                    continue

                input_vec = self._tile_to_dim(decision.input_data, dims)
                output = self.swarm.compute_with_specialist(input_vec, 'router')
                output_list = _serializable_input(output)
                if isinstance(output_list, list) and output_list and isinstance(output_list[0], list):
                    logits_source = [float(value) for value in output_list[0]]
                else:
                    logits_source = [float(value) for value in output_list[:len(specialist_names)]]
                logits = logits_source[:len(specialist_names)]
                weights = _softmax(logits)

                target_name = max(decision.specialist_weights.items(), key=lambda kv: kv[1])[0]
                try:
                    target_idx = specialist_names.index(target_name)
                except ValueError:
                    continue

                predicted_idx = _argmax(weights)
                if predicted_idx == target_idx:
                    correct += 1
                weight_sum += float(weights[target_idx])
                total += 1

            if total == 0:
                return {'accuracy': 0.0, 'avg_correct_weight': 0.0}

            return {
                'accuracy': correct / float(total),
                'avg_correct_weight': weight_sum / float(total),
            }
        finally:
            self._set_adapter_weights(adapter, A_backup, B_backup)

    def _train_router_gpu(
        self,
        train_decisions: List[RoutingDecision],
        val_decisions: List[RoutingDecision],
        epochs: int,
        learning_rate: float,
        log_prefix: str = "[RouterSpecialist]",
    ) -> Dict[str, Any]:
        """Train router adapter using sovereign GPU LoRA engine."""
        specialist_names = self._specialist_names()
        if len(specialist_names) == 0:
            raise RuntimeError("No specialists registered for router to learn")

        adapter = self.swarm.base.specialists['router']['adapter']
        dims = self.swarm.base.specialists['router']['dims']
        rank = adapter.rank
        alpha = adapter.alpha

        if len(train_decisions) == 0:
            raise RuntimeError("No training samples for router specialist")

        inputs_np, targets_np = self._prepare_router_arrays(train_decisions, dims, specialist_names)
        base_matrix = _host_tensor(self.swarm.base.get_base_at_dim(dims), rows=dims, cols=dims)
        _attach_ctypes_proxy(adapter.A)
        _attach_ctypes_proxy(adapter.B)
        _attach_ctypes_proxy(base_matrix)

        engine = LoRAGPUEngine()
        batch_cap = max(1, min(15, len(train_decisions)))
        buffers = engine.allocate_buffers(base_matrix, adapter.A, adapter.B, inputs_np, targets_np, max_batch=batch_cap)

        rng = random.Random(42)
        dataset_size = len(inputs_np)

        try:
            for epoch in range(1, epochs + 1):
                perm = list(range(dataset_size))
                rng.shuffle(perm)
                epoch_inputs = [inputs_np[idx] for idx in perm]
                epoch_targets = [targets_np[idx] for idx in perm]
                engine.update_dataset(buffers, epoch_inputs, epoch_targets)

                order = list(range(dataset_size))
                epoch_loss = 0.0
                batches = 0
                for batch_start in range(0, dataset_size, batch_cap):
                    batch_end = min(batch_start + batch_cap, dataset_size)
                    batch_indices = _IndexBatch(order[batch_start:batch_end])
                    loss = engine.train_batch(
                        buffers=buffers,
                        batch_indices=batch_indices,
                        dims=dims,
                        rank=rank,
                        alpha=alpha,
                        learning_rate=learning_rate,
                    )
                    epoch_loss += loss
                    batches += 1

                avg_loss = epoch_loss / float(batches or 1)
                print(f"{log_prefix} Epoch {epoch:03d}/{epochs} - loss={avg_loss:.6f}")

            A_trained, B_trained = engine.fetch_weights(buffers, dims, rank)
        finally:
            engine.free_buffers(buffers)

        if val_decisions:
            baseline_metrics = self._evaluate_router(adapter, adapter.A.copy(), adapter.B.copy(), val_decisions, specialist_names)
            candidate_metrics = self._evaluate_router(adapter, A_trained.copy(), B_trained.copy(), val_decisions, specialist_names)
        else:
            baseline_metrics = {'accuracy': 0.0, 'avg_correct_weight': 0.0}
            candidate_metrics = {'accuracy': 0.0, 'avg_correct_weight': 0.0}

        improvement = candidate_metrics['avg_correct_weight'] - baseline_metrics['avg_correct_weight']
        success = improvement >= 1e-4

        adapter.update_count += 1
        if success:
            self._set_adapter_weights(adapter, A_trained, B_trained)
            adapter.accepted_count += 1
            adapter.baseline_performance = candidate_metrics['avg_correct_weight']
            outcome = "accepted"
        else:
            adapter.rejected_count += 1
            outcome = "rejected"

        print(f"{log_prefix} Validation (baseline → candidate): "
              f"{baseline_metrics['avg_correct_weight']:.4f} → {candidate_metrics['avg_correct_weight']:.4f} "
              f"({outcome})")

        return {
            'success': success,
            'baseline_metrics': baseline_metrics,
            'candidate_metrics': candidate_metrics,
            'improvement': improvement,
            'epochs': epochs,
            'train_samples': len(train_decisions),
            'val_samples': len(val_decisions),
        }

    # ------------------------------------------------------------------
    # Public training APIs
    # ------------------------------------------------------------------
    def train_from_history(self, routing_history: List[RoutingDecision],
                           epochs: int = 5,
                           filter_threshold: float = 0.5,
                           learning_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Train router specialist from routing history.

        Args:
            routing_history: List of routing decisions
            epochs: Number of training epochs
            filter_threshold: Minimum performance to include
            learning_rate: Override specialist learning rate

        Returns:
            Training statistics
        """
        if 'router' not in self.swarm.base.specialists:
            raise ValueError("Router specialist not registered. Call register_router_specialist() first.")

        # Filter to successful decisions
        successful = [d for d in routing_history if d.outcome_performance >= filter_threshold]

        print(f"\n[RouterSpecialist] Training from routing history")
        print(f"  Total decisions: {len(routing_history)}")
        print(f"  Successful (≥{filter_threshold:.2f}): {len(successful)}")

        if len(successful) == 0:
            print("  ✗ No successful decisions to train from")
            return {'error': 'No successful decisions'}

        if len(successful) < 2:
            print("  ✗ Need at least 2 successful decisions for train/validation split")
            return {'error': 'Insufficient successful decisions'}

        specialist_names = self._specialist_names()
        val_count = max(1, int(len(successful) * 0.1)) if len(successful) > 10 else max(1, len(successful) // 2)
        if val_count >= len(successful):
            val_count = len(successful) - 1

        val_decisions = successful[-val_count:]
        train_decisions = successful[:-val_count]

        lr = learning_rate or self.swarm.config.specialist_learning_rate
        stats = self._train_router_gpu(
            train_decisions=train_decisions,
            val_decisions=val_decisions,
            epochs=epochs,
            learning_rate=lr,
            log_prefix="[Router]",
        )

        return {
            **stats,
            'specialists': specialist_names,
        }

    def update_from_new_decisions(self, new_decisions: List[RoutingDecision],
                                  min_performance: float = 0.5,
                                  epochs: int = 1,
                                  learning_rate: Optional[float] = None) -> bool:
        """
        Update router specialist from new routing decisions (continual learning).

        Args:
            new_decisions: New routing decisions
            min_performance: Minimum performance to include
            epochs: Training epochs
            learning_rate: Optional override for learning rate

        Returns:
            True if update accepted, False if rejected
        """
        if 'router' not in self.swarm.base.specialists:
            raise ValueError("Router specialist not registered")

        # Filter successful
        successful = [d for d in new_decisions if d.outcome_performance >= min_performance]

        if len(successful) == 0:
            print("[RouterSpecialist] No successful decisions to update from")
            return False

        if len(successful) < 2:
            print("[RouterSpecialist] Not enough successful decisions for update")
            return False

        # Small validation split for continual learning
        val_count = max(1, len(successful) // 4)
        if val_count >= len(successful):
            val_count = len(successful) - 1

        val_decisions = successful[-val_count:]
        train_decisions = successful[:-val_count]

        lr = learning_rate or self.swarm.config.specialist_learning_rate
        stats = self._train_router_gpu(
            train_decisions=train_decisions,
            val_decisions=val_decisions,
            epochs=epochs,
            learning_rate=lr,
            log_prefix="[RouterUpdate]",
        )

        if stats['success']:
            print(f"[RouterSpecialist] ✓ Update accepted from {len(train_decisions)} samples")
        else:
            print(f"[RouterSpecialist] ✗ Update rejected (no improvement)")

        return bool(stats['success'])


class RouterTransition:
    """
    Manages transition from heuristic to learned routing.

    Ensures smooth bootstrap without performance degradation.
    """

    def __init__(self, swarm):
        """Initialize transition manager."""
        self.swarm = swarm
        self.heuristic_performance = 0.0
        self.learned_performance = 0.0

    def evaluate_routing_strategy(self, test_tasks: List[Dict],
                                  strategy: str,
                                  outcome_fn: Callable) -> float:
        """
        Evaluate routing strategy on test tasks.

        Args:
            test_tasks: Test tasks
            strategy: 'heuristic' or 'learned'
            outcome_fn: Function to evaluate outcomes

        Returns:
            Average performance [0-1]
        """
        from knowledge3d.cranium.moe_router import MoERouter, RoutingConfig, RoutingStrategy

        if strategy == 'heuristic':
            router = MoERouter(self.swarm, config=RoutingConfig(strategy=RoutingStrategy.HEURISTIC))
        else:
            router = MoERouter(self.swarm, config=RoutingConfig(strategy=RoutingStrategy.LEARNED))

        performances = []

        for task in test_tasks:
            description = task.get('description')
            input_data = task.get('input')

            if strategy == 'heuristic':
                weights = router.route_blend(task_description=description) if description else {}
                if (not weights) and (input_data is not None):
                    weights = router.route_blend(input_data=input_data)
            else:
                weights = router.route_blend(input_data=input_data) if input_data is not None else {}
                if (not weights) and description:
                    weights = router.route_blend(task_description=description)

            if not weights:
                continue

            performance = outcome_fn(task, weights)
            performances.append(performance)

        return _mean(performances)

    def should_transition(self, test_tasks: List[Dict],
                         outcome_fn: Callable,
                         min_improvement: float = 0.0) -> bool:
        """
        Check if we should transition from heuristic to learned.

        Args:
            test_tasks: Test tasks
            outcome_fn: Evaluation function
            min_improvement: Minimum improvement required

        Returns:
            True if learned ≥ heuristic + min_improvement
        """
        print("\n[RouterTransition] Evaluating routing strategies...")

        self.heuristic_performance = self.evaluate_routing_strategy(
            test_tasks, 'heuristic', outcome_fn
        )

        self.learned_performance = self.evaluate_routing_strategy(
            test_tasks, 'learned', outcome_fn
        )

        improvement = self.learned_performance - self.heuristic_performance

        print(f"  Heuristic performance: {self.heuristic_performance:.3f}")
        print(f"  Learned performance: {self.learned_performance:.3f}")
        print(f"  Improvement: {improvement:+.3f}")

        should_switch = improvement >= min_improvement

        if should_switch:
            print(f"  ✓ Transition recommended: Learned ≥ Heuristic + {min_improvement:.3f}")
        else:
            print(f"  ✗ Not ready: Need +{min_improvement - improvement:.3f} more improvement")

        return should_switch
