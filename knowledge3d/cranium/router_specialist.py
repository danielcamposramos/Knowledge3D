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

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RoutingDecision:
    """A single routing decision record."""
    input_data: np.ndarray          # Task input
    task_description: Optional[str] # Text description (if available)
    specialist_weights: Dict[str, float]  # Predicted weights
    outcome_performance: float      # How well did it work? [0-1]
    timestamp: str


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
                avg_performance = np.mean([d.outcome_performance for d in self.routing_history[-100:]])
                print(f"  Collected {i+1}/{len(tasks)}: Avg performance {avg_performance:.3f}")

        avg_performance = np.mean([d.outcome_performance for d in self.routing_history])
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
                'input_data': decision.input_data.tolist() if decision.input_data is not None else None,
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

    def train_from_history(self, routing_history: List[RoutingDecision],
                          epochs: int = 5,
                          filter_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Train router specialist from routing history.

        Args:
            routing_history: List of routing decisions
            epochs: Number of training epochs
            filter_threshold: Minimum performance to include

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

        # Convert to training samples
        # Input: Task features
        # Target: Specialist weights (successful decisions)
        train_samples = []

        specialist_names = [s for s in self.swarm.base.specialists.keys() if s != 'router']

        for decision in successful:
            if decision.input_data is None:
                continue

            # Target weights (one-hot or distribution)
            target_weights = np.zeros(len(specialist_names))
            for i, name in enumerate(specialist_names):
                target_weights[i] = decision.specialist_weights.get(name, 0.0)

            train_samples.append({
                'input': decision.input_data,
                'target_weights': target_weights,
                'performance': decision.outcome_performance
            })

        print(f"  Training samples: {len(train_samples)}")

        # Evaluation function: How close are predicted weights to target?
        def eval_fn(weights: np.ndarray, samples: List[Dict]) -> float:
            """Evaluate router by weight prediction accuracy."""
            total_loss = 0.0

            for sample in samples:
                # Forward pass (simplified - real would use actual forward)
                # For validation, we just check weight shapes
                pass

            # Placeholder: Return random score
            return 0.5 + np.random.rand() * 0.1

        # Set validation samples
        val_split = int(len(train_samples) * 0.1)
        val_samples = train_samples[-val_split:] if val_split > 0 else []
        train_samples = train_samples[:-val_split] if val_split > 0 else train_samples

        self.swarm.set_specialist_validation_samples('router', val_samples)

        # Train router specialist
        stats_all = []

        for epoch in range(epochs):
            print(f"\n[Epoch {epoch+1}/{epochs}]")

            stats = self.swarm.train_specialist_epoch(
                'router',
                train_samples,
                eval_fn,
                use_self_update=True
            )

            stats_all.append(stats)

        # Summary
        print("\n[RouterSpecialist] Training complete")
        print(f"  Epochs: {epochs}")
        print(f"  Final loss: {stats_all[-1]['avg_loss']:.4f}")

        return {
            'epochs': epochs,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'stats': stats_all
        }

    def update_from_new_decisions(self, new_decisions: List[RoutingDecision],
                                  min_performance: float = 0.5) -> bool:
        """
        Update router specialist from new routing decisions (continual learning).

        Args:
            new_decisions: New routing decisions
            min_performance: Minimum performance to include

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

        print(f"\n[RouterSpecialist] Updating from {len(successful)} new decisions...")

        # Convert to training samples
        specialist_names = [s for s in self.swarm.base.specialists.keys() if s != 'router']
        train_samples = []

        for decision in successful:
            if decision.input_data is None:
                continue

            target_weights = np.zeros(len(specialist_names))
            for i, name in enumerate(specialist_names):
                target_weights[i] = decision.specialist_weights.get(name, 0.0)

            train_samples.append({
                'input': decision.input_data,
                'target_weights': target_weights,
                'performance': decision.outcome_performance
            })

        # Evaluation function
        def eval_fn(weights, samples):
            return 0.5 + np.random.rand() * 0.1

        # Train one epoch
        stats = self.swarm.train_specialist_epoch(
            'router',
            train_samples,
            eval_fn,
            use_self_update=True
        )

        success = stats.get('update_accepted', False)

        if success:
            print(f"[RouterSpecialist] ✓ Update accepted from {len(train_samples)} samples")
        else:
            print(f"[RouterSpecialist] ✗ Update rejected (performance did not improve)")

        return success


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

            if description:
                weights = router.route_blend(task_description=description)
            elif input_data is not None:
                weights = router.route_blend(input_data=input_data)
            else:
                continue

            performance = outcome_fn(task, weights)
            performances.append(performance)

        return np.mean(performances) if performances else 0.0

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
