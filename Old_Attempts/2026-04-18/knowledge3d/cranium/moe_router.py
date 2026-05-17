"""
MoE Router: Intelligent Specialist Selection

Routes inputs to appropriate specialists based on task characteristics.

Strategies:
1. Manual: User specifies specialist
2. Heuristic: Rule-based routing (keyword matching)
3. Learned: Meta-specialist learns optimal routing (future)

Architecture:
    Input → Feature Extraction → Router → Specialist Weights
                                          ↓
                                    MoE Blending → Output

Key Features:
- Automatic task type detection
- Confidence-based specialist blending
- Routing statistics and monitoring
- Support for multi-specialist inference

Usage:
    # Create router
    router = MoERouter(swarm)

    # Route single specialist
    specialist_name = router.route_single(input_data)
    output = swarm.forward(input_data, specialist=specialist_name)

    # Route with blending (multiple specialists)
    weights = router.route_blend(input_data)
    output = swarm.forward_moe(input_data, weights)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from enum import Enum
from dataclasses import dataclass


class RoutingStrategy(Enum):
    """Routing strategy."""
    MANUAL = "manual"         # User specifies specialist
    HEURISTIC = "heuristic"   # Rule-based routing
    LEARNED = "learned"       # Meta-specialist routing (future)


@dataclass
class RoutingConfig:
    """Router configuration."""
    strategy: RoutingStrategy = RoutingStrategy.HEURISTIC
    confidence_threshold: float = 0.8    # Single specialist if confidence > threshold
    blend_top_k: int = 2                 # Blend top-K specialists if < threshold
    enable_fallback: bool = True         # Fall back to base if no specialist matches
    track_statistics: bool = True        # Track routing decisions


class MoERouter:
    """
    Mixture-of-Experts router for specialist selection.

    Routes inputs to appropriate specialists based on task characteristics.
    """

    def __init__(self, swarm, config: Optional[RoutingConfig] = None):
        """
        Initialize router.

        Args:
            swarm: AdaptiveSwarmTRM instance
            config: Routing configuration
        """
        self.swarm = swarm
        self.config = config or RoutingConfig()

        # Routing statistics
        self.routing_stats = {
            'total_routes': 0,
            'specialist_counts': {},
            'blend_count': 0,
            'fallback_count': 0
        }

        # Task type keywords (for heuristic routing)
        self.task_keywords = {
            'ocr': ['character', 'text', 'recognize', 'read', 'ocr', 'glyph', 'font'],
            'speech': ['audio', 'speech', 'transcribe', 'voice', 'utterance', 'phoneme'],
            'multimodal': ['multimodal', 'cross-modal', 'fusion', 'trimodal', 'alignment', 'modal'],
            'math': ['equation', 'calculate', 'solve', 'math', 'arithmetic', 'algebra'],
            'code': ['function', 'code', 'program', 'compile', 'debug', 'syntax'],
            'reasoning': ['logic', 'deduce', 'infer', 'reason', 'conclude'],
            'qa': ['question', 'answer', 'explain', 'what', 'why', 'how']
        }

        # Get strategy value (handle both string and enum)
        strategy_value = self.config.strategy.value if isinstance(self.config.strategy, RoutingStrategy) else self.config.strategy
        print(f"[MoERouter] Initialized with {strategy_value} strategy")

    def route_single(self, input_data: Optional[np.ndarray] = None,
                    task_description: Optional[str] = None,
                    manual_specialist: Optional[str] = None) -> str:
        """
        Route to single specialist.

        Args:
            input_data: Input vector (for learned routing)
            task_description: Text description of task (for heuristic routing)
            manual_specialist: Manual specialist selection

        Returns:
            Specialist name (or 'base' for fallback)
        """
        self.routing_stats['total_routes'] += 1

        # Manual routing
        if manual_specialist is not None:
            self._update_stats(manual_specialist)
            return manual_specialist

        # Heuristic routing
        if self.config.strategy == RoutingStrategy.HEURISTIC:
            if task_description is not None:
                specialist = self._route_heuristic(task_description)

                if specialist is not None:
                    self._update_stats(specialist)
                    return specialist

        # Learned routing (future implementation)
        elif self.config.strategy == RoutingStrategy.LEARNED:
            if input_data is not None:
                specialist = self._route_learned(input_data)

                if specialist is not None:
                    self._update_stats(specialist)
                    return specialist

        # Fallback to base
        if self.config.enable_fallback:
            self.routing_stats['fallback_count'] += 1
            return 'base'

        # No fallback - return first specialist
        specialists = list(self.swarm.base.specialists.keys())
        if len(specialists) > 0:
            self._update_stats(specialists[0])
            return specialists[0]

        return 'base'

    def route_blend(self, input_data: Optional[np.ndarray] = None,
                   task_description: Optional[str] = None) -> Dict[str, float]:
        """
        Route with specialist blending (MoE).

        Returns weights for multiple specialists.

        Args:
            input_data: Input vector
            task_description: Text description

        Returns:
            Dict mapping specialist_name → weight [0-1]
        """
        self.routing_stats['total_routes'] += 1

        # Heuristic blending
        if self.config.strategy == RoutingStrategy.HEURISTIC:
            if task_description is not None:
                weights = self._route_heuristic_blend(task_description)

                if weights:
                    self.routing_stats['blend_count'] += 1
                    return weights

        # Learned blending (future)
        elif self.config.strategy == RoutingStrategy.LEARNED:
            if input_data is not None:
                weights = self._route_learned_blend(input_data)

                if weights:
                    self.routing_stats['blend_count'] += 1
                    return weights

        # Fallback: Uniform blend of all specialists
        specialists = list(self.swarm.base.specialists.keys())
        if len(specialists) == 0:
            return {}

        uniform_weight = 1.0 / len(specialists)
        return {name: uniform_weight for name in specialists}

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return self.routing_stats.copy()

    def reset_stats(self):
        """Reset routing statistics."""
        self.routing_stats = {
            'total_routes': 0,
            'specialist_counts': {},
            'blend_count': 0,
            'fallback_count': 0
        }

    def _route_heuristic(self, task_description: str) -> Optional[str]:
        """
        Heuristic routing based on keyword matching.

        Args:
            task_description: Task description text

        Returns:
            Specialist name or None
        """
        task_lower = task_description.lower()

        # Score each specialist by keyword matches
        scores = {}

        for specialist_name in self.swarm.base.specialists.keys():
            # Get keywords for this specialist type
            keywords = self.task_keywords.get(specialist_name, [])

            # Count matches
            score = sum(1 for keyword in keywords if keyword in task_lower)
            scores[specialist_name] = score

        # Return specialist with highest score
        if scores:
            best_specialist = max(scores.items(), key=lambda x: x[1])

            if best_specialist[1] > 0:  # At least one keyword match
                return best_specialist[0]

        return None

    def _route_heuristic_blend(self, task_description: str) -> Dict[str, float]:
        """
        Heuristic blending based on keyword matching.

        Args:
            task_description: Task description

        Returns:
            Specialist weights
        """
        task_lower = task_description.lower()

        # Score each specialist
        scores = {}

        for specialist_name in self.swarm.base.specialists.keys():
            keywords = self.task_keywords.get(specialist_name, [])
            score = sum(1 for keyword in keywords if keyword in task_lower)
            scores[specialist_name] = score

        # Normalize scores to weights
        total_score = sum(scores.values())

        if total_score == 0:
            # No matches - uniform blend
            num_specialists = len(self.swarm.base.specialists)
            return {name: 1.0 / num_specialists for name in self.swarm.base.specialists.keys()}

        # Convert scores to weights
        weights = {name: score / total_score for name, score in scores.items()}

        # Filter to top-K specialists
        if self.config.blend_top_k > 0:
            top_k = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:self.config.blend_top_k]
            weights = dict(top_k)

            # Renormalize
            total_weight = sum(weights.values())
            weights = {name: w / total_weight for name, w in weights.items()}

        return weights

    def _route_learned(self, input_data: np.ndarray) -> Optional[str]:
        """
        Learned routing using router specialist.

        The router is itself a specialist in the swarm that learns optimal routing.

        Args:
            input_data: Input vector

        Returns:
            Specialist name or None
        """
        if 'router' not in self.swarm.base.specialists:
            # Router specialist not trained yet, fall back to heuristic
            return None

        # Use router specialist to predict best specialist
        router_output = self.swarm.compute_with_specialist(input_data, 'router')

        # Router output is specialist weights - pick highest
        specialist_names = list(self.swarm.base.specialists.keys())
        specialist_names = [s for s in specialist_names if s != 'router']  # Exclude router itself

        if len(specialist_names) == 0:
            return None

        # Output dims = number of specialists (excluding router)
        weights = router_output[:len(specialist_names)]

        # Return specialist with highest weight
        best_idx = np.argmax(weights)
        return specialist_names[best_idx]

    def _route_learned_blend(self, input_data: np.ndarray) -> Dict[str, float]:
        """
        Learned blending using router specialist.

        Router specialist predicts weights for all specialists.

        Args:
            input_data: Input vector

        Returns:
            Specialist weights
        """
        if 'router' not in self.swarm.base.specialists:
            # Router specialist not trained yet
            return {}

        # Use router specialist to predict weights
        router_output = self.swarm.compute_with_specialist(input_data, 'router')

        # Router output is specialist weights (one per specialist, excluding router)
        specialist_names = list(self.swarm.base.specialists.keys())
        specialist_names = [s for s in specialist_names if s != 'router']

        if len(specialist_names) == 0:
            return {}

        # Extract weights (first N dimensions of output)
        weights_raw = router_output[:len(specialist_names)]

        # Softmax normalization
        weights_exp = np.exp(weights_raw - np.max(weights_raw))
        weights_normalized = weights_exp / np.sum(weights_exp)

        # Create weight dict
        weights = {name: float(w) for name, w in zip(specialist_names, weights_normalized)}

        return weights

    def _update_stats(self, specialist_name: str):
        """Update routing statistics."""
        if specialist_name not in self.routing_stats['specialist_counts']:
            self.routing_stats['specialist_counts'][specialist_name] = 0

        self.routing_stats['specialist_counts'][specialist_name] += 1


class TaskComplexityEstimator:
    """
    Estimates task complexity for automatic dimension selection.

    Uses heuristics to estimate complexity [0-1]:
    - Input length
    - Vocabulary diversity
    - Syntactic complexity
    - Known task type difficulty
    """

    # Task type base complexity
    TASK_COMPLEXITY = {
        'ocr': 0.3,        # Medium-low (visual pattern matching)
        'math': 0.7,       # Medium-high (multi-step reasoning)
        'code': 0.8,       # High (syntax + semantics + logic)
        'reasoning': 0.9,  # Very high (deep inference chains)
        'qa': 0.5          # Medium (retrieval + light reasoning)
    }

    @staticmethod
    def estimate(input_data: Optional[np.ndarray] = None,
                task_type: Optional[str] = None,
                text: Optional[str] = None) -> float:
        """
        Estimate task complexity.

        Args:
            input_data: Input vector
            task_type: Task type identifier
            text: Text input (for text-based complexity)

        Returns:
            Complexity estimate [0-1]
        """
        complexity = 0.5  # Default: medium

        # Task type base complexity
        if task_type is not None:
            complexity = TaskComplexityEstimator.TASK_COMPLEXITY.get(task_type, 0.5)

        # Adjust based on input characteristics
        if text is not None:
            # Length factor
            length_factor = min(len(text) / 1000, 1.0)  # Normalize to [0-1]

            # Vocabulary diversity (unique words / total words)
            words = text.lower().split()
            if len(words) > 0:
                diversity = len(set(words)) / len(words)
            else:
                diversity = 0.5

            # Combine factors
            complexity = (complexity + length_factor * 0.3 + diversity * 0.2) / 1.5

        # Input vector norm (higher norm = more complex)
        if input_data is not None:
            norm_factor = min(np.linalg.norm(input_data) / 100, 1.0)
            complexity = (complexity + norm_factor * 0.2) / 1.2

        # Clamp to [0-1]
        return max(0.0, min(1.0, complexity))


class RoutingAnalyzer:
    """
    Analyzes routing patterns and provides recommendations.

    Tracks:
    - Specialist utilization
    - Routing accuracy (if ground truth available)
    - Performance by specialist
    """

    def __init__(self):
        """Initialize analyzer."""
        self.routing_history = []

    def record_routing(self, input_id: str, specialist: str,
                      performance: Optional[float] = None):
        """
        Record routing decision.

        Args:
            input_id: Input identifier
            specialist: Selected specialist
            performance: Task performance (if available)
        """
        self.routing_history.append({
            'input_id': input_id,
            'specialist': specialist,
            'performance': performance
        })

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze routing patterns.

        Returns:
            Analysis results
        """
        if len(self.routing_history) == 0:
            return {'error': 'No routing history'}

        # Specialist utilization
        specialist_counts = {}
        for record in self.routing_history:
            spec = record['specialist']
            specialist_counts[spec] = specialist_counts.get(spec, 0) + 1

        # Performance by specialist
        specialist_performance = {}
        for record in self.routing_history:
            if record['performance'] is not None:
                spec = record['specialist']
                if spec not in specialist_performance:
                    specialist_performance[spec] = []
                specialist_performance[spec].append(record['performance'])

        # Average performance
        avg_performance = {}
        for spec, perfs in specialist_performance.items():
            avg_performance[spec] = np.mean(perfs) if perfs else 0.0

        return {
            'total_routings': len(self.routing_history),
            'specialist_utilization': specialist_counts,
            'specialist_avg_performance': avg_performance,
            'most_used_specialist': max(specialist_counts.items(), key=lambda x: x[1])[0] if specialist_counts else None,
            'best_performing_specialist': max(avg_performance.items(), key=lambda x: x[1])[0] if avg_performance else None
        }

    def save_analysis(self, path: Path):
        """Save analysis to file."""
        analysis = self.analyze()

        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"[RoutingAnalyzer] Analysis saved to {path}")
