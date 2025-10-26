"""
TRM Cranium: Triadic Reasoning Module Core

The "brain" of Knowledge3D, implementing sovereign reasoning architecture.

Components:
- TRM Engine: Core reasoning with RPN stacks
- Latency Guard: 500ms inference guarantee
- Adaptive Swarm: Self-updating multi-specialist system
- Matryoshka TRM: Variable dimensionality (64 dims → 16K dims)
- MoE Router: Intelligent specialist selection
- OCR Pipeline: Character detection (Phase F.2)

Architecture Philosophy:
- Sovereign: Zero PyTorch/TensorFlow dependencies
- Adaptive: Dimensions scale to task complexity
- Self-Updating: Validation-gated weight updates
- Memory Efficient: LoRA-style adapters (18× reduction)
"""

# Phase H: Adaptive Swarm Architecture
from .adaptive_swarm import (
    AdaptiveSwarmTRM,
    SwarmConfig,
    SwarmTrainingProtocol
)

from .matryoshka_trm import (
    MatryoshkaTRM,
    DimensionSelector
)

from .trm_adapters import (
    AdapterWeights,
    SelfUpdatingAdapter,
    AdapterConfig
)

from .moe_router import (
    MoERouter,
    RoutingStrategy,
    RoutingConfig,
    TaskComplexityEstimator,
    RoutingAnalyzer
)

from .router_specialist import (
    RouterBootstrap,
    RouterSpecialistTrainer,
    RouterTransition,
    RoutingDecision
)

__all__ = [
    # Adaptive Swarm
    'AdaptiveSwarmTRM',
    'SwarmConfig',
    'SwarmTrainingProtocol',

    # Matryoshka TRM
    'MatryoshkaTRM',
    'DimensionSelector',

    # Adapters
    'AdapterWeights',
    'SelfUpdatingAdapter',
    'AdapterConfig',

    # MoE Routing
    'MoERouter',
    'RoutingStrategy',
    'RoutingConfig',
    'TaskComplexityEstimator',
    'RoutingAnalyzer',

    # Router Specialist (The Key Insight)
    'RouterBootstrap',
    'RouterSpecialistTrainer',
    'RouterTransition',
    'RoutingDecision'
]
