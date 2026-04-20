# -*- coding: utf-8 -*-
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
- Sovereign: Zero PyTorch/TensorFlow dependencies in the hot path
- Adaptive: Dimensions scale to task complexity
- Self-Updating: Validation-gated weight updates
- Memory Efficient: LoRA-style adapters (18× reduction)

This module exposes high-level TRM entry points but keeps heavy
dependencies (NumPy/PyTorch) out of the import-time path. Symbols are
resolved lazily on first attribute access.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

# Always-loaded foundational knowledge
from knowledge3d.cranium.math_galaxy import get_math_galaxy


_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Adaptive Swarm
    "AdaptiveSwarmTRM": ("adaptive_swarm", "AdaptiveSwarmTRM"),
    "SwarmConfig": ("adaptive_swarm", "SwarmConfig"),
    "SwarmTrainingProtocol": ("adaptive_swarm", "SwarmTrainingProtocol"),
    # Matryoshka TRM
    "MatryoshkaTRM": ("matryoshka_trm", "MatryoshkaTRM"),
    "DimensionSelector": ("matryoshka_trm", "DimensionSelector"),
    # Adapters
    "AdapterWeights": ("trm_adapters", "AdapterWeights"),
    "SelfUpdatingAdapter": ("trm_adapters", "SelfUpdatingAdapter"),
    "AdapterConfig": ("trm_adapters", "AdapterConfig"),
    # MoE Routing
    "MoERouter": ("moe_router", "MoERouter"),
    "RoutingStrategy": ("moe_router", "RoutingStrategy"),
    "RoutingConfig": ("moe_router", "RoutingConfig"),
    "TaskComplexityEstimator": ("moe_router", "TaskComplexityEstimator"),
    "RoutingAnalyzer": ("moe_router", "RoutingAnalyzer"),
    # Router Specialist
    "RouterBootstrap": ("router_specialist", "RouterBootstrap"),
    "RouterSpecialistTrainer": ("router_specialist", "RouterSpecialistTrainer"),
    "RouterTransition": ("router_specialist", "RouterTransition"),
    "RoutingDecision": ("router_specialist", "RoutingDecision"),
    # Procedural stack
    "ProceduralCompiler": ("procedural_compiler", "ProceduralCompiler"),
    "ProceduralProgram": ("procedural_compiler", "ProceduralProgram"),
    "PrototypeTable": ("procedural_compiler", "PrototypeTable"),
    "ProceduralGalaxy": ("procedural_galaxy", "ProceduralGalaxy"),
    "PhaseGProceduralBridge": ("phase_g_procedural_bridge", "PhaseGProceduralBridge"),
    "ProceduralFidelityValidator": ("fidelity_validator", "ProceduralFidelityValidator"),
    "ProceduralFidelityResult": ("fidelity_validator", "ProceduralFidelityResult"),
    "AdaptiveDimensionCompressor": ("adaptive_procedural_bridge", "AdaptiveDimensionCompressor"),
    "PhaseHProceduralIntegration": ("phase_h_procedural_integration", "PhaseHProceduralIntegration"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve TRM symbols to avoid importing heavy stacks on import."""
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = __import__(f"{__name__}.{module_name}", fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy symbols via dir()."""
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))

MATH_GALAXY = get_math_galaxy()

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
    'RoutingDecision',

    # Procedural stack
    'ProceduralCompiler',
    'ProceduralProgram',
    'PrototypeTable',
    'ProceduralGalaxy',
    'PhaseGProceduralBridge',
    'ProceduralFidelityValidator',
    'ProceduralFidelityResult',
    'AdaptiveDimensionCompressor',
    'PhaseHProceduralIntegration',
    # Foundational knowledge
    'get_math_galaxy',
]
