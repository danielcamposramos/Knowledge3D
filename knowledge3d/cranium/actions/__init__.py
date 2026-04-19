"""Action-related helpers for the fused head output layer.

Post-purge surface (2026-04-18):
    The original numpy-dtype ``ActionBuffer`` / ``ACTION_BUFFER_DTYPE``
    pair lives in ``Old_Attempts/2026-04-18/`` as part of
    ``TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md``. The
    sovereign successor is a pure-ctypes ``ActionBufferStruct`` +
    ``ActionBuffer`` pair that preserves the 288-byte PTX contract
    (see :mod:`knowledge3d.cranium.actions.action_types`).

    The following helpers remain archived and should be re-driven
    directly from PTX when needed:

        confidence_propagation / ConfidencePropagator
        context_aware_alpha / ContextAwareAlpha
        multi_modal_confidence_propagation / MultiModalConfidencePropagator
        enhanced_multi_modal_confidence_propagation / EnhancedMultiModalConfidencePropagator
        adaptive_convergence_analyzer / AdaptiveConvergenceAnalyzer

    Their sovereign successors live in the PTX kernels
    ``confidence_propagation.ptx``, ``adaptive_convergence.ptx``, and
    ``decode_actions.ptx`` under ``knowledge3d/cranium/ptx/``.
"""

from .action_types import (  # noqa: F401
    ACTION_BUFFER_SIZE,
    ActionBuffer,
    ActionBufferStruct,
    ActionResult,
    ActionType,
)
from .alpha_rl_optimizer import AlphaRLOptimizer, AlphaRange, AlphaState  # noqa: F401
from .advanced_alpha_rl_optimizer import AdvancedAlphaRLOptimizer  # noqa: F401

__all__ = [
    "ACTION_BUFFER_SIZE",
    "ActionBuffer",
    "ActionBufferStruct",
    "ActionResult",
    "ActionType",
    "AlphaRLOptimizer",
    "AlphaRange",
    "AlphaState",
    "AdvancedAlphaRLOptimizer",
]
