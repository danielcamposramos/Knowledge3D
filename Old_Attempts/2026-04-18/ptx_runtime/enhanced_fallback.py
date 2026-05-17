"""Enhanced Fallback - Claude's Enhancement #4

Graduated fallback hierarchy for robust error recovery.
"""
import logging
from enum import IntEnum

logger = logging.getLogger(__name__)


class FallbackLevel(IntEnum):
    """Fallback levels with increasing robustness"""
    TEMPORAL_FULL = 0    # Full temporal mode (baseline)
    TEMPORAL_HALF = 1    # 50% sparsity reduction, retry temporal
    SPATIAL_CACHED = 2   # Spatial-only with cached weights
    SPATIAL_DENSE = 3    # Spatial with no sparsity (ultra-safe)


# Budget cost for each fallback level (in microseconds)
FALLBACK_BUDGET_US = {
    FallbackLevel.TEMPORAL_FULL: 0,    # baseline
    FallbackLevel.TEMPORAL_HALF: 5,    # +5 µs for reduced sparsity
    FallbackLevel.SPATIAL_CACHED: 0,   # same as current
    FallbackLevel.SPATIAL_DENSE: 8,    # +8 µs for dense matvec
}


class EnhancedFallback:
    """Graduated fallback engine with telemetry"""

    def __init__(self):
        self.fallback_counts = {level: 0 for level in FallbackLevel}
        self.fallback_success_rates = {level: [] for level in FallbackLevel}

    def attempt_fallback(self, level: FallbackLevel, bridge, input_emb, modal_sig, original_error=None):
        """
        Attempt recovery at specified fallback level.

        Args:
            level: Fallback level to attempt
            bridge: ThinkingTagBridge instance
            input_emb: Input embedding
            modal_sig: Modal signature
            original_error: Original error that triggered fallback

        Returns:
            (success: bool, result: ThinkingTagOutput or None)
        """
        self.fallback_counts[level] += 1

        logger.warning(f"Attempting fallback level {level.name} (attempt #{self.fallback_counts[level]})")

        try:
            if level == FallbackLevel.TEMPORAL_FULL:
                # This is the baseline - should not be called as fallback
                return False, None

            elif level == FallbackLevel.TEMPORAL_HALF:
                # Reduce sparsity by 50%, retry temporal path
                sparsity_level = bridge.adaptive_sparsity.calculate_sparsity(input_emb, modal_sig)
                reduced_sparsity = sparsity_level * 0.5  # 50% reduction

                # Query with reduced sparsity
                trajectories = bridge.resonance_field.query(
                    input_emb,
                    sparsity=reduced_sparsity,
                    time_window=0.5
                )

                enriched = bridge.cross_modal_engine.apply_resonance_pattern(trajectories, modal_sig)
                sparse_weights = bridge.adaptive_sparsity.apply_adaptive_sparsity(enriched, reduced_sparsity)
                context = bridge.temporal_reasoning.compute_deltas(enriched)

                output = bridge._execute_temporal_mlp(input_emb, sparse_weights, context)
                crystallized = bridge.graph_crystallizer.apply(output, bridge.ema_buffer)

                confidence_rays = bridge.vector_resonator.compute(confidence_vector=crystallized)
                uncertainty = bridge._compute_entropy(crystallized)
                coherence_scores = bridge.temporal_reasoning.estimate_coherence(context)

                from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagOutput
                result = ThinkingTagOutput(crystallized, confidence_rays, uncertainty, coherence_scores)
                result.tags = [("fallback_temporal_half", 1.0)]

                self.fallback_success_rates[level].append(1.0)
                return True, result

            elif level == FallbackLevel.SPATIAL_CACHED:
                # Spatial-only with cached weights (current fallback)
                trajectories = bridge.resonance_field.query(
                    input_emb, sparsity=0.1, region="thinking_weights"
                )
                sparse_weights = bridge._assemble_sparse_weights(trajectories)

                output = bridge._execute_spatial_mlp(input_emb, sparse_weights)
                probs = bridge._sigmoid_approx(output)

                from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagOutput
                result = ThinkingTagOutput(
                    probs,
                    np.ones_like(probs),
                    0.99,
                    np.zeros_like(probs)
                )
                result.tags = [("fallback_spatial_cached", 0.99)]

                self.fallback_success_rates[level].append(1.0)
                return True, result

            elif level == FallbackLevel.SPATIAL_DENSE:
                # Ultra-safe dense mode (no sparsity)
                trajectories = bridge.resonance_field.query(
                    input_emb, sparsity=0.0, region="thinking_weights"  # No sparsity
                )

                # Dense weight assembly (no sparsification)
                dense_weights = bridge._assemble_sparse_weights(trajectories)

                output = bridge._execute_spatial_mlp(input_emb, dense_weights)
                probs = bridge._sigmoid_approx(output)

                from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagOutput
                result = ThinkingTagOutput(
                    probs,
                    np.ones_like(probs),
                    0.95,
                    np.zeros_like(probs)
                )
                result.tags = [("fallback_spatial_dense", 0.95)]

                self.fallback_success_rates[level].append(1.0)
                return True, result

        except Exception as e:
            logger.error(f"Fallback level {level.name} failed: {e}")
            self.fallback_success_rates[level].append(0.0)
            return False, None

    def get_stats(self) -> dict:
        """Get fallback statistics"""
        import numpy as np

        stats = {
            "fallback_counts": {level.name: count for level, count in self.fallback_counts.items()},
            "fallback_success_rates": {}
        }

        for level, rates in self.fallback_success_rates.items():
            if rates:
                stats["fallback_success_rates"][level.name] = {
                    "avg": float(np.mean(rates)),
                    "count": len(rates)
                }
            else:
                stats["fallback_success_rates"][level.name] = {
                    "avg": 0.0,
                    "count": 0
                }

        total_fallbacks = sum(self.fallback_counts.values())
        stats["total_fallbacks"] = total_fallbacks

        return stats

    def get_fallback_rate(self) -> float:
        """Get overall fallback rate"""
        total = sum(self.fallback_counts.values())
        return total
