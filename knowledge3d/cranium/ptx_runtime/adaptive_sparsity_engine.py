import numpy as np
import logging

logger = logging.getLogger(__name__)

class AdaptiveSparsityEngine:
    def __init__(self, vector_resonator, atomic_fission_fusion):
        self.vector_resonator = vector_resonator
        self.atomic_fission_fusion = atomic_fission_fusion

    def calculate_sparsity(self, input_embedding, modal_signature):
        try:
            complexity = self.vector_resonator.calculate_complexity(
                input_embedding, modal_signature
            )
            logger.debug(f"Input complexity: {complexity}")
        except Exception as e:
            logger.warning(f"Complexity calculation failed: {e}")
            complexity = 0.5

        if complexity < 0.3:
            return 0.05
        elif complexity < 0.7:
            return 0.1
        else:
            return 0.2

    def apply_adaptive_sparsity(self, weights, sparsity_level):
        logger.debug(f"Applying sparsity: {sparsity_level}")
        try:
            return self.atomic_fission_fusion.create_sparse(
                weights, sparsity_level, preserve_important=True
            )
        except Exception as e:
            logger.error(f"Sparsity application failed: {e}")
            return weights  # Fallback to original
