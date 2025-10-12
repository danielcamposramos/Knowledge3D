"""Modal Affinity Matrix - Claude's Enhancement #5

Learned modal affinity matrix for cross-modal confidence boosting.
"""
import numpy as np
import logging
from .sovereign.loader import gpu_malloc, memcpy_htod, memcpy_dtoh

logger = logging.getLogger(__name__)


class ModalAffinityMatrix:
    """Learned modal affinity matrix with EMA updates"""

    def __init__(self):
        # Initialize with symmetric identity-like matrix
        self.affinity_matrix = np.array([
            [1.0, 0.5, 0.3],  # text -> text, image, audio
            [0.5, 1.0, 0.4],  # image -> text, image, audio
            [0.3, 0.4, 1.0]   # audio -> text, image, audio
        ], dtype=np.float32)

        self.success_rates = np.ones((3, 3), dtype=np.float32) * 0.5
        self.ema_alpha = 0.1  # EMA smoothing factor

        # Modal name to index mapping
        self.modal_to_idx = {'text': 0, 'image': 1, 'audio': 2}

        # GPU buffer for affinity matrix
        try:
            self.gpu_buffer = gpu_malloc(36)  # 3x3 float32 = 36 bytes
            self._update_gpu_buffer()
        except Exception as e:
            logger.warning(f"Could not allocate GPU buffer for affinity matrix: {e}")
            self.gpu_buffer = None

    def _update_gpu_buffer(self):
        """Update GPU buffer with current affinity matrix"""
        if self.gpu_buffer is not None:
            try:
                memcpy_htod(self.gpu_buffer, self.affinity_matrix.ctypes.data, 36)
            except Exception as e:
                logger.warning(f"Could not update GPU buffer: {e}")

    def get_affinity(self, source_modality: str, target_modality: str) -> float:
        """Get affinity between two modalities"""
        if source_modality not in self.modal_to_idx or target_modality not in self.modal_to_idx:
            return 0.5

        src_idx = self.modal_to_idx[source_modality]
        tgt_idx = self.modal_to_idx[target_modality]
        return float(self.affinity_matrix[src_idx, tgt_idx])

    def update_success(self, modal_signature: list, success_score: float):
        """Update success rates using EMA"""
        if success_score < 0.0 or success_score > 1.0:
            logger.warning(f"Invalid success score: {success_score}, clamping to [0,1]")
            success_score = max(0.0, min(1.0, success_score))

        # Update pairwise success rates
        for i, mod1 in enumerate(modal_signature):
            if mod1 not in self.modal_to_idx:
                continue

            idx1 = self.modal_to_idx[mod1]

            for j, mod2 in enumerate(modal_signature):
                if mod2 not in self.modal_to_idx:
                    continue

                idx2 = self.modal_to_idx[mod2]

                # EMA update
                old_rate = self.success_rates[idx1, idx2]
                new_rate = self.ema_alpha * success_score + (1 - self.ema_alpha) * old_rate
                self.success_rates[idx1, idx2] = new_rate

        # Blend affinity matrix with success rates
        self.affinity_matrix = 0.7 * self.affinity_matrix + 0.3 * self.success_rates

        # Update GPU buffer
        self._update_gpu_buffer()

        logger.debug(f"Updated affinity matrix for {modal_signature} with success {success_score:.3f}")

    def get_modal_boost(self, modal_signature: list) -> float:
        """Get confidence boost for modality combination"""
        if len(modal_signature) < 2:
            return 1.0

        total_affinity = 0.0
        count = 0

        # Calculate average pairwise affinity
        for i, mod1 in enumerate(modal_signature):
            for j, mod2 in enumerate(modal_signature):
                if i != j and mod1 in self.modal_to_idx and mod2 in self.modal_to_idx:
                    total_affinity += self.get_affinity(mod1, mod2)
                    count += 1

        if count == 0:
            return 1.0

        avg_affinity = total_affinity / count

        # Boost factor: 1.0 to 1.5 based on affinity
        boost = 1.0 + 0.5 * avg_affinity

        return float(boost)

    def get_gpu_buffer(self):
        """Get GPU buffer pointer"""
        return self.gpu_buffer

    def get_affinity_matrix(self) -> np.ndarray:
        """Get current affinity matrix"""
        return self.affinity_matrix.copy()

    def get_stats(self) -> dict:
        """Get affinity matrix statistics"""
        return {
            "affinity_matrix": self.affinity_matrix.tolist(),
            "success_rates": self.success_rates.tolist(),
            "ema_alpha": self.ema_alpha,
            "avg_affinity": float(np.mean(self.affinity_matrix)),
            "max_affinity": float(np.max(self.affinity_matrix)),
            "min_affinity": float(np.min(self.affinity_matrix))
        }
