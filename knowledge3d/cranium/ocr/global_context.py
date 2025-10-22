"""
GlobalContextEncoder: CLIP-large inspired global understanding.

Maps DeepSeek-OCR's CLIP-large (300M params) to K3D's sovereign stack.
Uses dense attention for document-level context.

Phase E: Leverage existing GalaxyResonanceEngine
Phase F: Enhanced with PTX dense attention kernels
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class GlobalContextEncoder:
    """
    CLIP-large inspired global understanding.

    DeepSeek approach:
    - Dense attention for global document context
    - Fuses visual tokens with semantic understanding
    - Output: Document-level embedding

    K3D Phase E implementation:
    - Leverage existing GalaxyResonanceEngine for semantic embeddings
    - Simple fusion of visual + semantic features
    - Phase F: PTX dense attention kernels
    """

    def __init__(self, galaxy_path: Optional[Path] = None):
        """
        Initialize global context encoder.

        Args:
            galaxy_path: Path to K3D Galaxy GLB (optional)
        """
        self.galaxy_path = galaxy_path
        self.galaxy = None
        self.embedding_dim = 512  # K3D standard embedding dimension

    def encode_global_context(
        self,
        compressed_tokens: np.ndarray,
        text_content: str
    ) -> np.ndarray:
        """
        Encode global document context from compressed visual tokens and text.

        Args:
            compressed_tokens: Compressed visual features (H/16, W/16, 256)
            text_content: Extracted text content from document

        Returns:
            Global context embedding (512,) float32
        """
        # Get semantic embedding via RPN (or Galaxy if available)
        text_embedding = self._get_semantic_embedding(text_content)

        # Flatten compressed tokens and extract visual features
        token_sequence = compressed_tokens.reshape(-1, compressed_tokens.shape[-1])

        if token_sequence.shape[0] == 0:
            # No visual tokens, return semantic embedding only
            return text_embedding

        # DeepSeek-style: Average pooling over spatial dimensions
        visual_features = token_sequence.mean(axis=0).astype(np.float32)

        # Ensure dimensions match for fusion
        if visual_features.shape[0] > self.embedding_dim:
            visual_features = visual_features[:self.embedding_dim]
        elif visual_features.shape[0] < self.embedding_dim:
            # Pad with zeros
            pad_size = self.embedding_dim - visual_features.shape[0]
            visual_features = np.pad(visual_features, (0, pad_size), mode='constant')

        # Fuse visual + semantic (DeepSeek-style multimodal fusion)
        # Phase E: Simple weighted average
        # Phase F: PTX cross-attention kernel
        global_context = 0.5 * visual_features + 0.5 * text_embedding

        # Normalize
        norm = np.linalg.norm(global_context)
        if norm > 1e-8:
            global_context = global_context / norm

        return global_context.astype(np.float32)

    def _get_semantic_embedding(self, text: str) -> np.ndarray:
        """
        Get semantic embedding from text using RPN or Galaxy.

        Args:
            text: Text content to embed

        Returns:
            Semantic embedding (512,) float32
        """
        if not text or not text.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Phase E: Use RPN engine directly
        # Phase F: Enhance with Galaxy resonance if available
        try:
            from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
            rpn = RPNEmbeddingEngine()

            # Truncate long text
            text_snippet = text[:512] if len(text) > 512 else text

            result = rpn.embed_sentence(text_snippet)

            # RPN returns 128-dim, extend to 512
            rpn_embedding = result if isinstance(result, np.ndarray) else result.get("embedding_128", np.zeros(128, dtype=np.float32))

            # Extend to 512 dimensions (repeat pattern)
            embedding_512 = np.zeros(self.embedding_dim, dtype=np.float32)
            for i in range(self.embedding_dim):
                embedding_512[i] = rpn_embedding[i % 128]

            # Normalize
            norm = np.linalg.norm(embedding_512)
            if norm > 1e-8:
                embedding_512 = embedding_512 / norm

            return embedding_512

        except Exception:
            # Fallback: zero embedding
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def load_galaxy(self, galaxy_path: Path) -> None:
        """
        Load K3D Galaxy for enhanced semantic resonance.

        Args:
            galaxy_path: Path to Galaxy GLB file

        Note: Phase F enhancement - not used in Phase E
        """
        self.galaxy_path = galaxy_path
        # Phase F: Load Galaxy and enable resonance-based semantic embedding
        # For Phase E, we use RPN only
