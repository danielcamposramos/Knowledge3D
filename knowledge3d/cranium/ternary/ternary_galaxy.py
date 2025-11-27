"""
Content-addressed storage for ternary codec data (seed + residual).

Seeds are procedural RPN programs; residuals are ternary vectors stored once and
referenced by content hash. All data is GPU-resident; host hashes rely on the
packed host bytes kept in TernaryVector.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

from knowledge3d.cranium.ternary.ternary_vector import TernaryVector


class TernaryGalaxy:
    """Galaxy-backed deduplication for ternary codec artifacts."""

    def __init__(self) -> None:
        self.seeds: Dict[str, str] = {}  # seed_id -> RPN seed program
        self.residuals: Dict[bytes, TernaryVector] = {}  # hash -> vector
        self.frame_refs: Dict[str, Tuple[str, bytes]] = {}  # frame_id -> (seed_id, residual_hash)

    def store_frame(self, frame_id: str, seed_rpn: str, residual: TernaryVector) -> None:
        seed_hash = hashlib.sha256(seed_rpn.encode("utf-8")).hexdigest()
        seed_id = f"seed_{seed_hash}"
        if seed_id not in self.seeds:
            self.seeds[seed_id] = seed_rpn

        residual_hash = self._hash_ternary(residual)
        if residual_hash not in self.residuals:
            self.residuals[residual_hash] = residual

        self.frame_refs[frame_id] = (seed_id, residual_hash)

    def load_frame(self, frame_id: str) -> Tuple[str, TernaryVector]:
        if frame_id not in self.frame_refs:
            raise KeyError(f"Frame {frame_id} not found")
        seed_id, residual_hash = self.frame_refs[frame_id]
        return self.seeds[seed_id], self.residuals[residual_hash]

    def _hash_ternary(self, vector: TernaryVector) -> bytes:
        """Hash packed ternary bytes (host copy)."""
        return hashlib.sha256(vector.packed_host).digest()

    def stats(self) -> Dict[str, int]:
        return {
            "unique_seeds": len(self.seeds),
            "unique_residuals": len(self.residuals),
            "total_frames": len(self.frame_refs),
            "deduplication_ratio": len(self.frame_refs) // max(1, len(self.residuals)),
        }


__all__ = ["TernaryGalaxy"]
