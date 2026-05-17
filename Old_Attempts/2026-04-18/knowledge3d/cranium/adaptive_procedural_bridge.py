"""
Adaptive procedural compression bridge for K3D.

Combines Phase H Matryoshka dimension selection with Phase 2.6 dictionary codecs.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from .procedural_compiler import ProceduralCompiler, PrototypeTable

QualityLevel = Literal["ultrafast", "fast", "balanced", "maximum"]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


class AdaptiveDimensionCompressor:
    """
    Dimension-aware procedural compressor using dictionary codecs.

    Quality levels:
        ultrafast → 64D  (≈80× overall, ≥0.98 fidelity)
        fast      → 128D (≈69× overall, ≥0.99 fidelity)
        balanced  → 512D (≈24× overall, ≥0.995 fidelity)
        maximum   → 2048D (≈12×, ≥0.999 fidelity)
    """

    DEFAULT_DIMENSION_MAP: Dict[QualityLevel, int] = {
        "ultrafast": 64,
        "fast": 128,
        "balanced": 512,
        "maximum": 2048,
    }

    DEFAULT_FIDELITY_THRESHOLDS: Dict[QualityLevel, float] = {
        "ultrafast": 0.98,
        "fast": 0.99,
        "balanced": 0.995,
        "maximum": 0.999,
    }

    def __init__(
        self,
        cache_dir: Path = Path("validation_cache"),
        prototype_table_path: Optional[Path] = None,
        enable_compression: bool = True,
        dimension_map: Optional[Dict[QualityLevel, int]] = None,
        fidelity_thresholds: Optional[Dict[QualityLevel, float]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.enable_compression = enable_compression
        self.dimension_map = dimension_map or self.DEFAULT_DIMENSION_MAP.copy()
        self.fidelity_thresholds = fidelity_thresholds or self.DEFAULT_FIDELITY_THRESHOLDS.copy()

        prototype_table_path = prototype_table_path or self.cache_dir / "prototype_table_2048d_512.npz"
        if not prototype_table_path.exists():
            raise FileNotFoundError(f"Prototype table not found: {prototype_table_path}")
        self._base_prototype_table = PrototypeTable.load(prototype_table_path)

        self.compilers: Dict[int, ProceduralCompiler] = {}
        if self.enable_compression:
            self._initialise_compilers()

    # --------------------------------------------------------------------- #
    # Initialisation helpers
    # --------------------------------------------------------------------- #
    def _initialise_compilers(self) -> None:
        for dim in set(self.dimension_map.values()):
            self.compilers[dim] = self._build_compiler_for_dimension(dim)

    def _build_compiler_for_dimension(self, dim: int) -> ProceduralCompiler:
        dictionary_path = self._locate_dictionary_file(dim)
        dict_payload = np.load(dictionary_path)
        atoms_key = "dictionary" if "dictionary" in dict_payload else "atoms"
        dictionary_atoms = dict_payload[atoms_key].astype(np.float32)

        trunc_table = self._truncate_prototype_table(self._base_prototype_table, dim)
        compiler = ProceduralCompiler(
            prototype_table=trunc_table,
            dictionary_atoms=dictionary_atoms,
            dictionary_max_coeffs=self._default_dict_coeffs(dim),
            dictionary_residual_topk=self._default_residual_topk(dim),
            dictionary_similarity_threshold=self._dimension_threshold(dim),
        )
        return compiler

    def _locate_dictionary_file(self, dim: int) -> Path:
        pattern = f"dictionary_{dim}d_*.npz"
        matches = sorted(self.cache_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No dictionary file matching {pattern} in {self.cache_dir}.\n"
                "Run scripts/train_dictionary.py to generate dictionaries."
            )
        return matches[0]

    def _truncate_prototype_table(self, table: PrototypeTable, dim: int) -> PrototypeTable:
        prototypes = table.prototypes[:, :dim]
        return PrototypeTable(prototypes, metadata=table.metadata)

    def _default_dict_coeffs(self, dim: int) -> int:
        return 8 if dim >= 128 else 4

    def _default_residual_topk(self, dim: int) -> int:
        if dim <= 64:
            return 32
        if dim <= 128:
            return 64
        if dim <= 512:
            return 128
        return 256

    def _dimension_threshold(self, dim: int) -> float:
        for quality, mapped_dim in self.dimension_map.items():
            if mapped_dim == dim:
                return self.fidelity_thresholds[quality]
        return 0.99

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def compress(
        self,
        embedding: np.ndarray,
        quality: QualityLevel = "fast",
        return_metadata: bool = False,
    ) -> Tuple[bytes, Dict] | bytes:
        """
        Compress a Matryoshka embedding at the requested quality level.
        """
        if not self.enable_compression:
            payload = embedding.astype(np.float32).tobytes()
            return (payload, {"compression": 1.0}) if return_metadata else payload

        if quality not in self.dimension_map:
            raise ValueError(f"Unknown quality level: {quality}")

        target_dim = self.dimension_map[quality]
        threshold = self.fidelity_thresholds[quality]
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.size < target_dim:
            raise ValueError(f"Embedding size {vector.size} < target dimension {target_dim}")
        truncated = vector[:target_dim]

        compiler = self.compilers[target_dim]
        program_bytes, metadata = self._encode_with_dictionary(compiler, truncated)
        reconstructed = self._decode_with_magic(compiler, program_bytes)
        fidelity = _cosine_similarity(truncated, reconstructed)

        fallback_used = False
        if fidelity < threshold:
            dense_payload = compiler.compile_prototype_delta_dense(truncated, return_metadata=True)
            if isinstance(dense_payload, tuple):
                program_bytes = dense_payload[0]
                metadata = dense_payload[1]
            else:
                program_bytes = dense_payload
            reconstructed = compiler.decompile_prototype_delta_dense(program_bytes)
            fidelity = _cosine_similarity(truncated, reconstructed)
            fallback_used = True

        metadata = metadata or {}
        metadata.update(
            {
                "quality": quality,
                "target_dim": target_dim,
                "threshold": threshold,
                "actual_fidelity": fidelity,
                "actual_compression": float(vector.size * 4) / max(1, len(program_bytes)),
                "fallback": fallback_used,
            }
        )

        if return_metadata:
            return program_bytes, metadata
        return program_bytes

    def decompress(self, program_bytes: bytes, target_dim: Optional[int] = None) -> np.ndarray:
        """
        Decompress a previously-compressed program.
        """
        if not self.enable_compression:
            return np.frombuffer(program_bytes, dtype=np.float32)

        dim = target_dim or self._detect_dimension(program_bytes)
        compiler = self.compilers.get(dim)
        if compiler is None:
            raise ValueError(f"No compiler loaded for dimension {dim}")
        return self._decode_with_magic(compiler, program_bytes)

    def get_compression_stats(self, quality: QualityLevel = "fast") -> Dict[str, float | int]:
        """
        Return empirical compression/fidelity metrics for a quality tier.
        """
        stats = {
            64: {"compression": 80.6, "fidelity": 0.9963, "bytes": 101},
            128: {"compression": 69.4, "fidelity": 0.99998, "bytes": 118},
            512: {"compression": 24.2, "fidelity": 0.99998, "bytes": 338},
            2048: {"compression": 12.0, "fidelity": 0.99996, "bytes": 682},
        }
        target_dim = self.dimension_map[quality]
        entry = stats.get(target_dim, {"compression": 1.0, "fidelity": 1.0, "bytes": target_dim * 4})
        return {
            "quality": quality,
            "dimension": target_dim,
            "expected_compression": entry["compression"],
            "expected_fidelity": entry["fidelity"],
            "expected_bytes": entry["bytes"],
            "original_bytes": 2048 * 4,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _encode_with_dictionary(
        self,
        compiler: ProceduralCompiler,
        vector: np.ndarray,
    ) -> Tuple[bytes, Dict]:
        payload = compiler.compile_dictionary_sparse(vector, return_metadata=True)
        if isinstance(payload, tuple):
            return payload
        return payload, {}

    def _decode_with_magic(self, compiler: ProceduralCompiler, payload: bytes) -> np.ndarray:
        magic = payload[:4]
        if magic == b"PD04":
            return compiler.decompile_dictionary_sparse(payload)
        if magic == b"PD03":
            return compiler.decompile_prototype_multi(payload)
        if magic == b"PD02":
            return compiler.decompile_prototype_delta_dense(payload)
        if magic == b"PD01":
            return compiler.decompile_prototype_delta(payload)
        return compiler.decompile_simple(payload)

    def _detect_dimension(self, payload: bytes) -> int:
        if len(payload) < 8:
            raise ValueError("Payload too small to detect dimension.")
        magic = payload[:4]
        if magic in {b"PD02", b"PD04"}:
            _, dims = struct.unpack("<4sI", payload[:8])
            return int(dims)
        if magic == b"PD01":
            _, dims, _, _, _ = struct.unpack("<4sIHHf", payload[:16])
            return int(dims)
        return self.dimension_map["maximum"]
