"""
Procedural compiler that converts dense embeddings into sovereign RPN programs.

The compiler emits instruction streams that can be executed by the modular RPN
kernel (see knowledge3d/cranium/kernels/modular_rpn_kernel.cu). Programs are
stored as compact byte blobs with three sections:

    [header | opcode bytes | scalar literals | vector literals]

Vectors are stored in float16 precision to keep programs in the 40–64 byte
range for typical 128‑dimensional embeddings.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


PROCEDURAL_VERSION = 1
PROTOTYPES = np.array(
    [
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)


@dataclass
class ProceduralProgram:
    """Serializable procedural program."""

    version: int
    opcodes: np.ndarray  # uint8
    scalars: np.ndarray  # float32
    vectors: np.ndarray  # float32, shape (n, 3)

    def to_bytes(self) -> bytes:
        """Serialize program to bytes: <BHHH | opcodes | scalars_fp16 | vectors_fp16>."""
        opcodes_u8 = self.opcodes.astype(np.uint8)
        scalars_fp16 = self.scalars.astype(np.float16)
        vectors_fp16 = self.vectors.astype(np.float16)
        header = struct.pack(
            "<BHHH",
            self.version,
            opcodes_u8.size,
            scalars_fp16.size,
            vectors_fp16.size,
        )
        return header + opcodes_u8.tobytes() + scalars_fp16.tobytes() + vectors_fp16.tobytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ProceduralProgram":
        version, op_count, scalar_count, vector_count = struct.unpack_from("<BHHH", payload, 0)
        offset = struct.calcsize("<BHHH")
        opcodes = np.frombuffer(payload, dtype=np.uint8, count=op_count, offset=offset)
        offset += op_count
        scalars = np.frombuffer(payload, dtype=np.float16, count=scalar_count, offset=offset).astype(np.float32)
        offset += scalar_count * np.dtype(np.float16).itemsize
        vectors = (
            np.frombuffer(payload, dtype=np.float16, count=vector_count, offset=offset)
            .astype(np.float32)
            .reshape(-1, 3)
        )
        return cls(version=version, opcodes=opcodes, scalars=scalars, vectors=vectors)


class PrototypeTable:
    """
    Stores high-dimensional prototype embeddings for differential encoding.

    Provides helper constructors for k-means training, persistence helpers, and
    nearest-prototype queries used by the procedural compiler.
    """

    def __init__(
        self,
        prototypes: np.ndarray,
        metadata: Optional[Dict] = None,
        basis: Optional[np.ndarray] = None,
    ) -> None:
        prototypes = np.asarray(prototypes, dtype=np.float32)
        if prototypes.ndim != 2:
            raise ValueError("Prototype array must be 2D (count x dimension).")
        self.prototypes = self._normalize_rows(prototypes)
        self.metadata = metadata or {}
        self._proto_norms = np.linalg.norm(self.prototypes, axis=1)
        self.basis = self._prepare_basis(basis)

    # ------------------------------------------------------------------ #
    @property
    def count(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.prototypes.shape[1])

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        """Persist prototype table (compressed) alongside metadata."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        basis = self.basis if self.basis is not None else np.empty((0, self.dimension), dtype=np.float32)
        np.savez_compressed(
            target,
            prototypes=self.prototypes,
            metadata=np.array([json.dumps(self.metadata)]),
            basis=basis,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeTable":
        payload = np.load(Path(path), allow_pickle=False)
        prototypes = payload["prototypes"].astype(np.float32)
        metadata_raw = payload["metadata"]
        metadata = json.loads(str(metadata_raw[0])) if metadata_raw.size else {}
        basis = payload["basis"] if "basis" in payload else None
        return cls(prototypes=prototypes, metadata=metadata, basis=basis)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_rows(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return array / norms

    def _prepare_basis(self, basis: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if basis is None:
            return None
        basis = np.asarray(basis, dtype=np.float32)
        if basis.size == 0:
            return None
        if basis.ndim != 2 or basis.shape[1] != self.dimension:
            raise ValueError("Basis must be (rank x dimension).")
        return self._normalize_rows(basis)

    @property
    def basis_rank(self) -> int:
        return 0 if self.basis is None else int(self.basis.shape[0])

    @property
    def has_basis(self) -> bool:
        return self.basis is not None and self.basis.size > 0

    @staticmethod
    def _cosine_distances(
        batch: np.ndarray,
        centroids: np.ndarray,
        centroid_norms: Optional[np.ndarray] = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        batch_norms = np.linalg.norm(batch, axis=1, keepdims=True)
        centroid_norms = centroid_norms if centroid_norms is not None else np.linalg.norm(centroids, axis=1)
        denom = np.clip(batch_norms * centroid_norms[np.newaxis, :], eps, None)
        sims = (batch @ centroids.T) / denom
        sims = np.clip(sims, -1.0, 1.0)
        return 1.0 - sims

    @classmethod
    def _assign_points(cls, data: np.ndarray, centroids: np.ndarray, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """Assign each data point to nearest centroid (returns labels + cosine distances)."""
        num_points = data.shape[0]
        centroid_norms = np.linalg.norm(centroids, axis=1)
        labels = np.empty(num_points, dtype=np.int32)
        distances = np.empty(num_points, dtype=np.float32)

        for start in range(0, num_points, batch_size):
            end = min(num_points, start + batch_size)
            batch = data[start:end]
            dists = cls._cosine_distances(batch, centroids, centroid_norms=centroid_norms)
            labels[start:end] = np.argmin(dists, axis=1)
            distances[start:end] = dists[np.arange(dists.shape[0]), labels[start:end]]

        distances = np.maximum(distances, 0.0)
        return labels, distances

    @classmethod
    def build_from_embeddings(
        cls,
        embeddings: np.ndarray,
        num_prototypes: int,
        *,
        max_iters: int = 50,
        batch_size: int = 2048,
        seed: int = 42,
        tol: float = 1e-4,
        basis_rank: int = 32,
    ) -> Tuple["PrototypeTable", Dict[str, float]]:
        """Train prototypes via k-means on provided embeddings."""
        data = np.asarray(embeddings, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("Embeddings must be 2D.")
        num_points, dim = data.shape
        if num_points < num_prototypes:
            raise ValueError("Not enough embeddings to build prototype table.")
        data = cls._normalize_rows(data)

        rng = np.random.default_rng(seed)
        init_idx = rng.choice(num_points, size=num_prototypes, replace=False)
        centroids = data[init_idx].copy()

        for iteration in range(max_iters):
            labels, _ = cls._assign_points(data, centroids, batch_size=batch_size)
            updated = centroids.copy()
            for proto_idx in range(num_prototypes):
                mask = labels == proto_idx
                if np.any(mask):
                    updated[proto_idx] = data[mask].mean(axis=0, dtype=np.float32)
                else:
                    updated[proto_idx] = data[rng.integers(0, num_points)]
            updated = cls._normalize_rows(updated)

            shift = float(np.linalg.norm(updated - centroids) / max(1, num_prototypes))
            centroids = updated
            if shift < tol:
                break

        labels, distances = cls._assign_points(data, centroids, batch_size=batch_size)
        avg_dist = float(np.mean(distances))
        max_dist = float(np.max(distances))
        basis = None
        effective_basis = min(basis_rank, dim) if basis_rank > 0 else 0
        if effective_basis > 0:
            try:
                _, _, vt = np.linalg.svd(centroids, full_matrices=False)
                basis = vt[:effective_basis].astype(np.float32)
            except np.linalg.LinAlgError:
                basis = None
        metadata = {
            "algorithm": "kmeans",
            "num_embeddings": int(num_points),
            "embedding_dim": int(dim),
            "num_prototypes": int(num_prototypes),
            "max_iters": int(max_iters),
            "avg_distance": avg_dist,
            "max_distance": max_dist,
            "avg_distance_normalized": avg_dist / 2.0,
            "max_distance_normalized": max_dist / 2.0,
            "seed": int(seed),
            "basis_rank": int(basis.shape[0]) if basis is not None else 0,
        }
        table = cls(prototypes=centroids.astype(np.float32), metadata=metadata, basis=basis)
        return table, metadata

    # ------------------------------------------------------------------ #
    def nearest(self, vector: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """Return (index, prototype_vector, distance) for closest prototype."""
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or vector.size != self.dimension:
            raise ValueError("Vector dimensionality mismatch for prototype lookup.")
        proto_norms = self._proto_norms
        vec_norm = float(np.linalg.norm(vector))
        denom = np.clip(vec_norm * proto_norms, 1e-8, None)
        sims = (self.prototypes @ vector) / denom
        sims = np.clip(sims, -1.0, 1.0)
        dists = 1.0 - sims
        idx = int(np.argmin(dists))
        return idx, self.prototypes[idx], float(dists[idx])

    def topk_nearest(self, vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return top-k prototype indices, vectors, and cosine distances."""
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or vector.size != self.dimension:
            raise ValueError("Vector dimensionality mismatch for prototype lookup.")
        sims = self.prototypes @ vector
        idx = np.argsort(sims)[::-1][:k]
        proto_vecs = self.prototypes[idx]
        proto_norms = self._proto_norms[idx]
        vec_norm = float(np.linalg.norm(vector))
        denom = np.clip(vec_norm * proto_norms, 1e-8, None)
        cos = np.clip(sims[idx] / denom, -1.0, 1.0)
        dists = 1.0 - cos
        return idx, proto_vecs, dists

    def evaluate_embeddings(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Compute coverage metrics for provided embeddings."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        _, dists = self._assign_points(embeddings, self.prototypes)
        avg_dist = float(np.mean(dists))
        max_dist = float(np.max(dists))
        return {
            "avg_distance": avg_dist,
            "max_distance": max_dist,
            "avg_distance_normalized": avg_dist / 2.0,
            "max_distance_normalized": max_dist / 2.0,
            "num_embeddings": int(embeddings.shape[0]),
        }


class ProceduralCompiler:
    """Compile dense embeddings into procedural opcode streams."""

    def __init__(
        self,
        chunk: int = 3,
        normalize: bool = True,
        prototype_table: Optional[PrototypeTable] = None,
        prototype_topk: int = 16,
        prototype_topk_step: int = 8,
        prototype_topk_cap: int = 128,
        prototype_cosine_threshold: float = 0.99,
        use_prototype_basis: bool = True,
        multi_max_prototypes: int = 3,
        multi_candidate_count: int = 16,
        multi_residual_topk: int = 256,
        multi_similarity_threshold: float = 0.99,
        dictionary_atoms: Optional[np.ndarray] = None,
        dictionary_max_coeffs: int = 8,
        dictionary_residual_topk: int = 128,
        dictionary_similarity_threshold: float = 0.99,
    ) -> None:
        self.chunk = chunk
        self.normalize = normalize
        self.prototype_table = prototype_table
        self.prototype_topk = max(1, prototype_topk)
        self.prototype_topk_step = max(1, prototype_topk_step)
        self.prototype_topk_cap = max(self.prototype_topk, prototype_topk_cap)
        self.prototype_cosine_threshold = prototype_cosine_threshold
        self.use_prototype_basis = use_prototype_basis
        self.multi_max_prototypes = max(1, multi_max_prototypes)
        self.multi_candidate_count = max(self.multi_max_prototypes, multi_candidate_count)
        self.multi_residual_topk = max(0, multi_residual_topk)
        self.multi_similarity_threshold = multi_similarity_threshold
        self.dictionary_atoms = None
        self.dictionary_atoms_dim: Optional[int] = None
        if dictionary_atoms is not None:
            self.attach_dictionary(dictionary_atoms)
        self.dictionary_max_coeffs = max(1, dictionary_max_coeffs)
        self.dictionary_residual_topk = max(0, dictionary_residual_topk)
        self.dictionary_similarity_threshold = dictionary_similarity_threshold

    # Opcodes used by procedural programs
    _OP_LITERAL_SCALAR = 0x00
    _OP_LITERAL_VECTOR = 0x01
    _OP_PROTOTYPE_LOAD = 0x40
    _OP_DELTA_APPLY = 0x41
    _OP_NORMALIZE = 0x23
    _SIMPLE_HEADER = struct.Struct("<If")
    _DELTA_SENTINEL = 0x80
    _PROTO_DELTA_MAGIC = b"PD01"
    _PROTO_BASIS_MAGIC = b"PB01"
    _PROTO_DENSE_MAGIC = b"PD02"
    _PROTO_MULTI_MAGIC = b"PD03"
    _PROTO_DICT_MAGIC = b"PD04"
    _PROTO_DELTA_HEADER = struct.Struct("<4sIHHf")
    _PROTO_DELTA_ENTRY = struct.Struct("<Hb")
    _PROTO_DENSE_HEADER = struct.Struct("<4sIHf")
    _PROTO_MULTI_HEADER = struct.Struct("<4sIBHf")
    _PROTO_MULTI_PROTO = struct.Struct("<Hf")
    _PROTO_DICT_HEADER = struct.Struct("<4sIHHff")
    _PROTO_DICT_ENTRY = struct.Struct("<Hh")

    def _pad_embedding(self, embedding: np.ndarray) -> np.ndarray:
        pad = (-embedding.size) % self.chunk
        if pad == 0:
            return embedding
        return np.concatenate([embedding, np.zeros(pad, dtype=embedding.dtype)])

    def _select_prototype(self, vec: np.ndarray) -> Tuple[int, np.ndarray]:
        distances = np.linalg.norm(PROTOTYPES - vec, axis=1)
        idx = int(np.argmin(distances))
        return idx, PROTOTYPES[idx]

    def compile_embedding(self, embedding: np.ndarray) -> ProceduralProgram:
        """Compile dense embedding into a procedural program."""
        embedding = np.asarray(embedding, dtype=np.float32)
        padded = self._pad_embedding(embedding)
        chunks = padded.reshape(-1, self.chunk)

        opcodes: List[int] = []
        scalars: List[float] = []
        vectors: List[Tuple[float, float, float]] = []

        for chunk_vec in chunks:
            if chunk_vec.size < 3:
                chunk_vec = np.pad(chunk_vec, (0, 3 - chunk_vec.size))

            proto_idx, proto = self._select_prototype(chunk_vec)
            delta = chunk_vec - proto

            # literal scalar -> prototype load
            opcodes.append(self._OP_LITERAL_SCALAR)
            scalars.append(float(proto_idx))
            opcodes.append(self._OP_PROTOTYPE_LOAD)

            # literal vector delta -> delta apply
            opcodes.append(self._OP_LITERAL_VECTOR)
            vectors.append(tuple(delta.tolist()))
            opcodes.append(self._OP_DELTA_APPLY)

            if self.normalize:
                opcodes.append(self._OP_NORMALIZE)

        return ProceduralProgram(
            version=PROCEDURAL_VERSION,
            opcodes=np.array(opcodes, dtype=np.uint8),
            scalars=np.array(scalars, dtype=np.float32),
            vectors=np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, 3), dtype=np.float32),
        )

    def compile_with_prototype(self, embedding: np.ndarray, prototype: np.ndarray) -> ProceduralProgram:
        """Compile by reusing a provided prototype embedding."""
        embedding = np.asarray(embedding, dtype=np.float32)
        prototype = np.asarray(prototype, dtype=np.float32)
        if prototype.size != embedding.size:
            raise ValueError("Prototype must match embedding dimensionality.")

        padded_embed = self._pad_embedding(embedding)
        padded_proto = self._pad_embedding(prototype)
        chunks = padded_embed.reshape(-1, self.chunk)
        proto_chunks = padded_proto.reshape(-1, self.chunk)

        opcodes: List[int] = []
        scalars: List[float] = []
        vectors: List[Tuple[float, float, float]] = []

        for chunk_vec, proto_vec in zip(chunks, proto_chunks, strict=False):
            if chunk_vec.size < 3:
                chunk_vec = np.pad(chunk_vec, (0, 3 - chunk_vec.size))
            if proto_vec.size < 3:
                proto_vec = np.pad(proto_vec, (0, 3 - proto_vec.size))

            delta = chunk_vec - proto_vec

            opcodes.append(self._OP_LITERAL_VECTOR)
            vectors.append(tuple(proto_vec.tolist()))

            opcodes.append(self._OP_LITERAL_VECTOR)
            vectors.append(tuple(delta.tolist()))

            opcodes.append(self._OP_DELTA_APPLY)
            if self.normalize:
                opcodes.append(self._OP_NORMALIZE)

        return ProceduralProgram(
            version=PROCEDURAL_VERSION,
            opcodes=np.array(opcodes, dtype=np.uint8),
            scalars=np.array(scalars, dtype=np.float32),
            vectors=np.array(vectors, dtype=np.float32),
        )

    # ---------------------------------------------------------------------- #
    # Interpreter
    # ---------------------------------------------------------------------- #
    def decompile_program(self, program_bytes: bytes) -> np.ndarray:
        """Execute a procedural program and reconstruct the embedding."""
        program = ProceduralProgram.from_bytes(program_bytes)
        stack: List[Tuple[str, np.ndarray | float]] = []

        scalar_iter = iter(program.scalars.tolist())
        vector_iter = iter(program.vectors.tolist())

        def push_scalar(value: float) -> None:
            stack.append(("scalar", value))

        def push_vector(value: Sequence[float]) -> None:
            stack.append(("vector", np.array(value, dtype=np.float32)))

        def pop_scalar() -> float:
            if not stack:
                raise RuntimeError("Stack underflow")
            typ, value = stack.pop()
            if typ != "scalar":
                raise RuntimeError("Type mismatch: expected scalar.")
            return float(value)

        def pop_vector() -> np.ndarray:
            if not stack:
                raise RuntimeError("Stack underflow")
            typ, value = stack.pop()
            if typ != "vector":
                raise RuntimeError("Type mismatch: expected vector.")
            return np.asarray(value, dtype=np.float32)

        for opcode in program.opcodes:
            if opcode == self._OP_LITERAL_SCALAR:
                push_scalar(next(scalar_iter, 0.0))
            elif opcode == self._OP_LITERAL_VECTOR:
                push_vector(next(vector_iter, (0.0, 0.0, 0.0)))
            elif opcode == self._OP_PROTOTYPE_LOAD:
                proto_idx = int(pop_scalar())
                proto = PROTOTYPES[max(0, min(len(PROTOTYPES) - 1, proto_idx))]
                push_vector(proto)
            elif opcode == self._OP_DELTA_APPLY:
                delta = pop_vector()
                base = pop_vector()
                push_vector(base + delta)
            elif opcode == self._OP_NORMALIZE:
                vec = pop_vector()
                norm = np.linalg.norm(vec)
                if norm < 1e-6:
                    push_vector(vec)
                else:
                    push_vector(vec / norm)
            else:
                raise RuntimeError(f"Unsupported opcode in CPU interpreter: 0x{opcode:02X}")

        vectors = [value for (typ, value) in stack if typ == "vector"]
        if not vectors:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(vectors, axis=0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Prototype delta compression (Phase 2.2)
    # ------------------------------------------------------------------ #
    def attach_prototype_table(self, table: PrototypeTable) -> None:
        """Attach or replace the active prototype table."""
        if table is None:
            raise ValueError("Prototype table cannot be None.")
        self.prototype_table = table

    def attach_dictionary(self, dictionary_atoms: np.ndarray) -> None:
        """Attach dictionary atoms for sparse coding codecs."""
        atoms = np.asarray(dictionary_atoms, dtype=np.float32)
        if atoms.ndim != 2:
            raise ValueError("Dictionary atoms must be 2D (atoms x dimension).")
        norms = np.linalg.norm(atoms, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.dictionary_atoms = atoms / norms
        self.dictionary_atoms_dim = self.dictionary_atoms.shape[1]

    def _require_prototype_table(self) -> PrototypeTable:
        if self.prototype_table is None:
            raise RuntimeError("Prototype table not configured. Load PrototypeTable before using prototype-delta mode.")
        return self.prototype_table

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def compile_prototype_delta(
        self,
        embedding: np.ndarray,
        *,
        return_metadata: bool = False,
    ) -> Union[bytes, Tuple[bytes, Dict[str, float]]]:
        """
        Differentially encode `embedding` using the nearest prototype and a sparse
        set of quantised delta corrections.

        Format:
            [magic='PD01' | dims:u32 | proto_id:u16 | nnz:u16 | scale:f32 | (idx:u16, val:int8)*]
        """
        table = self.prototype_table
        if table is None:
            fallback = self.compile_embedding_simple(embedding)
            metadata = {"mode": "prototype_sparse", "codec": "simple_fallback"}
            if return_metadata:
                return fallback, metadata
            return fallback
        table = self._require_prototype_table()
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        dims = vector.size
        if dims != table.dimension:
            raise ValueError(f"Embedding dimension {dims} != prototype table dimension {table.dimension}.")

        proto_idx, proto_vec, proto_distance = table.nearest(vector)
        if table.has_basis and self.use_prototype_basis:
            payload, metadata = self._encode_basis_delta(vector, proto_vec, proto_idx, proto_distance, table, dims)
        else:
            payload, metadata = self._encode_sparse_delta(vector, proto_vec, proto_idx, proto_distance, dims)

        metadata["compression_ratio"] = float(vector.nbytes) / max(1, len(payload))
        if return_metadata:
            return payload, metadata
        return payload

    def _encode_sparse_delta(
        self,
        vector: np.ndarray,
        proto_vec: np.ndarray,
        proto_idx: int,
        proto_distance: float,
        dims: int,
    ) -> Tuple[bytes, Dict[str, float]]:
        residual = vector - proto_vec
        abs_residual = np.abs(residual)
        order = np.argsort(abs_residual)[::-1]
        max_k = min(self.prototype_topk_cap, dims)
        k = min(self.prototype_topk, max_k)

        best_payload: Optional[bytes] = None
        metadata: Dict[str, float] = {}

        while True:
            if k == 0 or abs_residual[order[0]] < 1e-9:
                nnz_indices = np.empty(0, dtype=np.int32)
                quantized = np.empty(0, dtype=np.int8)
                scale = 1.0
            else:
                nnz_indices = np.sort(order[:k])
                values = residual[nnz_indices]
                max_abs = float(np.max(np.abs(values)))
                scale = max(max_abs / 127.0, 1e-6)
                quantized = np.clip(np.round(values / scale), -127, 127).astype(np.int8)

            reconstruction = np.array(proto_vec, copy=True)
            if nnz_indices.size:
                reconstruction[nnz_indices] += quantized.astype(np.float32) * float(scale)
            similarity = self._cosine_similarity(vector, reconstruction)

            header = self._PROTO_DELTA_HEADER.pack(
                self._PROTO_DELTA_MAGIC,
                dims,
                int(proto_idx),
                int(nnz_indices.size),
                float(scale),
            )
            payload = bytearray(header)
            for idx, q in zip(nnz_indices, quantized, strict=False):
                payload.extend(self._PROTO_DELTA_ENTRY.pack(int(idx), int(np.int8(q))))
            best_payload = bytes(payload)
            metadata = {
                "codec": "sparse",
                "prototype_id": int(proto_idx),
                "nnz": int(nnz_indices.size),
                "scale": float(scale),
                "similarity": float(similarity),
                "proto_distance": float(proto_distance),
            }

            if similarity >= self.prototype_cosine_threshold or k >= max_k:
                break
            k = min(k + self.prototype_topk_step, max_k)

        if not best_payload:
            raise RuntimeError("Failed to encode sparse prototype delta payload.")
        return best_payload, metadata

    def _encode_basis_delta(
        self,
        vector: np.ndarray,
        proto_vec: np.ndarray,
        proto_idx: int,
        proto_distance: float,
        table: PrototypeTable,
        dims: int,
    ) -> Tuple[bytes, Dict[str, float]]:
        basis = table.basis
        assert basis is not None
        coeffs = basis @ (vector - proto_vec)
        order = np.argsort(np.abs(coeffs))[::-1]
        max_k = min(self.prototype_topk_cap, basis.shape[0])
        k = min(self.prototype_topk, max_k)

        best_payload: Optional[bytes] = None
        metadata: Dict[str, float] = {}

        while True:
            if k == 0 or coeffs.size == 0:
                selected = np.empty(0, dtype=np.int32)
                quantized = np.empty(0, dtype=np.int8)
                scale = 1.0
            else:
                selected = np.sort(order[:k])
                values = coeffs[selected]
                max_abs = float(np.max(np.abs(values))) if values.size else 0.0
                scale = max(max_abs / 127.0, 1e-6)
                quantized = np.clip(np.round(values / scale), -127, 127).astype(np.int8) if values.size else np.empty(0, dtype=np.int8)

            reconstruction = np.array(proto_vec, copy=True)
            if selected.size:
                coeff = quantized.astype(np.float32) * float(scale)
                reconstruction += coeff @ basis[selected]
            similarity = self._cosine_similarity(vector, reconstruction)

            header = self._PROTO_DELTA_HEADER.pack(
                self._PROTO_BASIS_MAGIC,
                dims,
                int(proto_idx),
                int(selected.size),
                float(scale),
            )
            payload = bytearray(header)
            for idx, q in zip(selected, quantized, strict=False):
                payload.extend(self._PROTO_DELTA_ENTRY.pack(int(idx), int(np.int8(q))))
            best_payload = bytes(payload)
            metadata = {
                "codec": "basis",
                "prototype_id": int(proto_idx),
                "nnz": int(selected.size),
                "scale": float(scale),
                "similarity": float(similarity),
                "proto_distance": float(proto_distance),
            }

            if similarity >= self.prototype_cosine_threshold or k >= max_k:
                break
            k = min(k + self.prototype_topk_step, max_k)

        if not best_payload:
            raise RuntimeError("Failed to encode basis prototype delta payload.")
        return best_payload, metadata

    def compile_prototype_delta_dense(
        self,
        embedding: np.ndarray,
        *,
        return_metadata: bool = False,
    ) -> Union[bytes, Tuple[bytes, Dict[str, float]]]:
        """Dense quantised residual codec (PD02)."""
        if self.prototype_table is None:
            fallback = self.compile_embedding_simple(embedding)
            meta = {"mode": "prototype_delta_dense", "codec": "simple_fallback"}
            if return_metadata:
                return fallback, meta
            return fallback
        table = self._require_prototype_table()
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        dims = vector.size
        if dims != table.dimension:
            raise ValueError(f"Embedding dimension {dims} != prototype table dimension {table.dimension}.")

        proto_idx, proto_vec, proto_distance = table.nearest(vector)
        residual = vector - proto_vec
        max_abs = float(np.max(np.abs(residual)))
        scale = max(max_abs / 127.0, 1e-6)
        quantized = np.clip(np.round(residual / scale), -127, 127).astype(np.int8)

        header = self._PROTO_DENSE_HEADER.pack(
            self._PROTO_DENSE_MAGIC,
            dims,
            int(proto_idx),
            float(scale),
        )
        payload = header + quantized.tobytes()
        metadata = {
            "mode": "prototype_delta_dense",
            "codec": "dense",
            "prototype_id": int(proto_idx),
            "scale": float(scale),
            "proto_distance": float(proto_distance),
            "compression_ratio": float(vector.nbytes) / max(1, len(payload)),
        }
        if return_metadata:
            return payload, metadata
        return payload

    def _solve_prototype_blend(self, vector: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray]:
        table = self._require_prototype_table()
        num_candidates = min(table.count, self.multi_candidate_count)
        candidate_indices, _, _ = table.topk_nearest(vector, num_candidates)
        selected: List[int] = []
        weights = np.empty(0, dtype=np.float32)
        residual = np.array(vector, copy=True)
        best_residual_norm = np.linalg.norm(residual, ord=2)

        for _ in range(self.multi_max_prototypes):
            best_idx = None
            best_weights = None
            best_residual = None
            best_norm = best_residual_norm

            for idx in candidate_indices:
                if int(idx) in selected:
                    continue
                trial = selected + [int(idx)]
                P = table.prototypes[trial]
                try:
                    w, *_ = np.linalg.lstsq(P.T, vector, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                trial_residual = vector - P.T @ w
                norm = float(np.linalg.norm(trial_residual, ord=2))
                if norm < best_norm:
                    best_idx = int(idx)
                    best_weights = w.astype(np.float32)
                    best_residual = trial_residual.astype(np.float32)
                    best_norm = norm

            if best_idx is None:
                break
            selected = selected + [best_idx]
            weights = best_weights if best_weights is not None else np.zeros(len(selected), dtype=np.float32)
            residual = best_residual if best_residual is not None else residual
            best_residual_norm = best_norm

        return selected, weights, residual

    def compile_prototype_multi(
        self,
        embedding: np.ndarray,
        *,
        return_metadata: bool = False,
    ) -> Union[bytes, Tuple[bytes, Dict[str, float]]]:
        """Multi-prototype blend + sparse residual codec (PD03)."""
        if self.prototype_table is None:
            return self.compile_prototype_delta_dense(embedding, return_metadata=return_metadata)

        vector = np.asarray(embedding, dtype=np.float32).ravel()
        dims = vector.size
        selected_indices, weights, residual = self._solve_prototype_blend(vector)
        if not selected_indices:
            return self.compile_prototype_delta_dense(embedding, return_metadata=return_metadata)

        residual_entries, scale, sparse_indices, quantized = self._encode_sparse_entries(residual, self.multi_residual_topk)
        header = self._PROTO_MULTI_HEADER.pack(
            self._PROTO_MULTI_MAGIC,
            dims,
            len(selected_indices),
            sparse_indices.size,
            float(scale),
        )
        proto_section = bytearray()
        for idx, weight in zip(selected_indices, weights, strict=False):
            proto_section.extend(self._PROTO_MULTI_PROTO.pack(int(idx), float(weight)))

        payload = bytes(header) + bytes(proto_section) + residual_entries
        reconstructed = self.decompile_prototype_multi(payload)
        similarity = self._cosine_similarity(vector, reconstructed)

        metadata = {
            "mode": "prototype_delta_multi",
            "codec": "multi",
            "prototypes": selected_indices,
            "weights": weights.tolist(),
            "sparse_count": int(sparse_indices.size),
            "nnz": int(sparse_indices.size),
            "scale": float(scale),
            "similarity": float(similarity),
            "proto_distance": float(np.linalg.norm(residual, ord=2)),
            "compression_ratio": float(vector.nbytes) / max(1, len(payload)),
        }

        if similarity < self.multi_similarity_threshold:
            return self._fallback_dense_or_simple(vector, "prototype_delta_multi", return_metadata)

        if return_metadata:
            return payload, metadata
        return payload

    def compile_dictionary_sparse(
        self,
        embedding: np.ndarray,
        *,
        return_metadata: bool = False,
    ) -> Union[bytes, Tuple[bytes, Dict[str, float]]]:
        """Dictionary-based sparse codec (PD04)."""
        if self.dictionary_atoms is None:
            return self.compile_prototype_delta_dense(embedding, return_metadata=return_metadata)

        vector = np.asarray(embedding, dtype=np.float32).ravel()
        dims = vector.size
        if self.dictionary_atoms_dim is None or self.dictionary_atoms_dim != dims:
            return self._fallback_dense_or_simple(vector, "dictionary_sparse", return_metadata)
        selected, coeffs, residual = self._matching_pursuit(vector)
        if not selected:
            return self.compile_prototype_delta_dense(embedding, return_metadata=return_metadata)

        max_abs = float(np.max(np.abs(coeffs)))
        coeff_scale = max(max_abs / 32767.0, 1e-9)
        quantized = np.clip(np.round(coeffs / coeff_scale), -32767, 32767).astype(np.int16)

        residual_entries, residual_scale, _, _ = self._encode_sparse_entries(residual, self.dictionary_residual_topk)
        header = self._PROTO_DICT_HEADER.pack(
            self._PROTO_DICT_MAGIC,
            dims,
            len(selected),
            len(residual_entries) // self._PROTO_DELTA_ENTRY.size if residual_entries else 0,
            float(coeff_scale),
            float(residual_scale),
        )
        coeff_payload = bytearray()
        for idx, q in zip(selected, quantized, strict=False):
            coeff_payload.extend(self._PROTO_DICT_ENTRY.pack(int(idx), int(q)))

        payload = bytes(header) + bytes(coeff_payload) + residual_entries
        reconstructed = self.decompile_dictionary_sparse(payload)
        similarity = self._cosine_similarity(vector, reconstructed)

        metadata = {
            "mode": "dictionary_sparse",
            "codec": "dict",
            "coeff_count": int(len(selected)),
            "residual_count": int(len(residual_entries) // self._PROTO_DELTA_ENTRY.size if residual_entries else 0),
            "similarity": float(similarity),
            "compression_ratio": float(vector.nbytes) / max(1, len(payload)),
        }

        if similarity < self.dictionary_similarity_threshold:
            return self._fallback_dense_or_simple(vector, "dictionary_sparse", return_metadata)

        if return_metadata:
            return payload, metadata
        return payload

    def decompile_prototype_delta(self, program: bytes) -> np.ndarray:
        """Reconstruct embedding from prototype-delta payload."""
        if len(program) < self._PROTO_DELTA_HEADER.size:
            raise ValueError("Prototype delta payload truncated.")
        magic, dims, proto_id, nnz, scale = self._PROTO_DELTA_HEADER.unpack_from(program, 0)
        table = self._require_prototype_table()
        if proto_id >= table.count:
            raise ValueError(f"Prototype id {proto_id} out of range.")
        base = np.array(table.prototypes[proto_id], copy=True)
        cursor = self._PROTO_DELTA_HEADER.size

        if magic == self._PROTO_DELTA_MAGIC:
            for _ in range(nnz):
                if cursor + self._PROTO_DELTA_ENTRY.size > len(program):
                    raise ValueError("Prototype delta payload truncated while reading entries.")
                idx, q = self._PROTO_DELTA_ENTRY.unpack_from(program, cursor)
                cursor += self._PROTO_DELTA_ENTRY.size
                if idx >= dims or idx >= base.size:
                    continue
                base[idx] += float(np.int8(q)) * float(scale)
        elif magic == self._PROTO_BASIS_MAGIC:
            if not table.has_basis:
                raise RuntimeError("Prototype table missing basis for basis-coded payload.")
            basis = table.basis
            assert basis is not None
            for _ in range(nnz):
                if cursor + self._PROTO_DELTA_ENTRY.size > len(program):
                    raise ValueError("Prototype basis payload truncated while reading entries.")
                idx, q = self._PROTO_DELTA_ENTRY.unpack_from(program, cursor)
                cursor += self._PROTO_DELTA_ENTRY.size
                if idx >= basis.shape[0]:
                    continue
                base += float(np.int8(q)) * float(scale) * basis[idx]
        elif magic == self._PROTO_DENSE_MAGIC:
            dense_payload = program[self._PROTO_DENSE_HEADER.size :]
            quantized = np.frombuffer(dense_payload, dtype=np.int8, count=dims)
            if quantized.size != dims:
                raise ValueError("Dense prototype payload truncated.")
            base += quantized.astype(np.float32) * float(scale)
        else:
            raise ValueError("Unknown prototype delta magic header.")
        return base.astype(np.float32)

    def decompile_prototype_delta_dense(self, program: bytes) -> np.ndarray:
        """Decode PD02 dense payload."""
        if len(program) < self._PROTO_DENSE_HEADER.size:
            raise ValueError("Prototype delta dense payload truncated.")
        magic, dims, proto_id, scale = self._PROTO_DENSE_HEADER.unpack_from(program, 0)
        if magic != self._PROTO_DENSE_MAGIC:
            raise ValueError("Invalid dense prototype magic header.")
        table = self._require_prototype_table()
        if proto_id >= table.count:
            raise ValueError(f"Prototype id {proto_id} out of range.")
        proto_vec = np.array(table.prototypes[proto_id], copy=True)
        quantized = np.frombuffer(program[self._PROTO_DENSE_HEADER.size :], dtype=np.int8)
        if quantized.size != dims:
            raise ValueError("Dense payload length mismatch.")
        residual = quantized.astype(np.float32) * float(scale)
        return proto_vec + residual

    def decompile_prototype_multi(self, program: bytes) -> np.ndarray:
        """Decode PD03 multi-prototype payload."""
        if len(program) < self._PROTO_MULTI_HEADER.size:
            raise ValueError("Prototype multi payload truncated.")
        magic, dims, num_protos, nnz, scale = self._PROTO_MULTI_HEADER.unpack_from(program, 0)
        if magic != self._PROTO_MULTI_MAGIC:
            raise ValueError("Invalid multi-prototype magic header.")
        table = self._require_prototype_table()
        offset = self._PROTO_MULTI_HEADER.size
        base = np.zeros(dims, dtype=np.float32)

        for _ in range(num_protos):
            if offset + self._PROTO_MULTI_PROTO.size > len(program):
                raise ValueError("Prototype multi payload truncated during prototype section.")
            proto_id, weight = self._PROTO_MULTI_PROTO.unpack_from(program, offset)
            offset += self._PROTO_MULTI_PROTO.size
            if proto_id >= table.count:
                continue
            base += float(weight) * table.prototypes[proto_id]

        for _ in range(nnz):
            if offset + self._PROTO_DELTA_ENTRY.size > len(program):
                raise ValueError("Prototype multi payload truncated during residual section.")
            idx, q = self._PROTO_DELTA_ENTRY.unpack_from(program, offset)
            offset += self._PROTO_DELTA_ENTRY.size
            if idx >= dims:
                continue
            base[idx] += float(np.int8(q)) * float(scale)

        return base

    def decompile_dictionary_sparse(self, program: bytes) -> np.ndarray:
        """Decode PD04 dictionary sparse payload."""
        if len(program) < self._PROTO_DICT_HEADER.size:
            raise ValueError("Dictionary sparse payload truncated.")
        magic, dims, coeff_count, nnz, coeff_scale, residual_scale = self._PROTO_DICT_HEADER.unpack_from(program, 0)
        if magic != self._PROTO_DICT_MAGIC:
            raise ValueError("Invalid dictionary sparse magic header.")
        if self.dictionary_atoms is None:
            raise RuntimeError("Dictionary atoms not attached.")
        atoms = self.dictionary_atoms
        offset = self._PROTO_DICT_HEADER.size
        base = np.zeros(dims, dtype=np.float32)
        for _ in range(coeff_count):
            if offset + self._PROTO_DICT_ENTRY.size > len(program):
                raise ValueError("Dictionary sparse payload truncated during coeff section.")
            atom_idx, q = self._PROTO_DICT_ENTRY.unpack_from(program, offset)
            offset += self._PROTO_DICT_ENTRY.size
            if atom_idx >= atoms.shape[0]:
                continue
            base += float(q) * float(coeff_scale) * atoms[atom_idx]

        for _ in range(nnz):
            if offset + self._PROTO_DELTA_ENTRY.size > len(program):
                raise ValueError("Dictionary sparse payload truncated during residual section.")
            idx, q = self._PROTO_DELTA_ENTRY.unpack_from(program, offset)
            offset += self._PROTO_DELTA_ENTRY.size
            if idx >= dims:
                continue
            base[idx] += float(np.int8(q)) * float(residual_scale)

        return base

    # ------------------------------------------------------------------ #
    # Minimal compression path (Phase 1 Task 1.1)
    # ------------------------------------------------------------------ #
    def compile_embedding_simple(self, embedding: np.ndarray) -> bytes:
        """
        Compress an embedding using quantisation + delta encoding.

        The format is:
            <uint32 dims> <float32 scale> <int8 first_value> <delta stream>

        Delta stream stores int8 differences with a sentinel (0x80) that
        indicates the following int16 should be used whenever the delta
        exceeds the int8 range. This keeps the implementation simple while
        guaranteeing deterministic reconstruction.
        """
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        dims = vec.size
        if dims == 0:
            return self._SIMPLE_HEADER.pack(0, 1.0)

        max_abs = float(np.max(np.abs(vec)))
        scale = max(max_abs / 127.0, 1e-6)
        quantized = np.clip(np.round(vec / scale), -127, 127).astype(np.int16)

        payload = bytearray(self._SIMPLE_HEADER.pack(dims, scale))
        payload.extend(struct.pack("<b", int(quantized[0])))
        payload.extend(self._encode_delta_stream(quantized))
        return bytes(payload)

    def decompile_simple(self, program: bytes) -> np.ndarray:
        """
        Reverse of `compile_embedding_simple`.

        The method reconstructs the quantised int8 stream, performs the
        cumulative delta decode, and expands back to float32 using the stored
        scale factor.
        """
        if len(program) < self._SIMPLE_HEADER.size:
            raise ValueError("Procedural program too small for header.")

        dims, scale = self._SIMPLE_HEADER.unpack_from(program, 0)
        dims = int(dims)
        if dims == 0:
            return np.zeros(0, dtype=np.float32)

        cursor = self._SIMPLE_HEADER.size
        if cursor >= len(program):
            raise ValueError("Procedural program missing quantised payload.")

        first_val = struct.unpack_from("<b", program, cursor)[0]
        cursor += 1

        quantized = np.empty(dims, dtype=np.int16)
        quantized[0] = first_val

        idx = 1
        while idx < dims:
            if cursor >= len(program):
                raise ValueError("Delta stream truncated while decoding procedural program.")
            token = program[cursor]
            cursor += 1

            if token == self._DELTA_SENTINEL:
                if cursor + 2 > len(program):
                    raise ValueError("Sentinel delta missing payload in procedural program.")
                delta = struct.unpack_from("<h", program, cursor)[0]
                cursor += 2
            else:
                delta = token if token < 128 else token - 256

            quantized[idx] = quantized[idx - 1] + int(delta)
            idx += 1

        reconstructed = quantized.astype(np.float32) * float(scale)
        return reconstructed

    def _encode_delta_stream(self, quantized: np.ndarray) -> bytes:
        """Encode quantised values into delta stream bytes."""
        if quantized.size <= 1:
            return b""

        stream = bytearray()

        prev = int(quantized[0])
        for value in quantized[1:]:
            delta = int(value) - prev
            if -127 <= delta <= 127:
                stream.append(delta & 0xFF)
            else:
                stream.append(self._DELTA_SENTINEL)
                stream.extend(struct.pack("<h", int(delta)))
            prev = int(value)
        return bytes(stream)

    def _encode_sparse_entries(self, residual: np.ndarray, topk: int) -> Tuple[bytes, float, np.ndarray, np.ndarray]:
        """Encode sparse residual entries (indices, quantised values)."""
        if topk <= 0:
            return b"", 1.0, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int8)

        abs_res = np.abs(residual)
        if not np.any(abs_res):
            return b"", 1.0, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int8)

        count = min(topk, residual.size)
        topk_idx = np.argpartition(abs_res, -count)[-count:]
        topk_idx = topk_idx[np.argsort(abs_res[topk_idx])[::-1]]
        values = residual[topk_idx]
        nonzero_mask = np.abs(values) > 0.0
        topk_idx = topk_idx[nonzero_mask]
        values = values[nonzero_mask]
        if topk_idx.size == 0:
            return b"", 1.0, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int8)

        max_abs = float(np.max(np.abs(values)))
        scale = max(max_abs / 127.0, 1e-6)
        quantized = np.clip(np.round(values / scale), -127, 127).astype(np.int8)

        entries = bytearray()
        for idx, q in zip(topk_idx, quantized, strict=False):
            entries.extend(self._PROTO_DELTA_ENTRY.pack(int(idx), int(np.int8(q))))
        return bytes(entries), float(scale), topk_idx.astype(np.int32), quantized

    def _fallback_dense_or_simple(
        self,
        embedding: np.ndarray,
        mode_label: str,
        return_metadata: bool,
    ) -> Union[bytes, Tuple[bytes, Dict[str, float]]]:
        """Fallback to PD02 if dimensions align, else simple codec."""
        vector = np.asarray(embedding, dtype=np.float32)
        can_use_dense = (
            self.prototype_table is not None
            and self.prototype_table.dimension == vector.size
        )
        if can_use_dense:
            dense_program, dense_meta = self.compile_prototype_delta_dense(vector, return_metadata=True)
            dense_meta["mode"] = mode_label
            if return_metadata:
                return dense_program, dense_meta
            return dense_program
        program = self.compile_embedding_simple(vector)
        metadata = {"mode": mode_label, "codec": "simple_fallback"}
        if return_metadata:
            return program, metadata
        return program

    def _matching_pursuit(self, vector: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Greedy matching pursuit using attached dictionary atoms."""
        if self.dictionary_atoms is None:
            raise RuntimeError("Dictionary atoms not attached.")
        atoms = self.dictionary_atoms
        vector = np.asarray(vector, dtype=np.float32)
        if self.dictionary_atoms_dim is None or vector.size != self.dictionary_atoms_dim:
            raise ValueError(
                f"Vector dimensionality {vector.size} does not match dictionary ({self.dictionary_atoms_dim})."
            )
        residual = np.array(vector, copy=True)
        selected: List[int] = []
        coeffs: List[float] = []
        for _ in range(self.dictionary_max_coeffs):
            correlations = atoms @ residual
            idx = int(np.argmax(np.abs(correlations)))
            if idx in selected:
                break
            coeff = float(correlations[idx])
            if abs(coeff) < 1e-6:
                break
            selected.append(idx)
            coeffs.append(coeff)
            residual = residual - coeff * atoms[idx]
        return selected, np.array(coeffs, dtype=np.float32), residual
