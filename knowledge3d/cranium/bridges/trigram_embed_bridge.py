"""
GPU trigram embedding bridge for RPN engine.

Loads the checked-in trigram PTX module, manages the trigram embedding table on
device memory, and exposes helpers to embed sequences of trigram indices using
sovereign GPU kernels.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from knowledge3d.cranium.sovereign import loader


class TrigramEmbedBridge:
    """Sovereign GPU trigram embedding operator."""

    def __init__(self, arch: str = "sm_86"):
        kernel_dir = Path(__file__).parent.parent / "ptx"
        self._ptx_path = kernel_dir / "trigram_embed.ptx"
        if not self._ptx_path.exists():
            raise FileNotFoundError(f"Trigram embedding PTX not found: {self._ptx_path}")

        self._arch = arch
        self._module: Optional[loader.CUmodule] = None
        self._lookup_kernel: Optional[loader.CUfunction] = None
        self._normalize_kernel: Optional[loader.CUfunction] = None

        self.embed_table_gpu: Optional[loader.CUdeviceptr] = None
        self.vocab_size: int = 0
        self.embed_dim: int = 0

        self._load_ptx_module()

    def _load_ptx_module(self) -> None:
        try:
            self._module = loader.load_module_from_file(str(self._ptx_path))
            self._lookup_kernel = loader.get_function(self._module, "trigram_lookup_average")
            self._normalize_kernel = loader.get_function(self._module, "l2_normalize_embedding")
        except Exception as exc:
            self._module = None
            self._lookup_kernel = None
            self._normalize_kernel = None
            raise RuntimeError(
                f"Failed to load trigram embedding PTX ({self._ptx_path}). "
                "The live path is PTX-only; rebuild the checked-in PTX if needed."
            ) from exc

    def upload_embedding_table(self, embeddings: np.ndarray) -> None:
        """Upload (or refresh) the embedding table on GPU."""
        if embeddings.ndim != 2:
            raise ValueError(f"Embedding table must be 2D, got shape {embeddings.shape}")

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        vocab_size, embed_dim = embeddings.shape

        if self.embed_table_gpu is not None:
            loader.gpu_free(self.embed_table_gpu)
            self.embed_table_gpu = None

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)

        if vocab_size == 0:
            return

        self.embed_table_gpu = loader.gpu_malloc(embeddings.nbytes)
        loader.memcpy_htod(
            self.embed_table_gpu,
            embeddings.ctypes.data_as(ctypes.c_void_p),
            embeddings.nbytes,
        )

    def embed_indices(
        self,
        indices: Sequence[int],
        normalize: bool = True,
        return_cpu: bool = True,
    ) -> np.ndarray | loader.CUdeviceptr:
        """Embed a sequence of trigram table indices on the GPU."""
        if not indices:
            if return_cpu:
                return np.zeros(self.embed_dim, dtype=np.float32)
            zero_gpu = loader.gpu_malloc(self.embed_dim * 4)
            zeros = np.zeros(self.embed_dim, dtype=np.float32)
            loader.memcpy_htod(zero_gpu, zeros.ctypes.data_as(ctypes.c_void_p), zeros.nbytes)
            return zero_gpu

        if self.embed_table_gpu is None:
            raise RuntimeError("Embedding table not uploaded to GPU bridge.")

        ids_np = np.asarray(indices, dtype=np.int32)
        ids_gpu = loader.gpu_malloc(ids_np.nbytes)
        loader.memcpy_htod(ids_gpu, ids_np.ctypes.data_as(ctypes.c_void_p), ids_np.nbytes)

        output_gpu = loader.gpu_malloc(self.embed_dim * 4)

        threads = 256
        blocks = (self.embed_dim + threads - 1) // threads

        loader.launch(
            self._lookup_kernel,
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            params=[
                ctypes.c_uint64(ids_gpu.value),
                ctypes.c_uint64(self.embed_table_gpu.value),
                ctypes.c_uint64(output_gpu.value),
                ctypes.c_int(len(indices)),
                ctypes.c_int(self.embed_dim),
                ctypes.c_int(self.vocab_size),
            ],
        )
        loader.synchronize()

        loader.gpu_free(ids_gpu)

        if normalize and self.embed_dim > 0:
            norm_block = 256
            shared_bytes = norm_block * 4
            loader.launch(
                self._normalize_kernel,
                grid=(1, 1, 1),
                block=(norm_block, 1, 1),
                params=[
                    ctypes.c_uint64(output_gpu.value),
                    ctypes.c_int(self.embed_dim),
                ],
                shared_mem=shared_bytes,
            )
            loader.synchronize()

        if return_cpu:
            output = np.zeros(self.embed_dim, dtype=np.float32)
            loader.memcpy_dtoh(
                output.ctypes.data_as(ctypes.c_void_p),
                output_gpu,
                output.nbytes,
            )
            loader.gpu_free(output_gpu)
            return output

        return output_gpu

    def __del__(self):
        try:
            if self.embed_table_gpu is not None:
                loader.gpu_free(self.embed_table_gpu)
                self.embed_table_gpu = None
        except Exception:
            pass
