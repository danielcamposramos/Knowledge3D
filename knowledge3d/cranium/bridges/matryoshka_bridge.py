"""Matryoshka projection bridge — GPU matvec for adaptive embeddings.

Sovereign resurrection (2026-04-18): the archived module
(``Old_Attempts/2026-04-18/knowledge3d/cranium/bridges/matryoshka_bridge.py``)
used ``numpy`` only for its ``project_host`` host-staging helper. The hot-path
``project_device`` entrypoint is already pure ctypes, and ``project_host``
now accepts a sequence of floats (or a bytes/ctypes buffer) and returns a
``ctypes.c_float`` array. See the Absolute Sovereignty Purge
(``TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md``).
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable, Sequence, Union

from knowledge3d.cranium.sovereign import loader


class MatryoshkaProjectionBridge:
    """Thin wrapper around the matryoshka_project kernel."""

    def __init__(self, arch: str = "sm_86"):
        kernel_dir = Path(__file__).parent.parent / "ptx"
        self._ptx_path = kernel_dir / "matryoshka_project.ptx"
        if not self._ptx_path.exists():
            raise FileNotFoundError(f"Matryoshka PTX not found: {self._ptx_path}")

        self._arch = arch  # retained for clarity; PTX is precompiled
        self._module = None
        self._kernel = None
        self._load_ptx()

    def _load_ptx(self) -> None:
        self._module = loader.load_module_from_file(str(self._ptx_path))
        self._kernel = loader.get_function(self._module, "matryoshka_project")

    @property
    def kernel(self):
        if self._kernel is None:
            raise RuntimeError("MatryoshkaProjectionBridge kernel not initialised")
        return self._kernel

    def project_device(
        self,
        weights_ptr: Union[int, "loader.CUdeviceptr"],
        vector_ptr: "loader.CUdeviceptr",
        output_ptr: "loader.CUdeviceptr",
        target_dim: int,
        stride: int,
    ):
        """Project on GPU writing result to ``output_ptr``."""
        weights_addr = (
            weights_ptr.value
            if isinstance(weights_ptr, loader.CUdeviceptr)
            else int(weights_ptr)
        )

        threads = 256
        blocks = (target_dim + threads - 1) // threads

        params = [
            ctypes.c_uint64(weights_addr),
            ctypes.c_uint64(vector_ptr.value),
            ctypes.c_uint64(output_ptr.value),
            ctypes.c_int(target_dim),
            ctypes.c_int(stride),
        ]

        loader.launch(
            self.kernel,
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            params=params,
        )
        loader.synchronize()

        return output_ptr

    def project_host(
        self,
        weights_ptr: Union[int, "loader.CUdeviceptr"],
        vector: Iterable[float] | Sequence[float],
        target_dim: int,
        stride: int,
    ):
        """Upload ``vector``, execute GPU projection, return host ctypes float buffer.

        ``vector`` may be any iterable of floats (list/tuple/ctypes array).
        Short vectors are zero-padded, longer vectors truncated. Return value
        is a ``(ctypes.c_float * target_dim)`` buffer.
        """
        src = list(float(v) for v in vector)
        if len(src) < target_dim:
            src = src + [0.0] * (target_dim - len(src))
        elif len(src) > target_dim:
            src = src[:target_dim]
        vec_buf = (ctypes.c_float * target_dim)(*src)

        d_vector = loader.gpu_malloc(target_dim * 4)
        d_output = loader.gpu_malloc(target_dim * 4)

        try:
            loader.memcpy_htod(
                d_vector,
                ctypes.cast(vec_buf, ctypes.c_void_p),
                ctypes.sizeof(vec_buf),
            )
            self.project_device(weights_ptr, d_vector, d_output, target_dim, stride)
            result = (ctypes.c_float * target_dim)()
            loader.memcpy_dtoh(
                ctypes.cast(result, ctypes.c_void_p),
                d_output,
                ctypes.sizeof(result),
            )
        finally:
            loader.gpu_free(d_vector)
            loader.gpu_free(d_output)

        return result


__all__ = ["MatryoshkaProjectionBridge"]
