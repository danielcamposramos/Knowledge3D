"""
Matryoshka projection bridge - GPU matvec for adaptive embeddings.

Provides a sovereign bridge for the matryoshka_project kernel so higher-level
components can project embeddings at arbitrary Matryoshka dimensions without
leaving the GPU.
"""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import numpy as np

from knowledge3d.cranium.sovereign import loader


class MatryoshkaProjectionBridge:
    """Thin wrapper around the matryoshka_project kernel."""

    def __init__(self, arch: str = "sm_86"):
        kernel_dir = Path(__file__).parent.parent / "ptx"
        self._cu_path = kernel_dir / "matryoshka_project.cu"
        if not self._cu_path.exists():
            raise FileNotFoundError(f"Matryoshka kernel not found: {self._cu_path}")

        self._arch = arch
        self._module = None
        self._kernel = None
        self._compile_and_load()

    def _compile_and_load(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as tmp:
            ptx_path = Path(tmp.name)

        try:
            cmd = [
                "nvcc",
                "-ptx",
                str(self._cu_path),
                "-o",
                str(ptx_path),
                "-arch",
                self._arch,
                "-O3",
                "-allow-unsupported-compiler",  # Sovereignty: no GCC version constraints
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Failed to compile Matryoshka kernel ({self._cu_path}): {exc.stderr}"
                ) from exc

            self._module = loader.load_module_from_file(str(ptx_path))
            self._kernel = loader.get_function(self._module, "matryoshka_project")
        finally:
            ptx_path.unlink(missing_ok=True)

    @property
    def kernel(self):
        if self._kernel is None:
            raise RuntimeError("MatryoshkaProjectionBridge kernel not initialised")
        return self._kernel

    def project_device(
        self,
        weights_ptr: Union[int, loader.CUdeviceptr],
        vector_ptr: loader.CUdeviceptr,
        output_ptr: loader.CUdeviceptr,
        target_dim: int,
        stride: int,
    ) -> loader.CUdeviceptr:
        """Project on GPU writing result to `output_ptr`."""
        weights_addr = weights_ptr.value if isinstance(weights_ptr, loader.CUdeviceptr) else int(weights_ptr)

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
        weights_ptr: Union[int, loader.CUdeviceptr],
        vector: np.ndarray,
        target_dim: int,
        stride: int,
    ) -> np.ndarray:
        """
        Convenience wrapper that uploads `vector`, executes GPU projection, and returns a host array.
        """
        vec = np.asarray(vector, dtype=np.float32)
        if vec.size < target_dim:
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[: vec.size] = vec
            vec = padded
        elif vec.size > target_dim:
            vec = vec[:target_dim].copy()

        d_vector = loader.gpu_malloc(target_dim * 4)
        d_output = loader.gpu_malloc(target_dim * 4)

        try:
            loader.memcpy_htod(d_vector, vec.ctypes.data_as(ctypes.c_void_p), vec.nbytes)
            self.project_device(weights_ptr, d_vector, d_output, target_dim, stride)
            result = np.empty(target_dim, dtype=np.float32)
            loader.memcpy_dtoh(result.ctypes.data_as(ctypes.c_void_p), d_output, result.nbytes)
        finally:
            loader.gpu_free(d_vector)
            loader.gpu_free(d_output)

        return result
