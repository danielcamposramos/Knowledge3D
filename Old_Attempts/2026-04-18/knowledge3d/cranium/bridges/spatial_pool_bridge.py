"""
Spatial pooling bridge for sovereign GPU embeddings.

Provides a thin wrapper around the spatial_mean_pool kernel so higher-level
Python code can request spatial mean pooling without touching NumPy.
"""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import numpy as np

from knowledge3d.cranium.sovereign import loader


class SpatialMeanPooler:
    """
    Sovereign spatial mean pooling helper.

    Compiles `spatial_pool.cu` to PTX on-demand and exposes a convenience
    routine that executes the kernel entirely on the GPU, returning either a
    device pointer or host array as required.
    """

    def __init__(self, arch: str = "sm_86"):
        kernel_dir = Path(__file__).parent.parent / "ptx"
        self._cu_path = kernel_dir / "spatial_pool.cu"
        if not self._cu_path.exists():
            raise FileNotFoundError(f"Spatial pooling kernel not found: {self._cu_path}")

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
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Failed to compile spatial pooling kernel ({self._cu_path}): {exc.stderr}"
                ) from exc

            self._module = loader.load_module_from_file(str(ptx_path))
            self._kernel = loader.get_function(self._module, "spatial_mean_pool")
        finally:
            ptx_path.unlink(missing_ok=True)

    @property
    def kernel(self):
        if self._kernel is None:
            raise RuntimeError("SpatialMeanPooler kernel not initialised")
        return self._kernel

    def mean_pool_host(
        self,
        features_ptr: Union[int, loader.CUdeviceptr],
        H: int,
        W: int,
        C: int,
    ) -> np.ndarray:
        """
        Execute spatial mean pooling and return the result as a host NumPy array.
        """
        output = np.zeros(C, dtype=np.float32)
        d_output = loader.gpu_malloc(output.nbytes)

        try:
            self.mean_pool_device(features_ptr, d_output, H, W, C)
            loader.memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_output, output.nbytes)
        finally:
            loader.gpu_free(d_output)

        return output

    def mean_pool_device(
        self,
        features_ptr: Union[int, loader.CUdeviceptr],
        output_ptr: loader.CUdeviceptr,
        H: int,
        W: int,
        C: int,
    ) -> loader.CUdeviceptr:
        """
        Execute spatial mean pooling writing the result to `output_ptr`.
        """
        if isinstance(features_ptr, loader.CUdeviceptr):
            features_address = features_ptr.value
        else:
            features_address = int(features_ptr)

        grid_x = (C + 255) // 256
        params = [
            ctypes.c_uint64(features_address),
            ctypes.c_uint64(output_ptr.value),
            ctypes.c_int(H),
            ctypes.c_int(W),
            ctypes.c_int(C),
        ]

        loader.launch(
            self.kernel,
            grid=(grid_x, 1, 1),
            block=(256, 1, 1),
            params=params,
        )
        loader.synchronize()

        return output_ptr
