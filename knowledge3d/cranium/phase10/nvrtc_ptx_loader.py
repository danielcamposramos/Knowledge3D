from __future__ import annotations

import os
from typing import Optional

import numpy as np  # type: ignore


class NVRTCPTXLoader:
    """True CUDA driver loader for PTX using cuda-python (if available).

    If cuda-python or a CUDA device is not available, instantiation will raise
    or generate vertices will fallback (handled by caller).
    """

    def __init__(
        self,
        ptx_path: str = "knowledge3d/cranium/ptx/generate_shape_kernel.ptx",
        kernel_name: str = "generate_shape_kernel",
    ):
        self.ptx_path = ptx_path
        self.kernel_name = kernel_name.encode("utf-8")
        self._cuda = None
        self._nvrtc = None
        self._module = None
        self._kernel = None
        self._ctx = None
        self._load()

    def _load(self) -> None:
        try:
            from cuda import cuda  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"cuda-python not available: {e}")
        self._cuda = cuda

        if not os.path.exists(self.ptx_path):
            raise FileNotFoundError(f"PTX not found: {self.ptx_path}")
        ptx_code = open(self.ptx_path, "r", encoding="utf-8").read().encode("utf-8")

        err, = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuInit failed: {err}")
        err, dev = cuda.cuDeviceGet(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGet failed: {err}")
        err, ctx = cuda.cuCtxCreate(0, dev)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxCreate failed: {err}")
        self._ctx = ctx
        err, mod = cuda.cuModuleLoadData(ptx_code)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleLoadData failed: {err}")
        err, func = cuda.cuModuleGetFunction(mod, self.kernel_name)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction failed: {err}")
        self._module = mod
        self._kernel = func

    def generate_vertices(self, embedding: np.ndarray, vertex_count: int, shape_type_idx: int) -> np.ndarray:
        if self._kernel is None:
            raise RuntimeError("PTX kernel unavailable")
        cuda = self._cuda
        # Ensure float32 contiguous
        emb = np.ascontiguousarray(embedding.astype(np.float32))
        vertices_sz = int(vertex_count) * 3 * 4
        err, d_emb = cuda.cuMemAlloc(emb.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemAlloc emb failed: {err}")
        err, d_out = cuda.cuMemAlloc(vertices_sz)
        if err != cuda.CUresult.CUDA_SUCCESS:
            cuda.cuMemFree(d_emb)
            raise RuntimeError(f"cuMemAlloc out failed: {err}")
        try:
            err, = cuda.cuMemcpyHtoD(d_emb, emb.ctypes.data, emb.nbytes)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemcpyHtoD emb failed: {err}")
            import ctypes as _ct
            params = [
                _ct.c_void_p(int(d_emb)),
                _ct.c_void_p(int(d_out)),
                _ct.c_uint(int(vertex_count)),
                _ct.c_uint(int(shape_type_idx)),
            ]
            err, = cuda.cuLaunchKernel(
                self._kernel,
                1, 1, 1,
                int(vertex_count), 1, 1,
                0, 0,
                params, 0,
            )
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuLaunchKernel failed: {err}")
            out = np.zeros((int(vertex_count), 3), dtype=np.float32)
            err, = cuda.cuMemcpyDtoH(out.ctypes.data, d_out, vertices_sz)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemcpyDtoH failed: {err}")
            return out
        finally:
            cuda.cuMemFree(d_emb)
            cuda.cuMemFree(d_out)
