from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import numpy as np  # type: ignore


class GalaxyMemoryUpdater:
    """Blend galaxy embeddings using PTX when available."""

    def __init__(self, galaxy_dir: Path | None = None, ptx_path: str = "knowledge3d/cranium/ptx/galaxy_memory_updater.ptx"):
        self.galaxy_dir = Path(galaxy_dir) if galaxy_dir else None
        self.ptx_path = ptx_path
        self._cuda = None
        self._kernel = None
        self._module = None
        self._ctx = None
        self._load_kernel()

    def _load_kernel(self) -> None:
        try:
            from cuda import cuda  # type: ignore
        except Exception:
            return
        if not os.path.exists(self.ptx_path):
            return
        try:
            ptx_code = Path(self.ptx_path).read_bytes()
        except Exception:
            return
        err, = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            return
        err, dev = cuda.cuDeviceGet(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            return
        err, ctx = cuda.cuCtxCreate(0, dev)
        if err != cuda.CUresult.CUDA_SUCCESS:
            return
        self._ctx = ctx
        err, mod = cuda.cuModuleLoadData(ptx_code)
        if err != cuda.CUresult.CUDA_SUCCESS:
            cuda.cuCtxDestroy(ctx)
            return
        err, func = cuda.cuModuleGetFunction(mod, b"update_star_embedding_kernel")
        if err != cuda.CUresult.CUDA_SUCCESS:
            cuda.cuModuleUnload(mod)
            cuda.cuCtxDestroy(ctx)
            return
        self._cuda = cuda
        self._module = mod
        self._kernel = func

    def blend(self, old: np.ndarray, teacher: np.ndarray, blend_factor: float) -> np.ndarray:
        if self._kernel is None or self._cuda is None:
            return self._blend_numpy(old, teacher, blend_factor)
        return self._blend_cuda(old, teacher, blend_factor)

    def blend_sequence(self, base: np.ndarray, teachers: List[np.ndarray], blend_factor: float = 0.3) -> np.ndarray:
        out = np.array(base, dtype=np.float32)
        if not teachers:
            return out
        for teacher in teachers:
            out = self.blend(out, np.array(teacher, dtype=np.float32), blend_factor)
        return out

    def _blend_cuda(self, old: np.ndarray, teacher: np.ndarray, blend_factor: float) -> np.ndarray:
        cuda = self._cuda
        assert cuda is not None
        dim = old.size
        if dim == 0:
            return np.array([], dtype=np.float32)
        old_vec = np.ascontiguousarray(old.astype(np.float32))
        teacher_vec = np.ascontiguousarray(teacher.astype(np.float32))
        out_vec = np.zeros_like(old_vec)
        err, d_old = cuda.cuMemAlloc(old_vec.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            return self._blend_numpy(old, teacher, blend_factor)
        err, d_teacher = cuda.cuMemAlloc(teacher_vec.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            cuda.cuMemFree(d_old)
            return self._blend_numpy(old, teacher, blend_factor)
        err, d_out = cuda.cuMemAlloc(out_vec.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            cuda.cuMemFree(d_old)
            cuda.cuMemFree(d_teacher)
            return self._blend_numpy(old, teacher, blend_factor)
        try:
            err, = cuda.cuMemcpyHtoD(d_old, old_vec.ctypes.data, old_vec.nbytes)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemcpyHtoD old failed: {err}")
            err, = cuda.cuMemcpyHtoD(d_teacher, teacher_vec.ctypes.data, teacher_vec.nbytes)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemcpyHtoD teacher failed: {err}")
            import ctypes as _ct
            old_ptr = _ct.c_void_p(int(d_old))
            teacher_ptr = _ct.c_void_p(int(d_teacher))
            out_ptr = _ct.c_void_p(int(d_out))
            blend_c = _ct.c_float(blend_factor)
            dim_c = _ct.c_uint(dim)
            param_ptrs = (_ct.c_void_p * 5)(
                _ct.cast(_ct.pointer(old_ptr), _ct.c_void_p),
                _ct.cast(_ct.pointer(teacher_ptr), _ct.c_void_p),
                _ct.cast(_ct.pointer(out_ptr), _ct.c_void_p),
                _ct.cast(_ct.pointer(blend_c), _ct.c_void_p),
                _ct.cast(_ct.pointer(dim_c), _ct.c_void_p),
            )
            threads = 256
            blocks = (dim + threads - 1) // threads
            err, = cuda.cuLaunchKernel(
                self._kernel,
                int(blocks), 1, 1,
                threads, 1, 1,
                0, 0,
                param_ptrs, 0,
            )
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuLaunchKernel failed: {err}")
            err, = cuda.cuMemcpyDtoH(out_vec.ctypes.data, d_out, out_vec.nbytes)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemcpyDtoH failed: {err}")
            return out_vec
        except Exception:
            return self._blend_numpy(old, teacher, blend_factor)
        finally:
            cuda.cuMemFree(d_old)
            cuda.cuMemFree(d_teacher)
            cuda.cuMemFree(d_out)

    @staticmethod
    def _blend_numpy(old: np.ndarray, teacher: np.ndarray, blend_factor: float) -> np.ndarray:
        return (old * (1.0 - blend_factor)) + (teacher * blend_factor)
