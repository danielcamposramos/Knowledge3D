"""
PTX binding for MDCT forward/inverse using cuda-python (driver + nvrtc).

Computes DCT-IV (self-inverse) on GPU. Follows K3D's sovereign PTX runtime pattern
from knowledge3d/cranium/ptx_runtime/nvrtc_ptx_loader.py.
"""

from __future__ import annotations

import ctypes as _ct
import logging
import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_cuda() -> Tuple:
    try:
        from cuda import cuda, nvrtc  # type: ignore
        return cuda, nvrtc
    except Exception:
        try:
            from cuda.bindings import driver as cuda  # type: ignore
            from cuda.bindings import nvrtc  # type: ignore
            return cuda, nvrtc
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"cuda-python bindings unavailable: {exc}")


MDCT_KERNEL_SRC = r"""
extern "C" __global__ void ternary_mdct_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float sum = 0.0f;
    float norm = sqrtf(2.0f / (float)n);

    for (int k = 0; k < n; ++k) {
        float angle = 3.14159265358979f / (float)n * ((float)tid + 0.5f) * ((float)k + 0.5f);
        sum = fmaf(input[k], __cosf(angle), sum);
    }

    output[tid] = norm * sum;
}

extern "C" __global__ void ternary_mdct_inverse(
    const float* __restrict__ coeffs,
    float* __restrict__ output,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float sum = 0.0f;
    float norm = sqrtf(2.0f / (float)n);

    for (int k = 0; k < n; ++k) {
        float angle = 3.14159265358979f / (float)n * ((float)k + 0.5f) * ((float)tid + 0.5f);
        sum = fmaf(coeffs[k], __cosf(angle), sum);
    }

    output[tid] = norm * sum;
}
"""


class TernaryMDCTKernel:
    """GPU MDCT/iMDCT via cuda-python following K3D's sovereign PTX pattern."""

    def __init__(self, n: int = 1024, device_index: int = 0) -> None:
        self.n = n
        self.device_index = device_index
        self.cuda, self.nvrtc = _load_cuda()
        self._ctx: Optional[int] = None
        self._module: Optional[int] = None
        self._kernel_fwd: Optional[int] = None
        self._kernel_inv: Optional[int] = None
        self._d_in: Optional[int] = None
        self._d_out: Optional[int] = None
        self._init_cuda()

    def _init_cuda(self) -> None:
        cuda = self.cuda

        err, = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuInit failed: {err}")

        err, dev = cuda.cuDeviceGet(self.device_index)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGet failed: {err}")

        err, ctx = cuda.cuDevicePrimaryCtxRetain(dev)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {err}")

        err, = cuda.cuCtxSetCurrent(ctx)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxSetCurrent failed: {err}")

        self._ctx = ctx
        self._compile_and_load(dev)

        err, d_in = cuda.cuMemAlloc(self.n * 4)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemAlloc input failed: {err}")

        err, d_out = cuda.cuMemAlloc(self.n * 4)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemAlloc output failed: {err}")

        self._d_in = d_in
        self._d_out = d_out

    def _compile_and_load(self, dev) -> None:
        cuda = self.cuda
        nvrtc = self.nvrtc

        res, prog = nvrtc.nvrtcCreateProgram(MDCT_KERNEL_SRC.encode("utf-8"), b"mdct.cu", 0, [], [])
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcCreateProgram failed: {res}")

        maj_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        min_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        err, maj = cuda.cuDeviceGetAttribute(maj_attr, dev)
        err2, minu = cuda.cuDeviceGetAttribute(min_attr, dev)
        if err != cuda.CUresult.CUDA_SUCCESS or err2 != cuda.CUresult.CUDA_SUCCESS:
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError("Failed to query compute capability")

        arch = f"--gpu-architecture=compute_{maj}{minu}".encode("utf-8")
        opts = [arch, b"--fmad=false"]
        res, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            log_size_res, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
            log_text = ""
            if log_size_res == nvrtc.nvrtcResult.NVRTC_SUCCESS and log_size > 1:
                buf = bytearray(log_size)
                nvrtc.nvrtcGetProgramLog(prog, buf)
                log_text = buf.decode("utf-8", errors="replace")
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(f"nvrtcCompileProgram failed ({res}):\n{log_text}")

        res, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(f"nvrtcGetPTXSize failed: {res}")

        buf = bytearray(ptx_size)
        res, = nvrtc.nvrtcGetPTX(prog, buf)
        nvrtc.nvrtcDestroyProgram(prog)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcGetPTX failed: {res}")

        # K3D pattern: Load PTX as bytes
        err, module = cuda.cuModuleLoadData(bytes(buf))
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleLoadData failed: {err}")

        self._module = module

        err, fwd = cuda.cuModuleGetFunction(module, b"ternary_mdct_forward")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction forward failed: {err}")

        err, inv = cuda.cuModuleGetFunction(module, b"ternary_mdct_inverse")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction inverse failed: {err}")

        self._kernel_fwd = fwd
        self._kernel_inv = inv

    def _launch(self, func, d_in, d_out, n, grid, block) -> None:
        """Launch helper using K3D's proven argument marshalling pattern."""
        cuda = self.cuda

        # K3D pattern from nvrtc_ptx_loader.py lines 197-207
        in_arg = _ct.c_void_p(int(d_in))
        out_arg = _ct.c_void_p(int(d_out))
        n_arg = _ct.c_int(int(n))

        param_array = (_ct.c_void_p * 3)(
            _ct.cast(_ct.pointer(in_arg), _ct.c_void_p),
            _ct.cast(_ct.pointer(out_arg), _ct.c_void_p),
            _ct.cast(_ct.pointer(n_arg), _ct.c_void_p),
        )
        err, = cuda.cuLaunchKernel(
            func,
            int(grid[0]), int(grid[1]), int(grid[2]),
            int(block[0]), int(block[1]), int(block[2]),
            0,  # sharedMemBytes
            0,  # hStream
            param_array,
            0,  # extra (NULL)
        )
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel failed: {err}")

    def forward(self, frame: np.ndarray) -> np.ndarray:
        """Run MDCT on GPU."""
        if self._kernel_fwd is None or self._d_in is None or self._d_out is None:
            raise RuntimeError("Kernel not initialised")

        x = np.ascontiguousarray(frame.astype(np.float32))
        if x.size != self.n:
            raise ValueError(f"frame length must be {self.n}")

        cuda = self.cuda

        err, = cuda.cuMemcpyHtoD(self._d_in, x.ctypes.data, x.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyHtoD failed: {err}")

        block = (256, 1, 1)
        grid = (int(math.ceil(self.n / 256)), 1, 1)

        self._launch(self._kernel_fwd, self._d_in, self._d_out, self.n, grid, block)

        # Synchronize
        err, = cuda.cuCtxSynchronize()
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxSynchronize failed: {err}")

        out = np.empty(self.n, dtype=np.float32)
        err, = cuda.cuMemcpyDtoH(out.ctypes.data, self._d_out, out.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyDtoH failed: {err}")

        return out

    def inverse(self, coeffs: np.ndarray) -> np.ndarray:
        """Run inverse MDCT on GPU."""
        if self._kernel_inv is None or self._d_in is None or self._d_out is None:
            raise RuntimeError("Kernel not initialised")

        c = np.ascontiguousarray(coeffs.astype(np.float32))
        if c.size != self.n:
            raise ValueError(f"coeff length must be {self.n}")

        cuda = self.cuda

        err, = cuda.cuMemcpyHtoD(self._d_in, c.ctypes.data, c.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyHtoD failed: {err}")

        block = (256, 1, 1)
        grid = (int(math.ceil(self.n / 256)), 1, 1)

        self._launch(self._kernel_inv, self._d_in, self._d_out, self.n, grid, block)

        # Synchronize
        err, = cuda.cuCtxSynchronize()
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxSynchronize failed: {err}")

        out = np.empty(self.n, dtype=np.float32)
        err, = cuda.cuMemcpyDtoH(out.ctypes.data, self._d_out, out.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyDtoH failed: {err}")

        return out

    def close(self) -> None:
        """Free buffers."""
        if self._d_in is not None:
            self.cuda.cuMemFree(self._d_in)
            self._d_in = None
        if self._d_out is not None:
            self.cuda.cuMemFree(self._d_out)
            self._d_out = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
        try:
            if self._module is not None and self.cuda is not None:
                self.cuda.cuModuleUnload(self._module)
        except Exception:
            pass


__all__ = ["TernaryMDCTKernel"]
