"""
PTX binding for 8x8 DCT-II with optional ternary quantisation.

Uses cuda-python (driver + nvrtc) only. Kernel processes one 8x8 block per
thread block (64 threads). Intended for batched 8x8 blocks laid out as
(num_blocks, 8, 8) contiguous float32.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import ctypes

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_cuda():
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


DCT8_KERNEL = r"""
#define PI 3.14159265358979f
extern "C" __global__ void dct8x8_forward(const float* input, int num_blocks, float* output) {
    // blockIdx.x -> block index
    int b = blockIdx.x;
    if (b >= num_blocks) return;
    // threadIdx.x in [0,63] maps to (u,v) coefficient
    int tid = threadIdx.x;
    int u = tid / 8;
    int v = tid % 8;

    float norm = 0.5f * ((u == 0) ? rsqrtf(2.0f) : 1.0f) * ((v == 0) ? rsqrtf(2.0f) : 1.0f);
    float sum = 0.0f;
    const float* block = input + b * 64;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            float val = block[y * 8 + x];
            float cu = cosf((PI / 8.0f) * (x + 0.5f) * u);
            float cv = cosf((PI / 8.0f) * (y + 0.5f) * v);
            sum = fmaf(val, cu * cv, sum);
        }
    }
    output[b * 64 + tid] = norm * sum * 0.25f; // match orthonormal scaling
}

extern "C" __global__ void dct8x8_inverse(const float* coeffs, int num_blocks, float* output) {
    int b = blockIdx.x;
    if (b >= num_blocks) return;
    int tid = threadIdx.x;
    int y = tid / 8;
    int x = tid % 8;
    float sum = 0.0f;
    const float* block = coeffs + b * 64;
    for (int v = 0; v < 8; ++v) {
        for (int u = 0; u < 8; ++u) {
            float c = block[v * 8 + u];
            float alpha_u = (u == 0) ? rsqrtf(2.0f) : 1.0f;
            float alpha_v = (v == 0) ? rsqrtf(2.0f) : 1.0f;
            float cu = cosf((PI / 8.0f) * (x + 0.5f) * u);
            float cv = cosf((PI / 8.0f) * (y + 0.5f) * v);
            sum = fmaf(alpha_u * alpha_v * c, cu * cv, sum);
        }
    }
    output[b * 64 + tid] = 0.25f * sum;
}
"""


class TernaryDCT8x8Kernel:
    """GPU 8x8 DCT-II forward kernel."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.cuda, self.nvrtc = _load_cuda()
        self._ctx: Optional[int] = None
        self._module: Optional[int] = None
        self._kernel: Optional[int] = None
        self._kernel_inv: Optional[int] = None
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

    def _compile_and_load(self, dev) -> None:
        cuda = self.cuda
        nvrtc = self.nvrtc
        res, prog = nvrtc.nvrtcCreateProgram(DCT8_KERNEL.encode("utf-8"), b"dct8x8.cu", 0, [], [])
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcCreateProgram failed: {res}")
        major_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        minor_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        err, major = cuda.cuDeviceGetAttribute(major_attr, dev)
        err_minor, minor = cuda.cuDeviceGetAttribute(minor_attr, dev)
        if err != cuda.CUresult.CUDA_SUCCESS or err_minor != cuda.CUresult.CUDA_SUCCESS:
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError("Failed to query compute capability")
        arch = f"--gpu-architecture=compute_{major}{minor}".encode("utf-8")
        opts = [arch]
        res, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            log_size_res, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
            log_text = ""
            if log_size_res == nvrtc.nvrtcResult.NVRTC_SUCCESS and log_size > 1:
                log_buffer = bytearray(log_size)
                nvrtc.nvrtcGetProgramLog(prog, log_buffer)
                log_text = log_buffer.decode("utf-8", errors="replace")
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(f"nvrtcCompileProgram failed ({res}):\n{log_text}")
        res, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError(f"nvrtcGetPTXSize failed: {res}")
        ptx_buffer = bytearray(ptx_size)
        res, = nvrtc.nvrtcGetPTX(prog, ptx_buffer)
        nvrtc.nvrtcDestroyProgram(prog)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcGetPTX failed: {res}")
        err, module = cuda.cuModuleLoadData(bytes(ptx_buffer))
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleLoadData failed: {err}")
        self._module = module
        err, func = cuda.cuModuleGetFunction(module, b"dct8x8_forward")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction failed: {err}")
        self._kernel = func
        err, ifunc = cuda.cuModuleGetFunction(module, b"dct8x8_inverse")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction inverse failed: {err}")
        self._kernel_inv = ifunc

    def _launch(self, func, args_list, grid, block) -> None:
        cuda = self.cuda
        arg_objs = []
        arg_ptrs = (ctypes.c_void_p * len(args_list))()
        for i, a in enumerate(args_list):
            if isinstance(a, float):
                obj = ctypes.c_float(a)
            elif isinstance(a, int):
                obj = ctypes.c_int(a)
            else:
                obj = ctypes.c_void_p(int(a))
            arg_objs.append(obj)
            arg_ptrs[i] = ctypes.cast(ctypes.pointer(obj), ctypes.c_void_p)
        err, = cuda.cuLaunchKernel(
            func,
            int(grid[0]),
            int(grid[1]),
            int(grid[2]),
            int(block[0]),
            int(block[1]),
            int(block[2]),
            0,
            0,
            arg_ptrs,
            None,
        )
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel failed: {err}")

    def forward(self, blocks: np.ndarray) -> np.ndarray:
        """
        Run 8x8 DCT on a batch of blocks.

        Args:
            blocks: float32 array (num_blocks, 8, 8)
        Returns:
            float32 array of same shape flattened per block.
        """
        if self._kernel is None:
            raise RuntimeError("Kernel not initialised")
        b = np.ascontiguousarray(blocks.astype(np.float32))
        if b.ndim != 3 or b.shape[1:] != (8, 8):
            raise ValueError("blocks must have shape (num_blocks, 8, 8)")
        num_blocks = b.shape[0]
        cuda = self.cuda
        # Allocate
        err, d_input = cuda.cuMemAlloc(b.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemAlloc input failed: {err}")
        err, d_output = cuda.cuMemAlloc(b.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            cuda.cuMemFree(d_input)
            raise RuntimeError(f"cuMemAlloc output failed: {err}")
        try:
            err, = cuda.cuMemcpyHtoD(d_input, b.ctypes.data, b.nbytes)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemcpyHtoD failed: {err}")
            block = (64, 1, 1)
            grid = (num_blocks, 1, 1)
            args = [d_input, int(num_blocks), d_output]
            self._launch(self._kernel, args, grid, block)
            out = np.empty_like(b)
            err, = cuda.cuMemcpyDtoH(out.ctypes.data, d_output, out.nbytes)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemcpyDtoH failed: {err}")
            return out
        finally:
            cuda.cuMemFree(d_input)
            cuda.cuMemFree(d_output)

    def inverse(self, coeffs: np.ndarray) -> np.ndarray:
        """Run 8x8 inverse DCT on a batch."""
        if self._kernel_inv is None:
            raise RuntimeError("Kernel not initialised")
        c = np.ascontiguousarray(coeffs.astype(np.float32))
        if c.ndim != 3 or c.shape[1:] != (8, 8):
            raise ValueError("coeffs must have shape (num_blocks, 8, 8)")
        num_blocks = c.shape[0]
        cuda = self.cuda
        err, d_input = cuda.cuMemAlloc(c.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemAlloc input failed: {err}")
        err, d_output = cuda.cuMemAlloc(c.nbytes)
        if err != cuda.CUresult.CUDA_SUCCESS:
            cuda.cuMemFree(d_input)
            raise RuntimeError(f"cuMemAlloc output failed: {err}")
        try:
            err, = cuda.cuMemcpyHtoD(d_input, c.ctypes.data, c.nbytes)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemcpyHtoD failed: {err}")
            block = (64, 1, 1)
            grid = (num_blocks, 1, 1)
            args = [d_input, int(num_blocks), d_output]
            self._launch(self._kernel_inv, args, grid, block)
            out = np.empty_like(c)
            err, = cuda.cuMemcpyDtoH(out.ctypes.data, d_output, out.nbytes)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemcpyDtoH failed: {err}")
            return out
        finally:
            cuda.cuMemFree(d_input)
            cuda.cuMemFree(d_output)


__all__ = ["TernaryDCT8x8Kernel"]
