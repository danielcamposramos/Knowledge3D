"""
PTX binding for ternary quantize/dequantize (GPU-only, no CPU fallback).

This mirrors the sovereign cuda-python pattern used in MDCT/DCT bindings.
"""

from __future__ import annotations

import ctypes as _ct
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from knowledge3d.cranium.sovereign import loader


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


KERNEL_SRC = r"""
extern "C" __global__ void ternary_quantize(
    const float* __restrict__ input,
    int n,
    float threshold,
    signed char* __restrict__ output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = input[idx];
    signed char t = 0;
    if (v > threshold) t = 1;
    else if (v < -threshold) t = -1;
    output[idx] = t;
}

extern "C" __global__ void ternary_dequantize(
    const signed char* __restrict__ input,
    int n,
    float scale,
    float* __restrict__ output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    signed char q = input[idx];
    output[idx] = (float)q * scale;
}

extern "C" __global__ void max_abs_reduce(
    const float* __restrict__ input,
    int n,
    float* __restrict__ partial
) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    float x = 0.0f;
    if (i < n) {
        float v = input[i];
        x = v >= 0 ? v : -v;
    }
    sdata[tid] = x;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}
"""


class TernaryQuantizer:
    """GPU ternary quantize/dequantize with no CPU fallback."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.cuda, self.nvrtc = _load_cuda()
        self._ctx: Optional[int] = None
        self._module: Optional[int] = None
        self._quant: Optional[int] = None
        self._dequant: Optional[int] = None
        self._max_abs: Optional[int] = None
        self._init_cuda()

    def _init_cuda(self) -> None:
        # Use shared context from sovereign loader (fork-safe, proven in production)
        loader._ensure_init()

        cuda = self.cuda

        # Bind loader context to cuda-python
        err, ctx = cuda.cuCtxGetCurrent()
        if err != cuda.CUresult.CUDA_SUCCESS or ctx is None or int(ctx) == 0:
            raise RuntimeError("No CUDA context available after loader._ensure_init()")
        err, = cuda.cuCtxSetCurrent(ctx)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxSetCurrent failed: {err}")

        self._ctx = ctx

        # Get device from context
        err, dev = cuda.cuCtxGetDevice()
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxGetDevice failed: {err}")

        self._compile_and_load(dev)

    def _compile_and_load(self, dev) -> None:
        cuda = self.cuda
        nvrtc = self.nvrtc
        res, prog = nvrtc.nvrtcCreateProgram(KERNEL_SRC.encode("utf-8"), b"ternary_quant.cu", 0, [], [])
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
        err, module = cuda.cuModuleLoadData(bytes(buf))
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleLoadData failed: {err}")
        self._module = module
        err, qfun = cuda.cuModuleGetFunction(module, b"ternary_quantize")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction quantize failed: {err}")
        err, dfun = cuda.cuModuleGetFunction(module, b"ternary_dequantize")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction dequantize failed: {err}")
        err, maxfun = cuda.cuModuleGetFunction(module, b"max_abs_reduce")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction max_abs_reduce failed: {err}")
        self._quant = qfun
        self._dequant = dfun
        self._max_abs = maxfun

    def _launch_quant(self, d_in, d_out, n: int, threshold: float) -> None:
        cuda = self.cuda
        blk = (256, 1, 1)
        grd = (int((n + blk[0] - 1) // blk[0]), 1, 1)
        in_arg = _ct.c_void_p(int(getattr(d_in, "value", d_in)))
        n_arg = _ct.c_int(int(n))
        thr_arg = _ct.c_float(float(threshold))
        out_arg = _ct.c_void_p(int(getattr(d_out, "value", d_out)))
        params = (_ct.c_void_p * 4)(
            _ct.cast(_ct.pointer(in_arg), _ct.c_void_p),
            _ct.cast(_ct.pointer(n_arg), _ct.c_void_p),
            _ct.cast(_ct.pointer(thr_arg), _ct.c_void_p),
            _ct.cast(_ct.pointer(out_arg), _ct.c_void_p),
        )
        err, = cuda.cuLaunchKernel(
            self._quant,
            int(grd[0]), int(grd[1]), int(grd[2]),
            int(blk[0]), int(blk[1]), int(blk[2]),
            0, 0, params, 0,
        )
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel quantize failed: {err}")

    def _launch_dequant(self, d_in, d_out, n: int, scale: float) -> None:
        cuda = self.cuda
        blk = (256, 1, 1)
        grd = (int((n + blk[0] - 1) // blk[0]), 1, 1)
        in_arg = _ct.c_void_p(int(getattr(d_in, "value", d_in)))
        n_arg = _ct.c_int(int(n))
        scale_arg = _ct.c_float(float(scale))
        out_arg = _ct.c_void_p(int(getattr(d_out, "value", d_out)))
        params = (_ct.c_void_p * 4)(
            _ct.cast(_ct.pointer(in_arg), _ct.c_void_p),
            _ct.cast(_ct.pointer(n_arg), _ct.c_void_p),
            _ct.cast(_ct.pointer(scale_arg), _ct.c_void_p),
            _ct.cast(_ct.pointer(out_arg), _ct.c_void_p),
        )
        err, = cuda.cuLaunchKernel(
            self._dequant,
            int(grd[0]), int(grd[1]), int(grd[2]),
            int(blk[0]), int(blk[1]), int(blk[2]),
            0, 0, params, 0,
        )
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel dequantize failed: {err}")

    def _launch_max_abs(self, d_in, d_part, n: int, grid) -> None:
        cuda = self.cuda
        blk = (256, 1, 1)
        in_arg = _ct.c_void_p(int(getattr(d_in, "value", d_in)))
        n_arg = _ct.c_int(int(n))
        out_arg = _ct.c_void_p(int(getattr(d_part, "value", d_part)))
        params = (_ct.c_void_p * 3)(
            _ct.cast(_ct.pointer(in_arg), _ct.c_void_p),
            _ct.cast(_ct.pointer(n_arg), _ct.c_void_p),
            _ct.cast(_ct.pointer(out_arg), _ct.c_void_p),
        )
        err, = cuda.cuLaunchKernel(
            self._max_abs,
            int(grid[0]), int(grid[1]), int(grid[2]),
            int(blk[0]), int(blk[1]), int(blk[2]),
            blk[0] * 4,  # shared memory size
            0, params, 0,
        )
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel max_abs_reduce failed: {err}")

    def max_abs(self, coefficients: np.ndarray) -> float:
        """Compute max(|coefficients|) on GPU (no CPU fallback)."""
        x = np.ascontiguousarray(coefficients, dtype=np.float32)
        n = x.size
        blk = 256
        grid = (int((n + blk - 1) // blk), 1, 1)
        d_in = loader.gpu_malloc(x.nbytes)
        d_part = loader.gpu_malloc(grid[0] * 4)
        try:
            loader.memcpy_htod(
                dst_device=d_in,
                src_host=_ct.c_void_p(x.ctypes.data),
                size_bytes=x.nbytes,
            )
            self._launch_max_abs(d_in, d_part, n, grid)
            partial = np.empty(grid[0], dtype=np.float32)
            loader.memcpy_dtoh(
                dst_host=_ct.c_void_p(partial.ctypes.data),
                src_device=d_part,
                size_bytes=partial.nbytes,
            )
            return float(partial.max()) if partial.size > 0 else 0.0
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_part)

    def quantize(self, coefficients: np.ndarray, threshold: float) -> np.ndarray:
        """Quantize on GPU to int8 {-1,0,+1}."""
        x = np.ascontiguousarray(coefficients, dtype=np.float32)
        n = x.size
        d_in = loader.gpu_malloc(x.nbytes)
        d_out = loader.gpu_malloc(n)  # int8
        try:
            loader.memcpy_htod(
                dst_device=d_in,
                src_host=_ct.c_void_p(x.ctypes.data),
                size_bytes=x.nbytes,
            )
            self._launch_quant(d_in, d_out, n, threshold)
            out = np.empty(n, dtype=np.int8)
            loader.memcpy_dtoh(
                dst_host=_ct.c_void_p(out.ctypes.data),
                src_device=d_out,
                size_bytes=out.nbytes,
            )
            return out.reshape(x.shape)
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def dequantize(self, quantized: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize on GPU back to float32."""
        q = np.ascontiguousarray(quantized, dtype=np.int8)
        n = q.size
        d_in = loader.gpu_malloc(q.nbytes)
        d_out = loader.gpu_malloc(n * 4)
        try:
            loader.memcpy_htod(
                dst_device=d_in,
                src_host=_ct.c_void_p(q.ctypes.data),
                size_bytes=q.nbytes,
            )
            self._launch_dequant(d_in, d_out, n, scale)
            out = np.empty(n, dtype=np.float32)
            loader.memcpy_dtoh(
                dst_host=_ct.c_void_p(out.ctypes.data),
                src_device=d_out,
                size_bytes=out.nbytes,
            )
            return out.reshape(q.shape)
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)


__all__ = ["TernaryQuantizer"]
