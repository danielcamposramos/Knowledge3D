"""
GPU-only harmonic extraction and synthesis helpers (no CPU fallbacks).

Kernels:
- harmonic_topk: find top-K magnitude bins from a real-valued spectrum (e.g., MDCT).
- harmonic_synthesize: additive synthesis from (freq, amp, phase) tuples.
- subtract_residual: residual = original - approximation.
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
extern "C" __global__ void harmonic_topk(
    const float* __restrict__ coeffs,
    int n,
    int k,
    int* __restrict__ out_idx,
    float* __restrict__ out_mag
) {
    // Assumes n <= 1024 and launches one block with n threads (rounded up to warp).
    extern __shared__ float smag[];
    int tid = threadIdx.x;
    if (tid < n) {
        float v = coeffs[tid];
        smag[tid] = v >= 0 ? v : -v;
    } else {
        smag[tid] = -1.0f;
    }
    __syncthreads();
    for (int iter = 0; iter < k; ++iter) {
        // reduce max value + index
        float maxv = -1.0f;
        int maxidx = -1;
        for (int i = tid; i < n; i += blockDim.x) {
            float v = smag[i];
            if (v > maxv) {
                maxv = v;
                maxidx = i;
            }
        }
        // warp-level reduction
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, maxv, offset);
            int oidx = __shfl_down_sync(0xffffffff, maxidx, offset);
            if (other > maxv) { maxv = other; maxidx = oidx; }
        }
        // write candidate from lane 0 of each warp to shared
        if ((threadIdx.x & (warpSize - 1)) == 0) {
            smag[blockDim.x + threadIdx.x / warpSize] = maxv;
            // reuse smag tail for indices (offset by blockDim.x + max warps)
            smag[blockDim.x + 32 + threadIdx.x / warpSize] = __int_as_float(maxidx);
        }
        __syncthreads();
        // reduce across warps
        int warp_count = (blockDim.x + warpSize - 1) / warpSize;
        if (tid < warpSize) {
            float v = (tid < warp_count) ? smag[blockDim.x + tid] : -1.0f;
            int idx = (tid < warp_count) ? __float_as_int(smag[blockDim.x + 32 + tid]) : -1;
            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(0xffffffff, v, offset);
                int oidx = __shfl_down_sync(0xffffffff, idx, offset);
                if (other > v) { v = other; idx = oidx; }
            }
            if (tid == 0) {
                out_idx[iter] = idx;
                out_mag[iter] = v;
                if (idx >= 0 && idx < n) {
                    smag[idx] = -1.0f; // zero it for next iteration
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void harmonic_synthesize(
    const float* __restrict__ freq,
    const float* __restrict__ amp,
    const float* __restrict__ phase,
    int n_harm,
    float sample_rate,
    int num_samples,
    float* __restrict__ out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;
    float t = (float)idx / sample_rate;
    float acc = 0.0f;
    for (int h = 0; h < n_harm; ++h) {
        float omega = 6.283185307179586f * freq[h];
        acc += amp[h] * __cosf(omega * t + phase[h]);
    }
    out[idx] = acc;
}

extern "C" __global__ void subtract_residual(
    const float* __restrict__ original,
    const float* __restrict__ approx,
    int n,
    float* __restrict__ out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = original[idx] - approx[idx];
}
"""


class AudioHarmonicGPU:
    """GPU helper for harmonic extraction and synthesis."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.cuda, self.nvrtc = _load_cuda()
        self._ctx: Optional[int] = None
        self._module: Optional[int] = None
        self._topk: Optional[int] = None
        self._synth: Optional[int] = None
        self._sub: Optional[int] = None
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
        res, prog = nvrtc.nvrtcCreateProgram(KERNEL_SRC.encode("utf-8"), b"audio_harmonic.cu", 0, [], [])
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
        err, topk = cuda.cuModuleGetFunction(module, b"harmonic_topk")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction harmonic_topk failed: {err}")
        err, synth = cuda.cuModuleGetFunction(module, b"harmonic_synthesize")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction harmonic_synthesize failed: {err}")
        err, sub = cuda.cuModuleGetFunction(module, b"subtract_residual")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction subtract_residual failed: {err}")
        self._topk = topk
        self._synth = synth
        self._sub = sub

    def harmonic_topk(self, coeffs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Top-K magnitude indices and magnitudes on GPU. Assumes len<=1024."""
        x = np.ascontiguousarray(coeffs, dtype=np.float32)
        n = x.size
        if n > 1024:
            raise ValueError("harmonic_topk supports n<=1024")
        if k <= 0:
            raise ValueError("k must be positive")
        k = min(k, n)
        cuda = self.cuda
        d_in = loader.gpu_malloc(x.nbytes)
        d_idx = loader.gpu_malloc(k * 4)
        d_mag = loader.gpu_malloc(k * 4)
        try:
            loader.memcpy_htod(
                dst_device=d_in,
                src_host=_ct.c_void_p(x.ctypes.data),
                size_bytes=x.nbytes,
            )
            block_dim = 1
            while block_dim < n and block_dim < 1024:
                block_dim <<= 1
            grid = (1, 1, 1)
            blk = (block_dim, 1, 1)
            in_arg = _ct.c_void_p(int(getattr(d_in, "value", d_in)))
            n_arg = _ct.c_int(int(n))
            k_arg = _ct.c_int(int(k))
            idx_arg = _ct.c_void_p(int(getattr(d_idx, "value", d_idx)))
            mag_arg = _ct.c_void_p(int(getattr(d_mag, "value", d_mag)))
            params = (_ct.c_void_p * 5)(
                _ct.cast(_ct.pointer(in_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(n_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(k_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(idx_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(mag_arg), _ct.c_void_p),
            )
            err, = cuda.cuLaunchKernel(
                self._topk,
                int(grid[0]), int(grid[1]), int(grid[2]),
                int(blk[0]), int(blk[1]), int(blk[2]),
                blk[0] * 4 * 3,  # shared space for smag + tmp
                0, params, 0,
            )
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuLaunchKernel harmonic_topk failed: {err}")
            idx_host = np.empty(k, dtype=np.int32)
            mag_host = np.empty(k, dtype=np.float32)
            loader.memcpy_dtoh(
                dst_host=_ct.c_void_p(idx_host.ctypes.data),
                src_device=d_idx,
                size_bytes=idx_host.nbytes,
            )
            loader.memcpy_dtoh(
                dst_host=_ct.c_void_p(mag_host.ctypes.data),
                src_device=d_mag,
                size_bytes=mag_host.nbytes,
            )
            return idx_host, mag_host
        finally:
            loader.gpu_free(d_in); loader.gpu_free(d_idx); loader.gpu_free(d_mag)

    def synthesize(self, freq: np.ndarray, amp: np.ndarray, phase: np.ndarray, sample_rate: float, num_samples: int) -> np.ndarray:
        """Additive synth on GPU."""
        f = np.ascontiguousarray(freq, dtype=np.float32)
        a = np.ascontiguousarray(amp, dtype=np.float32)
        p = np.ascontiguousarray(phase, dtype=np.float32)
        if f.size != a.size or f.size != p.size:
            raise ValueError("freq, amp, phase size mismatch")
        n_harm = f.size
        cuda = self.cuda
        d_f = loader.gpu_malloc(f.nbytes)
        d_a = loader.gpu_malloc(a.nbytes)
        d_p = loader.gpu_malloc(p.nbytes)
        d_out = loader.gpu_malloc(num_samples * 4)
        try:
            for buf, data in [(d_f, f), (d_a, a), (d_p, p)]:
                loader.memcpy_htod(
                    dst_device=buf,
                    src_host=_ct.c_void_p(data.ctypes.data),
                    size_bytes=data.nbytes,
                )
            blk = (256, 1, 1)
            grid = (int((num_samples + blk[0] - 1) // blk[0]), 1, 1)
            f_arg = _ct.c_void_p(int(getattr(d_f, "value", d_f)))
            a_arg = _ct.c_void_p(int(getattr(d_a, "value", d_a)))
            p_arg = _ct.c_void_p(int(getattr(d_p, "value", d_p)))
            n_arg = _ct.c_int(int(n_harm))
            sr_arg = _ct.c_float(float(sample_rate))
            ns_arg = _ct.c_int(int(num_samples))
            out_arg = _ct.c_void_p(int(getattr(d_out, "value", d_out)))
            params = (_ct.c_void_p * 7)(
                _ct.cast(_ct.pointer(f_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(a_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(p_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(n_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(sr_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(ns_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(out_arg), _ct.c_void_p),
            )
            err, = cuda.cuLaunchKernel(
                self._synth,
                int(grid[0]), int(grid[1]), int(grid[2]),
                int(blk[0]), int(blk[1]), int(blk[2]),
                0, 0, params, 0,
            )
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuLaunchKernel harmonic_synthesize failed: {err}")
            out = np.empty(num_samples, dtype=np.float32)
            loader.memcpy_dtoh(
                dst_host=_ct.c_void_p(out.ctypes.data),
                src_device=d_out,
                size_bytes=out.nbytes,
            )
            return out
        finally:
            loader.gpu_free(d_f); loader.gpu_free(d_a); loader.gpu_free(d_p); loader.gpu_free(d_out)

    def subtract_residual(self, original: np.ndarray, approx: np.ndarray) -> np.ndarray:
        """Residual = original - approx on GPU."""
        o = np.ascontiguousarray(original, dtype=np.float32)
        a = np.ascontiguousarray(approx, dtype=np.float32)
        if o.size != a.size:
            raise ValueError("original and approx size mismatch")
        n = o.size
        cuda = self.cuda
        d_o = loader.gpu_malloc(o.nbytes)
        d_a = loader.gpu_malloc(a.nbytes)
        d_out = loader.gpu_malloc(o.nbytes)
        try:
            for buf, data in [(d_o, o), (d_a, a)]:
                loader.memcpy_htod(
                    dst_device=buf,
                    src_host=_ct.c_void_p(data.ctypes.data),
                    size_bytes=data.nbytes,
                )
            blk = (256, 1, 1)
            grid = (int((n + blk[0] - 1) // blk[0]), 1, 1)
            o_arg = _ct.c_void_p(int(getattr(d_o, "value", d_o)))
            a_arg = _ct.c_void_p(int(getattr(d_a, "value", d_a)))
            n_arg = _ct.c_int(int(n))
            out_arg = _ct.c_void_p(int(getattr(d_out, "value", d_out)))
            params = (_ct.c_void_p * 4)(
                _ct.cast(_ct.pointer(o_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(a_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(n_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(out_arg), _ct.c_void_p),
            )
            err, = cuda.cuLaunchKernel(
                self._sub,
                int(grid[0]), int(grid[1]), int(grid[2]),
                int(blk[0]), int(blk[1]), int(blk[2]),
                0, 0, params, 0,
            )
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuLaunchKernel subtract_residual failed: {err}")
            out = np.empty(n, dtype=np.float32)
            loader.memcpy_dtoh(
                dst_host=_ct.c_void_p(out.ctypes.data),
                src_device=d_out,
                size_bytes=out.nbytes,
            )
            return out
        finally:
            loader.gpu_free(d_o); loader.gpu_free(d_a); loader.gpu_free(d_out)


__all__ = ["AudioHarmonicGPU"]
