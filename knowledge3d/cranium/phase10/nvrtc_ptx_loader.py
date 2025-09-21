from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np  # type: ignore


class NVRTCPTXLoader:
    """Load a PTX kernel (or fall back to inline CUDA) for GPU vertex generation."""

    def __init__(
        self,
        ptx_path: str = "knowledge3d/cranium/ptx/generate_shape_kernel.ptx",
        *,
        fallback_inline: bool = True,
    ) -> None:
        try:
            from cuda import cuda, nvrtc  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"cuda-python bindings unavailable: {exc}")

        self._cuda = cuda
        self._nvrtc = nvrtc
        self._ctx: Optional[int] = None
        self._module: Optional[int] = None
        self._kernel: Optional[int] = None
        self._ptx_path = Path(ptx_path)
        self._fallback_inline = fallback_inline
        self._load_module()

    def _load_module(self) -> None:
        cuda = self._cuda
        nvrtc = self._nvrtc

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

        module = None
        if self._ptx_path.exists():
            try:
                ptx_bytes = self._ptx_path.read_bytes()
                err, module = cuda.cuModuleLoadData(ptx_bytes)
                if err != cuda.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f"cuModuleLoadData failed: {err}")
            except Exception as exc:
                if not self._fallback_inline:
                    raise RuntimeError(f"Failed to load PTX from {self._ptx_path}: {exc}") from exc
                module = None

        if module is None:
            module = self._compile_inline_module(dev)

        self._module = module
        err, func = cuda.cuModuleGetFunction(module, b"generate_shape_kernel")
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuModuleGetFunction failed: {err}")
        self._kernel = func

    def _compile_inline_module(self, dev) -> int:
        cuda = self._cuda
        nvrtc = self._nvrtc

        CUDA_KERNEL = Path(__file__).with_name("generate_shape_kernel.cu")
        if CUDA_KERNEL.exists():
            source = CUDA_KERNEL.read_text(encoding="utf-8")
        else:
            source = _INLINE_FALLBACK_KERNEL

        res, prog = nvrtc.nvrtcCreateProgram(source.encode("utf-8"), b"generate_shape_kernel.cu", 0, [], [])
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcCreateProgram failed: {res}")

        major_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        minor_attr = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        err, major = cuda.cuDeviceGetAttribute(major_attr, dev)
        err_minor, minor = cuda.cuDeviceGetAttribute(minor_attr, dev)
        if err != cuda.CUresult.CUDA_SUCCESS or err_minor != cuda.CUresult.CUDA_SUCCESS:
            nvrtc.nvrtcDestroyProgram(prog)
            raise RuntimeError("Failed to query device compute capability")

        arch = f"--gpu-architecture=compute_{major}{minor}".encode("utf-8")
        opts = [arch, b"--fmad=false"]
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
        return module

    def generate_vertices(self, embedding: np.ndarray, vertex_count: int, shape_type_idx: int) -> np.ndarray:
        if self._kernel is None:
            raise RuntimeError("PTX kernel not initialised")
        cuda = self._cuda

        emb = np.ascontiguousarray(embedding.astype(np.float32))
        if emb.ndim != 1:
            emb = emb.reshape(-1)

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
                raise RuntimeError(f"cuMemcpyHtoD failed: {err}")

            import ctypes as _ct

            emb_arg = _ct.c_void_p(int(d_emb))
            out_arg = _ct.c_void_p(int(d_out))
            count_arg = _ct.c_uint(int(vertex_count))
            shape_arg = _ct.c_uint(int(shape_type_idx))

            param_array = (_ct.c_void_p * 4)(
                _ct.cast(_ct.pointer(emb_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(out_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(count_arg), _ct.c_void_p),
                _ct.cast(_ct.pointer(shape_arg), _ct.c_void_p),
            )

            threads = max(1, min(int(vertex_count), 256))
            blocks = int(math.ceil(vertex_count / threads))

            err, = cuda.cuLaunchKernel(
                self._kernel,
                blocks, 1, 1,
                threads, 1, 1,
                0, 0,
                param_array, 0,
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

    def __del__(self) -> None:  # pragma: no cover
        try:
            if self._module is not None and self._cuda is not None:
                self._cuda.cuModuleUnload(self._module)
        except Exception:
            pass
        try:
            if self._ctx is not None and self._cuda is not None:
                self._cuda.cuCtxDestroy(self._ctx)
        except Exception:
            pass


_INLINE_FALLBACK_KERNEL = r"""
extern "C" __global__ void generate_shape_kernel(
    const float* __restrict__ embedding,
    float* __restrict__ vertices,
    unsigned int vertex_count,
    unsigned int shape_type
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= vertex_count) {
        return;
    }

    float scale = fabsf(embedding[0] + embedding[1] + embedding[2]);
    if (scale < 1e-3f) {
        scale = 1.0f;
    }

    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;

    if (shape_type == 0) {
        if (tid == 0) { vx = 1.0f;  vy = 1.0f;  vz = 1.0f; }
        else if (tid == 1) { vx = -1.0f; vy = -1.0f; vz = 1.0f; }
        else if (tid == 2) { vx = -1.0f; vy = 1.0f; vz = -1.0f; }
        else if (tid == 3) { vx = 1.0f;  vy = -1.0f; vz = -1.0f; }
    } else if (shape_type == 1) {
        int sx = (tid & 1) ? 1 : -1;
        int sy = (tid & 2) ? 1 : -1;
        int sz = (tid & 4) ? 1 : -1;
        vx = (float)sx;
        vy = (float)sy;
        vz = (float)sz;
    } else if (shape_type == 2) {
        if (tid == 0) { vx = 1.0f; }
        else if (tid == 1) { vx = -1.0f; }
        else if (tid == 2) { vy = 1.0f; }
        else if (tid == 3) { vy = -1.0f; }
        else if (tid == 4) { vz = 1.0f; }
        else if (tid == 5) { vz = -1.0f; }
    } else if (shape_type == 3) {
        const float phi = 1.6180339887498948482f;
        switch (tid) {
            case 0:  vx = phi;  vy = 1.0f;  vz = 0.0f; break;
            case 1:  vx = -phi; vy = 1.0f;  vz = 0.0f; break;
            case 2:  vx = phi;  vy = -1.0f; vz = 0.0f; break;
            case 3:  vx = -phi; vy = -1.0f; vz = 0.0f; break;
            case 4:  vx = 0.0f; vy = -1.0f; vz = phi;  break;
            case 5:  vx = 0.0f; vy = 1.0f;  vz = phi;  break;
            case 6:  vx = 0.0f; vy = -1.0f; vz = -phi; break;
            case 7:  vx = 0.0f; vy = 1.0f;  vz = -phi; break;
            case 8:  vx = phi;  vy = 0.0f;  vz = -1.0f; break;
            case 9:  vx = phi;  vy = 0.0f;  vz = 1.0f;  break;
            case 10: vx = -phi; vy = 0.0f;  vz = -1.0f; break;
            case 11: vx = -phi; vy = 0.0f;  vz = 1.0f;  break;
            default: break;
        }
        float norm = sqrtf(vx * vx + vy * vy + vz * vz);
        if (norm > 0.0f) {
            vx /= norm;
            vy /= norm;
            vz /= norm;
        }
    } else {
        float angle = (float)tid * (2.0f * 3.14159265358979323846f / max(1u, vertex_count));
        vx = cosf(angle);
        vy = sinf(angle);
        vz = 0.0f;
    }

    unsigned int offset = tid * 3;
    vertices[offset + 0] = vx * scale;
    vertices[offset + 1] = vy * scale;
    vertices[offset + 2] = vz * scale;
}
"""
