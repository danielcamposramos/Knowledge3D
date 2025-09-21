from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np  # type: ignore
from cuda import cuda, nvrtc  # type: ignore

from knowledge3d.cranium.ptx.galaxy_buffer import GalaxyGPUMemory, DeviceBuffer


def _compile_kernel(source: str, name: str) -> int:
    res, prog = nvrtc.nvrtcCreateProgram(source.encode("utf-8"), b"kernels.cu", 0, [], [])
    if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram failed: {res}")

    opts = [b"--gpu-architecture=compute_75"]
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

    err, func = cuda.cuModuleGetFunction(module, name.encode("utf-8"))
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleGetFunction failed: {err}")
    return func


_KERNEL_SOURCE = Path(__file__).with_name("kernels.cu").read_text(encoding="utf-8")
_APPLY_TRANSFORM = _compile_kernel(_KERNEL_SOURCE, "apply_transform")
_RECALC_NORMALS = _compile_kernel(_KERNEL_SOURCE, "recalc_normals")
_BLEND_EMBED = _compile_kernel(_KERNEL_SOURCE, "blend_embeddings")


def apply_mesh_transform(
    galaxy_memory: GalaxyGPUMemory,
    mesh_index: int,
    matrix: np.ndarray,
) -> None:
    if matrix.shape != (4, 4):
        raise ValueError("matrix must be 4x4")
    if not galaxy_memory.mesh_records:
        raise ValueError("no mesh data loaded")
    record = galaxy_memory.mesh_records[mesh_index]
    offsets, counts = _prepare_offsets_counts(galaxy_memory)

    matrix_dev = _alloc_upload(matrix.astype(np.float32).reshape(-1))
    offsets_dev = _alloc_upload(offsets)
    counts_dev = _alloc_upload(counts)

    threads = 256
    blocks = math.ceil(record.vertex_count / threads)

    params = (
        matrix_dev.ptr,
        galaxy_memory.vertices.ptr,
        offsets_dev.ptr,
        counts_dev.ptr,
        np.uint32(mesh_index),
    )
    _launch_kernel(_APPLY_TRANSFORM, blocks, threads, params)

    _free_device(matrix_dev)
    _free_device(offsets_dev)
    _free_device(counts_dev)


def recalc_mesh_normals(
    galaxy_memory: GalaxyGPUMemory,
    mesh_index: int,
) -> None:
    if not galaxy_memory.mesh_records:
        raise ValueError("no mesh data loaded")
    record = galaxy_memory.mesh_records[mesh_index]
    offsets, counts = _prepare_offsets_counts(galaxy_memory, index_mode=True)

    offsets_dev = _alloc_upload(offsets)
    counts_dev = _alloc_upload(counts)

    threads = 256
    blocks = math.ceil((record.index_count // 3) / threads)

    params = (
        galaxy_memory.vertices.ptr,
        galaxy_memory.indices.ptr,
        galaxy_memory.normals.ptr if hasattr(galaxy_memory, "normals") else 0,
        offsets_dev.ptr,
        counts_dev.ptr,
        np.uint32(mesh_index),
    )
    _launch_kernel(_RECALC_NORMALS, blocks, threads, params)

    _free_device(offsets_dev)
    _free_device(counts_dev)


def blend_node_embedding(
    galaxy_memory: GalaxyGPUMemory,
    node_index: int,
    source_embedding: np.ndarray,
    alpha: float,
) -> None:
    if galaxy_memory.embeddings.size == 0:
        raise ValueError("embedding buffer is empty")
    embedding_dim = source_embedding.shape[-1]
    source_dev = _alloc_upload(source_embedding.astype(np.float32).reshape(-1))

    threads = 256
    blocks = math.ceil(embedding_dim / threads)

    params = (
        source_dev.ptr,
        galaxy_memory.embeddings.ptr,
        np.float32(alpha),
        np.uint32(embedding_dim),
        np.uint32(node_index),
    )
    _launch_kernel(_BLEND_EMBED, blocks, threads, params)
    _free_device(source_dev)


def _alloc_upload(array: np.ndarray) -> DeviceBuffer:
    size = int(array.nbytes)
    if size == 0:
        return DeviceBuffer(ptr=0, size=0)
    err, dptr = cuda.cuMemAlloc(size)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemAlloc failed: {err}")
    err, = cuda.cuMemcpyHtoD(dptr, array.ctypes.data, size)
    if err != cuda.CUresult.CUDA_SUCCESS:
        cuda.cuMemFree(dptr)
        raise RuntimeError(f"cuMemcpyHtoD failed: {err}")
    return DeviceBuffer(ptr=int(dptr), size=size)


def _free_device(buffer: DeviceBuffer) -> None:
    if buffer.ptr:
        cuda.cuMemFree(buffer.ptr)


def _launch_kernel(func: int, blocks: int, threads: int, params: Tuple) -> None:
    args = tuple(np.array([p], dtype=np.uint64) if isinstance(p, int) else np.array([p], dtype=np.float32) for p in params)
    err, = cuda.cuLaunchKernel(
        func,
        blocks, 1, 1,
        threads, 1, 1,
        0, 0,
        tuple(arg.ctypes.data for arg in args),
        0,
    )
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel failed: {err}")


def _prepare_offsets_counts(galaxy_memory: GalaxyGPUMemory, *, index_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if index_mode:
        offsets = np.array([r.index_offset for r in galaxy_memory.mesh_records], dtype=np.uint32)
        counts = np.array([r.index_count for r in galaxy_memory.mesh_records], dtype=np.uint32)
    else:
        offsets = np.array([r.vertex_offset for r in galaxy_memory.mesh_records], dtype=np.uint32)
        counts = np.array([r.vertex_count for r in galaxy_memory.mesh_records], dtype=np.uint32)
    return offsets, counts
