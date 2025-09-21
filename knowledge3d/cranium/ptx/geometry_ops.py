from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore
from cuda import cuda, nvrtc  # type: ignore

from knowledge3d.cranium.ptx.galaxy_buffer import (
    GalaxyGPUMemory,
    DeviceBuffer,
    load_meshes_from_glb,
    release_galaxy_memory,
    save_embeddings_to_json,
    save_meshes_to_glb,
)


_MODULE_HANDLES: List[int] = []
_KERNEL_CACHE: Dict[str, int] = {}


def _ensure_cuda_context() -> int:
    err, ctx = cuda.cuCtxGetCurrent()
    if err == cuda.CUresult.CUDA_SUCCESS and ctx:
        return ctx
    err, = cuda.cuInit(0)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {err}")
    err, dev = cuda.cuDeviceGet(0)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet failed: {err}")
    err, ctx = cuda.cuCtxCreate(0, dev)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxCreate failed: {err}")
    return ctx


def _compile_kernel(source: str, name: str) -> int:
    _ensure_cuda_context()
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
    _MODULE_HANDLES.append(module)
    return func


_KERNEL_SOURCE = Path(__file__).with_name("kernels.cu").read_text(encoding="utf-8")


def _get_kernel(name: str) -> int:
    if name in _KERNEL_CACHE:
        return _KERNEL_CACHE[name]
    func = _compile_kernel(_KERNEL_SOURCE, name)
    _KERNEL_CACHE[name] = func
    return func


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

    kernel = _get_kernel("apply_transform")

    params = (
        matrix_dev.ptr,
        galaxy_memory.vertices.ptr,
        offsets_dev.ptr,
        counts_dev.ptr,
        np.uint32(mesh_index),
    )
    _launch_kernel(kernel, blocks, threads, params)

    _free_device(matrix_dev)
    _free_device(offsets_dev)
    _free_device(counts_dev)

    galaxy_memory.vertices_dirty = True
    galaxy_memory.normals_stale = True


def recalc_mesh_normals(
    galaxy_memory: GalaxyGPUMemory,
    mesh_index: int,
) -> None:
    if not galaxy_memory.mesh_records:
        raise ValueError("no mesh data loaded")
    if galaxy_memory.normals.ptr == 0 or galaxy_memory.normals.size == 0:
        raise ValueError("normal buffer unavailable; cannot recalc")
    record = galaxy_memory.mesh_records[mesh_index]
    offsets, counts = _prepare_offsets_counts(galaxy_memory, index_mode=True)

    offsets_dev = _alloc_upload(offsets)
    counts_dev = _alloc_upload(counts)

    err, = cuda.cuMemsetD32(galaxy_memory.normals.ptr, 0, galaxy_memory.normals.size // 4)
    if err != cuda.CUresult.CUDA_SUCCESS:
        _free_device(offsets_dev)
        _free_device(counts_dev)
        raise RuntimeError(f"cuMemsetD32 failed: {err}")

    threads = 256
    blocks = math.ceil((record.index_count // 3) / threads)

    kernel = _get_kernel("recalc_normals")

    params = (
        galaxy_memory.vertices.ptr,
        galaxy_memory.indices.ptr,
        galaxy_memory.normals.ptr,
        offsets_dev.ptr,
        counts_dev.ptr,
        np.uint32(mesh_index),
    )
    _launch_kernel(kernel, blocks, threads, params)

    _free_device(offsets_dev)
    _free_device(counts_dev)

    galaxy_memory.normals_dirty = True
    galaxy_memory.normals_stale = False


def scale_mesh_vertices(
    galaxy_memory: GalaxyGPUMemory,
    mesh_index: int,
    scale: np.ndarray,
) -> None:
    scale_vec = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scale_vec.size != 3:
        raise ValueError("scale must be length-3 vector")
    offsets, counts = _prepare_offsets_counts(galaxy_memory)
    scale_dev = _alloc_upload(scale_vec)
    offsets_dev = _alloc_upload(offsets)
    counts_dev = _alloc_upload(counts)

    record = galaxy_memory.mesh_records[mesh_index]
    threads = 256
    blocks = math.ceil(record.vertex_count / threads)

    params = (
        scale_dev.ptr,
        galaxy_memory.vertices.ptr,
        offsets_dev.ptr,
        counts_dev.ptr,
        np.uint32(mesh_index),
    )
    _launch_kernel(_get_kernel("scale_vertices"), blocks, threads, params)

    _free_device(scale_dev)
    _free_device(offsets_dev)
    _free_device(counts_dev)

    galaxy_memory.vertices_dirty = True
    galaxy_memory.normals_stale = True


def offset_mesh_vertices(
    galaxy_memory: GalaxyGPUMemory,
    mesh_index: int,
    offset: np.ndarray,
) -> None:
    offset_vec = np.asarray(offset, dtype=np.float32).reshape(-1)
    if offset_vec.size != 3:
        raise ValueError("offset must be length-3 vector")
    offsets, counts = _prepare_offsets_counts(galaxy_memory)
    offset_dev = _alloc_upload(offset_vec)
    offsets_dev = _alloc_upload(offsets)
    counts_dev = _alloc_upload(counts)

    record = galaxy_memory.mesh_records[mesh_index]
    threads = 256
    blocks = math.ceil(record.vertex_count / threads)

    params = (
        offset_dev.ptr,
        galaxy_memory.vertices.ptr,
        offsets_dev.ptr,
        counts_dev.ptr,
        np.uint32(mesh_index),
    )
    _launch_kernel(_get_kernel("offset_vertices"), blocks, threads, params)

    _free_device(offset_dev)
    _free_device(offsets_dev)
    _free_device(counts_dev)

    galaxy_memory.vertices_dirty = True
    galaxy_memory.normals_stale = True


def blend_node_embedding(
    galaxy_memory: GalaxyGPUMemory,
    node_index: int,
    source_embedding: np.ndarray,
    alpha: float,
) -> None:
    if galaxy_memory.embeddings.size == 0:
        raise ValueError("embedding buffer is empty")
    embedding_dim = source_embedding.shape[-1]
    if source_embedding.ndim != 1:
        raise ValueError("source_embedding must be 1D")
    if galaxy_memory.embedding_shape:
        if len(galaxy_memory.embedding_shape) != 2:
            raise ValueError("embedding buffer shape is invalid")
        node_count, device_dim = galaxy_memory.embedding_shape
        if embedding_dim != device_dim:
            raise ValueError("source embedding dimension mismatch")
        if node_index >= node_count:
            raise IndexError("node_index out of range")
    source_dev = _alloc_upload(source_embedding.astype(np.float32).reshape(-1))

    threads = 256
    blocks = math.ceil(embedding_dim / threads)

    kernel = _get_kernel("blend_embeddings")

    params = (
        source_dev.ptr,
        galaxy_memory.embeddings.ptr,
        np.float32(alpha),
        np.uint32(embedding_dim),
        np.uint32(node_index),
    )
    _launch_kernel(kernel, blocks, threads, params)
    _free_device(source_dev)

    galaxy_memory.embeddings_dirty = True


def normalize_node_embedding(
    galaxy_memory: GalaxyGPUMemory,
    node_index: int,
) -> None:
    if galaxy_memory.embeddings.ptr == 0 or galaxy_memory.embeddings.size == 0:
        raise ValueError("embedding buffer is empty")
    if not galaxy_memory.embedding_shape or len(galaxy_memory.embedding_shape) != 2:
        raise ValueError("embedding shape metadata unavailable")
    node_count, embedding_dim = galaxy_memory.embedding_shape
    if node_index >= node_count:
        raise IndexError("node_index out of range")

    threads = min(256, int(embedding_dim))
    if threads == 0:
        return
    shared_mem = threads * np.dtype(np.float32).itemsize
    params = (
        galaxy_memory.embeddings.ptr,
        np.uint32(embedding_dim),
        np.uint32(node_index),
    )
    _launch_kernel(_get_kernel("normalize_embedding"), 1, threads, params, shared_mem=shared_mem)
    galaxy_memory.embeddings_dirty = True


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


def _launch_kernel(func: int, blocks: int, threads: int, params: Tuple, *, shared_mem: int = 0) -> None:
    arg_arrays = []
    for value in params:
        if isinstance(value, (int, np.integer)):
            arg_arrays.append(np.array([int(value)], dtype=np.uint64))
        elif isinstance(value, (float, np.floating)):
            arg_arrays.append(np.array([float(value)], dtype=np.float32))
        else:  # pragma: no cover - unexpected parameter type
            raise TypeError(f"Unsupported kernel parameter type: {type(value)!r}")
    err, = cuda.cuLaunchKernel(
        func,
        blocks, 1, 1,
        threads, 1, 1,
        shared_mem,
        0,
        tuple(arg.ctypes.data for arg in arg_arrays),
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


class PTXGeometrySession:
    """High-level helper that manages GPU geometry state for a GLB scene."""

    def __init__(self) -> None:
        self._galaxy: Optional[GalaxyGPUMemory] = None

    # ------------------------------------------------------------------
    def load_scene(self, glb_path: str, *, embedding_json: Optional[str] = None) -> GalaxyGPUMemory:
        """Load a GLB + optional embedding JSON into GPU memory, replacing any existing scene."""

        self.close()
        self._galaxy = load_meshes_from_glb(glb_path, embedding_json=embedding_json)
        return self._galaxy

    # ------------------------------------------------------------------
    def apply_transform(
        self,
        mesh_index: int,
        matrix: np.ndarray,
        *,
        recalc_normals: bool = False,
    ) -> None:
        galaxy = self._require_memory()
        matrix_arr = np.asarray(matrix, dtype=np.float32)
        if matrix_arr.shape != (4, 4):
            raise ValueError("matrix must be shape (4, 4)")
        apply_mesh_transform(galaxy, mesh_index, matrix_arr)
        if recalc_normals:
            recalc_mesh_normals(galaxy, mesh_index)

    def recalc_normals(self, mesh_index: int) -> None:
        galaxy = self._require_memory()
        recalc_mesh_normals(galaxy, mesh_index)

    def blend_embedding(self, node_index: int, embedding: np.ndarray, alpha: float) -> None:
        galaxy = self._require_memory()
        emb_arr = np.asarray(embedding, dtype=np.float32)
        if emb_arr.ndim != 1:
            raise ValueError("embedding must be 1D")
        blend_node_embedding(galaxy, node_index, emb_arr, alpha)

    def apply_transform_for_mesh(
        self,
        mesh_id: int,
        matrix: np.ndarray,
        *,
        primitive_index: Optional[int] = None,
        recalc_normals: bool = False,
    ) -> None:
        galaxy = self._require_memory()
        matrix_arr = np.asarray(matrix, dtype=np.float32)
        if matrix_arr.shape != (4, 4):
            raise ValueError("matrix must be shape (4, 4)")

        matched = False
        for idx, record in enumerate(galaxy.mesh_records):
            if record.mesh_index != mesh_id:
                continue
            if primitive_index is not None and record.primitive_index != primitive_index:
                continue
            apply_mesh_transform(galaxy, idx, matrix_arr)
            matched = True
            if recalc_normals and galaxy.normals.ptr:
                try:
                    recalc_mesh_normals(galaxy, idx)
                except ValueError:
                    # Normals may be missing for this primitive; ignore.
                    pass

        if not matched:
            raise ValueError(f"No mesh record found for mesh_id={mesh_id} (primitive_index={primitive_index})")

    def translate_mesh(
        self,
        mesh_id: int,
        translation: np.ndarray | List[float],
        *,
        primitive_index: Optional[int] = None,
        recalc_normals: bool = False,
    ) -> None:
        vec = np.asarray(translation, dtype=np.float32)
        if vec.shape != (3,):
            raise ValueError("translation must be length-3 vector")
        galaxy = self._require_memory()
        matching = [idx for idx, record in enumerate(galaxy.mesh_records) if record.mesh_index == mesh_id]
        if primitive_index is not None:
            matching = [idx for idx in matching if galaxy.mesh_records[idx].primitive_index == primitive_index]
            if not matching:
                raise IndexError("primitive_index out of range")
        if not matching:
            raise IndexError("mesh_id has no primitives in GPU memory")
        for idx in matching:
            offset_mesh_vertices(galaxy, idx, vec)
            if recalc_normals:
                try:
                    recalc_mesh_normals(galaxy, idx)
                except ValueError:
                    pass

    def scale_mesh(
        self,
        mesh_id: int,
        scale: np.ndarray | List[float],
        *,
        primitive_index: Optional[int] = None,
        recalc_normals: bool = False,
    ) -> None:
        vec = np.asarray(scale, dtype=np.float32)
        if vec.shape != (3,):
            raise ValueError("scale must be length-3 vector")
        galaxy = self._require_memory()
        matching = [idx for idx, record in enumerate(galaxy.mesh_records) if record.mesh_index == mesh_id]
        if primitive_index is not None:
            matching = [idx for idx in matching if galaxy.mesh_records[idx].primitive_index == primitive_index]
            if not matching:
                raise IndexError("primitive_index out of range")
        if not matching:
            raise IndexError("mesh_id has no primitives in GPU memory")
        for idx in matching:
            scale_mesh_vertices(galaxy, idx, vec)
            if recalc_normals:
                try:
                    recalc_mesh_normals(galaxy, idx)
                except ValueError:
                    pass

    def normalize_embedding(self, node_index: int) -> None:
        galaxy = self._require_memory()
        normalize_node_embedding(galaxy, node_index)

    # ------------------------------------------------------------------
    def save(
        self,
        *,
        target_glb: Optional[str] = None,
        target_embeddings: Optional[str] = None,
    ) -> None:
        galaxy = self._require_memory()
        save_meshes_to_glb(galaxy, target_path=target_glb)
        save_embeddings_to_json(galaxy, target_path=target_embeddings)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._galaxy is not None:
            release_galaxy_memory(self._galaxy)
            self._galaxy = None

    # Context manager helpers ------------------------------------------------
    def __enter__(self) -> "PTXGeometrySession":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        self.close()

    # Internal ----------------------------------------------------------------
    def _require_memory(self) -> GalaxyGPUMemory:
        if self._galaxy is None:
            raise RuntimeError("No Galaxy scene loaded. Call load_scene first.")
        return self._galaxy

    @property
    def galaxy_memory(self) -> GalaxyGPUMemory:
        return self._require_memory()
