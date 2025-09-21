from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


_COMPONENT_SIZES = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _component_count(accessor_type: str) -> int:
    return _COMPONENT_SIZES.get(accessor_type, 3)

import numpy as np  # type: ignore
from pygltflib import GLTF2  # type: ignore

try:
    from cuda import cuda  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("cuda-python bindings unavailable: install cuda-python in k3d-cranium env") from exc


@dataclass
class DeviceBuffer:
    ptr: int
    size: int


@dataclass
class MeshRecord:
    mesh_index: int
    primitive_index: int
    vertex_offset: int
    vertex_count: int
    index_offset: int
    index_count: int
    index_component_type: int
    material_id: Optional[int]


@dataclass
class GalaxyGPUMemory:
    ctx: int
    vertices: DeviceBuffer
    indices: DeviceBuffer
    embeddings: DeviceBuffer
    normals: DeviceBuffer
    node_table: List[Dict[str, int]] = field(default_factory=list)
    mesh_records: List[MeshRecord] = field(default_factory=list)
    glb_path: Optional[str] = None
    embeddings_path: Optional[str] = None
    embedding_shape: Tuple[int, ...] = field(default_factory=tuple)
    vertices_dirty: bool = False
    indices_dirty: bool = False
    normals_dirty: bool = False
    normals_stale: bool = False
    embeddings_dirty: bool = False

    def __del__(self) -> None:  # pragma: no cover
        try:
            if self.vertices.ptr:
                cuda.cuMemFree(self.vertices.ptr)
        except Exception:
            pass
        try:
            if self.indices.ptr:
                cuda.cuMemFree(self.indices.ptr)
        except Exception:
            pass
        try:
            if self.embeddings.ptr:
                cuda.cuMemFree(self.embeddings.ptr)
        except Exception:
            pass
        try:
            if self.normals.ptr:
                cuda.cuMemFree(self.normals.ptr)
        except Exception:
            pass
        try:
            if self.ctx:
                cuda.cuCtxDestroy(self.ctx)
        except Exception:
            pass


def _ensure_context() -> int:
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


def _alloc_and_upload(array: np.ndarray) -> DeviceBuffer:
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


def load_meshes_from_glb(
    glb_path: str,
    *,
    embedding_json: Optional[str] = None,
) -> GalaxyGPUMemory:
    """Load GLB mesh buffers + optional embeddings into GPU memory.

    Parameters
    ----------
    glb_path: str
        Path to galaxy GLB containing embedded buffers.
    embedding_json: Optional[str]
        Optional path to JSON containing node embeddings. If omitted, embeddings buffer is zero sized.
    """

    glb = GLTF2().load(glb_path)

    # Gather mesh data from bufferViews
    vertex_chunks: List[np.ndarray] = []
    index_chunks: List[np.ndarray] = []
    mesh_records: List[MeshRecord] = []

    binary_blob = glb.binary_blob()

    def slice_view(view: "BufferView") -> memoryview:
        buf = glb.buffers[view.buffer]
        if getattr(buf, "uri", None):
            data = glb.get_data_from_buffer_uri(buf.uri)
        elif binary_blob is not None:
            data = binary_blob
        else:
            data = getattr(buf, "data", None)
        if data is None:
            raise ValueError("Unable to resolve buffer data; ensure GLB has embedded buffers")
        start = view.byteOffset or 0
        end = start + (view.byteLength or 0)
        return memoryview(data)[start:end]

    def attribute_index(attributes: Union[Dict[str, int], object], key: str) -> Optional[int]:
        if isinstance(attributes, dict):
            return attributes.get(key)
        return getattr(attributes, key, None)

    for mesh_index, mesh in enumerate(glb.meshes or []):
        for prim_index, prim in enumerate(mesh.primitives):
            attr_index = attribute_index(prim.attributes, "POSITION")
            if attr_index is None:
                continue
            accessor = glb.accessors[attr_index]
            view = glb.bufferViews[accessor.bufferView]
            raw = np.frombuffer(slice_view(view), dtype=np.float32)
            comp = _component_count(accessor.type)
            verts = raw.reshape((-1, comp))
            vertex_offset = sum(chunk.size for chunk in vertex_chunks) // 3
            vertex_chunks.append(verts.reshape(-1))

            if prim.indices is not None:
                index_accessor = glb.accessors[prim.indices]
                index_view = glb.bufferViews[index_accessor.bufferView]
                idx_dtype = np.uint16 if index_accessor.componentType == 5123 else np.uint32
                raw_idx = np.frombuffer(slice_view(index_view), dtype=idx_dtype)
                index_array = raw_idx.astype(np.uint32)
            else:
                index_array = np.arange(verts.shape[0], dtype=np.uint32)
                index_accessor = None
                index_view = None
            index_offset = sum(chunk.size for chunk in index_chunks)
            index_chunks.append(index_array)

            mesh_records.append(
                MeshRecord(
                    mesh_index=mesh_index,
                    primitive_index=prim_index,
                    vertex_offset=vertex_offset,
                    vertex_count=verts.shape[0],
                    index_offset=index_offset,
                    index_count=index_array.size,
                    index_component_type=(index_accessor.componentType if index_accessor else 5125),
                    material_id=getattr(prim, "material", None),
                )
            )

    vertices_array = np.concatenate(vertex_chunks).astype(np.float32) if vertex_chunks else np.array([], dtype=np.float32)
    indices_array = np.concatenate(index_chunks).astype(np.uint32) if index_chunks else np.array([], dtype=np.uint32)

    normal_chunks: List[np.ndarray] = []
    for mesh in glb.meshes or []:
        for prim in mesh.primitives:
            normal_attr = attribute_index(prim.attributes, "NORMAL")
            if normal_attr is None:
                continue
            accessor = glb.accessors[normal_attr]
            view = glb.bufferViews[accessor.bufferView]
            raw = np.frombuffer(slice_view(view), dtype=np.float32)
            comp = _component_count(accessor.type)
            normal_chunks.append(raw.reshape(-1))
    normals_array = np.concatenate(normal_chunks).astype(np.float32) if normal_chunks else np.array([], dtype=np.float32)

    embeddings_array = np.array([], dtype=np.float32)
    embedding_shape: Tuple[int, ...] = tuple()
    node_table: List[Dict[str, int]] = []
    if embedding_json and Path(embedding_json).exists():
        data = json.loads(Path(embedding_json).read_text())
        embeddings = data.get("embeddings", [])
        node_table = data.get("nodes", [])
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            embedding_shape = tuple(embeddings_array.shape)

    ctx = _ensure_context()
    vertices_dev = _alloc_and_upload(vertices_array)
    indices_dev = _alloc_and_upload(indices_array)
    embeddings_dev = _alloc_and_upload(embeddings_array)
    normals_dev = _alloc_and_upload(normals_array)

    return GalaxyGPUMemory(
        ctx=ctx,
        vertices=vertices_dev,
        indices=indices_dev,
        embeddings=embeddings_dev,
        normals=normals_dev,
        node_table=node_table,
        mesh_records=mesh_records,
        glb_path=str(glb_path),
        embeddings_path=str(Path(embedding_json)) if embedding_json else None,
        embedding_shape=embedding_shape,
    )


def _download_buffer(buffer: DeviceBuffer, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if buffer.ptr == 0 or buffer.size == 0:
        return np.array([], dtype=dtype)
    host = np.empty(buffer.size // dtype.itemsize, dtype=dtype)
    err, = cuda.cuMemcpyDtoH(host.ctypes.data, buffer.ptr, buffer.size)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemcpyDtoH failed: {err}")
    return host


def save_meshes_to_glb(galaxy_memory: GalaxyGPUMemory, target_path: Optional[str] = None) -> None:
    """Download GPU buffers and write back to GLB on disk."""

    if galaxy_memory.glb_path is None:
        raise ValueError("GalaxyGPUMemory missing original GLB path")

    if not any([galaxy_memory.vertices_dirty, galaxy_memory.indices_dirty, galaxy_memory.normals_dirty]) and target_path is None:
        # Nothing changed; skip disk churn unless explicitly requested.
        return

    glb = GLTF2().load(galaxy_memory.glb_path)

    vertices_host = _download_buffer(galaxy_memory.vertices, np.float32)
    indices_host = _download_buffer(galaxy_memory.indices, np.uint32)
    normals_host = _download_buffer(galaxy_memory.normals, np.float32)

    vertex_cursor = 0
    index_cursor = 0
    normal_cursor = 0
    record_iter = iter(galaxy_memory.mesh_records)

    for mesh in glb.meshes or []:
        for prim in mesh.primitives:
            try:
                record = next(record_iter)
            except StopIteration as exc:  # pragma: no cover
                raise RuntimeError("Mesh record mismatch during save") from exc

            pos_accessor = glb.accessors[prim.attributes["POSITION"]]
            pos_view = glb.bufferViews[pos_accessor.bufferView]
            count = pos_accessor.count * pos_accessor.type.count
            byte_length = count * 4
            if byte_length:
                data = vertices_host[vertex_cursor:vertex_cursor + count].astype(np.float32).tobytes()
                buffer = glb.buffers[pos_view.buffer]
                original = buffer.data or b""
                buffer.data = original[:pos_view.byteOffset] + data + original[pos_view.byteOffset + byte_length:]
                buffer.byteLength = len(buffer.data)
                vertex_cursor += count

            idx_accessor = glb.accessors[prim.indices]
            idx_view = glb.bufferViews[idx_accessor.bufferView]
            idx_count = idx_accessor.count
            idx_dtype = np.uint16 if record.index_component_type == 5123 else np.uint32
            idx_itemsize = np.dtype(idx_dtype).itemsize
            idx_bytes = idx_count * idx_itemsize
            if idx_bytes:
                idx_data = indices_host[index_cursor:index_cursor + idx_count].astype(idx_dtype).tobytes()
                buffer = glb.buffers[idx_view.buffer]
                original = buffer.data or b""
                buffer.data = original[:idx_view.byteOffset] + idx_data + original[idx_view.byteOffset + idx_bytes:]
                buffer.byteLength = len(buffer.data)
                index_cursor += idx_count

            normal_attr = prim.attributes.get("NORMAL")
            if normal_attr is not None and normals_host.size and galaxy_memory.normals_dirty and not galaxy_memory.normals_stale:
                norm_accessor = glb.accessors[normal_attr]
                norm_view = glb.bufferViews[norm_accessor.bufferView]
                norm_count = norm_accessor.count * norm_accessor.type.count
                norm_bytes = norm_count * 4
                if norm_bytes:
                    norm_data = normals_host[normal_cursor:normal_cursor + norm_count].astype(np.float32).tobytes()
                    buffer = glb.buffers[norm_view.buffer]
                    original = buffer.data or b""
                    buffer.data = original[:norm_view.byteOffset] + norm_data + original[norm_view.byteOffset + norm_bytes:]
                    buffer.byteLength = len(buffer.data)
                    normal_cursor += norm_count

    output = target_path or galaxy_memory.glb_path
    glb.save(output)

    galaxy_memory.vertices_dirty = False
    galaxy_memory.indices_dirty = False
    galaxy_memory.normals_dirty = False


def save_embeddings_to_json(galaxy_memory: GalaxyGPUMemory, target_path: Optional[str] = None) -> None:
    """Persist embedding buffer + node table back to JSON."""

    if galaxy_memory.embeddings.size == 0 or galaxy_memory.embeddings.ptr == 0:
        return

    if not galaxy_memory.embeddings_dirty and target_path is None:
        return

    path_str: Optional[str] = target_path or galaxy_memory.embeddings_path
    if path_str is None:
        raise ValueError("No embeddings path provided; pass target_path")

    embeddings_host = _download_buffer(galaxy_memory.embeddings, np.float32)
    if galaxy_memory.embedding_shape:
        embeddings_host = embeddings_host.reshape(galaxy_memory.embedding_shape)

    payload = {}
    path = Path(path_str)
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:  # pragma: no cover
            payload = {}

    payload["embeddings"] = embeddings_host.tolist()
    if galaxy_memory.node_table:
        payload["nodes"] = galaxy_memory.node_table

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    galaxy_memory.embeddings_dirty = False


def release_galaxy_memory(galaxy_memory: GalaxyGPUMemory) -> None:
    """Free GPU resources associated with a GalaxyGPUMemory instance."""

    try:
        if galaxy_memory.vertices.ptr:
            cuda.cuMemFree(galaxy_memory.vertices.ptr)
    finally:
        galaxy_memory.vertices = DeviceBuffer(ptr=0, size=0)

    try:
        if galaxy_memory.indices.ptr:
            cuda.cuMemFree(galaxy_memory.indices.ptr)
    finally:
        galaxy_memory.indices = DeviceBuffer(ptr=0, size=0)

    try:
        if galaxy_memory.embeddings.ptr:
            cuda.cuMemFree(galaxy_memory.embeddings.ptr)
    finally:
        galaxy_memory.embeddings = DeviceBuffer(ptr=0, size=0)

    try:
        if galaxy_memory.normals.ptr:
            cuda.cuMemFree(galaxy_memory.normals.ptr)
    finally:
        galaxy_memory.normals = DeviceBuffer(ptr=0, size=0)

    if galaxy_memory.ctx:
        cuda.cuCtxDestroy(galaxy_memory.ctx)
        galaxy_memory.ctx = 0
