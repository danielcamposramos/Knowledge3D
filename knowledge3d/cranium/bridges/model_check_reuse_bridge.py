"""Bounded GPU model-checking reuse bridge."""

from __future__ import annotations

import ctypes
from pathlib import Path

from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.sovereign import loader


class ModelCheckResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("visited_count", ctypes.c_uint32),
        ("frontier_peak", ctypes.c_uint32),
        ("witness_state", ctypes.c_uint32),
    ]


class ModelCheckReuseBridge:
    STATUS_UNKNOWN = 0
    STATUS_PASS = 1
    STATUS_FAIL = 2

    def __init__(self) -> None:
        self._source_path = Path(__file__).resolve().parents[1] / "cuda" / "model_check_reuse.cu"
        self._ptx_path = Path(__file__).resolve().parents[1] / "ptx" / "model_check_reuse.ptx"
        self._ptx_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_ptx()
        self._module = loader.load_module_from_file(str(self._ptx_path))
        self._kernel = loader.get_function(self._module, "k3d_model_check_reuse")

    def _ensure_ptx(self) -> None:
        if self._ptx_path.exists() and self._ptx_path.stat().st_mtime >= self._source_path.stat().st_mtime:
            return
        ptx_text = compile_cuda_file(self._source_path, arch="sm_86", use_fast_math=False)
        self._ptx_path.write_text(ptx_text, encoding="utf-8")

    def run(
        self,
        *,
        state_props: list[int],
        adjacency: list[int],
        num_states: int,
        max_degree: int,
        root_state: int,
        target_mask: int,
        forbidden_mask: int,
    ) -> ModelCheckResult:
        props_arr = (ctypes.c_uint32 * max(1, len(state_props)))(*([*state_props] if state_props else [0]))
        adj_arr = (ctypes.c_uint32 * max(1, len(adjacency)))(*([*adjacency] if adjacency else [0xFFFFFFFF]))
        result_host, result_device = loader.mapped_host_alloc(ctypes.sizeof(ModelCheckResult))
        result = ModelCheckResult.from_address(int(result_host.value))
        result.status = self.STATUS_UNKNOWN
        result.visited_count = 0
        result.frontier_peak = 0
        result.witness_state = 0xFFFFFFFF

        d_props = loader.gpu_malloc(ctypes.sizeof(props_arr))
        d_adj = loader.gpu_malloc(ctypes.sizeof(adj_arr))
        try:
            loader.memcpy_htod(d_props, ctypes.cast(props_arr, ctypes.c_void_p), ctypes.sizeof(props_arr))
            loader.memcpy_htod(d_adj, ctypes.cast(adj_arr, ctypes.c_void_p), ctypes.sizeof(adj_arr))
            loader.launch(
                self._kernel,
                (1, 1, 1),
                (1, 1, 1),
                [
                    ctypes.c_uint64(int(d_props.value)),
                    ctypes.c_uint64(int(d_adj.value)),
                    ctypes.c_uint32(max(0, int(num_states))),
                    ctypes.c_uint32(max(0, int(max_degree))),
                    ctypes.c_uint32(max(0, int(root_state))),
                    ctypes.c_uint32(max(0, int(target_mask))),
                    ctypes.c_uint32(max(0, int(forbidden_mask))),
                    ctypes.c_uint64(int(result_device.value)),
                ],
            )
            loader.synchronize()
            return ModelCheckResult.from_buffer_copy(bytes(ctypes.string_at(int(result_host.value), ctypes.sizeof(ModelCheckResult))))
        finally:
            loader.gpu_free(d_props)
            loader.gpu_free(d_adj)
            loader.mapped_host_free(result_host)


__all__ = ["ModelCheckReuseBridge", "ModelCheckResult"]
