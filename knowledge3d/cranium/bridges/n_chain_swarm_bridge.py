"""Persistent cooperative N-chain swarm bridge.

This bridge only pumps control words and launches the sovereign CUDA kernel.
Lane selection, halting, and cross-lane work remain in the GPU kernel.
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Any

from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.bridges.sleep_perf_consumer_bridge import (
    SwarmPerfCalibration,
)


class SwarmKernelControl(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_uint32),
        ("halting_counter", ctypes.c_uint32),
        ("tick_epoch", ctypes.c_uint32),
        ("halt_epoch", ctypes.c_uint32),
    ]


class SwarmTickControl(ctypes.Structure):
    _fields_ = [
        ("vram_free_mib", ctypes.c_uint32),
        ("t_remaining_us", ctypes.c_uint32),
        ("n_cand_frustum", ctypes.c_uint32),
        ("h_belief_q10", ctypes.c_uint32),
        ("n_floor", ctypes.c_uint32),
        ("n_hard_max", ctypes.c_uint32),
        ("sleep_calibration_n_hint", ctypes.c_uint32),
        ("paradigm_mask", ctypes.c_uint32),
    ]


class ReasoningLaneOutput(ctypes.Structure):
    _fields_ = [
        ("halt_flag", ctypes.c_uint32),
        ("result_handle", ctypes.c_uint32),
        ("belief_q15", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),
        ("payload", ctypes.c_uint8 * 48),
    ]


class NChainSwarmBridge:
    """Bridge for `k3d_swarm_sovereign` with dynamic N selection."""

    N_HARD_MAX = 1024
    N_FLOOR = 1
    N_DEFAULT = 9
    LANE_PERF_RING_ENTRIES = 1 << 20
    LANE_PERF_RECORD_BYTES = 32
    LANE_OUTPUT_RECORD_BYTES = 64
    GALAXY_ATLAS_BYTES = 4096
    GRID = (256, 1, 1)
    BLOCK = (128, 1, 1)
    SHARED_BYTES = 0
    FLAG_RUN = 0
    FLAG_COMPLETE = 1
    FLAG_SHUTDOWN = 2

    def __init__(self) -> None:
        self._source_path = Path(__file__).resolve().parents[1] / "cuda" / "k3d_swarm_persistent.cu"
        self._ptx_path = Path(__file__).resolve().parents[1] / "ptx" / "k3d_swarm_persistent.ptx"
        self._ptx_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_ptx()
        self._module = loader.load_module_from_file(str(self._ptx_path))
        self._kernel = loader.get_function(self._module, "k3d_swarm_sovereign")

        self._kernel_control_host, self._kernel_control_device = loader.mapped_host_alloc(ctypes.sizeof(SwarmKernelControl))
        self._kernel_control = SwarmKernelControl.from_address(int(self._kernel_control_host.value))
        self._kernel_control.state = self.FLAG_COMPLETE
        self._kernel_control.halting_counter = 0
        self._kernel_control.tick_epoch = 0
        self._kernel_control.halt_epoch = 0

        self._tick_control_host, self._tick_control_device = loader.mapped_host_alloc(ctypes.sizeof(SwarmTickControl))
        self._tick_control = SwarmTickControl.from_address(int(self._tick_control_host.value))
        self._tick_control.vram_free_mib = 0
        self._tick_control.t_remaining_us = 20_000
        self._tick_control.n_cand_frustum = self.N_DEFAULT
        self._tick_control.h_belief_q10 = 0
        self._tick_control.n_floor = self.N_FLOOR
        self._tick_control.n_hard_max = self.N_HARD_MAX
        self._tick_control.sleep_calibration_n_hint = 0
        self._tick_control.paradigm_mask = 0

        self._d_n_active_host, self._d_n_active = loader.mapped_host_alloc(4)
        self._n_active = ctypes.c_uint32.from_address(int(self._d_n_active_host.value))
        self._n_active.value = self.N_DEFAULT
        self._d_halting_counter = loader.gpu_malloc(4)
        loader.memset_d32(self._d_halting_counter, 0, 1)
        # First-class halting scalar readback (PTX writes Q15 fixed-point
        # max-belief into this scalar at the halt-flip; host divides by
        # 32768.0 to yield a float in [0.0, 1.0]).
        # See TEMP/CLAUDE_HALTING_READBACK_HOOK_SPEC_04.21.2026.md §3.3
        self._d_halting_value_q15_host, self._d_halting_value_q15 = loader.mapped_host_alloc(4)
        self._halting_value_q15 = ctypes.c_uint32.from_address(
            int(self._d_halting_value_q15_host.value)
        )
        self._halting_value_q15.value = 0

        self._calibration_host, self._calibration_device = loader.mapped_host_alloc(ctypes.sizeof(SwarmPerfCalibration))
        self._calibration = SwarmPerfCalibration.from_address(int(self._calibration_host.value))
        self._zero_calibration()

        self._d_perf_ring = loader.gpu_malloc(self.LANE_PERF_RING_ENTRIES * self.LANE_PERF_RECORD_BYTES)
        self._d_perf_ring_head = loader.gpu_malloc(4)
        loader.memset_d32(self._d_perf_ring_head, 0, 1)
        lane_output_bytes = self.N_HARD_MAX * ctypes.sizeof(ReasoningLaneOutput)
        self._lane_outputs_host, self._lane_outputs_device = loader.mapped_host_alloc(lane_output_bytes)
        self._lane_outputs = (ReasoningLaneOutput * self.N_HARD_MAX).from_address(int(self._lane_outputs_host.value))
        ctypes.memset(int(self._lane_outputs_host.value), 0, lane_output_bytes)

        self._galaxy_atlas_host, self._d_galaxy_atlas = loader.mapped_host_alloc(self.GALAXY_ATLAS_BYTES)
        self._galaxy_atlas_capacity = self.GALAXY_ATLAS_BYTES
        ctypes.memset(int(self._galaxy_atlas_host.value), 0, self.GALAXY_ATLAS_BYTES)
        self._launched = False
        self._cached_vram_free_mib = 4096
        self._refresh_vram_cache()

    def _zero_calibration(self) -> None:
        self._calibration.n_hint = 0
        self._calibration.sample_count_total = 0
        self._calibration.last_tick_epoch = 0
        self._calibration.utility_peak_q20 = 0
        for index in range(16):
            self._calibration.bucket_samples[index] = 0
            self._calibration.bucket_utility_q20[index] = 0
        for index in range(4):
            self._calibration._pad[index] = 0

    def _refresh_vram_cache(self) -> None:
        try:
            loader.ensure_init()
            used_bytes, total_bytes = loader.get_vram_usage()
            self._cached_vram_free_mib = max(1, (int(total_bytes) - int(used_bytes)) >> 20)
        except Exception:
            pass

    def _ensure_ptx(self) -> None:
        newest_source = max(
            self._source_path.stat().st_mtime,
            (self._source_path.parent / "halting_gate.cu").stat().st_mtime,
            (self._source_path.parent / "n_selector.cu").stat().st_mtime,
            (self._source_path.parent / "reasoning_tick_io.cuh").stat().st_mtime,
            (self._source_path.parent / "reasoning_tick_entrypoints.cuh").stat().st_mtime,
            (self._source_path.parent / "swarm_perf_calibration_reader.cuh").stat().st_mtime,
        )
        if self._ptx_path.exists() and self._ptx_path.stat().st_mtime >= newest_source:
            return
        ptx_text = compile_cuda_file(
            self._source_path,
            arch="sm_86",
            use_fast_math=False,
            extra_nvcc_flags=["-DK3D_REASONING_OPCODES_V1"],
        )
        self._ptx_path.write_text(ptx_text, encoding="utf-8")

    @property
    def n_active(self) -> int:
        return int(self._n_active.value)

    @property
    def halting_flag(self) -> int:
        return int(self._kernel_control.state)

    @property
    def tick_epoch(self) -> int:
        return int(self._kernel_control.tick_epoch)

    @property
    def halt_epoch(self) -> int:
        return int(self._kernel_control.halt_epoch)

    def read_lane_output(self, lane_index: int = 0) -> dict[str, int]:
        lane = self._lane_outputs[int(lane_index)]
        payload_words = ctypes.cast(lane.payload, ctypes.POINTER(ctypes.c_uint32))
        return {
            "halt_flag": int(lane.halt_flag),
            "result_handle": int(lane.result_handle),
            "belief_q15": int(lane.belief_q15),
            "payload0": int(payload_words[0]),
            "payload1": int(payload_words[1]),
            "payload2": int(payload_words[2]),
            "payload3": int(payload_words[3]),
        }

    def _load_galaxy_atlas(self, payload: bytes | bytearray | memoryview | None) -> None:
        raw = bytes(payload or b"\x00")
        if not raw:
            raw = b"\x00"
        if len(raw) > self._galaxy_atlas_capacity:
            raise ValueError(f"galaxy_atlas_too_large:{len(raw)}>{self._galaxy_atlas_capacity}")
        ctypes.memset(int(self._galaxy_atlas_host.value), 0, self._galaxy_atlas_capacity)
        ctypes.memmove(int(self._galaxy_atlas_host.value), raw, len(raw))

    def read_calibration(self) -> dict[str, Any]:
        return {
            "n_hint": int(self._calibration.n_hint),
            "sample_count_total": int(self._calibration.sample_count_total),
            "last_tick_epoch": int(self._calibration.last_tick_epoch),
            "utility_peak_q20": int(self._calibration.utility_peak_q20),
            "bucket_samples": [int(self._calibration.bucket_samples[index]) for index in range(16)],
            "bucket_utility_q20": [int(self._calibration.bucket_utility_q20[index]) for index in range(16)],
        }

    def read_perf_ring_head(self) -> int:
        payload = (ctypes.c_uint32 * 1)()
        loader.memcpy_dtoh(ctypes.cast(payload, ctypes.c_void_p), self._d_perf_ring_head, ctypes.sizeof(payload))
        return int(payload[0])

    def launch(self) -> None:
        """Launch the persistent cooperative kernel once."""
        if self._launched:
            return
        self._kernel_control.state = self.FLAG_COMPLETE
        self._kernel_control.halting_counter = 0
        self._kernel_control.tick_epoch = 0
        self._kernel_control.halt_epoch = 0
        self._n_active.value = self.N_DEFAULT
        loader.memset_d32(self._d_halting_counter, 0, 1)
        self._halting_value_q15.value = 0
        loader.launch_cooperative(
            self._kernel,
            grid=self.GRID,
            block=self.BLOCK,
            shared_mem=self.SHARED_BYTES,
            params=[
                ctypes.c_uint64(int(self._d_galaxy_atlas.value)),
                ctypes.c_uint64(int(self._kernel_control_device.value)),
                ctypes.c_uint64(int(self._tick_control_device.value)),
                ctypes.c_uint64(int(self._d_halting_counter.value)),
                ctypes.c_uint64(int(self._d_n_active.value)),
                ctypes.c_uint64(int(self._lane_outputs_device.value)),
                ctypes.c_uint64(int(self._d_perf_ring.value)),
                ctypes.c_uint64(int(self._d_perf_ring_head.value)),
                ctypes.c_uint32(self.LANE_PERF_RING_ENTRIES - 1),
                ctypes.c_uint64(int(self._calibration_device.value)),
                ctypes.c_uint64(int(self._d_halting_value_q15.value)),
            ],
        )
        self._launched = True

    def tick(self, work_packet: Any | None = None, *, timeout_s: float = 2.0) -> dict[str, int]:
        """Submit one sovereign tick and wait for the GPU halting flag."""
        if not self._launched:
            self.launch()
        self._refresh_vram_cache()
        free_mib = self._cached_vram_free_mib
        packet = work_packet if isinstance(work_packet, dict) else {}
        self._tick_control.vram_free_mib = free_mib
        self._tick_control.t_remaining_us = max(48, int(packet.get("t_remaining_us", 20_000)))
        self._tick_control.n_cand_frustum = max(self.N_FLOOR, int(packet.get("n_cand_frustum", self.N_DEFAULT)))
        self._tick_control.h_belief_q10 = max(0, int(packet.get("h_belief_q10", 0)))
        self._tick_control.n_floor = max(self.N_FLOOR, int(packet.get("n_floor", self.N_FLOOR)))
        self._tick_control.n_hard_max = max(self._tick_control.n_floor, min(self.N_HARD_MAX, int(packet.get("n_hard_max", self.N_HARD_MAX))))
        self._tick_control.sleep_calibration_n_hint = int(self._calibration.n_hint)
        self._tick_control.paradigm_mask = int(packet.get("paradigm_mask", self._tick_control.paradigm_mask))
        self._load_galaxy_atlas(packet.get("galaxy_atlas"))
        ctypes.memset(int(self._lane_outputs_host.value), 0, self.N_HARD_MAX * ctypes.sizeof(ReasoningLaneOutput))
        self._n_active.value = self.N_DEFAULT
        self._kernel_control.halting_counter = 0
        loader.memset_d32(self._d_halting_counter, 0, 1)
        self._halting_value_q15.value = 0
        halt_epoch_before = int(self._kernel_control.halt_epoch)
        self._kernel_control.tick_epoch += 1
        self._kernel_control.state = self.FLAG_RUN
        deadline = time.perf_counter() + float(timeout_s)
        while int(self._kernel_control.state) != self.FLAG_COMPLETE:
            if time.perf_counter() >= deadline:
                raise TimeoutError("n_chain_swarm_tick_timeout")
            time.sleep(0.0005)
        settle_deadline = time.perf_counter() + 0.1
        while int(self._kernel_control.halt_epoch) <= halt_epoch_before and time.perf_counter() < settle_deadline:
            time.sleep(0.0005)
        self._kernel_control.halting_counter = int(self._n_active.value)
        # First-class halting scalar: PTX wrote Q15 max-belief at halt-flip.
        # See TEMP/CLAUDE_HALTING_READBACK_HOOK_SPEC_04.21.2026.md §2, §3.1, §3.3
        halting_value = float(int(self._halting_value_q15.value)) / 32768.0
        if halting_value < 0.0:
            halting_value = 0.0
        elif halting_value > 1.0:
            halting_value = 1.0
        return {
            "halting_flag": int(self._kernel_control.state),
            "halting_counter": int(self._kernel_control.halting_counter),
            "n_active": int(self._n_active.value),
            "halting_value": halting_value,
            "tick_epoch": int(self._kernel_control.tick_epoch),
            "halt_epoch": int(self._kernel_control.halt_epoch),
            "calibration_hint": int(self._calibration.n_hint),
        }

    def shutdown(self) -> None:
        """Signal the persistent kernel to exit and release buffers."""
        if self._launched:
            self._kernel_control.state = self.FLAG_SHUTDOWN
            loader.synchronize()
            self._launched = False

    def cleanup(self) -> None:
        self.shutdown()
        if getattr(self, "_d_galaxy_atlas", None):
            loader.mapped_host_free(self._galaxy_atlas_host)
            self._d_galaxy_atlas = None
            self._galaxy_atlas_host = None
        if getattr(self, "_d_perf_ring", None):
            loader.gpu_free(self._d_perf_ring)
            self._d_perf_ring = None
        if getattr(self, "_d_perf_ring_head", None):
            loader.gpu_free(self._d_perf_ring_head)
            self._d_perf_ring_head = None
        if getattr(self, "_lane_outputs_host", None):
            loader.mapped_host_free(self._lane_outputs_host)
            self._lane_outputs_host = None
        if getattr(self, "_d_halting_counter", None):
            loader.gpu_free(self._d_halting_counter)
            self._d_halting_counter = None
        if getattr(self, "_d_halting_value_q15_host", None):
            loader.mapped_host_free(self._d_halting_value_q15_host)
            self._d_halting_value_q15_host = None
            self._d_halting_value_q15 = None
        if getattr(self, "_kernel_control_host", None):
            loader.mapped_host_free(self._kernel_control_host)
            self._kernel_control_host = None
        if getattr(self, "_tick_control_host", None):
            loader.mapped_host_free(self._tick_control_host)
            self._tick_control_host = None
        if getattr(self, "_d_n_active_host", None):
            loader.mapped_host_free(self._d_n_active_host)
            self._d_n_active_host = None
        if getattr(self, "_calibration_host", None):
            loader.mapped_host_free(self._calibration_host)
            self._calibration_host = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.cleanup()
        except Exception:
            pass


__all__ = ["NChainSwarmBridge", "SwarmKernelControl", "SwarmTickControl"]
