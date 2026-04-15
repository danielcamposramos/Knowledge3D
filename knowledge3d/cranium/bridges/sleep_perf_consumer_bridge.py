"""Sleep-time sovereign consumer for the lane perf ring."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess
from typing import Any

from knowledge3d.cranium.sovereign import loader


class SwarmPerfCalibration(ctypes.Structure):
    _fields_ = [
        ("n_hint", ctypes.c_uint32),
        ("sample_count_total", ctypes.c_uint32),
        ("last_tick_epoch", ctypes.c_uint32),
        ("utility_peak_q20", ctypes.c_uint32),
        ("bucket_samples", ctypes.c_uint32 * 16),
        ("bucket_utility_q20", ctypes.c_uint32 * 16),
        ("_pad", ctypes.c_uint32 * 4),
    ]


CUDA_DIR = Path(__file__).resolve().parents[1] / "cuda"
PTX_DIR = Path(__file__).resolve().parents[1] / "ptx"
CUDA_SOURCE = CUDA_DIR / "sleep_perf_consumer.cu"
CUDA_HEADERS = (
    CUDA_DIR / "lane_perf_ring.cu",
    CUDA_DIR / "swarm_perf_calibration_reader.cuh",
)
PTX_PATH = PTX_DIR / "sleep_perf_consumer.ptx"


class SleepPerfConsumerBridge:
    """Launch the sovereign sleep perf consumer kernel."""

    def __init__(self) -> None:
        self.kernel = loader.load_ptx_file(str(self.ensure_ptx()), "k3d_sleep_perf_consume")

    @staticmethod
    def ensure_ptx() -> Path:
        PTX_DIR.mkdir(parents=True, exist_ok=True)
        newest = max([CUDA_SOURCE.stat().st_mtime, *[header.stat().st_mtime for header in CUDA_HEADERS]])
        if PTX_PATH.exists() and PTX_PATH.stat().st_mtime >= newest:
            return PTX_PATH
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("nvcc_not_found_for_sleep_perf_consumer")
        subprocess.run(
            [
                nvcc,
                "-ptx",
                "-arch=sm_86",
                "--compiler-bindir",
                "/usr/bin/gcc-13",
                "-o",
                str(PTX_PATH),
                str(CUDA_SOURCE),
            ],
            check=True,
        )
        return PTX_PATH

    def consume(
        self,
        *,
        perf_ring_ptr: Any,
        ring_size: int,
        ring_head: int,
        calibration_ptr: Any,
    ) -> None:
        loader.launch(
            self.kernel,
            (1, 1, 1),
            (1024, 1, 1),
            [
                perf_ring_ptr,
                ctypes.c_uint32(max(0, int(ring_size))),
                ctypes.c_uint32(max(0, int(ring_head))),
                calibration_ptr,
            ],
        )
        loader.synchronize()


__all__ = ["SleepPerfConsumerBridge", "SwarmPerfCalibration"]
