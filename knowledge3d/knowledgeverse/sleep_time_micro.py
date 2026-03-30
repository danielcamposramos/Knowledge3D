"""Micro sleep-time launcher for ARC3 inter-frame consolidation."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

from knowledge3d.cranium.sovereign import loader


CUDA_DIR = Path(__file__).resolve().parents[1] / "cranium" / "cuda"
PTX_DIR = Path(__file__).resolve().parents[1] / "cranium" / "ptx"
CUDA_SOURCE = CUDA_DIR / "sleep_time_micro.cu"
CUDA_HEADER = CUDA_DIR / "device_functions.cuh"
PTX_PATH = PTX_DIR / "sleep_time_micro.ptx"


class SleepTimeMicro:
    """Launch a short consolidation pass between ARC3 frames."""

    def __init__(self) -> None:
        self.kernel = loader.load_ptx_file(str(self.ensure_ptx()), "sleep_time_micro")

    @staticmethod
    def ensure_ptx() -> Path:
        PTX_DIR.mkdir(parents=True, exist_ok=True)
        newest = max(
            CUDA_SOURCE.stat().st_mtime,
            CUDA_HEADER.stat().st_mtime if CUDA_HEADER.exists() else 0.0,
        )
        if PTX_PATH.exists() and PTX_PATH.stat().st_mtime >= newest:
            return PTX_PATH
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("nvcc_not_found_for_sleep_time_micro")
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

    def consolidate(
        self,
        brain_gpu_ptr,
        outcome_signal: int,
        *,
        galaxy_ptr=None,
        chosen_star_index: int = 0,
    ) -> None:
        loader.launch(
            self.kernel,
            (1, 1, 1),
            (128, 1, 1),
            [
                brain_gpu_ptr,
                ctypes.c_int(max(-1, min(1, int(outcome_signal)))),
                galaxy_ptr if galaxy_ptr is not None else ctypes.c_void_p(),
                ctypes.c_uint(int(max(0, chosen_star_index))),
            ],
        )
        loader.synchronize()


__all__ = ["SleepTimeMicro"]
