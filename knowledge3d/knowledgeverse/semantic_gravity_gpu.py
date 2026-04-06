"""GPU semantic gravity tick over the VRAM-resident star table."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

from knowledge3d.cranium.sovereign import loader


CUDA_DIR = Path(__file__).resolve().parents[1] / "cranium" / "cuda"
PTX_DIR = Path(__file__).resolve().parents[1] / "cranium" / "ptx"
CUDA_SOURCE = CUDA_DIR / "semantic_gravity_tick.cu"
CUDA_HEADER = CUDA_DIR / "device_functions.cuh"
PTX_PATH = PTX_DIR / "semantic_gravity_tick.ptx"


class SemanticGravityGPU:
    """Thin sovereign wrapper for in-place semantic gravity drift on the star table."""

    def __init__(self) -> None:
        self.kernel = loader.load_ptx_file(str(self.ensure_ptx()), "semantic_gravity_tick")

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
            raise RuntimeError("nvcc_not_found_for_semantic_gravity_tick")
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

    def evolve_table(
        self,
        star_table,
        *,
        dt: float = 0.01,
        damping: float = 0.95,
    ) -> dict[str, int | str]:
        galaxy_ptr = getattr(star_table, "gpu_ptr", None) if star_table is not None else None
        total = max(0, int(getattr(star_table, "star_count", 0) or 0)) if star_table is not None else 0
        if galaxy_ptr is None or total <= 0:
            return {"status": "noop", "updated_rows": 0}
        loader.launch(
            self.kernel,
            ((total + 127) // 128, 1, 1),
            (128, 1, 1),
            [
                galaxy_ptr,
                ctypes.c_uint(total),
                getattr(star_table, "router_offsets_ptr", ctypes.c_void_p()),
                getattr(star_table, "router_counts_ptr", ctypes.c_void_p()),
                getattr(star_table, "executor_offsets_ptr", ctypes.c_void_p()),
                getattr(star_table, "executor_counts_ptr", ctypes.c_void_p()),
                getattr(star_table, "validator_offsets_ptr", ctypes.c_void_p()),
                getattr(star_table, "validator_counts_ptr", ctypes.c_void_p()),
                getattr(star_table, "anti_pattern_offsets_ptr", ctypes.c_void_p()),
                getattr(star_table, "anti_pattern_counts_ptr", ctypes.c_void_p()),
                getattr(star_table, "ref_indices_ptr", ctypes.c_void_p()),
                ctypes.c_float(float(dt)),
                ctypes.c_float(float(damping)),
            ],
        )
        loader.synchronize()
        return {"status": "ok", "updated_rows": total}


__all__ = ["SemanticGravityGPU"]
