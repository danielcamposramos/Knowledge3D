"""Bridge for the OOM spill planning kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cupy as cp

from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()


class SpillPlanner:
    """Compute spill plans using the gre_oom_spill PTX kernel."""

    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "kernels" / "gre_oom_spill.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX kernel: {ptx_path}")
        self.module = cp.RawModule(path=str(ptx_path))
        self.kernel = self.module.get_function("gre_oom_spill")

    def plan(
        self,
        oldest_index: int,
        atom_size_bytes: int,
        available_bytes: int,
        request_count: int,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> Tuple[int, int]:
        """Return (atoms_to_spill, bytes_required)."""

        stats = cp.asarray([oldest_index, atom_size_bytes], dtype=cp.uint64)
        out = cp.zeros(2, dtype=cp.uint64)

        self.kernel(
            (1,),
            (32,),
            (
                stats.data.ptr,
                cp.uint64(available_bytes),
                cp.uint32(request_count),
                out.data.ptr,
            ),
            stream=stream,
        )

        if stream is not None:
            stream.synchronize()

        atoms, bytes_required = out.get()
        return int(atoms), int(bytes_required)


__all__ = ["SpillPlanner"]
