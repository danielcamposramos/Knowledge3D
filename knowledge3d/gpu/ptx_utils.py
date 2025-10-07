from __future__ import annotations

"""
Lightweight PTX/CuPy utilities used by the fused-head pipeline.

The helpers are intentionally minimal – they only wrap CuPy's ``RawModule`` and
cache modules between launches so repeated calls remain cheap.  All callers are
expected to pass CuPy-compatible device pointers (e.g. ``cupy.ndarray.data`` or
``torch.cuda.Tensor.data_ptr()``).
"""

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

try:
    import cupy as cp  # type: ignore
except Exception as exc:  # pragma: no cover - CuPy unavailable
    raise RuntimeError(
        "CuPy is required for GPU PTX helpers. Install cupy-cudaXX inside the "
        "k3d GPU environment."
    ) from exc

Grid = Tuple[int, int, int]
Block = Tuple[int, int, int]


@lru_cache(maxsize=32)
def _load_module(cubin_path: str) -> cp.RawModule:
    path = Path(cubin_path)
    if not path.exists():
        raise FileNotFoundError(f"PTX/CUBIN file not found: {path}")
    return cp.RawModule(path=str(path))


def launch_ptx_kernel(
    cubin_path: str,
    kernel_name: str,
    *kernel_args: Iterable,
    grid: Grid = (1, 1, 1),
    block: Block = (32, 1, 1),
    stream: Optional[cp.cuda.Stream] = None,
) -> None:
    """
    Launch a PTX kernel stored in ``cubin_path``.

    Args:
        cubin_path: Path to the compiled PTX/CUBIN file.
        kernel_name: Symbol exported by the module.
        kernel_args: Kernel arguments (device pointers / scalars).
        grid: CUDA grid dimensions (default 1×1×1).
        block: CUDA block dimensions (default 32×1×1).
        stream: Optional CuPy stream for asynchronous launches.
    """

    module = _load_module(cubin_path)
    kernel = module.get_function(kernel_name)
    args = tuple(kernel_args)

    if stream is None:
        kernel(grid, block, args)
    else:
        kernel(grid, block, args, stream=stream)

