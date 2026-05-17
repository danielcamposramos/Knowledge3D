"""DotMap codec bridge — pure-ctypes launcher over dotmap_codec.ptx.

Wires opcode 0x2A2 (arc3_frame_to_dotmap) from the ARC3 screen pipeline.
Mirrors the structure of knowledge3d/cranium/bridges/arc3_screen_bridge.py.

Kernel dispatched here:
    dot_place_procedural (0x217) — content-adaptive dot placement using
        inverse-CDF golden-ratio quasi-random sampling.

The bridge accepts a float density map on GPU (f32 [W*H]) plus a desired
dot count, then returns a float dot-coordinate buffer (f32 [target_dots*2])
with the actual dot count written back.

Sovereignty: zero numpy / cupy / scipy / sympy. All allocation via
knowledge3d.cranium.sovereign.loader (gpu_malloc / memcpy_htod / launch).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from knowledge3d.cranium.sovereign import loader


class DotMapBridge:
    """Pure-ctypes launcher for the DotMap codec PTX kernel.

    Usage::

        bridge = DotMapBridge()
        # density_dev: CUdeviceptr to f32[W*H] importance map
        dots_dev, actual_count = bridge.frame_to_dotmap(
            density_dev, total_mass=float(W*H), target_dots=1024, W=64, H=64
        )
        # dots_dev: CUdeviceptr to f32[actual_count*2] (x,y pairs)
        # caller frees dots_dev with loader.gpu_free()
    """

    def __init__(self) -> None:
        ptx_path = (
            Path(__file__).parent.parent
            / "codecs"
            / "kernels"
            / "dotmap_codec.ptx"
        )
        if not ptx_path.exists():
            raise FileNotFoundError(
                f"DotMap codec PTX not found at {ptx_path.resolve()}"
            )

        self._module = loader.load_module_from_file(str(ptx_path))

        # Bind the dot placement kernel.
        self._kernels = {
            "dot_place": loader.get_function(
                self._module, "dot_place_procedural"
            ),
        }

    # ------------------------------------------------------------------
    # 0x2A2 / 0x217 — Frame to DotMap
    # ------------------------------------------------------------------

    def frame_to_dotmap(
        self,
        density_dev: object,
        total_mass: float,
        target_dots: int,
        W: int,
        H: int,
    ) -> tuple:
        """Run content-adaptive dot placement on a density map.

        Args:
            density_dev: CUdeviceptr to f32[W*H] per-cell importance values.
            total_mass:  precomputed sum(density) — avoids a reduction pass.
            target_dots: desired number of dots to place.
            W:           source grid width (content-driven, not display res).
            H:           source grid height.

        Returns:
            (dots_dev, actual_count) where:
              dots_dev     — CUdeviceptr to f32[actual_count * 2] (x,y pairs).
              actual_count — int, equals target_dots on success.
            Caller must free dots_dev with loader.gpu_free().
        """
        # Allocate output: [target_dots * 2] f32 for (x, y) coordinates.
        dots_dev = loader.gpu_malloc(target_dots * 2 * ctypes.sizeof(ctypes.c_float))
        # Allocate scalar dot_count output.
        count_dev = loader.gpu_malloc(ctypes.sizeof(ctypes.c_int32))

        try:
            # Grid: one thread per dot.
            block = 256
            grid = (target_dots + block - 1) // block
            loader.launch(
                self._kernels["dot_place"],
                grid=(grid, 1, 1),
                block=(block, 1, 1),
                params=[
                    ctypes.c_uint64(density_dev.value),
                    ctypes.c_uint64(dots_dev.value),
                    ctypes.c_uint64(count_dev.value),
                    ctypes.c_float(float(total_mass)),
                    ctypes.c_int(int(target_dots)),
                    ctypes.c_int(int(W)),
                    ctypes.c_int(int(H)),
                ],
            )
            loader.synchronize()

            # Read back the actual dot count.
            h_count = ctypes.c_int32(0)
            loader.memcpy_dtoh(
                ctypes.cast(ctypes.byref(h_count), ctypes.c_void_p),
                count_dev,
                ctypes.sizeof(ctypes.c_int32),
            )
            actual = int(h_count.value)
        finally:
            loader.gpu_free(count_dev)

        return dots_dev, actual


__all__ = ["DotMapBridge"]
