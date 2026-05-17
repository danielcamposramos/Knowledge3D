"""Projection screen bridge — pure-ctypes launcher over projection_screen.ptx.

Wires opcode 0x2A3 (arc3_project_to_screen) from the ARC3 screen pipeline.
Mirrors the structure of knowledge3d/cranium/bridges/arc3_screen_bridge.py.

Kernel dispatched here:
    screen_project_kernel (0x277) — blit a viewport RGBA buffer into a
        screen framebuffer at a specified rectangle with nearest-neighbor
        scaling.

Sovereignty: zero numpy / cupy / scipy / sympy. All allocation via
knowledge3d.cranium.sovereign.loader (gpu_malloc / memcpy_htod / launch).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from knowledge3d.cranium.sovereign import loader


class ProjectionScreenBridge:
    """Pure-ctypes launcher for the projection screen PTX kernel.

    Usage::

        bridge = ProjectionScreenBridge()
        # viewport_dev: CUdeviceptr to uint8 RGBA[Vw*Vh*4]
        # screen_dev:   CUdeviceptr to uint8 RGBA[Sw*Sh*4] (pre-allocated)
        bridge.project_to_screen(
            viewport_dev, screen_dev,
            Vw=64, Vh=64, Sw=512, Sh=512,
            rect=(0, 0, 256, 256),
        )
    """

    def __init__(self) -> None:
        ptx_path = (
            Path(__file__).parent.parent
            / "codecs"
            / "kernels"
            / "projection_screen.ptx"
        )
        if not ptx_path.exists():
            raise FileNotFoundError(
                f"Projection screen PTX not found at {ptx_path.resolve()}"
            )

        self._module = loader.load_module_from_file(str(ptx_path))

        # Bind the screen projection kernel.
        self._kernels = {
            "screen_project": loader.get_function(
                self._module, "screen_project_kernel"
            ),
        }

    # ------------------------------------------------------------------
    # 0x2A3 / 0x277 — Project viewport to screen
    # ------------------------------------------------------------------

    def project_to_screen(
        self,
        viewport_dev: object,
        screen_dev: object,
        Vw: int,
        Vh: int,
        Sw: int,
        Sh: int,
        rect: tuple,
    ) -> object:
        """Blit viewport RGBA into a sub-rectangle of a screen framebuffer.

        Nearest-neighbor scaling maps viewport pixels to the destination rect.
        The caller pre-allocates the screen buffer; this kernel writes only
        into the specified rectangle.

        Args:
            viewport_dev: CUdeviceptr to uint8 RGBA buffer of Vw*Vh*4 bytes.
            screen_dev:   CUdeviceptr to uint8 RGBA buffer of Sw*Sh*4 bytes.
                          Must be pre-allocated by caller.
            Vw, Vh:       Viewport dimensions in pixels.
            Sw, Sh:       Screen framebuffer dimensions in pixels.
            rect:         (rect_x, rect_y, rect_w, rect_h) — destination
                          rectangle in screen coordinates.

        Returns:
            screen_dev (same pointer, written in-place).
        """
        rect_x, rect_y, rect_w, rect_h = rect

        bx = (rect_w + 15) // 16
        by = (rect_h + 15) // 16
        loader.launch(
            self._kernels["screen_project"],
            grid=(bx, by, 1),
            block=(16, 16, 1),
            params=[
                ctypes.c_uint64(viewport_dev.value),
                ctypes.c_uint64(screen_dev.value),
                ctypes.c_int(int(Vw)),
                ctypes.c_int(int(Vh)),
                ctypes.c_int(int(Sw)),
                ctypes.c_int(int(Sh)),
                ctypes.c_int(int(rect_x)),
                ctypes.c_int(int(rect_y)),
                ctypes.c_int(int(rect_w)),
                ctypes.c_int(int(rect_h)),
            ],
        )
        loader.synchronize()
        return screen_dev


__all__ = ["ProjectionScreenBridge"]
