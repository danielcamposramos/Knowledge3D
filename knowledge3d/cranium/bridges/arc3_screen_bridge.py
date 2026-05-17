"""ARC3 screen bridge — pure-ctypes launcher over arc3_screen_bridge.ptx.

Wires opcodes 0x2A0-0x2A9 from the ARC3 screen pipeline. Mirrors the
structure of knowledge3d/cranium/codecs/ternary_codec_ops.py.

Five novel GPU kernels bound here:
    arc3_frame_decode_kernel   (0x2A0) — palette idx → RGBA raster
    arc3_click_invert_kernel   (0x2A4) — screen pixel → grid cell
    arc3_action_emit_kernel    (0x2A5) — grid cell + action id → record
    arc3_diff_highlight_kernel (0x2A7) — XOR changed-cell RGBA overlay
    arc3_lives_hud_kernel      (0x2A8) — HUD strip (lives + moves bar)

Delegating opcodes (no GPU kernel here — dispatched by execute_codec):
    0x2A1 arc3_palette_set     → constant-memory upload via get_global
    0x2A2 arc3_frame_to_dotmap → dotmap_codec.cu::dot_place_procedural
    0x2A3 arc3_project_to_screen → projection_screen.cu::screen_project_kernel
    0x2A6 arc3_replay_step     → host-side frame index advance + decode
    0x2A9 arc3_game_id_bind    → host-side hash store (no GPU state needed)

Sovereignty: zero numpy / cupy / scipy / sympy. All allocation via
knowledge3d.cranium.sovereign.loader (gpu_malloc / memcpy_htod / launch).
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional, Tuple

from knowledge3d.cranium.sovereign import loader


# ---------------------------------------------------------------------------
# Bridge class
# ---------------------------------------------------------------------------

class Arc3ScreenBridge:
    """Pure-ctypes launcher for ARC3 screen pipeline PTX kernels.

    Usage::

        bridge = Arc3ScreenBridge()
        bridge.upload_palette([0xFF0000FF, 0x00FF00FF, ...])   # 16 RGBA entries
        rgba_dev = bridge.decode_frame(frame_idx_dev, W=64, H=64)
        cell_gx, cell_gy = bridge.invert_click(sx, sy, rect=(0, 0, 512, 512), grid=(4, 4))
    """

    # ARC3 defines 16 discrete palette entries (per arc3_frame_encoder.cu:5).
    PALETTE_SIZE = 16

    def __init__(self) -> None:
        ptx_path = (
            Path(__file__).parent.parent
            / "codecs"
            / "kernels"
            / "arc3_screen_bridge.ptx"
        )
        if not ptx_path.exists():
            raise FileNotFoundError(
                f"ARC3 screen bridge PTX not found at {ptx_path.resolve()}"
            )

        self._module = loader.load_module_from_file(str(ptx_path))

        # Bind the 5 novel kernels.
        self._kernels = {
            "frame_decode": loader.get_function(
                self._module, "arc3_frame_decode_kernel"
            ),
            "click_invert": loader.get_function(
                self._module, "arc3_click_invert_kernel"
            ),
            "action_emit": loader.get_function(
                self._module, "arc3_action_emit_kernel"
            ),
            "diff_highlight": loader.get_function(
                self._module, "arc3_diff_highlight_kernel"
            ),
            "lives_hud": loader.get_function(
                self._module, "arc3_lives_hud_kernel"
            ),
        }

        # Fetch constant-memory symbol pointer for palette upload.
        # get_global returns (CUdeviceptr, size_in_bytes).
        self._palette_ptr, self._palette_size_bytes = loader.get_global(
            self._module, "c_arc3_palette"
        )

        # Current game id (host-side string, no GPU state required).
        self._game_id: Optional[str] = None

        # Replay: current frame index tracked host-side.
        self._replay_frame_idx: int = 0

    # ------------------------------------------------------------------
    # 0x2A1 — Palette upload (constant memory)
    # ------------------------------------------------------------------

    def upload_palette(self, rgba_entries: list) -> None:
        """Upload 16 packed RGBA uint32 entries into c_arc3_palette constant mem.

        Each entry is packed as (R << 0) | (G << 8) | (B << 16) | (A << 24),
        matching the packing in arc3_frame_decode_kernel.

        Args:
            rgba_entries: list of exactly 16 ints (packed uint32 RGBA).
        """
        if len(rgba_entries) != self.PALETTE_SIZE:
            raise ValueError(
                f"Palette must have exactly {self.PALETTE_SIZE} entries, "
                f"got {len(rgba_entries)}"
            )
        Uint32Array = ctypes.c_uint32 * self.PALETTE_SIZE
        h_palette = Uint32Array(*[int(v) & 0xFFFFFFFF for v in rgba_entries])
        expected_bytes = self.PALETTE_SIZE * ctypes.sizeof(ctypes.c_uint32)
        if self._palette_size_bytes != expected_bytes:
            raise RuntimeError(
                f"c_arc3_palette symbol size mismatch: PTX reports "
                f"{self._palette_size_bytes} bytes, expected {expected_bytes}"
            )
        # Write directly into constant memory via memcpy_htod to the symbol ptr.
        loader.memcpy_htod(
            self._palette_ptr,
            ctypes.cast(h_palette, ctypes.c_void_p),
            expected_bytes,
        )

    # ------------------------------------------------------------------
    # 0x2A0 — Frame decode
    # ------------------------------------------------------------------

    def decode_frame(
        self,
        frame_idx_dev: object,
        W: int,
        H: int,
    ) -> object:
        """Decode a palette-indexed ARC3 frame to RGBA raster on GPU.

        Args:
            frame_idx_dev: CUdeviceptr to a uint8 buffer of W*H palette indices.
            W: frame width in pixels (typically 64).
            H: frame height in pixels (typically 64).

        Returns:
            CUdeviceptr to a uint8 RGBA buffer of W*H*4 bytes.
            Caller is responsible for freeing with loader.gpu_free().
        """
        rgba_dev = loader.gpu_malloc(W * H * 4 * ctypes.sizeof(ctypes.c_uint8))
        bx = (W + 15) // 16
        by = (H + 15) // 16
        loader.launch(
            self._kernels["frame_decode"],
            grid=(bx, by, 1),
            block=(16, 16, 1),
            params=[
                ctypes.c_uint64(frame_idx_dev.value),
                ctypes.c_uint64(rgba_dev.value),
                ctypes.c_int(W),
                ctypes.c_int(H),
            ],
        )
        loader.synchronize()
        return rgba_dev

    # ------------------------------------------------------------------
    # 0x2A4 — Click invert
    # ------------------------------------------------------------------

    def invert_click(
        self,
        sx: int,
        sy: int,
        rect: Tuple[int, int, int, int],
        grid: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Map a screen pixel to a grid cell (inverse of screen projection).

        Args:
            sx, sy: screen pixel coordinates.
            rect: (rect_x, rect_y, rect_w, rect_h) bounding rectangle.
            grid: (grid_w, grid_h) number of grid cells.

        Returns:
            (gx, gy) grid cell coordinates. Returns (-1, -1) for out-of-rect clicks.
        """
        rect_x, rect_y, rect_w, rect_h = rect
        grid_w, grid_h = grid
        d_cell = loader.gpu_malloc(2 * ctypes.sizeof(ctypes.c_int32))
        try:
            loader.launch(
                self._kernels["click_invert"],
                grid=(1, 1, 1),
                block=(1, 1, 1),
                params=[
                    ctypes.c_int(sx),
                    ctypes.c_int(sy),
                    ctypes.c_int(rect_x),
                    ctypes.c_int(rect_y),
                    ctypes.c_int(rect_w),
                    ctypes.c_int(rect_h),
                    ctypes.c_int(grid_w),
                    ctypes.c_int(grid_h),
                    ctypes.c_uint64(d_cell.value),
                ],
            )
            loader.synchronize()
            Int2 = ctypes.c_int32 * 2
            h_cell = Int2()
            loader.memcpy_dtoh(
                ctypes.cast(h_cell, ctypes.c_void_p),
                d_cell,
                2 * ctypes.sizeof(ctypes.c_int32),
            )
            return int(h_cell[0]), int(h_cell[1])
        finally:
            loader.gpu_free(d_cell)

    # ------------------------------------------------------------------
    # 0x2A5 — Action emit
    # ------------------------------------------------------------------

    def emit_action(
        self,
        action_id: int,
        gx: int,
        gy: int,
    ) -> Tuple[int, int, int]:
        """Pack (action_id, gx, gy) into a 3-int action record via GPU kernel.

        Action ids: 1=Up 2=Down 3=Left 4=Right 5=Perform 6=Click 7=Undo
        For non-Click actions gx/gy are stored for trace completeness.

        Returns:
            (action_id, gx, gy) tuple — round-tripped through GPU for trace.
        """
        d_record = loader.gpu_malloc(3 * ctypes.sizeof(ctypes.c_int32))
        try:
            loader.launch(
                self._kernels["action_emit"],
                grid=(1, 1, 1),
                block=(1, 1, 1),
                params=[
                    ctypes.c_int(action_id),
                    ctypes.c_int(gx),
                    ctypes.c_int(gy),
                    ctypes.c_uint64(d_record.value),
                ],
            )
            loader.synchronize()
            Int3 = ctypes.c_int32 * 3
            h_record = Int3()
            loader.memcpy_dtoh(
                ctypes.cast(h_record, ctypes.c_void_p),
                d_record,
                3 * ctypes.sizeof(ctypes.c_int32),
            )
            return int(h_record[0]), int(h_record[1]), int(h_record[2])
        finally:
            loader.gpu_free(d_record)

    # ------------------------------------------------------------------
    # 0x2A7 — Diff highlight
    # ------------------------------------------------------------------

    def diff_highlight(
        self,
        frame_a_dev: object,
        frame_b_dev: object,
        W: int,
        H: int,
        hi_rgba: int = 0xC0FFFF00,
    ) -> object:
        """Produce a W*H*4 RGBA overlay marking cells that differ between frames.

        Unchanged cells → alpha=0 (transparent). Changed cells → alpha=0xC0
        with the highlight color hi_rgba.

        Args:
            frame_a_dev: CUdeviceptr to uint8[W*H] palette indices (before).
            frame_b_dev: CUdeviceptr to uint8[W*H] palette indices (after).
            W, H: frame dimensions.
            hi_rgba: highlight color packed as (R|G<<8|B<<16|alpha<<24).
                     Default is yellow (0x00FFFF00 but alpha overridden by kernel to 0xC0).

        Returns:
            CUdeviceptr to uint8 RGBA overlay buffer of W*H*4 bytes.
            Caller frees with loader.gpu_free().
        """
        overlay_dev = loader.gpu_malloc(W * H * 4 * ctypes.sizeof(ctypes.c_uint8))
        bx = (W + 15) // 16
        by = (H + 15) // 16
        loader.launch(
            self._kernels["diff_highlight"],
            grid=(bx, by, 1),
            block=(16, 16, 1),
            params=[
                ctypes.c_uint64(frame_a_dev.value),
                ctypes.c_uint64(frame_b_dev.value),
                ctypes.c_uint64(overlay_dev.value),
                ctypes.c_int(W),
                ctypes.c_int(H),
                ctypes.c_uint32(int(hi_rgba) & 0xFFFFFFFF),
            ],
        )
        loader.synchronize()
        return overlay_dev

    # ------------------------------------------------------------------
    # 0x2A8 — Lives HUD
    # ------------------------------------------------------------------

    def lives_hud(
        self,
        Hw: int,
        Hh: int,
        lives_rem: int,
        lives_tot: int,
        moves_rem: int,
        moves_tot: int,
    ) -> object:
        """Render HUD strip: red life squares (top) + blue moves bar (bottom).

        Args:
            Hw, Hh: HUD patch dimensions in pixels.
            lives_rem: lives remaining.
            lives_tot: total lives.
            moves_rem: moves remaining.
            moves_tot: total moves budget.

        Returns:
            CUdeviceptr to uint8 RGBA buffer of Hw*Hh*4 bytes.
            Caller frees with loader.gpu_free().
        """
        hud_dev = loader.gpu_malloc(Hw * Hh * 4 * ctypes.sizeof(ctypes.c_uint8))
        bx = (Hw + 15) // 16
        by = (Hh + 15) // 16
        loader.launch(
            self._kernels["lives_hud"],
            grid=(bx, by, 1),
            block=(16, 16, 1),
            params=[
                ctypes.c_uint64(hud_dev.value),
                ctypes.c_int(Hw),
                ctypes.c_int(Hh),
                ctypes.c_int(lives_rem),
                ctypes.c_int(lives_tot),
                ctypes.c_int(moves_rem),
                ctypes.c_int(moves_tot),
            ],
        )
        loader.synchronize()
        return hud_dev

    # ------------------------------------------------------------------
    # 0x2A6 — Replay step (host-side frame index advance + decode)
    # ------------------------------------------------------------------

    def replay_step(
        self,
        frames_dev: object,
        W: int,
        H: int,
    ) -> Tuple[object, int]:
        """Advance replay cursor and decode current frame.

        Args:
            frames_dev: CUdeviceptr to stacked uint8[N][W*H] palette index buffer.
            W, H: frame dimensions.

        Returns:
            (rgba_dev, frame_idx) — decoded RGBA buffer + current frame index.
            rgba_dev must be freed by caller.
        """
        frame_offset = self._replay_frame_idx * W * H
        # Compute device pointer to current frame slice.
        frame_slice_ptr_val = frames_dev.value + frame_offset
        # Wrap device pointer as CUdeviceptr for decode_frame.
        from knowledge3d.cranium.sovereign.loader import CUdeviceptr
        frame_slice_dev = CUdeviceptr(frame_slice_ptr_val)
        rgba_dev = self.decode_frame(frame_slice_dev, W, H)
        idx = self._replay_frame_idx
        self._replay_frame_idx += 1
        return rgba_dev, idx

    def reset_replay(self) -> None:
        """Reset the replay frame cursor to 0."""
        self._replay_frame_idx = 0

    # ------------------------------------------------------------------
    # 0x2A9 — Game ID bind (host-side only)
    # ------------------------------------------------------------------

    def bind_game_id(self, game_id: str) -> None:
        """Store the current ARC3 game id (hash suffix included).

        No GPU state — the id is used by the trace layer to tag records.

        Args:
            game_id: e.g. "ls20-9607627b"
        """
        self._game_id = str(game_id)

    @property
    def game_id(self) -> Optional[str]:
        """Currently bound ARC3 game id, or None if not yet set."""
        return self._game_id


__all__ = ["Arc3ScreenBridge"]
