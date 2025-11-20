"""
Procedural video texture generator.

Generates deterministic procedural frames from compact numeric seeds using a
small set of patterns (Perlin-style noise, Voronoi cells, fractal patterns) and
maps the result to colour via a palette derived from the seed itself.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np


class ProceduralVideoGenerator:
    """
    Generate video frames procedurally from compact seeds.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = int(width)
        self.height = int(height)

    def _rng_from_seed(self, seed: np.ndarray) -> np.random.Generator:
        """Deterministically derive RNG from seed contents."""
        seed_bytes = np.asarray(seed, dtype=np.float32).tobytes()
        digest = hashlib.sha256(seed_bytes).digest()
        seed_int = int.from_bytes(digest[:8], "little", signed=False)
        return np.random.default_rng(seed_int)

    def generate_frame(self, seed: np.ndarray, time_param: float = 0.0) -> np.ndarray:
        """
        Generate video frame from procedural seed.

        Args:
            seed: RPN embedding (1D, 64D-2048D suggested).
            time_param: Animation time in [0, 1] range (looping).

        Returns:
            frame: RGB image (height, width, 3) uint8.
        """
        if time_param < 0 or time_param > 1:
            raise ValueError("time_param must be in [0, 1]")
        seed_arr = np.asarray(seed, dtype=np.float32).flatten()
        if seed_arr.size == 0:
            raise ValueError("seed must not be empty")

        rng = self._rng_from_seed(seed_arr)
        pattern_selector = int(abs(seed_arr[0]) * 10) % 3

        u, v = np.meshgrid(
            np.linspace(0.0, 1.0, self.width, endpoint=False, dtype=np.float32),
            np.linspace(0.0, 1.0, self.height, endpoint=False, dtype=np.float32),
        )

        # Animate by offsetting coordinates.
        offset = (time_param % 1.0) * 2.0
        u = (u + offset) % 1.0
        v = (v + offset) % 1.0

        if pattern_selector == 0:
            pattern = self.perlin_noise(u, v, seed_arr[:16])
        elif pattern_selector == 1:
            pattern = self.voronoi_cells(u, v, seed_arr[:16])
        else:
            pattern = self.fractal_pattern(u, v, seed_arr[:16])

        palette = self._palette_from_seed(seed_arr, rng)
        frame = self.map_to_color(pattern, palette)
        return frame.astype(np.uint8)

    def perlin_noise(self, u: np.ndarray, v: np.ndarray, seed_params: np.ndarray) -> np.ndarray:
        """
        Generate value-noise (Perlin-like) texture.

        Args:
            u, v: Normalised coordinates (0-1), shape (H, W).
            seed_params: Parameters from RPN seed.
        """
        rng = self._rng_from_seed(seed_params)
        freq = int(max(1, min(16, abs(seed_params[0]) * 6 + 4)))
        grid_shape = (freq + 1, freq + 1)
        grid = rng.random(grid_shape, dtype=np.float32)

        x = u * freq
        y = v * freq
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.clip(x0 + 1, 0, freq)
        y1 = np.clip(y0 + 1, 0, freq)

        sx = x - x0
        sy = y - y0

        def fade(t: np.ndarray) -> np.ndarray:
            return t * t * t * (t * (t * 6 - 15) + 10)

        sx_f = fade(sx)
        sy_f = fade(sy)

        v00 = grid[y0, x0]
        v10 = grid[y0, x1]
        v01 = grid[y1, x0]
        v11 = grid[y1, x1]

        ix0 = v00 + sx_f * (v10 - v00)
        ix1 = v01 + sx_f * (v11 - v01)
        noise = ix0 + sy_f * (ix1 - ix0)
        return np.clip(noise, 0.0, 1.0).astype(np.float32)

    def voronoi_cells(self, u: np.ndarray, v: np.ndarray, seed_params: np.ndarray) -> np.ndarray:
        """Generate Voronoi cell texture."""
        rng = self._rng_from_seed(seed_params)
        num_points = int(max(2, min(32, abs(seed_params[1]) * 10 + 8)))
        points = rng.random((num_points, 2), dtype=np.float32)

        coords = np.stack([u, v], axis=-1)
        # Compute squared distances to all points.
        diff = coords[:, :, None, :] - points[None, None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        min_dist = np.min(dist_sq, axis=-1)
        # Normalise distance to [0, 1] using maximum possible distance (sqrt(2)).
        norm = np.sqrt(np.clip(min_dist, 0.0, 2.0))
        return np.clip(norm / np.sqrt(2.0), 0.0, 1.0).astype(np.float32)

    def fractal_pattern(self, u: np.ndarray, v: np.ndarray, seed_params: np.ndarray) -> np.ndarray:
        """Generate a simple Mandelbrot-like fractal pattern."""
        max_iter = int(max(10, min(64, abs(seed_params[2]) * 20 + 20)))
        scale = 1.5 + abs(seed_params[3]) * 1.5
        cx = (u - 0.5) * scale - 0.5
        cy = (v - 0.5) * scale
        x = np.zeros_like(cx)
        y = np.zeros_like(cy)
        mask = np.ones_like(cx, dtype=bool)
        iter_counts = np.zeros_like(cx, dtype=np.float32)

        for i in range(max_iter):
            x_new = x * x - y * y + cx
            y_new = 2 * x * y + cy
            x, y = x_new, y_new
            escaped = (x * x + y * y) > 4.0
            newly_escaped = escaped & mask
            iter_counts[newly_escaped] = i
            mask = mask & (~escaped)
            if not mask.any():
                break

        iter_counts[mask] = max_iter
        norm = iter_counts / float(max_iter)
        return np.clip(norm, 0.0, 1.0).astype(np.float32)

    def map_to_color(self, grayscale: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """
        Map grayscale values to RGB via palette interpolation.

        Args:
            grayscale: Values (0-1), shape (H, W).
            palette: RGB colors, shape (N, 3) in [0, 255].
        """
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError("palette must have shape (N, 3)")
        gray = np.clip(np.asarray(grayscale, dtype=np.float32), 0.0, 1.0)
        num_colors = palette.shape[0]
        if num_colors < 2:
            raise ValueError("palette must contain at least two colors")

        scaled = gray * (num_colors - 1)
        idx0 = np.floor(scaled).astype(int)
        idx1 = np.clip(idx0 + 1, 0, num_colors - 1)
        t = (scaled - idx0).astype(np.float32)

        c0 = palette[idx0]
        c1 = palette[idx1]
        rgb = (c0 * (1.0 - t)[..., None] + c1 * t[..., None]).astype(np.uint8)
        return rgb

    def _palette_from_seed(self, seed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Derive a small palette from the seed for deterministic colours."""
        if seed.size < 9:
            seed = np.pad(seed, (0, 9 - seed.size), constant_values=0.0)
        colors = []
        for i in range(0, 9, 3):
            base = seed[i : i + 3]
            color = (np.abs(np.sin(base * 12.9898 + i)) * 255.0).astype(np.float32)
            jitter = rng.normal(0.0, 5.0, size=3).astype(np.float32)
            colors.append(np.clip(color + jitter, 0.0, 255.0))
        return np.stack(colors, axis=0).astype(np.float32)
