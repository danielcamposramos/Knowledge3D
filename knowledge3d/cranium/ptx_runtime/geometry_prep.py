"""Canonical geometry-prep runtime for contour/profile workflows.

This module sits above the sovereign drawing-transform kernels and provides
the mesh-facing metadata needed for 2D -> 3D preparation: bounded crops,
profile occupancy, and silhouette hints that can later feed extrusion/lathe
composition or opcode promotion decisions.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from .drawing_transform_kernels import (
    crop_gpu,
    find_bbox_gpu,
    profile_scan_gpu,
    row_profile_scan_gpu,
    smooth_profile_gpu,
)


def _as_int32_grid(grid: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(grid, dtype=np.int32))
    if arr.ndim != 2:
        raise ValueError(f"expected 2D int32 grid, got shape={arr.shape}")
    return arr


@dataclass(frozen=True)
class BoundingBox:
    min_y: int
    min_x: int
    max_y: int
    max_x: int

    @property
    def is_empty(self) -> bool:
        return self.max_y < self.min_y or self.max_x < self.min_x

    @property
    def height(self) -> int:
        return 0 if self.is_empty else (self.max_y - self.min_y + 1)

    @property
    def width(self) -> int:
        return 0 if self.is_empty else (self.max_x - self.min_x + 1)

    @property
    def area(self) -> int:
        return self.height * self.width

    def clamp(self, *, max_height: int, max_width: int) -> "BoundingBox":
        if self.is_empty:
            return self
        return BoundingBox(
            min_y=max(0, min(self.min_y, max_height - 1)),
            min_x=max(0, min(self.min_x, max_width - 1)),
            max_y=max(0, min(self.max_y, max_height - 1)),
            max_x=max(0, min(self.max_x, max_width - 1)),
        )

    def pad(self, pad: int, *, max_height: int, max_width: int) -> "BoundingBox":
        if self.is_empty:
            return self
        return BoundingBox(
            min_y=max(0, self.min_y - pad),
            min_x=max(0, self.min_x - pad),
            max_y=min(max_height - 1, self.max_y + pad),
            max_x=min(max_width - 1, self.max_x + pad),
        )


@dataclass(frozen=True)
class PreparedProfile:
    source_shape: tuple[int, int]
    bbox: BoundingBox
    region: np.ndarray
    contour_color: int
    nonzero_count: int
    mask_density: float
    aspect_ratio: float
    column_fill: tuple[int, ...]
    row_fill: tuple[int, ...]
    top_contour: tuple[int, ...]
    bottom_contour: tuple[int, ...]
    left_contour: tuple[int, ...]
    right_contour: tuple[int, ...]
    top_contour_smoothed: tuple[int, ...]
    bottom_contour_smoothed: tuple[int, ...]
    left_contour_smoothed: tuple[int, ...]
    right_contour_smoothed: tuple[int, ...]

    @property
    def is_empty(self) -> bool:
        return self.nonzero_count == 0


def _bbox_from_tuple(values: tuple[int, int, int, int]) -> BoundingBox:
    min_y, min_x, max_y, max_x = (int(v) for v in values)
    return BoundingBox(min_y=min_y, min_x=min_x, max_y=max_y, max_x=max_x)


def _occupied_mask(region: np.ndarray, color: int) -> np.ndarray:
    if color == 0:
        return region != 0
    return region == color


class GeometryPrep:
    """Always-on geometry preparation surface for contour/profile workflows."""

    _WARMED_PID: int | None = None
    _WARMUP_REPORT: dict[str, float | int | str] | None = None

    def find_bbox(self, grid: np.ndarray, color: int = 0) -> BoundingBox:
        return _bbox_from_tuple(find_bbox_gpu(_as_int32_grid(grid), color))

    def crop(self, grid: np.ndarray, *, y: int, x: int, h: int, w: int) -> np.ndarray:
        return crop_gpu(_as_int32_grid(grid), y, x, h, w)

    def extract_bbox(self, grid: np.ndarray, color: int = 0, *, pad: int = 0) -> tuple[BoundingBox, np.ndarray]:
        host = _as_int32_grid(grid)
        bbox = self.find_bbox(host, color)
        if bbox.is_empty:
            return bbox, np.zeros((1, 1), dtype=np.int32)
        if pad:
            bbox = bbox.pad(pad, max_height=host.shape[0], max_width=host.shape[1])
        region = crop_gpu(host, bbox.min_y, bbox.min_x, bbox.height, bbox.width)
        return bbox, region

    def prepare_profile(self, grid: np.ndarray, color: int = 0, *, pad: int = 0) -> PreparedProfile:
        host = _as_int32_grid(grid)
        bbox, region = self.extract_bbox(host, color, pad=pad)
        mask = _occupied_mask(region, color)
        nonzero_count = int(np.count_nonzero(mask))
        total_cells = int(mask.size)
        mask_density = float(nonzero_count / total_cells) if total_cells else 0.0
        aspect_ratio = float(region.shape[1] / max(1, region.shape[0]))
        top_scan, bottom_scan, column_scan = profile_scan_gpu(region, color)
        left_scan, right_scan, row_scan = row_profile_scan_gpu(region, color)
        top_smooth = smooth_profile_gpu(top_scan, passes=2, invalid_value=-1)
        bottom_smooth = smooth_profile_gpu(bottom_scan, passes=2, invalid_value=-1)
        left_smooth = smooth_profile_gpu(left_scan, passes=2, invalid_value=-1)
        right_smooth = smooth_profile_gpu(right_scan, passes=2, invalid_value=-1)
        column_fill = tuple(int(v) for v in column_scan.tolist())
        row_fill = tuple(int(v) for v in row_scan.tolist())
        top_contour = tuple(int(v) for v in top_scan.tolist())
        bottom_contour = tuple(int(v) for v in bottom_scan.tolist())
        left_contour = tuple(int(v) for v in left_scan.tolist())
        right_contour = tuple(int(v) for v in right_scan.tolist())
        return PreparedProfile(
            source_shape=tuple(host.shape),
            bbox=bbox,
            region=region,
            contour_color=int(color),
            nonzero_count=nonzero_count,
            mask_density=mask_density,
            aspect_ratio=aspect_ratio,
            column_fill=column_fill,
            row_fill=row_fill,
            top_contour=top_contour,
            bottom_contour=bottom_contour,
            left_contour=left_contour,
            right_contour=right_contour,
            top_contour_smoothed=tuple(int(v) for v in top_smooth.tolist()),
            bottom_contour_smoothed=tuple(int(v) for v in bottom_smooth.tolist()),
            left_contour_smoothed=tuple(int(v) for v in left_smooth.tolist()),
            right_contour_smoothed=tuple(int(v) for v in right_smooth.tolist()),
        )

    def warmup_runtime(self) -> dict[str, float | int | str]:
        current_pid = os.getpid()
        if self.__class__._WARMED_PID == current_pid and self.__class__._WARMUP_REPORT is not None:
            return dict(self.__class__._WARMUP_REPORT)

        sample = np.zeros((24, 24), dtype=np.int32)
        sample[5:18, 9:15] = 1

        total_start = time.perf_counter()
        bbox_start = time.perf_counter()
        bbox = self.find_bbox(sample)
        bbox_ms = (time.perf_counter() - bbox_start) * 1000.0

        extract_start = time.perf_counter()
        extracted_bbox, region = self.extract_bbox(sample, pad=1)
        extract_ms = (time.perf_counter() - extract_start) * 1000.0

        profile_start = time.perf_counter()
        profile = self.prepare_profile(sample, pad=1)
        profile_ms = (time.perf_counter() - profile_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        report: dict[str, float | int | str] = {
            "status": "ready",
            "pid": current_pid,
            "bbox_warmup_ms": bbox_ms,
            "extract_warmup_ms": extract_ms,
            "profile_warmup_ms": profile_ms,
            "total_warmup_ms": total_ms,
            "warm_bbox": [bbox.min_y, bbox.min_x, bbox.max_y, bbox.max_x],
            "warm_region_shape": [int(region.shape[0]), int(region.shape[1])],
            "warm_profile_nonzero": int(profile.nonzero_count),
            "warm_profile_bbox": [
                extracted_bbox.min_y,
                extracted_bbox.min_x,
                extracted_bbox.max_y,
                extracted_bbox.max_x,
            ],
        }
        self.__class__._WARMED_PID = current_pid
        self.__class__._WARMUP_REPORT = dict(report)
        return report


__all__ = [
    "BoundingBox",
    "PreparedProfile",
    "GeometryPrep",
]
