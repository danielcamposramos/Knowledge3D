"""Device lesson ring and GPU lesson-application kernel."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import struct
import subprocess

from knowledge3d.cranium.sovereign import loader


LESSON_RECORD_BYTES = 64
LESSON_FAMILY_ID_OFFSET = 0
LESSON_ROUTER_INDEX_OFFSET = 4
LESSON_EXECUTOR_INDEX_OFFSET = 8
LESSON_VALIDATOR_INDEX_OFFSET = 12
LESSON_WINNER_INDEX_OFFSET = 16
LESSON_WINNER_ROLE_OFFSET = 20
LESSON_EXPECTED_HASH_OFFSET = 24
LESSON_PREDICTED_HASH_OFFSET = 32
LESSON_REWARD_OFFSET = 40
LESSON_ANTI_PATTERN_OFFSET = 44
LESSON_ROUTE_DEPTH_OFFSET = 48
LESSON_ROUTE_TRACE_HASH_OFFSET = 56

LESSON_STATS_POSITIVE_STEPS_OFFSET = 0
LESSON_STATS_NEGATIVE_STEPS_OFFSET = 4
LESSON_STATS_ANTI_PATTERN_HITS_OFFSET = 8
LESSON_STATS_LAST_POSITIVE_LOSS_OFFSET = 12
LESSON_STATS_LAST_NEGATIVE_LOSS_OFFSET = 16
LESSON_STATS_BYTES = 32

CUDA_DIR = Path(__file__).resolve().parents[1] / "cranium" / "cuda"
PTX_DIR = Path(__file__).resolve().parents[1] / "cranium" / "ptx"
CUDA_SOURCE = CUDA_DIR / "semantic_lesson_tick.cu"
CUDA_HEADER = CUDA_DIR / "device_functions.cuh"
PTX_PATH = PTX_DIR / "semantic_lesson_tick.ptx"


def _bytes_ptr(payload: bytearray) -> ctypes.c_void_p:
    return ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(payload)))


class VRAMLessonRing:
    """Device-resident lesson ring with append counter and summary stats."""

    def __init__(self, capacity: int = 131_072) -> None:
        self.capacity = max(1, int(capacity))
        self.buffer_bytes = self.capacity * LESSON_RECORD_BYTES
        self.buffer = loader.gpu_malloc(self.buffer_bytes)
        self.counter = loader.gpu_malloc(4)
        self.stats = loader.gpu_malloc(LESSON_STATS_BYTES)
        self.reset()

    def reset(self) -> None:
        zero_buffer = bytearray(self.buffer_bytes)
        zero_counter = bytearray(4)
        zero_stats = bytearray(LESSON_STATS_BYTES)
        loader.memcpy_htod(self.buffer, _bytes_ptr(zero_buffer), len(zero_buffer))
        loader.memcpy_htod(self.counter, _bytes_ptr(zero_counter), len(zero_counter))
        loader.memcpy_htod(self.stats, _bytes_ptr(zero_stats), len(zero_stats))

    def load_stats(self, stats: dict[str, float | int] | None) -> None:
        payload = bytearray(LESSON_STATS_BYTES)
        source = dict(stats or {})
        struct.pack_into(
            "<I",
            payload,
            LESSON_STATS_POSITIVE_STEPS_OFFSET,
            max(0, int(source.get("positive_steps", 0) or 0)),
        )
        struct.pack_into(
            "<I",
            payload,
            LESSON_STATS_NEGATIVE_STEPS_OFFSET,
            max(0, int(source.get("negative_steps", 0) or 0)),
        )
        struct.pack_into(
            "<I",
            payload,
            LESSON_STATS_ANTI_PATTERN_HITS_OFFSET,
            max(0, int(source.get("anti_pattern_hits", 0) or 0)),
        )
        struct.pack_into(
            "<f",
            payload,
            LESSON_STATS_LAST_POSITIVE_LOSS_OFFSET,
            float(source.get("last_positive_loss", 0.0) or 0.0),
        )
        struct.pack_into(
            "<f",
            payload,
            LESSON_STATS_LAST_NEGATIVE_LOSS_OFFSET,
            float(source.get("last_negative_loss", 0.0) or 0.0),
        )
        loader.memcpy_htod(self.stats, _bytes_ptr(payload), len(payload))

    def reset_counter(self) -> None:
        zero_counter = bytearray(4)
        loader.memcpy_htod(self.counter, _bytes_ptr(zero_counter), len(zero_counter))

    def close(self) -> None:
        for attr in ("buffer", "counter", "stats"):
            ptr = getattr(self, attr, None)
            if ptr is not None:
                loader.gpu_free(ptr)
                setattr(self, attr, None)

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def read_count(self) -> int:
        raw = bytearray(4)
        loader.memcpy_dtoh(_bytes_ptr(raw), self.counter, len(raw))
        return int(struct.unpack_from("<I", raw, 0)[0])

    def read_stats(self) -> dict[str, float | int]:
        raw = bytearray(LESSON_STATS_BYTES)
        loader.memcpy_dtoh(_bytes_ptr(raw), self.stats, len(raw))
        return {
            "positive_steps": int(struct.unpack_from("<I", raw, LESSON_STATS_POSITIVE_STEPS_OFFSET)[0]),
            "negative_steps": int(struct.unpack_from("<I", raw, LESSON_STATS_NEGATIVE_STEPS_OFFSET)[0]),
            "anti_pattern_hits": int(struct.unpack_from("<I", raw, LESSON_STATS_ANTI_PATTERN_HITS_OFFSET)[0]),
            "last_positive_loss": float(struct.unpack_from("<f", raw, LESSON_STATS_LAST_POSITIVE_LOSS_OFFSET)[0]),
            "last_negative_loss": float(struct.unpack_from("<f", raw, LESSON_STATS_LAST_NEGATIVE_LOSS_OFFSET)[0]),
        }


class SemanticLessonGPU:
    """Thin sovereign wrapper that applies device lesson records to star priors."""

    def __init__(self) -> None:
        self.kernel = loader.load_ptx_file(str(self.ensure_ptx()), "semantic_lesson_tick")

    @staticmethod
    def ensure_ptx() -> Path:
        PTX_DIR.mkdir(parents=True, exist_ok=True)
        newest_source_mtime = max(
            CUDA_SOURCE.stat().st_mtime,
            CUDA_HEADER.stat().st_mtime if CUDA_HEADER.exists() else 0.0,
        )
        if PTX_PATH.exists() and PTX_PATH.stat().st_mtime >= newest_source_mtime:
            return PTX_PATH
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("nvcc_not_found_for_semantic_lesson_tick")
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

    def apply(self, star_table, lesson_ring: VRAMLessonRing) -> int:
        galaxy_ptr = getattr(star_table, "gpu_ptr", None) if star_table is not None else None
        galaxy_star_count = int(getattr(star_table, "star_count", 0) or 0) if star_table is not None else 0
        total_lessons = min(int(lesson_ring.read_count()), int(lesson_ring.capacity))
        if total_lessons <= 0 or galaxy_ptr is None or int(galaxy_star_count) <= 0:
            return 0
        loader.launch(
            self.kernel,
            ((total_lessons + 127) // 128, 1, 1),
            (128, 1, 1),
            [
                galaxy_ptr,
                ctypes.c_uint(int(galaxy_star_count)),
                lesson_ring.buffer,
                ctypes.c_uint(int(total_lessons)),
                lesson_ring.stats,
            ],
        )
        loader.synchronize()
        lesson_ring.reset_counter()
        return total_lessons


__all__ = [
    "LESSON_RECORD_BYTES",
    "LESSON_STATS_BYTES",
    "SemanticLessonGPU",
    "VRAMLessonRing",
]
