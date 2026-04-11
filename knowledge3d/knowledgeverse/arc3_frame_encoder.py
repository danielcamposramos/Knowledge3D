"""Encode ARC-AGI-3 frames to 64-float embeddings on GPU."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import struct
import subprocess

from knowledge3d.cranium.sovereign import loader


CUDA_DIR = Path(__file__).resolve().parents[1] / "cranium" / "cuda"
PTX_DIR = Path(__file__).resolve().parents[1] / "cranium" / "ptx"
CUDA_SOURCE = CUDA_DIR / "arc3_frame_encoder.cu"
CUDA_HEADER = CUDA_DIR / "device_functions.cuh"
PTX_PATH = PTX_DIR / "arc3_frame_encoder.ptx"
EMBEDDING_DIMS = 64


def _bytes_ptr(payload: bytearray) -> ctypes.c_void_p:
    return ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(payload)))


class ARC3FrameEncoder:
    def __init__(self) -> None:
        self.kernel = loader.load_ptx_file(str(self._ensure_ptx()), "arc3_encode_frame")

    @staticmethod
    def _ensure_ptx() -> Path:
        PTX_DIR.mkdir(parents=True, exist_ok=True)
        newest = max(
            CUDA_SOURCE.stat().st_mtime,
            CUDA_HEADER.stat().st_mtime if CUDA_HEADER.exists() else 0.0,
        )
        if PTX_PATH.exists() and PTX_PATH.stat().st_mtime >= newest:
            return PTX_PATH
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("nvcc_not_found_for_arc3_frame_encoder")
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

    def encode(self, frame: list[list[int]]) -> list[float]:
        height = len(frame)
        width = len(frame[0]) if height > 0 else 0
        if height == 0 or width == 0:
            return [0.0] * EMBEDDING_DIMS
        for row in frame:
            if len(row) != width:
                raise ValueError("arc3_frame_must_be_rectangular")

        flat = bytearray()
        for row in frame:
            flat.extend((int(cell) & 0xFF) for cell in row)

        frame_gpu = loader.gpu_malloc(len(flat))
        embed_gpu = loader.gpu_malloc(EMBEDDING_DIMS * 4)
        try:
            loader.memcpy_htod(frame_gpu, _bytes_ptr(flat), len(flat))
            loader.launch(
                self.kernel,
                (1, 1, 1),
                (1, 1, 1),
                [frame_gpu, embed_gpu, ctypes.c_uint(width), ctypes.c_uint(height)],
            )
            embed_host = bytearray(EMBEDDING_DIMS * 4)
            loader.memcpy_dtoh(_bytes_ptr(embed_host), embed_gpu, len(embed_host))
            return list(struct.unpack(f"<{EMBEDDING_DIMS}f", embed_host))
        finally:
            loader.gpu_free(frame_gpu)
            loader.gpu_free(embed_gpu)


__all__ = ["ARC3FrameEncoder"]
