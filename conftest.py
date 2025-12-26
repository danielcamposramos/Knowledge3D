"""
Repository-level pytest configuration.

This file exists primarily to keep `pytest` collection runnable in minimal /
CPU-only environments by skipping optional-dependency test modules early
(*before* they import GPU/vision/audio dependencies).
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from typing import Any


def _module_available(module_name: str) -> bool:
    # Avoid importing optional dependencies at collection time; some modules
    # (notably GPU stacks) can trigger driver/library initialization.
    return find_spec(module_name) is not None


_HAS_CUPY = _module_available("cupy")
_HAS_CV2 = _module_available("cv2")
_HAS_SOUNDFILE = _module_available("soundfile")
_HAS_WIKIPEDIAAPI = _module_available("wikipediaapi")


def _cuda_context_available() -> bool:
    if not _HAS_CUPY:
        return False
    try:
        import cupy as cp  # type: ignore

        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


_PROBE_CUDA = os.environ.get("K3D_PYTEST_PROBE_CUDA", "0").strip() == "1"
_HAS_CUDA = _cuda_context_available() if _PROBE_CUDA else False


def pytest_ignore_collect(collection_path: Any, config: Any) -> bool:  # noqa: ANN401
    """
    Skip optional-dependency test modules when the dependency isn't available.

    Note: `tests/conftest.py` has additional fixtures; this repo-level hook is
    needed because `knowledge3d/cranium/tests/*` sits outside the `tests/` tree.
    """
    path_str = str(collection_path)

    # GPU / CuPy-only tests.
    if not (_HAS_CUPY and _HAS_CUDA) and "knowledge3d/cranium/tests/" in path_str:
        return True

    # Optional deps: audio / vision / wikipedia ingestion tests (may live outside `tests/`).
    if not _HAS_SOUNDFILE and path_str.endswith("test_step15_audio_sovereign.py"):
        return True
    if not _HAS_CV2 and path_str.endswith("test_step15_visual_sovereign.py"):
        return True
    if not _HAS_WIKIPEDIAAPI and path_str.endswith("test_step15_wikipedia_benchmark.py"):
        return True

    return False
