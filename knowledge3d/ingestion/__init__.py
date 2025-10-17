"""
Knowledge ingestion subsystem.

Provides pipelines that transform external datasets into K3D-compatible
representations (Galaxy nodes, House artifacts, Garden growth inputs).

Modules are imported lazily to avoid pulling optional dependencies (e.g. librosa)
when only a subset of pipelines are required.
"""

from importlib import import_module
from typing import Any

__all__ = ["language", "lexicons", "documents"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
