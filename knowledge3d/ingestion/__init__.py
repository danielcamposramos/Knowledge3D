"""
Knowledge ingestion subsystem.

Provides pipelines that transform external datasets into K3D-compatible
representations (Galaxy nodes, House artifacts, Garden growth inputs).
"""

from .language import *  # noqa: F401,F403

__all__ = ["language"]
