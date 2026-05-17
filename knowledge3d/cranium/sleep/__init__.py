"""Sleep-time consolidation utilities.

Post-purge surface (2026-04-18):
    ``model_sleep`` (ModelSleepCycle) and ``knowledge_sleep``
    (KnowledgeSleepCycle) were moved to ``Old_Attempts/2026-04-18/``
    per ``TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md`` §4.1.
    The ``glyph_consolidator`` helper was also moved.

    Daniel's ruling (2026-04-18): no Python fallbacks anywhere — including
    sleep-time — see ``feedback_no_fallbacks_ever_including_sleeptime.md``.
    The sovereign successors are:

        knowledge3d/cranium/ptx/sleep_cluster_refiner.ptx
        knowledge3d/cranium/ptx/sleep_glyph_consolidator.ptx
        knowledge3d/cranium/ptx/sleep_time_micro.ptx
        knowledge3d/cranium/ptx/sleep_perf_consumer.ptx

    Surviving Python surface is the idle detector and scheduler only —
    the consolidation work itself runs on-GPU through those kernels.
"""

from .scheduler import SleepScheduler  # noqa: F401
from .memory_pressure_trigger import MemoryPressureSnapshot  # noqa: F401

__all__ = [
    "SleepScheduler",
    "MemoryPressureSnapshot",
]
